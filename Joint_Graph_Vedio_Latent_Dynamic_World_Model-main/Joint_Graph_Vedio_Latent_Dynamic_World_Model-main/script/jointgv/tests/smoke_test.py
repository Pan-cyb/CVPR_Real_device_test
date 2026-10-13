from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv import (
    GradientRatioOTController,
    JointGraphVideoWorldModel,
    MDNOutput,
)
from jointgv.blocks import ExplicitGraphObserver
from jointgv.model import compute_loss, compute_sequence_loss
from jointgv.online import DreamerBehavior, SlidingWindowReplay
from train_online_dmc import (
    behavior_losses,
    critic_loss_from_cache,
    dreamer_online_loss,
    policy_actions,
)


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay = SlidingWindowReplay(sequence_length=4, capacity=100)
    replay.start_with_action_dim(0, np.zeros((8, 8, 3), dtype=np.uint8), 2)
    for step in range(10):
        replay.append(
            0,
            np.full((8, 8, 3), step, dtype=np.uint8),
            np.zeros(2, dtype=np.float32),
            1.0 if step == 5 else 0.0,
            False,
        )
    positive_batch = replay.sample(
        16,
        torch.device("cpu"),
        positive_fraction=1.0,
    )
    assert (positive_batch["rewards"][:, 1:] > 0).any(dim=1).all()
    assert (positive_batch["replay_group"] == 1).all()
    replay.start_with_action_dim(
        1, np.zeros((8, 8, 3), dtype=np.uint8), 2
    )
    for step in range(8):
        replay.append(
            1,
            np.full((8, 8, 3), step, dtype=np.uint8),
            np.asarray([1.0, -1.0], dtype=np.float32),
            0.0,
            False,
        )
    stratified_batch = replay.sample(
        8,
        torch.device("cpu"),
        positive_fraction=0.5,
        high_action_zero_fraction=0.5,
        high_action_threshold=0.6,
    )
    assert int((stratified_batch["replay_group"] == 1).sum()) == 4
    assert int((stratified_batch["replay_group"] == 2).sum()) == 4

    # Dynamic OT balancing measures both objectives on exactly the same
    # parameters, reaches the analytic gradient ratio, and round-trips its
    # EMA/update state for deterministic resume.
    balance_p = torch.nn.Parameter(torch.tensor(1.0, device=device))
    balance_q = torch.nn.Parameter(torch.tensor(2.0, device=device))
    balance = GradientRatioOTController(
        initial_weight=0.1,
        target_ratio=0.1,
        ema_decay=0.0,
        update_every=1,
        min_weight=0.01,
        max_weight=10.0,
    )
    assert balance.step(
        (2.0 * balance_p + balance_q).square(),
        (balance_p - balance_q).square(),
        (balance_p, balance_q),
    )
    expected_weight = 0.1 * (320.0 / 8.0) ** 0.5
    assert abs(balance.weight - expected_weight) < 1e-6
    balance_diagnostics = balance.diagnostics(warmup_scale=0.5)
    assert abs(balance_diagnostics["ot_to_base_grad_ratio"] - 0.05) < 1e-6
    restored_balance = GradientRatioOTController(
        initial_weight=0.1,
        target_ratio=0.1,
        ema_decay=0.0,
        update_every=1,
        min_weight=0.01,
        max_weight=10.0,
    )
    restored_balance.load_state_dict(balance.state_dict())
    assert restored_balance.state_dict() == balance.state_dict()

    model = JointGraphVideoWorldModel(
        action_dim=3,
        token_dim=32,
        node_dim=32,
        ot_dim=16,
        rnn_dim=64,
        mixtures=5,
    ).to(device)
    images = torch.rand(2, 5, 3, 64, 64, device=device)
    actions = torch.rand(2, 4, 3, device=device) * 2 - 1

    output = model.forward_sequence(images, actions, sample_vae=False)
    graph_targets = {
        "node_state": torch.rand(2, 5, 8, 13, device=device),
        "node_mask": torch.ones(2, 5, 8, device=device),
        "relation": torch.randint(0, 4, (2, 5, 8, 8), device=device),
    }
    loss = compute_sequence_loss(
        model, output, images, graph_targets=graph_targets
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    mdn = output["mdn"]
    assert isinstance(mdn, MDNOutput)
    assert mdn.logmix.shape == (2, 4, 16, 32, 5)
    assert output["predicted_image"].shape == (2, 4, 3, 64, 64)
    assert output["predicted_tokens"].shape == (2, 4, 16, 32)
    assert output["observed_relation_logits"].shape == (2, 5, 8, 8, 4)
    assert output["graph_attention"].shape == (2, 5, 8, 16)
    assert output["delta_g"].abs().max().item() <= 0.25 + 1e-6
    assert output["correction"].abs().max().item() <= 0.25 + 1e-6
    assert 0.0 <= loss.ot.item() <= 4.0 + 1e-5
    assert loss.ot_variance.item() >= 0.0
    assert model.graph_plugin.z_ot.bias is None
    assert model.graph_plugin.g_ot.bias is None
    assert model.graph_plugin.ot_centering is True

    uncentered_model = JointGraphVideoWorldModel(
        action_dim=3,
        token_dim=32,
        node_dim=32,
        graph_nodes=8,
        ot_dim=16,
        rnn_dim=64,
        ot_centering=False,
    )
    assert uncentered_model.graph_plugin.ot_centering is False

    expanded_correction_model = JointGraphVideoWorldModel(
        action_dim=3,
        token_dim=32,
        node_dim=32,
        graph_nodes=8,
        ot_dim=16,
        rnn_dim=64,
        max_graph_correction=0.5,
        graph_correction_gate_init=0.25,
    )
    assert torch.allclose(
        torch.sigmoid(expanded_correction_model.graph_plugin.correction_gate_logit),
        torch.tensor(0.25),
    )
    for name in (
        "ot_cross",
        "ot_self_z",
        "ot_self_g",
        "ot_z_proj_rms",
        "ot_g_proj_rms",
        "ot_z_dispersion",
        "ot_g_dispersion",
        "ot_z_pair_cos",
        "ot_g_pair_cos",
        "delta_z_rms",
        "delta_g_rms",
        "delta_z_dispersion",
        "delta_g_dispersion",
        "delta_z_pair_cos",
        "delta_g_pair_cos",
        "graph_context_rms",
        "graph_correction_gate",
    ):
        assert name in output and torch.isfinite(output[name]).all(), name
    assert -1.0 - 1e-5 <= output["ot_z_pair_cos"].item() <= 1.0 + 1e-5
    assert -1.0 - 1e-5 <= output["ot_g_pair_cos"].item() <= 1.0 + 1e-5
    assert mdn.state[0].shape == (1, 2, 64)
    assert torch.isfinite(loss.total)
    assert any(parameter.grad is not None for parameter in model.mdn_lstm.parameters())
    assert any(parameter.grad is not None for parameter in model.graph_dynamics.parameters())
    assert any(parameter.grad is not None for parameter in model.graph_builder.parameters())
    optimizer.step()

    # The explicit observer has a dependency-free debug backbone so its role
    # queries, temporal recurrence, VTransE head, and one-round IMP interface
    # can be checked without loading the 766 MB RelTR checkpoint in CI.
    explicit = ExplicitGraphObserver(
        node_dim=32,
        nodes=8,
        backbone="debug_conv",
        pretrained_backbone=False,
        freeze_backbone=True,
    ).to(device)
    explicit_output = explicit.forward_sequence(images, sample=False)
    assert explicit_output["nodes"].shape == (2, 5, 8, 32)
    assert explicit_output["relation_logits"].shape == (2, 5, 8, 8, 4)
    assert explicit_output["positions_2d"].shape == (2, 5, 8, 2)
    assert explicit_output["attention"].shape == (2, 5, 8, 256)
    assert torch.isfinite(explicit_output["state"]).all()
    explicit_model = JointGraphVideoWorldModel(
        action_dim=3,
        token_dim=32,
        node_dim=32,
        ot_dim=16,
        rnn_dim=64,
        graph_observer="explicit",
        graph_backbone="debug_conv",
        graph_backbone_pretrained=False,
        graph_role_names=(
            "base",
            "link_1",
            "link_2",
            "end_effector",
            "target",
            "padding",
            "padding",
            "padding",
        ),
    ).to(device)
    explicit_joint_output = explicit_model.forward_sequence(
        images, actions, sample_vae=False
    )
    assert explicit_joint_output["graph_attention"].shape == (2, 5, 8, 256)
    assert torch.isfinite(explicit_joint_output["ot_loss"])
    assert explicit_model.graph_plugin.active_graph_node_count == 5
    assert torch.count_nonzero(
        explicit_joint_output["delta_g"][..., 5:, :]
    ).item() == 0
    assert torch.equal(
        explicit_joint_output["graph_next"][..., 5:, :],
        explicit_joint_output["graph"][..., 5:, :],
    )
    assert torch.equal(
        explicit_joint_output["next_relation_logits"][..., 5:, :, :],
        explicit_joint_output["relation_logits"][..., 5:, :, :],
    )
    assert torch.equal(
        explicit_joint_output["next_relation_logits"][..., :, 5:, :],
        explicit_joint_output["relation_logits"][..., :, 5:, :],
    )

    # Projection normalization and bounded graph updates must remain finite
    # even when an upstream latent increment has an extreme magnitude.
    with torch.no_grad():
        stress_delta = torch.randn_like(output["base_delta"]) * 1e6
        stress = model.graph_plugin(
            stress_delta,
            actions,
            output["graph"],
            output["relation_logits"],
        )
        stress_correction = model.graph_plugin.correct(
            stress_delta, stress.context
        )
    assert torch.isfinite(stress.ot_loss)
    assert torch.isfinite(stress.ot_variance_loss)
    assert 0.0 <= stress.ot_loss.item() <= 4.0 + 1e-5
    assert stress.delta_g.abs().max().item() <= 0.25 + 1e-6
    assert stress.context.abs().max().item() <= 1.0 + 1e-6
    assert stress_correction.abs().max().item() <= 0.25 + 1e-6

    # After one update graph-on changes mixture means, while graph-off must
    # preserve the unmodified MDN distribution exactly.
    graph_on = model.forward_sequence(images, actions, sample_vae=False)
    graph_off = model.forward_sequence(images, actions, use_graph=False, sample_vae=False)
    graph_on_mdn = graph_on["mdn"]
    graph_off_mdn = graph_off["mdn"]
    assert isinstance(graph_on_mdn, MDNOutput) and isinstance(graph_off_mdn, MDNOutput)
    assert not torch.equal(graph_on["conditioned_mean"], graph_on_mdn.mean)
    assert torch.equal(graph_off["conditioned_mean"], graph_off_mdn.mean)

    rollout = model.rollout(images[:, 0], actions, deterministic=True)
    assert rollout["tokens"].shape == (2, 4, 16, 32)
    assert rollout["graphs"].shape == (2, 4, 8, 32)
    assert rollout["relations"].shape == (2, 4, 8, 8, 4)
    assert rollout["images"].shape == (2, 4, 3, 64, 64)
    one_step = model(images[:, 0], actions[:, 0], sample_vae=False)
    one_step_loss = compute_loss(model, one_step, images[:, 0], images[:, 1])
    assert torch.isfinite(one_step_loss.total)

    burchim_images = images[:1, :3]
    burchim_actions = actions[:1, :2]
    control_targets = {
        "reward": torch.rand(1, 2, device=device),
        "continuation": torch.ones(1, 2, device=device),
    }
    dreamer = JointGraphVideoWorldModel(
        action_dim=3,
        backbone="dreamerv3",
        node_dim=32,
        ot_dim=16,
        max_graph_correction=0.5,
        graph_correction_gate_init=0.25,
    ).to(device)
    dreamer_output = dreamer.forward_sequence(
        burchim_images, burchim_actions, sample_vae=False
    )
    dreamer_loss = compute_sequence_loss(
        dreamer,
        dreamer_output,
        burchim_images,
        targets=control_targets,
        graph_supervision_weight=0.0,
    )
    dreamer_loss.total.backward()
    assert dreamer.backbone_name == "dreamerv3"
    assert dreamer_output["backbone"] == "dreamerv3"
    assert not dreamer.visual_is_variational
    assert dreamer_output["reward_prediction"].shape == (1, 2)
    assert dreamer_output["predicted_tokens"].shape == (1, 2, 1536)
    assert dreamer_output["base_delta"].shape == (1, 2, 1536)
    assert dreamer_output["graph_context"].shape == (1, 2, 32)
    assert dreamer.graph_dynamics_plugin.alignment_mode == "global"
    assert not hasattr(dreamer.dreamerv3_core, "latent_delta_head")
    assert dreamer_output["delta_g"].abs().max().item() <= 0.25 + 1e-6
    assert 0.0 <= dreamer_loss.ot.item() <= 4.0 + 1e-5
    assert dreamer_loss.ot_variance.item() >= 0.0
    assert dreamer.dreamerv3_core.graph_plugin.z_ot.bias is None
    assert dreamer.dreamerv3_core.graph_plugin.g_ot.bias is None
    assert torch.allclose(
        torch.sigmoid(dreamer.dreamerv3_core.graph_gate_logit),
        torch.tensor(0.25, device=device),
    )
    assert torch.isfinite(dreamer_output["prior_deter_rms"])
    assert torch.isfinite(dreamer_output["graph_correction_ratio"])
    assert dreamer_output["graph_correction_ratio"].item() >= 0.0
    assert (
        dreamer.graph_dynamics_plugin
        is dreamer.dreamerv3_core.graph_plugin
    )
    collapsed = torch.zeros(2, 8, 16, device=device)
    assert (
        dreamer.dreamerv3_core.graph_plugin._variance_floor_loss(collapsed)
        > 0
    )
    assert torch.isfinite(dreamer_output["ot_cross"])
    assert torch.isfinite(dreamer_output["ot_z_dispersion"])
    assert torch.isfinite(dreamer_output["ot_g_dispersion"])
    assert torch.isfinite(dreamer_output["graph_correction"]).all()
    assert 0.0 <= dreamer_output["graph_pool_attention_entropy"].item() <= 1.0
    assert torch.isfinite(dreamer_loss.total)
    assert torch.isfinite(dreamer_loss.representation_kl_raw)
    dreamer_rollout = dreamer.rollout(
        burchim_images[:, 0], burchim_actions, deterministic=True, decode=False
    )
    assert dreamer_rollout["tokens"].shape == (1, 2, 1536)
    assert dreamer_rollout["rewards"].shape == (1, 2)
    assert any(
        parameter.grad is not None
        for parameter in dreamer.dreamerv3_core.rssm.parameters()
    )

    graph_off = dreamer.forward_sequence(
        burchim_images,
        burchim_actions,
        use_graph=False,
        sample_vae=False,
    )
    assert "graph_next" not in graph_off

    # One online DreamerV3 world-model + actor--critic step and one policy
    # action through the train_online_dmc.py helpers, graph off and on.
    online_actions = torch.cat(
        (torch.zeros_like(burchim_actions[:, :1]), burchim_actions), dim=1
    )
    online_firsts = torch.zeros(1, 3, device=device)
    online_firsts[:, 0] = 1.0
    online_rewards = torch.zeros(1, 3, device=device)
    online_dones = torch.zeros(1, 3, device=device)
    frame = burchim_images[0, 0].mul(255.0).round().byte().cpu().numpy()
    for online_graph in (False, True):
        dreamer.zero_grad(set_to_none=True)
        online = dreamer.dreamerv3_core.forward_online_sequence(
            burchim_images, online_actions, online_firsts, use_graph=online_graph
        )
        world_loss, _ = dreamer_online_loss(
            online,
            burchim_images,
            online_rewards,
            online_dones,
            use_graph=online_graph,
            graph_scale=1.0,
            alignment_scale=1.0,
            alignment_weight=0.1,
            variance_weight=0.1,
        )
        world_loss.backward()
        behavior = DreamerBehavior(3, dreamer.dreamerv3_core.feat_dim).to(device)
        actor_loss, critic_cache, _ = behavior_losses(
            dreamer,
            behavior,
            online,
            online_dones,
            use_graph=online_graph,
            horizon=2,
            gamma=0.997,
            lambda_td=0.95,
            entropy_weight=3e-4,
            slow_regularization=1.0,
        )
        actor_loss.backward()
        critic_loss = critic_loss_from_cache(behavior, critic_cache, 1.0)
        critic_loss.backward()
        assert torch.isfinite(world_loss)
        assert torch.isfinite(actor_loss) and torch.isfinite(critic_loss)
        env_actions, *_ = policy_actions(
            dreamer,
            behavior,
            [frame],
            device,
            use_graph=online_graph,
            recurrent=[None],
            previous_actions=None,
            graph_states=[None],
            relation_states=[None],
        )
        assert env_actions.shape == (1, 3)

    vae_dreamer = JointGraphVideoWorldModel(
        action_dim=3,
        backbone="dreamerv3_vae",
        token_dim=32,
        node_dim=32,
        ot_dim=16,
    ).to(device)
    vae_output = vae_dreamer.forward_sequence(
        burchim_images, burchim_actions, sample_vae=False
    )
    vae_loss = compute_sequence_loss(
        vae_dreamer,
        vae_output,
        burchim_images,
        targets=control_targets,
        graph_supervision_weight=0.0,
    )
    vae_loss.total.backward()
    assert vae_dreamer.backbone_name == "dreamerv3_vae"
    assert vae_output["backbone"] == "dreamerv3_vae"
    assert vae_dreamer.visual_is_variational
    assert vae_output["predicted_tokens"].shape == (1, 2, 1536)
    assert vae_output["predicted_image"].shape == (1, 2, 3, 64, 64)
    assert torch.isfinite(vae_loss.total)
    vae_rollout = vae_dreamer.rollout(
        burchim_images[:, 0], burchim_actions, deterministic=True
    )
    assert vae_rollout["images"].shape == (1, 2, 3, 64, 64)

    # dino_model="debug_conv" is a tiny untrained stand-in for DINOv2 so this
    # test never touches the network; real runs always use a pretrained
    # torch.hub facebookresearch/dinov2 variant instead.
    dino = JointGraphVideoWorldModel(
        action_dim=3,
        backbone="dino_wm",
        token_dim=32,
        node_dim=32,
        ot_dim=16,
        dino_model="debug_conv",
        dino_image_size=32,
        dino_predictor_depth=2,
        dino_predictor_heads=4,
        dino_max_frames=4,
    ).to(device)
    dino_output = dino.forward_sequence(
        burchim_images, burchim_actions, sample_vae=False
    )
    dino.eval()
    dino.set_graph_conditioning_strength(0.0)
    dino_zero_graph = dino.forward_sequence(
        burchim_images, burchim_actions, sample_vae=False
    )
    dino_graph_off = dino.forward_sequence(
        burchim_images, burchim_actions, use_graph=False, sample_vae=False
    )
    assert torch.allclose(
        dino_zero_graph["predicted_tokens"],
        dino_graph_off["predicted_tokens"],
        atol=1e-6,
    )
    dino.set_graph_conditioning_strength(1.0)
    dino.train()
    dino_loss = compute_sequence_loss(
        dino,
        dino_output,
        burchim_images,
        targets=control_targets,
        graph_supervision_weight=0.0,
    )
    dino_loss.total.backward()
    assert dino.backbone_name == "dino_wm"
    assert dino_output["backbone"] == "dino_wm"
    assert not dino.visual_is_variational
    assert dino.graph_dynamics_plugin.alignment_mode == "sinkhorn"
    assert dino.rollout_latent_matches_visual_tokens
    assert dino_output["predicted_tokens"].shape == (1, 2, 16, 32)
    assert dino_output["base_delta"].shape == (1, 2, 16, 32)
    assert dino_output["predicted_image"].shape == (1, 2, 3, 64, 64)
    assert torch.isfinite(dino_loss.total)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in dino.backbone.predictor.parameters()
    )
    dino.zero_grad(set_to_none=True)
    dino_rollout = dino.rollout(
        burchim_images[:, 0], burchim_actions, deterministic=True
    )
    assert dino_rollout["tokens"].shape == (1, 2, 16, 32)
    assert dino_rollout["images"].shape == (1, 2, 3, 64, 64)

    print(
        "David Ha + DreamerV3 native/VAE + DINO-WM smoke test passed:",
        {
            "david_total": round(loss.total.item(), 5),
            "dreamerv3_total": round(dreamer_loss.total.item(), 5),
            "dreamerv3_vae_total": round(vae_loss.total.item(), 5),
            "dino_wm_total": round(dino_loss.total.item(), 5),
            "ot": round(loss.ot.item(), 5),
        },
        f"device={device}",
    )


if __name__ == "__main__":
    main()
