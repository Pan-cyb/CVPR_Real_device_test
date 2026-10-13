"""Pre-flight check for --graph-pose-correction / --correction-unroll-horizon.

Runs a tiny random DINO-WM model (debug_conv encoder, proxy graph observer)
for a few seconds and asserts:

1. During the warm-up delay (conditioning strength 0) the DINO predictor
   receives exactly the same gradient as in a graph-off pass, even with every
   new loss switched on.
2. The correction-unroll loss reaches only the correction's own parameters:
   no gradient in the DINO predictor, graph dynamics, observer, OT
   projections, or position head.
3. The position loss trains the position head.
4. Pose delivery uses the DINO row-major patch order with x = column,
   y = row (the graph_node_pixel convention).
5. rollout() runs with context frames, graph on and off.

Uses random weights and data only; it trains nothing and reads no files.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv import JointGraphVideoWorldModel, compute_sequence_loss
from jointgv.losses import _correction_unroll


def build_model(device: torch.device) -> JointGraphVideoWorldModel:
    model = JointGraphVideoWorldModel(
        action_dim=3,
        backbone="dino_wm",
        token_dim=32,
        node_dim=32,
        ot_dim=16,
        dino_model="debug_conv",
        dino_image_size=32,
        dino_predictor_depth=2,
        dino_predictor_heads=4,
        dino_max_frames=16,
        graph_history=4,
        graph_pose_correction=True,
        max_graph_correction=1.0,
    ).to(device)
    model.vae.requires_grad_(False)
    # The correction's output layer starts at zero, which would make every
    # upstream correction gradient zero; perturb it so the checks are real.
    torch.nn.init.normal_(
        model.graph_plugin.pose_correction_mlp[-1].weight, std=0.05
    )
    return model


def batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    frames, nodes = 9, 8
    images = torch.rand(2, frames, 3, 64, 64, device=device)
    actions = torch.rand(2, frames - 1, 3, device=device) * 2 - 1
    targets = {
        "reward": torch.rand(2, frames - 1, device=device),
        "continuation": torch.ones(2, frames - 1, device=device),
        "node_state": torch.randn(2, frames, nodes, 13, device=device),
        "node_mask": torch.ones(2, frames, nodes, device=device),
        "relation": torch.randint(0, 4, (2, frames, nodes, nodes), device=device),
        "node_pixel": torch.rand(2, frames, nodes, 2, device=device) * 2 - 1,
        "node_visible": torch.ones(2, frames, nodes, device=device),
    }
    return images, actions, targets


def grads(module: torch.nn.Module) -> list[torch.Tensor]:
    return [
        parameter.grad.detach().clone()
        if parameter.grad is not None
        else torch.zeros_like(parameter)
        for parameter in module.parameters()
    ]


def is_zero(module: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool((parameter.grad == 0).all())
        for parameter in module.parameters()
    )


def has_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in module.parameters()
    )


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    plugin = model.graph_plugin
    images, actions, targets = batch(device)
    # eval() only disables dropout so the two passes are comparable;
    # gradients are computed as usual.
    model.eval()

    # 1. Delay: backbone gradient identical to graph-off.
    model.set_graph_conditioning_strength(0.0)
    output = model.forward_sequence(
        images,
        actions,
        sample_vae=False,
        graph_multistep_horizon=3,
        correction_unroll_horizon=3,
    )
    loss = compute_sequence_loss(
        model,
        output,
        images,
        targets=targets,
        ot_weight=0.0,
        ot_variance_weight=0.0,
        vae_reconstruction_weight=0.0,
        beta_kl=0.0,
        graph_weight=1.0,
        graph_supervision_weight=1.0,
        motion_weighted_transition=True,
        graph_multistep_embedding_weight=1.0,
        graph_multistep_state_weight=1.0,
        graph_position_weight=1.0,
        correction_unroll_weight=1.0,
    )
    model.zero_grad(set_to_none=True)
    loss.total.backward()
    graph_on = grads(model.backbone)
    assert has_grad(plugin.position_head), "position loss did not train the head"
    assert has_grad(plugin.pose_correction_mlp), "unroll did not train the correction"
    assert float(loss.correction_unroll.detach()) > 0
    assert float(loss.graph_position.detach()) > 0

    off = model.forward_sequence(images, actions, use_graph=False, sample_vae=False)
    off_loss = compute_sequence_loss(
        model,
        off,
        images,
        targets=targets,
        ot_weight=0.0,
        ot_variance_weight=0.0,
        vae_reconstruction_weight=0.0,
        beta_kl=0.0,
        graph_weight=0.0,
        graph_supervision_weight=0.0,
        motion_weighted_transition=True,
    )
    model.zero_grad(set_to_none=True)
    off_loss.total.backward()
    graph_off = grads(model.backbone)
    worst = max(
        float((a - b).abs().max()) for a, b in zip(graph_on, graph_off)
    )
    # Only float summation-order noise is acceptable.
    assert all(
        torch.allclose(a, b, rtol=1e-5, atol=1e-8)
        for a, b in zip(graph_on, graph_off)
    ), f"backbone gradient differs from graph-off by {worst}"
    print(f"[1] delay: DINO predictor gradient identical to graph-off (max diff {worst:.1e})")

    # 2. Unroll loss reaches only the correction.
    model.zero_grad(set_to_none=True)
    output = model.forward_sequence(
        images, actions, sample_vae=False, correction_unroll_horizon=3
    )
    unroll, unroll_base = _correction_unroll(output, True)
    unroll.backward()
    for name, module in (
        ("DINO predictor", model.backbone),
        ("graph dynamics", plugin.dynamics),
        ("observer", plugin.builder),
        ("OT z projection", plugin.z_ot),
        ("OT g projection", plugin.g_ot),
        ("OT value", plugin.g_value),
        ("position head", plugin.position_head),
    ):
        assert is_zero(module), f"unroll loss leaked into {name}"
    for name, module in (
        ("correction MLP", plugin.pose_correction_mlp),
        ("per-token gate", plugin.pose_gate),
        ("pose payload", plugin.pose_value),
    ):
        assert has_grad(module), f"unroll loss did not reach {name}"
    print(
        "[2] unroll: only correction MLP / gate / pose payload get gradient "
        f"(corrected={float(unroll):.4f} uncorrected={float(unroll_base):.4f})"
    )

    # 3. Pose delivery geometry: node 0 at the top-right corner.
    side = int(round(model.token_count**0.5))
    original = plugin.node_position
    plugin.node_position = lambda nodes: torch.cat(
        (
            torch.tensor([0.8, -0.8], device=nodes.device).expand(
                *nodes.shape[:-2], 1, 2
            ),
            torch.full(
                (*nodes.shape[:-2], nodes.shape[-2] - 1, 2), 5.0, device=nodes.device
            ),
        ),
        dim=-2,
    )
    with torch.no_grad():
        pose = plugin.pose_context(torch.randn(1, 8, 32, device=device))
    plugin.node_position = original
    strongest = int(pose.norm(dim=-1)[0].argmax())
    assert strongest == side - 1, f"expected patch {side - 1}, got {strongest}"
    print(f"[3] pose goes to patch {strongest} = row 0, column {side - 1} (top-right)")

    # 4. Rollouts.
    model.set_graph_conditioning_strength(1.0)
    with torch.no_grad():
        for use_graph in (True, False):
            rollout = model.rollout(
                images[:, 3],
                actions[:, 3:8],
                use_graph=use_graph,
                context_images=images[:, :3],
            )
            assert rollout["tokens"].shape == (2, 5, model.token_count, 32)
            assert rollout["images"].shape == (2, 5, 3, 64, 64)
            assert torch.isfinite(rollout["tokens"]).all()
    print("[4] rollout with context frames runs, graph on and off")

    # 5. Plug-in (P) at full strength with every loss on, OT included:
    #    the backbone gradient is still exactly the graph-off gradient.
    def full_loss(output, ot_weight: float, plugin_weight: float):
        return compute_sequence_loss(
            model,
            output,
            images,
            targets=targets,
            ot_weight=ot_weight,
            ot_variance_weight=ot_weight,
            vae_reconstruction_weight=0.0,
            beta_kl=0.0,
            graph_weight=1.0,
            graph_supervision_weight=1.0,
            motion_weighted_transition=True,
            graph_multistep_embedding_weight=1.0,
            graph_multistep_state_weight=1.0,
            graph_position_weight=1.0,
            correction_unroll_weight=1.0,
            plugin_correction_weight=plugin_weight,
        )

    model.plugin_correction = True
    model.set_graph_conditioning_strength(1.0)
    model.zero_grad(set_to_none=True)
    output = model.forward_sequence(
        images,
        actions,
        sample_vae=False,
        graph_multistep_horizon=3,
        correction_unroll_horizon=3,
    )
    loss = full_loss(output, ot_weight=0.1, plugin_weight=1.0)
    loss.total.backward()
    plugin_grads = grads(model.backbone)
    worst = max(float((a - b).abs().max()) for a, b in zip(plugin_grads, graph_off))
    assert all(
        torch.allclose(a, b, rtol=1e-5, atol=1e-8)
        for a, b in zip(plugin_grads, graph_off)
    ), f"plug-in backbone gradient differs from graph-off by {worst}"
    assert float(loss.plugin_correction.detach()) > 0
    assert has_grad(plugin.pose_correction_mlp), "plug-in loss did not train the correction"
    assert has_grad(plugin.z_ot), "OT is not trained in plug-in mode"
    print(
        "[5] plug-in, full strength, OT on: DINO gradient identical to graph-off "
        f"(max diff {worst:.1e}); correction and OT still train"
    )

    # 6. J: --correction-backbone-weight w scales only the corrected
    #    prediction's gradient into DINO (linear in w), forward unchanged.
    model.plugin_correction = False
    backbone_grads = {}
    predictions = {}
    for weight in (0.0, 0.25, 1.0):
        model.correction_backbone_weight = weight
        model.zero_grad(set_to_none=True)
        output = model.forward_sequence(images, actions, sample_vae=False)
        full_loss(output, ot_weight=0.0, plugin_weight=0.0).total.backward()
        backbone_grads[weight] = grads(model.backbone)
        predictions[weight] = output["predicted_tokens"].detach()
    model.correction_backbone_weight = 1.0
    assert all(
        torch.allclose(predictions[weight], predictions[1.0], atol=1e-6)
        for weight in (0.0, 0.25)
    ), "correction-backbone-weight changed the forward pass"
    assert all(
        torch.allclose(g1 - g025, 0.75 * (g1 - g0), rtol=1e-4, atol=1e-7)
        for g0, g025, g1 in zip(
            backbone_grads[0.0], backbone_grads[0.25], backbone_grads[1.0]
        )
    ), "correction-backbone-weight does not scale the correction path linearly"
    assert any(
        not torch.allclose(g0, g1) for g0, g1 in zip(backbone_grads[0.0], backbone_grads[1.0])
    ), "correction path carries no backbone gradient at w=1"
    print("[6] J: backbone gradient = own terms + w x correction path; forward unchanged")

    # 7. Plug-in mode does not change what a rollout computes.
    with torch.no_grad():
        joint_rollout = model.rollout(
            images[:, 3], actions[:, 3:8], use_graph=True, context_images=images[:, :3]
        )
        model.plugin_correction = True
        plugin_rollout = model.rollout(
            images[:, 3], actions[:, 3:8], use_graph=True, context_images=images[:, :3]
        )
        backbone_rollout = model.rollout(
            images[:, 3], actions[:, 3:8], use_graph=False, context_images=images[:, :3]
        )
        model.plugin_correction = False
    assert torch.allclose(plugin_rollout["tokens"], joint_rollout["tokens"], atol=1e-5)
    assert not torch.allclose(plugin_rollout["tokens"], backbone_rollout["tokens"])
    print("[7] plug-in rollout = corrected rollout (and differs from backbone only)")

    # 8. Backbone history in rollouts (visual_context): the first step is
    #    exactly the teacher-forced prediction from the same 4 real frames.
    with torch.no_grad():
        teacher = model.forward_sequence(
            images[:, :5], actions[:, :4], use_graph=False, sample_vae=False
        )
        seeded = model.rollout(
            images[:, 3],
            actions[:, 3:8],
            use_graph=False,
            decode=False,
            context_images=images[:, :3],
            context_actions=actions[:, :3],
            visual_context=True,
        )
        plain = model.rollout(
            images[:, 3],
            actions[:, 3:8],
            use_graph=False,
            decode=False,
            context_images=images[:, :3],
        )
    assert torch.allclose(
        seeded["tokens"][:, 0], teacher["predicted_tokens"][:, 3], atol=1e-5
    ), "visual context does not reproduce the teacher-forced prediction"
    assert not torch.allclose(seeded["tokens"][:, 0], plain["tokens"][:, 0])
    print(
        "[8] visual context: first rollout step = teacher-forced prediction "
        "from the same 4 real frames (and differs from the 1-frame start)"
    )
    print(f"all checks passed on {device}")


if __name__ == "__main__":
    main()
