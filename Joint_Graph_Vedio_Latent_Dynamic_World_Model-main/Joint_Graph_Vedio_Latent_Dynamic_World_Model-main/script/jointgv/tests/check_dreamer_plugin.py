"""Pre-flight check for the DreamerV3 plug-in (P) path.

Builds a random native-vision DreamerV3 world model with the plug-in head
(proxy graph observer, 4-frame graph history, pose correction) and asserts:

1. P mode, every plug-in loss on (graph, supervision, OT, K-step graph,
   position, one-step corrected KL, correction unroll): every backbone
   parameter -- encoder, RSSM (including the prior head mlp_img2), decoder,
   reward and continuation heads -- receives the gradient of a graph-off
   pass (within the noise of repeating the graph-off pass itself; the
   world-model losses are bitwise equal), and the correction still trains.
2. The correction-unroll loss reaches only the correction's own parameters.
3. The position loss trains the position head and never the RSSM.
4. Deterministic rollouts: at correction strength 0 the graph-on rollout is
   identical to graph off; at strength 1 it differs.
5. Visual context: the backbone observes the context frames first, which
   changes the prediction relative to a one-frame start.
6. The historical path is untouched: a model without the plug-in head has
   none of its parameters and still runs its joint graph-on pass.

Uses random weights and data only; it trains nothing and reads no files.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv import JointGraphVideoWorldModel, compute_sequence_loss
from jointgv.losses import _graph_position

FRAMES, NODES, ACTIONS = 9, 8, 6


def build_model(device: torch.device, plugin: bool = True) -> JointGraphVideoWorldModel:
    model = JointGraphVideoWorldModel(
        action_dim=ACTIONS,
        backbone="dreamerv3",
        node_dim=32,
        ot_dim=16,
        graph_nodes=NODES,
        graph_observer="proxy",
        graph_history=4 if plugin else 1,
        graph_pose_correction=plugin,
        graph_global_correction=plugin,
        max_graph_correction=0.5,
    ).to(device)
    if plugin:
        head = model.dreamerv3_core.graph_plugin
        # Zero-initialized output layers would make every upstream
        # correction gradient zero; perturb them so the checks are real.
        torch.nn.init.normal_(head.global_correction_mlp[-1].weight, std=0.05)
        torch.nn.init.normal_(head.global_correction_gate[-1].weight, std=0.05)
    return model


def batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    images = torch.rand(2, FRAMES, 3, 64, 64, device=device)
    actions = torch.rand(2, FRAMES - 1, ACTIONS, device=device) * 2 - 1
    targets = {
        "reward": torch.rand(2, FRAMES - 1, device=device),
        "continuation": torch.ones(2, FRAMES - 1, device=device),
        "node_state": torch.randn(2, FRAMES, NODES, 13, device=device),
        "node_mask": torch.ones(2, FRAMES, NODES, device=device),
        "relation": torch.randint(0, 4, (2, FRAMES, NODES, NODES), device=device),
        "node_pixel": torch.rand(2, FRAMES, NODES, 2, device=device) * 2 - 1,
        "node_visible": torch.ones(2, FRAMES, NODES, device=device),
    }
    return images, actions, targets


def backbone_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    """Everything except the plug-in and the unused historical graph head."""
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith(
            (
                "dreamerv3_core.graph_plugin.",
                "dreamerv3_core.graph_to_deter.",
                "dreamerv3_core.graph_gate_logit",
            )
        )
    ]


def grads(parameters) -> list[torch.Tensor]:
    return [
        parameter.grad.detach().clone()
        if parameter.grad is not None
        else torch.zeros_like(parameter)
        for _, parameter in parameters
    ]


def worst_relative(
    parameters, first: list[torch.Tensor], second: list[torch.Tensor]
) -> tuple[float, str]:
    """Largest per-tensor max|a-b| / max|b| and the parameter it occurs at."""
    worst, where = 0.0, ""
    for (name, _), a, b in zip(parameters, first, second):
        scale = float(b.abs().max()) or 1.0
        difference = float((a - b).abs().max()) / scale
        if difference > worst:
            worst, where = difference, name
    return worst, where


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


def loss_of(model, output, images, targets, plugin_on: bool):
    weight = 1.0 if plugin_on else 0.0
    return compute_sequence_loss(
        model,
        output,
        images,
        targets=targets,
        graph_weight=0.2 * weight,
        graph_supervision_weight=weight,
        ot_weight=0.1 * weight,
        ot_variance_weight=0.1 * weight,
        beta_kl=0.0,
        graph_multistep_embedding_weight=weight,
        graph_multistep_state_weight=weight,
        graph_position_weight=weight,
        correction_unroll_weight=weight,
        plugin_correction_weight=weight,
    )


def main() -> None:
    # Deterministic cuDNN keeps the gradient comparison in [1] at the
    # float-noise level instead of cuDNN's atomic-add accumulation order.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    core = model.dreamerv3_core
    plugin = core.graph_plugin
    images, actions, targets = batch(device)
    model.plugin_correction = True
    model.set_graph_conditioning_strength(1.0)
    backbone = backbone_parameters(model)

    # 1. P mode: backbone gradient identical to graph-off.
    torch.manual_seed(1)
    output = model.forward_sequence(
        images,
        actions,
        use_graph=True,
        sample_vae=False,
        graph_multistep_horizon=3,
        correction_unroll_horizon=4,
    )
    loss = loss_of(model, output, images, targets, plugin_on=True)
    model.zero_grad(set_to_none=True)
    loss.total.backward()
    plugin_grads = grads(backbone)
    for name in (
        "plugin_correction",
        "correction_unroll",
        "graph_position",
        "graph_multistep_embedding",
        "ot",
    ):
        assert float(getattr(loss, name).detach()) > 0, f"{name} is zero"
    for name, module in (
        ("global correction MLP", plugin.global_correction_mlp),
        ("per-dimension gate", plugin.global_correction_gate),
        ("pose payload", plugin.pose_value),
        ("graph dynamics", plugin.dynamics),
        ("position head", plugin.position_head),
    ):
        assert has_grad(module), f"P losses did not reach {name}"

    def graph_off_pass():
        torch.manual_seed(1)
        off = model.forward_sequence(
            images, actions, use_graph=False, sample_vae=False
        )
        off_loss = loss_of(model, off, images, targets, plugin_on=False)
        model.zero_grad(set_to_none=True)
        off_loss.total.backward()
        return off_loss, grads(backbone)

    off_loss, off_grads = graph_off_pass()
    # The same graph-off pass twice: GPU accumulation noise (e.g. cuDNN
    # weight gradients of the first convolution over the full image).
    _, repeat_grads = graph_off_pass()
    noise, noise_name = worst_relative(backbone, repeat_grads, off_grads)
    worst, worst_name = worst_relative(backbone, plugin_grads, off_grads)
    # A real leak (e.g. graph losses reaching the encoder) is orders of
    # magnitude above this; the earlier proxy-token leak was ~1e-2.
    tolerance = max(10.0 * noise, 1e-6)
    assert worst <= tolerance, (
        f"backbone gradient differs from graph-off by {worst:.1e} (relative) "
        f"at {worst_name}; graph-off repeated differs by {noise:.1e} "
        f"at {noise_name}"
    )
    world_terms = ("reconstruction", "transition", "reward", "continuation")
    for name in world_terms:
        assert torch.equal(getattr(loss, name), getattr(off_loss, name)), name
    print(
        "[1] P mode, every plug-in loss on: encoder/RSSM/prior head/decoder/"
        f"heads gradient = graph-off (max relative diff {worst:.1e}; the "
        f"graph-off pass repeated differs by {noise:.1e}); world-model losses "
        "identical; correction, gate, pose payload, graph dynamics and "
        "position head train"
    )

    # 2. Unroll loss reaches only the correction.
    model.zero_grad(set_to_none=True)
    output = model.forward_sequence(
        images, actions, use_graph=True, sample_vae=False, correction_unroll_horizon=4
    )
    output["correction_unroll_loss"].backward()
    for name, module in (
        ("RSSM", core.rssm),
        ("encoder", core.visual),
        ("decoder", core.decoder),
        ("graph dynamics", plugin.dynamics),
        ("observer", plugin.builder),
        ("alignment z projection", plugin.z_ot),
        ("alignment g projection", plugin.g_ot),
        ("position head", plugin.position_head),
    ):
        assert is_zero(module), f"unroll loss leaked into {name}"
    for name, module in (
        ("global correction MLP", plugin.global_correction_mlp),
        ("per-dimension gate", plugin.global_correction_gate),
        ("pose payload", plugin.pose_value),
    ):
        assert has_grad(module), f"unroll loss did not reach {name}"
    print(
        "[2] unroll: only correction MLP / gate / pose payload get gradient "
        f"(corrected={float(output['correction_unroll_loss'].detach()):.3f} "
        f"uncorrected={float(output['correction_unroll_base_loss']):.3f})"
    )

    # 3. Position loss trains the position head.
    model.zero_grad(set_to_none=True)
    output = model.forward_sequence(images, actions, use_graph=True, sample_vae=False)
    _graph_position(output, targets).backward()
    assert has_grad(plugin.position_head), "position loss did not train the head"
    assert is_zero(core.rssm), "position loss leaked into the RSSM"
    print("[3] position loss trains the position head (never the RSSM)")

    # 4. Deterministic rollouts: strength 0 equals graph off.
    model.eval()
    with torch.no_grad():
        start = 3
        kwargs = {"context_images": images[:, :start]}
        model.set_graph_conditioning_strength(0.0)
        zero = model.rollout(images[:, start], actions[:, start:8], use_graph=True, **kwargs)
        off = model.rollout(images[:, start], actions[:, start:8], use_graph=False, **kwargs)
        model.set_graph_conditioning_strength(1.0)
        full = model.rollout(images[:, start], actions[:, start:8], use_graph=True, **kwargs)
    assert zero["images"].shape == (2, 5, 3, 64, 64)
    assert torch.equal(zero["images"], off["images"]), "strength 0 != graph off"
    assert torch.equal(zero["tokens"], off["tokens"]), "strength 0 != graph off"
    assert not torch.equal(full["tokens"], off["tokens"]), "correction had no effect"
    assert full["graphs"].shape[:2] == (2, 5)
    print(
        "[4] deterministic rollout: strength 0 identical to graph off, "
        "strength 1 differs"
    )

    # 5. Visual context.
    with torch.no_grad():
        context = {
            "context_images": images[:, :start],
            "context_actions": actions[:, :start],
        }
        with_context = model.rollout(
            images[:, start], actions[:, start:8], use_graph=True,
            visual_context=True, **context,
        )
        again = model.rollout(
            images[:, start], actions[:, start:8], use_graph=True,
            visual_context=True, **context,
        )
    assert torch.equal(with_context["tokens"], again["tokens"]), "not deterministic"
    assert not torch.equal(with_context["tokens"], full["tokens"]), (
        "visual context did not change the start state"
    )
    print("[5] visual context: RSSM observes the context frames first (deterministic)")

    # 6. Historical path untouched.
    legacy = build_model(device, plugin=False)
    new_names = [
        name
        for name, _ in legacy.named_parameters()
        if name.startswith(
            tuple(
                f"dreamerv3_core.graph_plugin.{key}"
                for key in ("global_correction", "pose_value", "position_head")
            )
        )
    ]
    assert not new_names, f"plug-in parameters in a historical model: {new_names}"
    assert not legacy.dreamerv3_core.global_correction
    legacy_output = legacy.forward_sequence(
        images, actions, use_graph=True, sample_vae=False
    )
    assert "graph_correction" in legacy_output
    assert "plugin_corrected_kl" not in legacy_output
    print("[6] historical DreamerV3 path: no plug-in parameters, joint pass unchanged")
    print(f"all checks passed on {device}")


if __name__ == "__main__":
    main()
