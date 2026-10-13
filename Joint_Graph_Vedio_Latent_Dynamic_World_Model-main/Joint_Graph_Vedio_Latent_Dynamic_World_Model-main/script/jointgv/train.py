from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jointgv import (
    GradientRatioOTController,
    JointGraphVideoWorldModel,
    JointLoss,
    compute_sequence_loss,
)
from jointgv.backbones import resolve_backbone_adapter
from jointgv.data import EpisodeSequences
from jointgv.dmc_protocol import (
    EXPLICIT_GRAPH_SCHEMA_VERSION,
    validate_dmc_dataset,
)
from jointgv.rollout_metrics import (
    gaussian_window,
    load_reference_observer,
    rollout_scores,
)
from jointgv.world_model import REMOVED_CONFIG_KEYS


LOSS_NAMES = tuple(field.name for field in fields(JointLoss))
OT_DIAGNOSTIC_NAMES = (
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
    "graph_correction_rms",
    "prior_deter_rms",
    "base_transition_rms",
    "graph_correction_ratio",
    "graph_correction_gate",
    "graph_pool_attention_entropy",
    "ot_weighted_contribution",
    "ot_variance_weighted_contribution",
    "ot_projector_grad_norm",
    "dynamic_ot_weight",
    "effective_ot_weight",
    "base_shared_grad_norm",
    "ot_shared_grad_norm",
    "base_shared_grad_ema",
    "ot_shared_grad_ema",
    "ot_to_base_grad_ratio",
    "ot_balance_updates",
    "weighted_ot_fraction",
    "backbone_objective",
    "jointgv_weighted_contribution",
)
# Gradient-norm clip of the plug-in's graph and correction groups under
# --separate-graph-grad-clip (the DINO-WM value, shared by every backbone).
PLUGIN_GRAD_CLIP = 1.0

CRITICAL_ACTIVATIONS = (
    "predicted_tokens",
    "base_predicted_tokens",
    "graph_next",
    "delta_g",
    "graph_context",
    "ot_loss",
    "ot_variance_loss",
    "reward_prediction",
    "continuation_logit",
    "z_mu",
    "z_logvar",
)


class MetricAverages:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weight = 0

    def update(self, values: dict[str, float], weight: int) -> None:
        self.weight += weight
        for name, value in values.items():
            self.sums[name] = self.sums.get(name, 0.0) + value * weight

    def compute(self) -> dict[str, float]:
        if self.weight == 0:
            raise RuntimeError("Cannot average an empty epoch")
        return {
            name: value / self.weight for name, value in self.sums.items()
        }


def linear_warmup(epoch: int, warmup_epochs: int, delay_epochs: int = 0) -> float:
    if warmup_epochs <= 0:
        return 1.0
    if epoch <= delay_epochs:
        return 0.0
    return min(1.0, (epoch - delay_epochs) / warmup_epochs)


def loss_scalars(loss: JointLoss) -> dict[str, float]:
    return {
        name: float(getattr(loss, name).detach().item())
        for name in LOSS_NAMES
    }


def ot_diagnostic_scalars(
    output: dict[str, Any],
    *,
    ot_scale: float,
    ot_weight: float,
    ot_variance_weight: float,
) -> dict[str, float]:
    """Detach observational OT statistics without changing the objective."""
    values: dict[str, float] = {}
    for name in OT_DIAGNOSTIC_NAMES:
        value = output.get(name)
        if isinstance(value, Tensor) and value.numel() == 1:
            values[name] = float(value.detach().item())
    correction = output.get("graph_correction")
    if isinstance(correction, Tensor):
        values["graph_correction_rms"] = float(
            correction.detach().square().mean().sqrt().item()
        )
    ot_loss = output.get("ot_loss")
    if isinstance(ot_loss, Tensor):
        weighted_ot = ot_loss.detach() * (ot_weight * ot_scale)
        values["ot_weighted_contribution"] = float(weighted_ot.item())
        values["effective_ot_weight"] = float(ot_weight * ot_scale)
    ot_variance_loss = output.get("ot_variance_loss")
    if isinstance(ot_variance_loss, Tensor):
        values["ot_variance_weighted_contribution"] = float(
            (
                ot_variance_loss.detach()
                * (ot_variance_weight * ot_scale)
            ).item()
        )
    invalid = [
        name for name, value in values.items() if not math.isfinite(value)
    ]
    if invalid:
        raise FloatingPointError(
            "Non-finite OT diagnostics: " + ", ".join(invalid)
        )
    return values


def ot_projector_grad_norm(
    model: JointGraphVideoWorldModel,
) -> float | None:
    """Gradient norm received by z_ot/g_ot through all active paths."""
    plugin = model.graph_dynamics_plugin
    gradients = [
        parameter.grad
        for module in (plugin.z_ot, plugin.g_ot)
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return None
    squared_norm = sum(
        float(gradient.detach().double().square().sum().item())
        for gradient in gradients
    )
    return math.sqrt(squared_norm)


def ot_balance_shared_parameters(
    model: JointGraphVideoWorldModel,
) -> tuple[torch.nn.Parameter, ...]:
    """Parameters potentially shared by the backbone and raw OT paths.

    The controller later keeps only parameters for which both objectives
    produce a gradient.  Keeping that intersection at measurement time is
    important because several Graph Plugin heads are private to one path.
    """
    plugin = model.graph_dynamics_plugin
    modules: list[torch.nn.Module] = [
        plugin.dynamics,
        plugin.z_ot,
        plugin.g_ot,
        plugin.g_value,
        plugin.correction,
    ]
    if model.dreamerv3_core is not None:
        modules.append(model.dreamerv3_core.graph_to_deter)
    unique: dict[int, torch.nn.Parameter] = {}
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                unique[id(parameter)] = parameter
    return tuple(unique.values())


@dataclass(frozen=True)
class GraphLossConfig:
    """Graph-side loss settings; defaults reproduce the historical loss."""

    embedding_weight: float = 0.2
    multistep_horizon: int = 0
    multistep_weight: float = 1.0
    position_weight: float = 0.0
    correction_unroll_horizon: int = 0
    correction_unroll_weight: float = 1.0
    plugin: bool = False


@dataclass(frozen=True)
class VisualMultistepConfig:
    """K-step self-fed training of the visual backbone's own prediction."""

    horizon: int = 0
    context: int = 3
    weight: float = 1.0


GRAPH_LOSS_TERMS = (
    "graph",
    "graph_supervision",
    "graph_multistep_embedding",
    "graph_multistep_state",
    "graph_position",
    "correction_unroll",
    "plugin_correction",
)


def graph_loss_weights(
    *,
    use_graph: bool,
    has_graph_targets: bool,
    graph_scale: float,
    graph_supervision_weight: float,
    config: GraphLossConfig,
) -> dict[str, float]:
    """Weights of every graph-side term, shared by the loss and its split.

    The K-step terms mirror the single-step pair (embedding vs observed
    embedding, decoded state vs MuJoCo state), scaled by multistep_weight.
    The correction-unroll term trains only the correction (backbone and
    graph run without gradient there). None of these terms reach the
    visual backbone, so all follow the undelayed graph_scale ramp.
    """
    if not use_graph:
        return dict.fromkeys(GRAPH_LOSS_TERMS, 0.0)
    supervision = (
        graph_supervision_weight * graph_scale if has_graph_targets else 0.0
    )
    embedding = config.embedding_weight * graph_scale
    multistep = config.multistep_weight if config.multistep_horizon > 0 else 0.0
    unroll = (
        config.correction_unroll_weight
        if config.correction_unroll_horizon > 0
        else 0.0
    )
    return {
        "graph": embedding,
        "graph_supervision": supervision,
        "graph_multistep_embedding": multistep * embedding,
        "graph_multistep_state": multistep * supervision,
        "graph_position": (
            config.position_weight * graph_scale if has_graph_targets else 0.0
        ),
        "correction_unroll": unroll * graph_scale,
        # Plug-in: the corrected prediction's loss, at the weight the joint
        # path gives it; reaches only the correction.
        "plugin_correction": 1.0 if config.plugin else 0.0,
    }


def split_backbone_and_jointgv_objectives(
    loss: JointLoss,
    *,
    use_graph: bool,
    graph_scale: float,
    ot_scale: float,
    ot_weight: float,
    ot_variance_weight: float,
    has_graph_targets: bool,
    graph_supervision_weight: float,
    graph_config: GraphLossConfig = GraphLossConfig(),
) -> tuple[Tensor, Tensor]:
    """Separate original world-model terms from explicit JointGV losses."""
    if not use_graph:
        return loss.total, loss.total.new_zeros(())
    weights = graph_loss_weights(
        use_graph=use_graph,
        has_graph_targets=has_graph_targets,
        graph_scale=graph_scale,
        graph_supervision_weight=graph_supervision_weight,
        config=graph_config,
    )
    jointgv = (
        (ot_weight * ot_scale) * loss.ot
        + (ot_variance_weight * ot_scale) * loss.ot_variance
    )
    for name, weight in weights.items():
        jointgv = jointgv + weight * getattr(loss, name)
    return loss.total - jointgv, jointgv


def ensure_finite_activations(
    output: dict[str, Any],
    *,
    epoch: int,
    batch_id: int,
    max_abs: float,
) -> None:
    tensors: list[tuple[str, Tensor]] = []
    for name in CRITICAL_ACTIVATIONS:
        value = output.get(name)
        if isinstance(value, Tensor):
            tensors.append((name, value))
    if not tensors:
        raise RuntimeError("No critical activations were produced")
    finite = torch.stack(
        [torch.isfinite(value).all() for _, value in tensors]
    ).all()
    if not bool(finite.item()):
        invalid = [
            name
            for name, value in tensors
            if not bool(torch.isfinite(value).all().item())
        ]
        raise FloatingPointError(
            f"Non-finite activations at epoch={epoch} batch={batch_id}: "
            + ", ".join(invalid)
        )
    largest = max(
        float(value.detach().abs().amax().item())
        for _, value in tensors
        if value.numel()
    )
    if largest > max_abs:
        raise FloatingPointError(
            f"Activation magnitude {largest:.3e} exceeds {max_abs:.3e} "
            f"at epoch={epoch} batch={batch_id}"
        )


def ensure_finite_loss(
    values: dict[str, float],
    *,
    epoch: int,
    batch_id: int,
    max_total: float,
) -> None:
    invalid = [name for name, value in values.items() if not math.isfinite(value)]
    if invalid:
        raise FloatingPointError(
            f"Non-finite losses at epoch={epoch} batch={batch_id}: "
            + ", ".join(invalid)
        )
    if abs(values["total"]) > max_total:
        raise FloatingPointError(
            f"Total loss {values['total']:.6f} exceeds safety limit "
            f"{max_total:.6f} at epoch={epoch} batch={batch_id}"
        )


def ensure_finite_gradients(
    model: JointGraphVideoWorldModel,
    *,
    epoch: int,
    batch_id: int,
) -> None:
    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all().item())
    ]
    if invalid:
        preview = ", ".join(invalid[:12])
        if len(invalid) > 12:
            preview += f", ... (+{len(invalid) - 12} more)"
        raise FloatingPointError(
            f"Non-finite parameter gradients at epoch={epoch} "
            f"batch={batch_id}: {preview}"
        )


def move_batch(
    batch: tuple[Tensor, Tensor, dict[str, Tensor]], device: torch.device
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    images, actions, targets = batch
    return (
        images.to(device),
        actions.to(device),
        {key: value.to(device) for key, value in targets.items()},
    )


def compute_batch_loss(
    model: JointGraphVideoWorldModel,
    batch: tuple[Tensor, Tensor, dict[str, Tensor]],
    device: torch.device,
    *,
    use_graph: bool,
    graph_scale: float,
    ot_scale: float,
    ot_weight: float,
    ot_variance_weight: float,
    has_graph_targets: bool,
    graph_supervision_weight: float,
    vae_kl_weight: float,
    vae_reconstruction_weight: float,
    sample_vae: bool,
    epoch: int,
    batch_id: int,
    max_loss: float,
    max_activation: float,
    motion_weighted_transition: bool = False,
    graph_config: GraphLossConfig = GraphLossConfig(),
    visual_config: VisualMultistepConfig = VisualMultistepConfig(),
) -> tuple[JointLoss, int, dict[str, float]]:
    images, actions, targets = move_batch(batch, device)
    output = model.forward_sequence(
        images,
        actions,
        use_graph=use_graph,
        sample_vae=sample_vae,
        graph_multistep_horizon=(
            graph_config.multistep_horizon if use_graph else 0
        ),
        visual_multistep_horizon=visual_config.horizon,
        visual_multistep_context=visual_config.context,
        correction_unroll_horizon=(
            graph_config.correction_unroll_horizon if use_graph else 0
        ),
    )
    ensure_finite_activations(
        output,
        epoch=epoch,
        batch_id=batch_id,
        max_abs=max_activation,
    )
    graph_weights = graph_loss_weights(
        use_graph=use_graph,
        has_graph_targets=has_graph_targets,
        graph_scale=graph_scale,
        graph_supervision_weight=graph_supervision_weight,
        config=graph_config,
    )
    loss = compute_sequence_loss(
        model,
        output,
        images,
        graph_weight=graph_weights["graph"],
        ot_weight=ot_weight * ot_scale if use_graph else 0.0,
        ot_variance_weight=(
            ot_variance_weight * ot_scale if use_graph else 0.0
        ),
        targets=targets,
        graph_supervision_weight=graph_weights["graph_supervision"],
        beta_kl=vae_kl_weight,
        vae_reconstruction_weight=vae_reconstruction_weight,
        motion_weighted_transition=motion_weighted_transition,
        graph_multistep_embedding_weight=graph_weights[
            "graph_multistep_embedding"
        ],
        graph_multistep_state_weight=graph_weights["graph_multistep_state"],
        visual_multistep_weight=(
            visual_config.weight if visual_config.horizon > 0 else 0.0
        ),
        graph_position_weight=graph_weights["graph_position"],
        correction_unroll_weight=graph_weights["correction_unroll"],
        plugin_correction_weight=graph_weights["plugin_correction"],
    )
    scalars = loss_scalars(loss)
    ensure_finite_loss(
        scalars,
        epoch=epoch,
        batch_id=batch_id,
        max_total=(
            max(max_loss, 1e5)
            if output.get("loss_family") == "dreamerv3"
            else max_loss
        ),
    )
    diagnostics = (
        ot_diagnostic_scalars(
            output,
            ot_scale=ot_scale,
            ot_weight=ot_weight,
            ot_variance_weight=ot_variance_weight,
        )
        if use_graph
        else {}
    )
    if use_graph:
        weighted_ot = abs(diagnostics["ot_weighted_contribution"])
        diagnostics["weighted_ot_fraction"] = weighted_ot / max(
            abs(scalars["total"]), 1e-12
        )
        backbone_objective, jointgv_objective = (
            split_backbone_and_jointgv_objectives(
                loss,
                use_graph=use_graph,
                graph_scale=graph_scale,
                ot_scale=ot_scale,
                ot_weight=ot_weight,
                ot_variance_weight=ot_variance_weight,
                has_graph_targets=has_graph_targets,
                graph_supervision_weight=graph_supervision_weight,
                graph_config=graph_config,
            )
        )
        diagnostics["backbone_objective"] = float(
            backbone_objective.detach().item()
        )
        diagnostics["jointgv_weighted_contribution"] = float(
            jointgv_objective.detach().item()
        )
    return loss, images.shape[0], diagnostics


@torch.no_grad()
def validate_epoch(
    model: JointGraphVideoWorldModel,
    loader: DataLoader,
    device: torch.device,
    *,
    use_graph: bool,
    graph_scale: float,
    ot_scale: float,
    ot_weight: float,
    ot_variance_weight: float,
    has_graph_targets: bool,
    graph_supervision_weight: float,
    vae_kl_weight: float,
    vae_reconstruction_weight: float,
    epoch: int,
    max_loss: float,
    max_activation: float,
    motion_weighted_transition: bool = False,
    graph_config: GraphLossConfig = GraphLossConfig(),
    correction_scale: float | None = None,
    visual_config: VisualMultistepConfig = VisualMultistepConfig(),
) -> dict[str, float]:
    model.eval()
    strength = graph_scale if correction_scale is None else correction_scale
    model.set_graph_conditioning_strength(strength if use_graph else 0.0)
    averages = MetricAverages()
    for batch_id, batch in enumerate(loader, start=1):
        loss, batch_size, diagnostics = compute_batch_loss(
            model,
            batch,
            device,
            use_graph=use_graph,
            graph_scale=graph_scale,
            ot_scale=ot_scale,
            ot_weight=ot_weight,
            ot_variance_weight=ot_variance_weight,
            has_graph_targets=has_graph_targets,
            graph_supervision_weight=graph_supervision_weight,
            vae_kl_weight=vae_kl_weight,
            vae_reconstruction_weight=vae_reconstruction_weight,
            sample_vae=False,
            epoch=epoch,
            batch_id=batch_id,
            max_loss=max_loss,
            max_activation=max_activation,
            motion_weighted_transition=motion_weighted_transition,
            graph_config=graph_config,
            visual_config=visual_config,
        )
        averages.update(
            {**loss_scalars(loss), **diagnostics}, batch_size
        )
    return averages.compute()


def rollout_prefixes(horizon: int) -> tuple[int, ...]:
    """Standard long-horizon reporting points within one rollout."""
    return tuple(sorted({min(8, horizon), min(16, horizon), horizon}))


# Structure-aware rollout metrics with their own best checkpoint
# (best_rollout_<name>.pt); True means higher is better.
EXTRA_ROLLOUT_SELECTIONS = {"motion_mse": False, "ssim": True, "joint_px": False}
ROLLOUT_HIGHER_IS_BETTER = {"image_mse": False, **EXTRA_ROLLOUT_SELECTIONS}


def empty_rollout_bests(names) -> dict[str, dict[str, float]]:
    return {
        name: {
            "value": -math.inf if ROLLOUT_HIGHER_IS_BETTER[name] else math.inf,
            "epoch": 0,
        }
        for name in names
    }


def plugin_node_comparison(
    model: JointGraphVideoWorldModel,
    loader: DataLoader,
    device: torch.device,
    **rollout_kwargs: Any,
) -> dict[str, dict[str, float]]:
    """Rollouts of one checkpoint with the correction fully on and off.

    Used at the last warm-up-delay epoch: the backbone has been trained
    exactly as graph-off so far and the correction only on the unroll, so
    "plugin" is the graph module plugged into an untouched backbone -- the
    reference later joint-training epochs are compared against.
    """
    return {
        "plugin": validate_rollout(
            model, loader, device, use_graph=True, strength=1.0, **rollout_kwargs
        ),
        "backbone": validate_rollout(
            model, loader, device, use_graph=True, strength=0.0, **rollout_kwargs
        ),
    }


def save_plugin_node(
    comparison: dict[str, dict[str, float]],
    *,
    epoch: int,
    source: Path,
    output: Path,
    horizon: int,
) -> None:
    """Keep the checkpoint as plugin_node.pt and its scores in plugin_node.json."""
    atomic_copy(source, output / "plugin_node.pt")
    (output / "plugin_node.json").write_text(
        json.dumps({"epoch": epoch, **comparison}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    cells = []
    for name in ("image_mse", "motion_mse", "ssim", "joint_px"):
        key = f"{name}_h{horizon}"
        if key in comparison["plugin"]:
            cells.append(
                f"{key}: plugin={comparison['plugin'][key]:.6f} "
                f"backbone={comparison['backbone'][key]:.6f}"
            )
    print(
        f"plugin node epoch {epoch} |--> saved plugin_node.pt  " + "  ".join(cells)
    )


def update_extra_rollout_bests(
    best_by_metric: dict[str, dict[str, float]],
    rollout_metrics: dict[str, float],
    *,
    horizon: int,
    epoch: int,
) -> list[str]:
    """Record new per-metric bests; the caller copies best_rollout_<name>.pt."""
    improved = []
    for name, state in best_by_metric.items():
        value = rollout_metrics[f"{name}_h{horizon}"]
        higher = ROLLOUT_HIGHER_IS_BETTER[name]
        if (value > state["value"]) if higher else (value < state["value"]):
            state["value"] = value
            state["epoch"] = epoch
            improved.append(name)
    return improved


def build_rollout_loader(
    root: Path,
    episode_files: list[Path],
    *,
    horizon: int,
    windows_per_episode: int,
    batch_size: int,
    workers: int,
    context_frames: int = 0,
    pixel_targets: bool = False,
    spread: bool = False,
) -> DataLoader:
    """Use the same deterministic windows from each validation episode.

    By default these are the first ``windows_per_episode`` non-overlapping
    windows (historical protocol). ``spread`` spaces them evenly over the
    whole episode instead.

    Limiting the number of windows per episode keeps recursive validation
    inexpensive and prevents long episodes from dominating the checkpoint
    selection metric merely because they contain more overlapping windows.
    ``pixel_targets`` keeps graph_node_pixel/visible labels for the
    joint-position metric; the windows themselves are unchanged.
    """
    dataset = EpisodeSequences(
        root,
        sequence_length=horizon + context_frames,
        stride=horizon + context_frames,
        include_graph_targets=None if pixel_targets else False,
        include_control_targets=True,
        episode_files=episode_files,
    )
    if pixel_targets and not dataset.has_graph_pixels:
        raise ValueError(
            "rollout joint-position metric needs graph_node_pixel labels"
        )
    counts: dict[int, int] = {}
    selected: list[int] = []
    if spread:
        # Evenly spaced over each episode's non-overlapping windows (first,
        # ..., last; the middle one for a single window) instead of the
        # first ones, so checkpoint selection is not decided by the start
        # of every episode only.
        per_episode: dict[int, list[int]] = {}
        for item, (file_id, _) in enumerate(dataset.index):
            per_episode.setdefault(file_id, []).append(item)
        for file_id, items in per_episode.items():
            if len(items) <= windows_per_episode:
                picks = list(range(len(items)))
            elif windows_per_episode == 1:
                picks = [len(items) // 2]
            else:
                picks = sorted(
                    {
                        int(round(position))
                        for position in np.linspace(
                            0, len(items) - 1, windows_per_episode
                        )
                    }
                )
            selected.extend(items[pick] for pick in picks)
            counts[file_id] = len(picks)
    else:
        for item, (file_id, _) in enumerate(dataset.index):
            count = counts.get(file_id, 0)
            if count < windows_per_episode:
                selected.append(item)
                counts[file_id] = count + 1
    eligible_episodes = len(counts)
    skipped_short_episodes = len(episode_files) - eligible_episodes
    if not selected:
        raise RuntimeError(
            "No validation episode is long enough for "
            f"context={context_frames} + horizon={horizon}"
        )
    if skipped_short_episodes:
        print(
            "recursive validation eligibility:",
            f"eligible_episodes={eligible_episodes}",
            f"skipped_short_episodes={skipped_short_episodes}",
            f"minimum_transitions={horizon + context_frames}",
        )
    return DataLoader(
        Subset(dataset, selected),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )


@torch.no_grad()
def validate_rollout(
    model: JointGraphVideoWorldModel,
    loader: DataLoader,
    device: torch.device,
    *,
    horizon: int,
    use_graph: bool,
    context_frames: int = 0,
    observer: torch.nn.Module | None = None,
    window: Tensor | None = None,
    strength: float = 1.0,
    visual_context: bool = False,
) -> dict[str, float]:
    """Measure recursive open-loop prediction without CEM or observations.

    Only the first image initializes the model. Recorded validation actions
    drive the rollout; all subsequent visual and graph states are the model's
    own predictions. Prefix metrics average every predicted step up to H.
    The ``context_frames`` real frames before the start always seed the graph
    history; with ``visual_context`` (DreamerV3, --rollout-visual-context)
    the backbone also observes them and the actions between them, so both
    tracks start from the same real frames.
    With a reference ``observer`` (and the loader's pixel labels), also
    reports motion-weighted MSE, SSIM and joint-position error in pixels.  For
    the RM65-B cart-pole schema it also reports hinge/tip/vector pixel errors,
    wrapped pole-angle MAE, and wrapped pole angular-change error
    (jointgv.rollout_metrics). ``strength`` is the graph correction strength
    to evaluate at; training passes the epoch's own correction_scale, so
    epochs inside the warm-up delay are scored as the backbone alone.
    """
    model.eval()
    model.set_graph_conditioning_strength(strength if use_graph else 0.0)
    averages = MetricAverages()
    prefixes = rollout_prefixes(horizon)
    for batch in loader:
        images, actions, targets = move_batch(batch, device)
        start = context_frames
        context_actions = actions[:, :start] if visual_context and start else None
        actions = actions[:, start : start + horizon]
        prediction = model.rollout(
            images[:, start],
            actions,
            use_graph=use_graph,
            deterministic=True,
            decode=True,
            context_images=images[:, :start] if start else None,
            context_actions=context_actions,
            visual_context=context_actions is not None,
        )
        target_images = images[:, start + 1 : start + horizon + 1]
        target_tokens = None
        if model.rollout_latent_matches_visual_tokens:
            flat_targets = target_images.flatten(0, 1)
            target_tokens = model.encode_visual_tokens(flat_targets)
            target_tokens = target_tokens.reshape(
                images.shape[0], horizon, model.token_count, model.token_dim
            )
        scores: dict[str, Tensor] = {}
        if observer is not None:
            scores = rollout_scores(
                prediction["images"],
                images[:, start],
                target_images,
                images[:, start : start + horizon],
                targets["node_pixel"][:, start + 1 : start + horizon + 1],
                targets["node_visible"][:, start + 1 : start + horizon + 1],
                observer,
                window,
                first_pixel=targets["node_pixel"][:, start],
                first_visible=targets["node_visible"][:, start],
            )
        values: dict[str, float] = {}
        for prefix in prefixes:
            for name, value in scores.items():
                values[f"{name}_h{prefix}"] = float(
                    torch.nanmean(value[:, :prefix]).item()
                )
            values[f"image_mse_h{prefix}"] = float(
                (prediction["images"][:, :prefix] - target_images[:, :prefix])
                .square()
                .mean()
                .item()
            )
            if target_tokens is not None:
                values[f"latent_mse_h{prefix}"] = float(
                    (
                        prediction["tokens"][:, :prefix]
                        - target_tokens[:, :prefix]
                    )
                    .square()
                    .mean()
                    .item()
                )
            values[f"reward_mse_h{prefix}"] = float(
                (
                    prediction["rewards"][:, :prefix]
                    - targets["reward"][:, start : start + prefix]
                )
                .square()
                .mean()
                .item()
            )
            values[f"continuation_mse_h{prefix}"] = float(
                (
                    prediction["continuations"][:, :prefix]
                    - targets["continuation"][:, start : start + prefix]
                )
                .square()
                .mean()
                .item()
            )
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Non-finite validation rollout metric")
        averages.update(values, images.shape[0])
    return averages.compute()


def atomic_save(checkpoint: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def git_state() -> dict[str, Any]:
    """Commit and dirty flag of the repository this script runs from."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit or None, "git_dirty": bool(status)}


def record_launch(output: Path, launch: dict[str, Any]) -> None:
    """Append this launch to output/run_config.json (one entry per start)."""
    path = output / "run_config.json"
    history: dict[str, Any] = {"launches": []}
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.replace(path.with_name("run_config.corrupt.json"))
    history.setdefault("launches", []).append(launch)
    path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def canonical_model_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized["backbone"] = resolve_backbone_adapter(
        str(normalized["backbone"])
    ).name
    normalized.setdefault("freeze_visual", False)
    normalized.setdefault("graph_observer", "proxy")
    normalized.setdefault("graph_backbone", "reltr_resnet50")
    # Weight origin affects initialization/download behavior, not structure;
    # resumed checkpoints always restore the serialized observer weights.
    normalized.pop("graph_backbone_pretrained", None)
    normalized.setdefault("freeze_graph_backbone", True)
    # Centering was mandatory before this option was introduced.
    normalized.setdefault("ot_centering", True)
    normalized.setdefault(
        "graph_correction_gate_init", 0.11920292202211755
    )
    # Checkpoint location and feature chunk size do not change tensor shapes.
    normalized.pop("reltr_checkpoint", None)
    normalized.pop("reltr_batch_size", None)
    normalized.setdefault("reltr_image_size", 224)
    # Options of the removed IRIS backbone, still stored in older configs.
    for key in REMOVED_CONFIG_KEYS:
        normalized.pop(key, None)
    # Semantic role names contain no parameters and are restored explicitly
    # from v12 checkpoints; they do not change model tensor shapes.
    normalized.pop("graph_role_names", None)
    normalized.setdefault("dino_model", "dinov2_vits14")
    normalized.setdefault("dino_image_size", 112)
    normalized.setdefault("dino_freeze_encoder", True)
    normalized.setdefault("dino_predictor_depth", 6)
    normalized.setdefault("dino_predictor_heads", 8)
    normalized.setdefault("dino_mlp_ratio", 4.0)
    normalized.setdefault("dino_dropout", 0.1)
    normalized.setdefault("dino_max_frames", 40)
    normalized.setdefault("graph_history", 1)
    normalized.setdefault("graph_pose_correction", False)
    normalized.setdefault("graph_pose_sigma", 0.15)
    normalized.setdefault("graph_global_correction", False)
    return normalized


def load_graph_observer_checkpoint(
    model: JointGraphVideoWorldModel,
    path: Path,
    device: torch.device,
) -> None:
    # Observer pretraining checkpoints contain optimizer states; keep the
    # payload on CPU and copy only model tensors into the already allocated
    # world model to avoid a duplicate CUDA checkpoint peak.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(
            "Graph observer checkpoint must be produced by "
            "pretrain_graph_observer.py"
        )
    if int(payload.get("checkpoint_version", 1)) < 3:
        raise ValueError(
            "This world-model path requires a v3 geom-centric dense "
            "Observer checkpoint. Retrain with the current "
            "pretrain_graph_observer.py and use a fully warmed checkpoint."
        )
    if float(payload.get("temporal_strength", 0.0)) < 1.0:
        raise ValueError(
            "Explicit Graph Observer checkpoint was saved before temporal "
            "warm-up completed. Use a v3 best.pt or best_long_rollout.pt "
            "whose temporal_strength is 1.0."
        )
    config = payload.get("observer_config", {})
    expected = {
        "nodes": model.graph_builder.nodes,
        "node_dim": model.graph_builder.node_dim,
        "teacher_state_dim": model.graph_builder.teacher_state_dim,
    }
    for key, value in expected.items():
        if key in config and config[key] != value:
            raise ValueError(
                f"Graph observer {key}={config[key]} does not match "
                f"world model {key}={value}"
            )
    model.graph_builder.load_state_dict(payload["model"])
    if "temporal_max_correction" in config:
        model.graph_builder.temporal_max_correction = float(
            config["temporal_max_correction"]
        )
    if "max_position_residual" in config:
        model.graph_builder.max_position_residual = float(
            config["max_position_residual"]
        )
    if config.get("role_names") is not None:
        model.graph_builder.role_names = tuple(config["role_names"])
    model.graph_builder.set_temporal_strength(
        float(payload.get("temporal_strength", 1.0))
    )
    print(
        f"loaded explicit graph observer checkpoint: {path} "
        f"(checkpoint_epoch={payload.get('epoch', 'unknown')}, "
        "temporal_strength="
        f"{float(payload.get('temporal_strength', 0.0)):.3f})"
    )
    print(
        "active graph nodes for GN/OT: "
        f"{model.graph_dynamics_plugin.active_graph_node_count}/"
        f"{model.graph_dynamics_plugin.graph_nodes}; "
        "padding deltas are fixed to zero"
    )


def load_visual_checkpoint(
    model: JointGraphVideoWorldModel,
    path: Path,
    device: torch.device,
) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Visual checkpoint must contain a state dictionary")
    checkpoint_kind = payload.get("visual_kind", "vae")
    if checkpoint_kind != model.visual_kind:
        raise ValueError(
            f"Visual checkpoint kind {checkpoint_kind!r} does not match "
            f"backbone visual kind {model.visual_kind!r}"
        )
    checkpoint_dim = int(payload.get("token_dim", model.token_dim))
    if checkpoint_dim != model.token_dim:
        raise ValueError(
            f"Visual checkpoint token_dim={checkpoint_dim} does not match "
            f"model token_dim={model.token_dim}"
        )
    if checkpoint_kind == "dino_patch":
        # A silent DINO variant/resolution mismatch would invalidate a
        # matched graph-off/on comparison (different patch grid or native
        # feature dim) without ever raising a shape error, since both sides
        # expose the same configured token width (native or projected).
        checkpoint_dino_model = payload.get("dino_model")
        if checkpoint_dino_model != model.visual_codec.dino_model_name:
            raise ValueError(
                f"Visual checkpoint dino_model={checkpoint_dino_model!r} does "
                f"not match backbone dino_model="
                f"{model.visual_codec.dino_model_name!r}"
            )
        checkpoint_dino_image_size = int(
            payload.get("dino_image_size", model.visual_codec.dino_image_size)
        )
        if checkpoint_dino_image_size != model.visual_codec.dino_image_size:
            raise ValueError(
                "Visual checkpoint dino_image_size="
                f"{checkpoint_dino_image_size} does not match backbone "
                f"dino_image_size={model.visual_codec.dino_image_size}"
            )
    state = payload.get("model", payload)
    model.visual_codec.load_state_dict(state)
    print(f"loaded visual checkpoint: {path}")


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    ordered = [name for name in LOSS_NAMES if name in metrics]
    for name in (
        "grad_norm",
        "grad_norm_backbone",
        "grad_norm_graph",
        "grad_norm_correction",
    ):
        if name in metrics:
            ordered.append(name)
    return prefix + " " + " ".join(
        f"{name}={metrics[name]:.6f}" for name in ordered
    )


def format_ot_diagnostics(
    prefix: str, metrics: dict[str, float]
) -> str | None:
    ordered = [name for name in OT_DIAGNOSTIC_NAMES if name in metrics]
    if not ordered:
        return None
    return prefix + "_ot " + " ".join(
        f"{name}={metrics[name]:.6e}" for name in ordered
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("runs/jointgv"))
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume model, optimizer, epoch, and best metric from a checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vae-kl-weight", type=float, default=1e-3)
    parser.add_argument(
        "--motion-weighted-transition-loss",
        action="store_true",
        help=(
            "Reweight the token-prediction loss by each token's own "
            "ground-truth temporal change, so a small fast-moving foreground "
            "(e.g. a thin robot arm) is not averaged away by a large static "
            "background. Only affects backbones with a deterministic "
            "per-token prediction (e.g. dino_wm); a no-op for the MDN-LSTM "
            "(david_ha) mixture path. Default off preserves every existing "
            "backbone's current loss exactly."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed Python, NumPy and PyTorch (CPU and CUDA) before anything "
            "else. Unset keeps the historical behaviour, i.e. PyTorch's "
            "built-in default seed 67280421310721 (runs before this flag "
            "used it). The effective seed and the RNG state are saved in "
            "every checkpoint, and --resume restores the RNG state."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Also request deterministic cuDNN/cuBLAS kernels (slower). "
            "Without it, a fixed seed gives the same initialization and data "
            "order, but GPU kernels may still differ in the last bits."
        ),
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--graph-warmup-epochs", type=int, default=10)
    parser.add_argument("--ot-warmup-epochs", type=int, default=20)
    parser.add_argument(
        "--graph-warmup-delay-epochs",
        type=int,
        default=0,
        help=(
            "Keep graph conditioning strength and the OT loss weight at "
            "exactly 0 through this many epochs (the shared backbone then "
            "trains identically to a graph-off run: the correction term is "
            "0 * gate * bounded(...), which is both exactly zero and "
            "carries exactly zero gradient), THEN start the existing "
            "--graph-warmup-epochs/--ot-warmup-epochs ramps from 0 at "
            "epoch delay+1. Graph-only losses (graph, graph_supervision, "
            "graph multistep) are NOT delayed: they never reach the shared "
            "backbone, so GraphDynamics trains from epoch 1 on its own "
            "--graph-warmup-epochs ramp and is already trained when the "
            "correction first switches on."
        ),
    )
    parser.add_argument(
        "--ot-weight",
        type=float,
        default=0.1,
        help=(
            "Fixed final Sinkhorn weight, or the initial weight when "
            "--dynamic-ot-weight is enabled."
        ),
    )
    parser.add_argument(
        "--dynamic-ot-weight",
        action="store_true",
        help=(
            "Adapt the full-strength OT weight from EMA shared-gradient "
            "norms; epoch OT warm-up still applies."
        ),
    )
    parser.add_argument(
        "--ot-target-grad-ratio",
        type=float,
        default=0.1,
        help="Target weighted-OT/shared-backbone gradient ratio.",
    )
    parser.add_argument(
        "--ot-balance-update-every",
        type=int,
        default=50,
        help="Re-estimate the dynamic OT weight every N minibatches.",
    )
    parser.add_argument(
        "--ot-balance-ema",
        type=float,
        default=0.95,
        help="EMA decay for shared gradient norms.",
    )
    parser.add_argument("--ot-weight-min", type=float, default=0.1)
    parser.add_argument("--ot-weight-max", type=float, default=1000.0)
    parser.add_argument(
        "--ot-variance-weight",
        type=float,
        default=0.1,
        help=(
            "Final anti-collapse projection-variance weight after OT "
            "warm-up."
        ),
    )
    parser.add_argument(
        "--disable-ot-centering",
        action="store_true",
        help=(
            "Keep the shared/global projection direction before OT. "
            "Per-token/node unit normalization remains enabled."
        ),
    )
    parser.add_argument(
        "--rollout-eval-start",
        type=int,
        default=0,
        help=(
            "First epoch eligible for recursive validation; 0 starts after "
            "the longer Graph/OT warm-up."
        ),
    )
    parser.add_argument(
        "--rollout-eval-every",
        type=int,
        default=5,
        help="Evaluate recursive validation every N epochs; 0 disables it.",
    )
    parser.add_argument("--rollout-horizon", type=int, default=32)
    parser.add_argument(
        "--rollout-context-frames",
        type=int,
        default=0,
        help=(
            "Real frames preceding each recursive-validation start, used "
            "only to seed graph history (--graph-history > 1). The visual "
            "backbone still starts from one frame. Graph-on and graph-off "
            "runs must use the same value so they are scored on identical "
            "frames. 0 keeps the historical windows."
        ),
    )
    parser.add_argument(
        "--rollout-observer-checkpoint",
        type=Path,
        default=None,
        help=(
            "Frozen pretrain_graph_observer.py checkpoint used only as a "
            "measuring tool in recursive validation: adds motion-weighted "
            "MSE, SSIM and joint-position error (pixels, observer read on "
            "the predicted frames vs graph_node_pixel), and keeps one best "
            "checkpoint per metric (best_rollout_motion_mse.pt, "
            "best_rollout_ssim.pt, best_rollout_joint_px.pt) next to the "
            "unchanged image-MSE best_rollout.pt. Never trained, never part "
            "of the model. Unset keeps the historical metrics only."
        ),
    )
    parser.add_argument(
        "--rescore-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Where --rescore-rollout-epochs reads epoch_XXXX.pt from; default "
            "is this run's own checkpoints directory. Point it at the parent "
            "run when a new run branches from one of its epochs."
        ),
    )
    parser.add_argument(
        "--rescore-rollout-epochs",
        type=int,
        nargs="+",
        default=None,
        help=(
            "With --resume: before continuing, re-run recursive validation "
            "on these saved epoch checkpoints under the current rollout "
            "metrics (results in rollout_rescore.jsonl) so best_rollout.pt "
            "is chosen consistently after a selection-metric change."
        ),
    )
    parser.add_argument(
        "--rollout-windows-per-episode",
        type=int,
        default=1,
        help="Deterministic open-loop windows sampled from each validation episode.",
    )
    parser.add_argument(
        "--rollout-window-spread",
        action="store_true",
        help=(
            "Space the --rollout-windows-per-episode validation windows evenly "
            "over each episode (first, ..., last; the middle one if 1) instead "
            "of taking the first ones, so best-rollout selection also sees the "
            "middle and end of episodes. Default off keeps the historical "
            "first-window protocol."
        ),
    )
    parser.add_argument(
        "--rollout-visual-context",
        action="store_true",
        help=(
            "DreamerV3 only. Recursive validation also gives the backbone the "
            "--rollout-context-frames real frames before the start (and the "
            "actions between them) through its posterior -- the same frames "
            "that seed the graph history -- so both tracks start from the same "
            "real information (evaluate_ablation.py --visual-context). Default "
            "off keeps the historical protocol: the backbone starts from the "
            "start frame alone. Switching it on a resumed run needs "
            "--rescore-rollout-epochs."
        ),
    )
    parser.add_argument("--max-loss", type=float, default=100.0)
    parser.add_argument("--max-activation", type=float, default=1e4)
    parser.add_argument("--max-graph-delta", type=float, default=0.25)
    parser.add_argument("--max-graph-correction", type=float, default=0.25)
    parser.add_argument(
        "--graph-correction-gate-init",
        type=float,
        default=0.11920292202211755,
        help=(
            "Initial graph-correction gate probability in (0, 1); "
            "the model stores and learns its corresponding logit."
        ),
    )
    parser.add_argument("--no-epoch-checkpoints", action="store_true")
    parser.add_argument(
        "--backbone",
        choices=(
            "david_ha",
            "dreamerv3",
            "dreamerv3_vae",
            "dino_wm",
        ),
        default="david_ha",
    )
    parser.add_argument(
        "--visual-checkpoint",
        type=Path,
        help="Load a pretrained visual codec before world-model training.",
    )
    parser.add_argument(
        "--freeze-visual",
        action="store_true",
        help="Freeze a loaded visual codec and use deterministic VAE means.",
    )
    parser.add_argument("--rnn-dim", type=int, default=256)
    parser.add_argument("--mixtures", type=int, default=5)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument(
        "--dino-model",
        type=str,
        default="dinov2_vits14",
        help=(
            "torch.hub facebookresearch/dinov2 variant name (downloaded and "
            "cached on first use), or 'debug_conv' for a tiny untrained "
            "stand-in used only by smoke tests."
        ),
    )
    parser.add_argument(
        "--dino-image-size",
        type=int,
        default=112,
        help=(
            "Resize DMC frames to this size before the frozen DINO encoder; "
            "must be a multiple of the encoder's patch size. The default "
            "(112 = 8x14) gives 64 patch tokens."
        ),
    )
    parser.add_argument(
        "--dino-predictor-depth", type=int, default=6,
        help="Causal Transformer layers predicting the next frame's patches.",
    )
    parser.add_argument("--dino-predictor-heads", type=int, default=8)
    parser.add_argument("--dino-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dino-dropout", type=float, default=0.1)
    parser.add_argument(
        "--dino-max-frames",
        type=int,
        default=40,
        help=(
            "Positional-embedding capacity in frames for the DINO-WM "
            "predictor; must be >= --sequence-length. Rollouts beyond this "
            "many frames fall back to a sliding window."
        ),
    )
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--graph-nodes", type=int, default=8)
    parser.add_argument("--message-steps", type=int, default=3)
    parser.add_argument("--graph-supervision-weight", type=float, default=1.0)
    parser.add_argument(
        "--graph-embedding-weight",
        type=float,
        default=0.2,
        help=(
            "Weight of the graph embedding loss (predicted next graph "
            "embedding vs the observed one). 0.2 is the historical fixed "
            "value; it constrains every embedding dimension, not only the "
            "ones the teacher-state head decodes."
        ),
    )
    parser.add_argument(
        "--graph-history",
        type=int,
        default=1,
        help=(
            "Frames of graph history fed to graph dynamics (current frame "
            "plus history-1 finite-difference velocities). 1 keeps the "
            "original single-frame GraphDynamics exactly."
        ),
    )
    parser.add_argument(
        "--graph-multistep-horizon",
        type=int,
        default=0,
        help=(
            "K: also unroll graph dynamics K steps on its own predictions "
            "and supervise every step (embedding + MuJoCo state). 0 disables."
        ),
    )
    parser.add_argument(
        "--visual-multistep-horizon",
        type=int,
        default=0,
        help=(
            "K: also unroll the visual backbone K steps on its own token "
            "predictions (no graph correction) and supervise every step "
            "with the transition loss. dino_wm only. 0 disables. The "
            "graph-off counterpart of --graph-multistep-horizon, for "
            "separating graph gains from multi-step training gains."
        ),
    )
    parser.add_argument(
        "--visual-multistep-context",
        type=int,
        default=3,
        help=(
            "Real steps of context before the visual unroll starts; 3 "
            "matches the graph unroll start for --graph-history 4."
        ),
    )
    parser.add_argument(
        "--visual-multistep-weight",
        type=float,
        default=1.0,
        help="Weight of the visual K-step loss.",
    )
    parser.add_argument(
        "--separate-graph-grad-clip",
        action="store_true",
        help=(
            "Clip gradients as three separate groups, each to the usual max "
            "norm: backbone (everything outside the graph plugin), graph "
            "(observer heads, graph dynamics, OT projections, position "
            "head), and the correction itself. With one global clip, large "
            "gradients in one group scale down the others' updates too, even "
            "though no gradient flows between them. No effect on graph-off "
            "runs (the graph groups have no gradients). Off keeps the "
            "historical single global clip."
        ),
    )
    parser.add_argument(
        "--graph-pose-correction",
        action="store_true",
        help=(
            "Correction redesign (dino_wm, graph on): (1) a head decodes "
            "each node's 2D image position (supervised by graph_node_pixel) "
            "and each node's pose is delivered to the patches near that "
            "position, independent of where the visual prediction draws the "
            "part; (2) the correction also sees the backbone's own predicted "
            "tokens; (3) the single global gate becomes a per-token gate "
            "computed from those inputs. The OT alignment, OT loss and the "
            "motion context are unchanged. Off keeps the original correction."
        ),
    )
    parser.add_argument(
        "--graph-pose-sigma",
        type=float,
        default=0.15,
        help=(
            "Gaussian width of the pose delivery, in [-1, 1] image "
            "coordinates (one patch is 2/16 = 0.125 at 224px DINO input)."
        ),
    )
    parser.add_argument(
        "--graph-position-weight",
        type=float,
        default=1.0,
        help="Weight of the node 2D position loss (with --graph-pose-correction).",
    )
    parser.add_argument(
        "--correction-unroll-horizon",
        type=int,
        default=0,
        help=(
            "K: also train the correction inside a rollout-style K-step "
            "unroll (visual and graph each fed their own outputs, corrected "
            "prediction fed back), starting at frame --graph-history - 1. "
            "Backbone and graph run without gradient there, so only the "
            "correction learns from it and the backbone stays single-step "
            "(K=0) trained. Not delayed by --graph-warmup-delay-epochs; "
            "ramps with --graph-warmup-epochs. dino_wm only. 0 disables."
        ),
    )
    parser.add_argument(
        "--correction-unroll-weight",
        type=float,
        default=1.0,
        help="Weight of the correction-unroll loss.",
    )
    parser.add_argument(
        "--plugin-correction",
        action="store_true",
        help=(
            "Plug-in training (dino_wm, graph on): the backbone never "
            "receives gradient from OT, the graph context or the correction; "
            "its loss is exactly the graph-off loss on its own prediction. "
            "The correction learns from the corrected prediction's loss "
            "(detached backbone) plus the unroll, at full strength with no "
            "delay/ramp. The backbone-only rollout of such a run is therefore "
            "a graph-off baseline: every recursive validation scores both "
            "tracks and keeps best checkpoints for each "
            "(best_backbone_rollout_*.pt next to best_rollout*.pt). The "
            "dynamic OT controller is frozen at its current weight (the "
            "backbone shares no parameter with OT)."
        ),
    )
    parser.add_argument(
        "--correction-backbone-weight",
        type=float,
        default=1.0,
        help=(
            "Joint training: scale of the gradient the corrected "
            "prediction's loss sends into the backbone (forward unchanged; "
            "the correction still learns from the full loss; OT is "
            "unaffected). 1.0 = historical joint training, 0.25 halves the "
            "backbone's reliance on the correction. dino_wm only."
        ),
    )
    parser.add_argument(
        "--score-resume-checkpoint",
        action="store_true",
        help=(
            "With --resume: before continuing, score the resumed checkpoint "
            "with recursive validation (both tracks in plug-in mode) and "
            "seed the best-rollout checkpoints with it (copied from --resume); "
            "at the warm-up-delay epoch or in plug-in mode it is also saved "
            "as plugin_node.pt."
        ),
    )
    parser.add_argument(
        "--dino-gradient-checkpointing",
        action="store_true",
        help=(
            "Recompute DINO-WM predictor activations during backward to "
            "save memory (same results, slower). Useful with "
            "--visual-multistep-horizon."
        ),
    )
    parser.add_argument(
        "--graph-multistep-weight",
        type=float,
        default=1.0,
        help="Scale of the K-step graph terms relative to the single-step ones.",
    )
    parser.add_argument(
        "--graph-observer",
        choices=("proxy", "explicit"),
        default="proxy",
        help="Visual-token proxy or pretrained RGB Explicit Graph Observer.",
    )
    parser.add_argument(
        "--graph-backbone",
        choices=("reltr_resnet50", "debug_conv"),
        default="reltr_resnet50",
    )
    parser.add_argument(
        "--reltr-checkpoint",
        type=Path,
        default=Path("Models/reltr/checkpoint0149.pth"),
        help="Official RelTR Visual Genome checkpoint.",
    )
    parser.add_argument("--reltr-image-size", type=int, default=224)
    parser.add_argument(
        "--reltr-batch-size",
        type=int,
        default=8,
        help="Maximum RGB frames per frozen RelTR feature-extraction chunk.",
    )
    parser.add_argument(
        "--graph-observer-checkpoint",
        type=Path,
        help="Checkpoint from pretrain_graph_observer.py.",
    )
    parser.add_argument(
        "--no-pretrained-graph-backbone",
        action="store_true",
        help="Do not load official RelTR weights (smoke tests only).",
    )
    parser.add_argument(
        "--train-graph-backbone",
        action="store_true",
        help="Unfreeze the visual graph backbone; frozen is the default.",
    )
    parser.add_argument("--disable-mujoco-supervision", action="store_true")
    parser.add_argument("--disable-graph-conditioning", action="store_true")
    parser.add_argument(
        "--allow-partial-dmc",
        action="store_true",
        help="Allow truncated DMC smoke episodes; never use for reported runs.",
    )
    args = parser.parse_args()

    if args.deterministic:
        # Must be set before the first cuBLAS call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed % 2**32)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # Unseeded, this is PyTorch's built-in default (67280421310721).
    run_seed = int(torch.initial_seed())
    print(
        f"seed={run_seed}",
        "(--seed)" if args.seed is not None else "(PyTorch default; --seed not set)",
        f"deterministic={args.deterministic}",
    )

    selected_adapter = resolve_backbone_adapter(args.backbone)

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.sequence_length < 1 or args.stride < 1:
        parser.error("--sequence-length and --stride must be positive")
    if min(
        args.dino_predictor_depth,
        args.dino_predictor_heads,
        args.dino_max_frames,
    ) < 1:
        parser.error("DINO-WM predictor dimensions must be positive")
    if args.token_dim % args.dino_predictor_heads != 0:
        parser.error("--token-dim must be divisible by --dino-predictor-heads")
    if args.dino_mlp_ratio <= 0 or args.dino_dropout < 0 or args.dino_dropout >= 1:
        parser.error("--dino-mlp-ratio must be positive and --dino-dropout in [0, 1)")
    if (
        selected_adapter.transition == "dino_wm"
        and args.sequence_length > args.dino_max_frames
    ):
        parser.error(
            "--sequence-length must not exceed --dino-max-frames "
            "(the predictor's positional-embedding capacity)"
        )
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.graph_warmup_epochs < 0 or args.ot_warmup_epochs < 0:
        parser.error("warm-up epochs must be nonnegative")
    if args.graph_history < 1:
        parser.error("--graph-history must be at least 1")
    if args.graph_multistep_horizon < 0 or args.rollout_context_frames < 0:
        parser.error(
            "--graph-multistep-horizon and --rollout-context-frames must be "
            "nonnegative"
        )
    if args.graph_embedding_weight < 0 or args.graph_multistep_weight < 0:
        parser.error("graph loss weights must be nonnegative")
    if (
        args.visual_multistep_horizon < 0
        or args.visual_multistep_context < 0
        or args.visual_multistep_weight < 0
    ):
        parser.error("visual multistep settings must be nonnegative")
    if (
        args.visual_multistep_horizon > 0
        or args.dino_gradient_checkpointing
    ) and selected_adapter.transition != "dino_wm":
        parser.error(
            "--visual-multistep-horizon and --dino-gradient-checkpointing "
            "require --backbone dino_wm"
        )
    if args.graph_pose_sigma <= 0 or args.graph_position_weight < 0:
        parser.error(
            "--graph-pose-sigma must be positive and --graph-position-weight "
            "nonnegative"
        )
    if args.correction_unroll_horizon < 0 or args.correction_unroll_weight < 0:
        parser.error("correction unroll settings must be nonnegative")
    # DreamerV3 shares the plug-in's history, K-step graph loss, pose,
    # correction unroll and clip groups only on its plug-in (P) path.
    dreamer_plugin = (
        selected_adapter.transition == "dreamerv3" and args.plugin_correction
    )
    if args.graph_pose_correction or args.correction_unroll_horizon > 0:
        if selected_adapter.transition != "dino_wm" and not dreamer_plugin:
            parser.error(
                "--graph-pose-correction and --correction-unroll-horizon "
                "require --backbone dino_wm, or dreamerv3 with "
                "--plugin-correction"
            )
        if args.disable_graph_conditioning:
            parser.error(
                "--graph-pose-correction and --correction-unroll-horizon "
                "require graph conditioning"
            )
    if args.graph_pose_correction and args.disable_mujoco_supervision:
        parser.error(
            "--graph-pose-correction needs graph_node_pixel targets; drop "
            "--disable-mujoco-supervision"
        )
    if (
        args.correction_unroll_horizon > 0
        and args.graph_history >= args.sequence_length
    ):
        parser.error(
            "--correction-unroll-horizon starts at frame --graph-history - 1, "
            "which must be inside --sequence-length"
        )
    if (
        args.separate_graph_grad_clip
        and selected_adapter.transition == "dreamerv3"
        and not dreamer_plugin
    ):
        parser.error(
            "--separate-graph-grad-clip is wired for dino_wm / david_ha and "
            "for dreamerv3 with --plugin-correction"
        )
    if (
        args.visual_multistep_horizon > 0
        and args.visual_multistep_context >= args.sequence_length
    ):
        parser.error(
            "--visual-multistep-context must be smaller than "
            "--sequence-length"
        )
    if (
        (args.graph_history > 1 or args.graph_multistep_horizon > 0)
        and selected_adapter.transition == "dreamerv3"
        and not dreamer_plugin
    ):
        parser.error(
            "--graph-history > 1 and --graph-multistep-horizon are wired for "
            "dino_wm / david_ha and for dreamerv3 with --plugin-correction"
        )
    if args.graph_warmup_delay_epochs < 0:
        parser.error("--graph-warmup-delay-epochs must be nonnegative")
    if args.reltr_image_size < 64 or args.reltr_batch_size < 1:
        parser.error("RelTR image size must be >=64 and batch size positive")
    if args.rollout_eval_start < 0 or args.rollout_eval_every < 0:
        parser.error("rollout evaluation schedule must be nonnegative")
    if args.plugin_correction and selected_adapter.transition not in {
        "dino_wm",
        "dreamerv3",
    }:
        parser.error("--plugin-correction requires --backbone dino_wm or dreamerv3")
    if args.rollout_visual_context:
        # Kept to DreamerV3 so no DINO-WM run's validation can change.
        if selected_adapter.transition != "dreamerv3":
            parser.error("--rollout-visual-context is wired for --backbone dreamerv3 only")
        if args.rollout_context_frames < 1:
            parser.error("--rollout-visual-context needs --rollout-context-frames >= 1")
    if (
        args.correction_backbone_weight != 1.0
        and selected_adapter.transition != "dino_wm"
    ):
        parser.error("--correction-backbone-weight requires --backbone dino_wm")
    if args.plugin_correction or args.correction_backbone_weight != 1.0:
        if args.disable_graph_conditioning:
            parser.error(
                "--plugin-correction / --correction-backbone-weight require "
                "graph conditioning"
            )
    if not 0.0 <= args.correction_backbone_weight <= 1.0:
        parser.error("--correction-backbone-weight must be in [0, 1]")
    if args.plugin_correction and args.correction_backbone_weight != 1.0:
        parser.error(
            "--plugin-correction already gives the backbone no correction "
            "gradient; do not combine it with --correction-backbone-weight"
        )
    if args.score_resume_checkpoint and args.resume is None:
        parser.error("--score-resume-checkpoint requires --resume")
    if args.rescore_rollout_epochs and args.resume is None:
        parser.error("--rescore-rollout-epochs requires --resume")
    if (
        args.rescore_rollout_epochs
        and args.no_epoch_checkpoints
        and args.rescore_checkpoint_dir is None
    ):
        parser.error("--rescore-rollout-epochs needs saved epoch checkpoints")
    if args.rollout_horizon < 1 or args.rollout_windows_per_episode < 1:
        parser.error("rollout horizon and windows per episode must be positive")
    if args.max_loss <= 0 or args.max_activation <= 0:
        parser.error("finite-check limits must be positive")
    if args.max_graph_delta <= 0 or args.max_graph_correction <= 0:
        parser.error("graph delta/correction limits must be positive")
    if not 0.0 < args.graph_correction_gate_init < 1.0:
        parser.error("--graph-correction-gate-init must be in (0, 1)")
    if args.vae_kl_weight < 0:
        parser.error("--vae-kl-weight must be nonnegative")
    if args.ot_weight < 0 or args.ot_variance_weight < 0:
        parser.error("OT weights must be nonnegative")
    if args.dynamic_ot_weight:
        if args.disable_graph_conditioning:
            parser.error(
                "--dynamic-ot-weight requires graph conditioning"
            )
        if args.ot_target_grad_ratio <= 0:
            parser.error("--ot-target-grad-ratio must be positive")
        if args.ot_balance_update_every < 1:
            parser.error("--ot-balance-update-every must be positive")
        if not 0.0 <= args.ot_balance_ema < 1.0:
            parser.error("--ot-balance-ema must be in [0, 1)")
        if args.ot_weight_min < 0 or args.ot_weight_max < args.ot_weight_min:
            parser.error("dynamic OT weight bounds are invalid")
        if not args.ot_weight_min <= args.ot_weight <= args.ot_weight_max:
            parser.error(
                "--ot-weight must lie within dynamic OT weight bounds"
            )
    if args.freeze_visual and args.visual_checkpoint is None and args.resume is None:
        parser.error("--freeze-visual requires --visual-checkpoint or --resume")
    if args.resume is not None and args.visual_checkpoint is not None:
        parser.error("--resume already contains visual weights; omit --visual-checkpoint")
    if args.resume is not None and args.graph_observer_checkpoint is not None:
        parser.error(
            "--resume already contains graph observer weights; omit "
            "--graph-observer-checkpoint"
        )
    if args.graph_observer_checkpoint is not None and args.graph_observer != "explicit":
        parser.error("--graph-observer-checkpoint requires --graph-observer explicit")
    if args.graph_observer == "explicit" and args.disable_mujoco_supervision:
        print(
            "WARNING: explicit observer has no MuJoCo graph supervision in "
            "this run; use a pretrained observer checkpoint."
        )
    if (
        selected_adapter.transition == "dreamerv3"
        and args.sequence_length != 64
        and not args.allow_partial_dmc
    ):
        parser.error(
            "burchim DreamerV3 reportable runs require --sequence-length 64; "
            "use --allow-partial-dmc only for a short smoke run"
        )

    protocol_info = validate_dmc_dataset(
        args.data,
        allow_partial=args.allow_partial_dmc,
        require_graph_schema_version=(
            EXPLICIT_GRAPH_SCHEMA_VERSION
            if args.graph_observer == "explicit"
            else None
        ),
    )
    all_train_files = sorted(args.data.glob("*.npz"))
    if not all_train_files:
        parser.error(f"No .npz episodes found in {args.data}")

    validation_protocol: dict[str, Any] | None = None
    if args.validation_data is not None:
        if args.validation_data.resolve() == args.data.resolve():
            parser.error("--validation-data must differ from --data")
        validation_protocol = validate_dmc_dataset(
            args.validation_data,
            allow_partial=args.allow_partial_dmc,
            require_graph_schema_version=(
                EXPLICIT_GRAPH_SCHEMA_VERSION
                if args.graph_observer == "explicit"
                else None
            ),
        )
        validation_files = sorted(args.validation_data.glob("*.npz"))
        if not validation_files:
            parser.error(
                "--validation-data contains no top-level .npz episodes: "
                f"{args.validation_data}. Collect the validation split first "
                "or correct the path."
            )
        training_files = all_train_files
    elif args.validation_fraction > 0.0 and len(all_train_files) >= 2:
        validation_count = max(
            1, round(len(all_train_files) * args.validation_fraction)
        )
        validation_count = min(validation_count, len(all_train_files) - 1)
        training_files = all_train_files[:-validation_count]
        validation_files = all_train_files[-validation_count:]
        validation_protocol = protocol_info
    else:
        training_files = all_train_files
        validation_files = []

    print(
        "verified DMC protocol:",
        protocol_info["protocol"],
        f"train_episodes={len(training_files)}",
        f"validation_episodes={len(validation_files)}",
        f"partial={protocol_info['incomplete_dmc_files']}",
    )
    if not validation_files:
        print(
            "WARNING: no validation episodes; best.pt will use a training metric. "
            "Use --validation-data for reportable runs."
        )

    include_graph_targets = (
        False if args.disable_mujoco_supervision else None
    )
    dataset = EpisodeSequences(
        args.data,
        args.sequence_length,
        args.stride,
        include_graph_targets=include_graph_targets,
        include_control_targets=True,
        episode_files=training_files,
    )
    loader = DataLoader(
        dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    validation_dataset = (
        EpisodeSequences(
            args.validation_data or args.data,
            args.sequence_length,
            args.stride,
            include_graph_targets=include_graph_targets,
            include_control_targets=True,
            episode_files=validation_files,
        )
        if validation_files
        else None
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        if validation_dataset is not None
        else None
    )
    rollout_eval_start = (
        args.rollout_eval_start
        if args.rollout_eval_start > 0
        else max(1, args.graph_warmup_epochs, args.ot_warmup_epochs)
    )
    rollout_loader = None
    if validation_files and args.rollout_eval_every > 0:
        validation_root = args.validation_data or args.data
        try:
            rollout_loader = build_rollout_loader(
                validation_root,
                validation_files,
                horizon=args.rollout_horizon,
                windows_per_episode=args.rollout_windows_per_episode,
                batch_size=args.batch_size,
                workers=args.workers,
                context_frames=args.rollout_context_frames,
                pixel_targets=args.rollout_observer_checkpoint is not None,
                spread=args.rollout_window_spread,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            parser.error(f"Cannot build validation rollout set: {error}")
        print(
            "recursive validation:",
            f"start_epoch={rollout_eval_start}",
            f"every={args.rollout_eval_every}",
            f"horizon={args.rollout_horizon}",
            f"windows={len(rollout_loader.dataset)}",
            "window_placement="
            + ("spread over episode" if args.rollout_window_spread else "episode start"),
            "planner=none",
        )

    first_sample = dataset[0]
    action_dim = first_sample[1].shape[-1]
    if validation_dataset is not None:
        validation_action_dim = validation_dataset[0][1].shape[-1]
        if validation_action_dim != action_dim:
            parser.error("Training and validation action dimensions differ")
        if validation_dataset.has_graph_targets != dataset.has_graph_targets:
            parser.error("Training and validation graph targets differ")
    if args.graph_pose_correction and not (
        dataset.has_graph_pixels
        and (validation_dataset is None or validation_dataset.has_graph_pixels)
    ):
        parser.error(
            "--graph-pose-correction needs graph_node_pixel/graph_node_visible "
            "in every training and validation episode"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "action_dim": action_dim,
        "backbone": args.backbone,
        "token_dim": args.token_dim,
        "node_dim": args.node_dim,
        "graph_nodes": args.graph_nodes,
        "message_steps": args.message_steps,
        "rnn_dim": args.rnn_dim,
        "mixtures": args.mixtures,
        "max_graph_delta": args.max_graph_delta,
        "max_graph_correction": args.max_graph_correction,
        "graph_correction_gate_init": args.graph_correction_gate_init,
        "ot_centering": not args.disable_ot_centering,
        "freeze_visual": args.freeze_visual,
        "graph_observer": args.graph_observer,
        "graph_backbone": args.graph_backbone,
        "graph_backbone_pretrained": (
            args.graph_observer == "explicit"
            and args.resume is None
            and args.graph_observer_checkpoint is None
            and not args.no_pretrained_graph_backbone
        ),
        "freeze_graph_backbone": not args.train_graph_backbone,
        "reltr_checkpoint": str(args.reltr_checkpoint),
        "reltr_image_size": args.reltr_image_size,
        "reltr_batch_size": args.reltr_batch_size,
        "graph_role_names": None,
        "dino_model": args.dino_model,
        "dino_image_size": args.dino_image_size,
        "dino_freeze_encoder": True,
        "dino_predictor_depth": args.dino_predictor_depth,
        "dino_predictor_heads": args.dino_predictor_heads,
        "dino_mlp_ratio": args.dino_mlp_ratio,
        "dino_dropout": args.dino_dropout,
        "dino_max_frames": args.dino_max_frames,
        "graph_history": args.graph_history,
        "graph_pose_correction": args.graph_pose_correction,
        "graph_pose_sigma": args.graph_pose_sigma,
        # DreamerV3 plug-in (P): the shared plugin's global correction head.
        "graph_global_correction": dreamer_plugin,
    }
    if dataset.has_graph_targets:
        first_targets = first_sample[2]
        model_config["graph_nodes"] = first_targets["node_state"].shape[-2]
        model_config["teacher_state_dim"] = first_targets["node_state"].shape[-1]

    use_graph = not args.disable_graph_conditioning
    model = JointGraphVideoWorldModel(**model_config).to(device)
    print(
        "Latent/graph alignment:",
        f"mode={model.graph_dynamics_plugin.alignment_mode}",
        (
            f"centering={'on' if model_config['ot_centering'] else 'off'}"
            if model.graph_dynamics_plugin.alignment_mode == "sinkhorn"
            else "centering=not_applicable_to_global_delta"
        ),
        "unit_normalization=on",
    )
    print(
        "Graph correction:",
        f"max={args.max_graph_correction:.6f}",
        f"gate_init={args.graph_correction_gate_init:.6f}",
        "full_strength_initial_component_cap="
        f"{args.max_graph_correction * args.graph_correction_gate_init:.6f}",
        f"pose={'on' if args.graph_pose_correction else 'off'}",
        f"pose_sigma={args.graph_pose_sigma:.3f}",
        f"unroll_horizon={args.correction_unroll_horizon}",
    )
    if args.visual_checkpoint is not None:
        try:
            load_visual_checkpoint(
                model, args.visual_checkpoint, device
            )
        except (KeyError, RuntimeError, ValueError) as error:
            parser.error(str(error))
    if args.graph_observer_checkpoint is not None:
        try:
            load_graph_observer_checkpoint(
                model, args.graph_observer_checkpoint, device
            )
        except (KeyError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        model_config["graph_role_names"] = tuple(
            model.graph_builder.role_names
        )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if selected_adapter.transition == "dreamerv3":
        optimizer = torch.optim.Adam(
            trainable_parameters, lr=args.lr, eps=1e-8
        )
    else:
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    ot_controller = (
        GradientRatioOTController(
            initial_weight=args.ot_weight,
            target_ratio=args.ot_target_grad_ratio,
            ema_decay=args.ot_balance_ema,
            update_every=args.ot_balance_update_every,
            min_weight=args.ot_weight_min,
            max_weight=args.ot_weight_max,
        )
        if args.dynamic_ot_weight
        else None
    )
    ot_shared_parameters = (
        ot_balance_shared_parameters(model)
        if ot_controller is not None
        else ()
    )
    if ot_controller is not None:
        print(
            "Dynamic OT balance:",
            f"initial_weight={ot_controller.weight:.6f}",
            f"target_grad_ratio={args.ot_target_grad_ratio:.6f}",
            f"update_every={args.ot_balance_update_every}",
            f"ema={args.ot_balance_ema:.6f}",
            f"bounds=[{args.ot_weight_min:.6f}, "
            f"{args.ot_weight_max:.6f}]",
            f"candidate_shared_parameters={len(ot_shared_parameters)}",
        )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    if not args.no_epoch_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    best_metric_value = float("inf")
    best_epoch = 0
    rollout_metric_name = f"image_mse_h{args.rollout_horizon}"
    rollout_observer = None
    rollout_window = None
    # One extra best checkpoint per structure-aware metric (value, epoch).
    best_rollout_by_metric: dict[str, dict[str, float]] = {}
    # Plug-in runs also track the backbone-only rollout (graph-off
    # baseline): best_backbone_rollout_<name>.pt, image_mse included.
    best_backbone_rollout_by_metric: dict[str, dict[str, float]] = {}
    if rollout_loader is not None and args.rollout_observer_checkpoint is not None:
        rollout_observer = load_reference_observer(
            args.rollout_observer_checkpoint, device
        )
        rollout_window = gaussian_window(device)
        best_rollout_by_metric = empty_rollout_bests(EXTRA_ROLLOUT_SELECTIONS)
        print(
            "recursive validation reference observer:",
            args.rollout_observer_checkpoint,
            "extra best checkpoints:",
            ", ".join(f"best_rollout_{name}.pt" for name in best_rollout_by_metric),
        )
    if rollout_loader is not None and args.plugin_correction:
        best_backbone_rollout_by_metric = empty_rollout_bests(
            ["image_mse"]
            + (list(EXTRA_ROLLOUT_SELECTIONS) if rollout_observer is not None else [])
        )
    # Best validation single-step error of the graph-corrected prediction
    # (best.pt keeps tracking the backbone objective).
    best_graph_single_step: dict[str, float] = {"value": math.inf, "epoch": 0}
    model.plugin_correction = args.plugin_correction
    model.correction_backbone_weight = args.correction_backbone_weight
    if args.plugin_correction:
        print(
            "plug-in correction: backbone trained on the graph-off loss only; "
            "correction at full strength; rollouts scored with and without it"
        )
    elif args.correction_backbone_weight != 1.0:
        print(
            "joint correction: backbone receives "
            f"{args.correction_backbone_weight:g}x the corrected prediction's gradient"
        )
    best_rollout_value = float("inf")
    best_rollout_epoch = 0
    rollout_config = {
        "start_epoch": rollout_eval_start,
        "every": args.rollout_eval_every,
        "horizon": args.rollout_horizon,
        "windows_per_episode": args.rollout_windows_per_episode,
        "metric": rollout_metric_name,
        "planner": "none",
    }
    if args.rollout_context_frames:
        # Only recorded when nonzero so historical checkpoints still resume.
        rollout_config["context_frames"] = args.rollout_context_frames
    if args.rollout_window_spread:
        # Same rule: absent from historical configs.
        rollout_config["window_spread"] = True
    if args.rollout_visual_context:
        # Same rule.
        rollout_config["visual_context"] = True
    if args.rollout_observer_checkpoint is not None:
        rollout_config["observer_checkpoint"] = str(
            args.rollout_observer_checkpoint
        )
    graph_config = GraphLossConfig(
        embedding_weight=args.graph_embedding_weight,
        multistep_horizon=args.graph_multistep_horizon,
        multistep_weight=args.graph_multistep_weight,
        position_weight=(
            args.graph_position_weight if args.graph_pose_correction else 0.0
        ),
        correction_unroll_horizon=args.correction_unroll_horizon,
        correction_unroll_weight=args.correction_unroll_weight,
        plugin=args.plugin_correction,
    )
    visual_config = VisualMultistepConfig(
        horizon=args.visual_multistep_horizon,
        context=args.visual_multistep_context,
        weight=args.visual_multistep_weight,
    )
    if args.dino_gradient_checkpointing:
        model.backbone.predictor.gradient_checkpointing = True
    graph_clip_parameters: list[torch.nn.Parameter] = []
    correction_clip_parameters: list[torch.nn.Parameter] = []
    backbone_clip_parameters: list[torch.nn.Parameter] = []
    if args.separate_graph_grad_clip:
        plugin_module = model.graph_dynamics_plugin
        graph_ids = {id(parameter) for parameter in plugin_module.parameters()}
        correction_ids = {
            id(parameter)
            for parameter in plugin_module.correction_parameters()
        }
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in correction_ids:
                correction_clip_parameters.append(parameter)
            elif id(parameter) in graph_ids:
                graph_clip_parameters.append(parameter)
            else:
                backbone_clip_parameters.append(parameter)
    start_epoch = 1
    resume_rng_state = None
    if args.resume is not None:
        if not args.resume.is_file():
            parser.error(f"Resume checkpoint does not exist: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_config = checkpoint.get("model_config")
        if checkpoint_config is not None:
            checkpoint_version = int(
                checkpoint.get("checkpoint_version", 0)
            )
            if checkpoint_version < 8:
                parser.error(
                    "Pre-v8 checkpoints predate bias-free centered OT "
                    "projectors and Dreamer attention pooling. Start a new "
                    "run instead of resuming an incompatible model."
                )
            if args.graph_observer == "explicit" and checkpoint_version < 12:
                parser.error(
                    "Pre-v12 JointGV checkpoints predate active-node masking "
                    "for Graph Dynamics and OT. Start a new explicit run."
                )
            if (
                selected_adapter.transition == "dreamerv3"
                and checkpoint_version < 13
            ):
                parser.error(
                    "Pre-v13 Dreamer checkpoints use the removed RSSM-to-"
                    "pseudo-token MLP. Start a new run for genuine global "
                    "delta alignment."
                )
        if (
            checkpoint_config is not None
            and canonical_model_config(checkpoint_config)
            != canonical_model_config(model_config)
        ):
            parser.error(
                "--resume checkpoint model_config does not match the "
                "current data/model arguments"
            )
        model.load_state_dict(checkpoint["model"])
        if args.graph_observer == "explicit":
            graph_role_names = checkpoint_config.get("graph_role_names")
            if not graph_role_names:
                parser.error(
                    "v12 explicit checkpoint is missing graph_role_names"
                )
            model.graph_builder.role_names = tuple(graph_role_names)
            model_config["graph_role_names"] = tuple(graph_role_names)
        optimizer.load_state_dict(checkpoint["optimizer"])
        stored_ot_controller = checkpoint.get("dynamic_ot_controller")
        if ot_controller is None and stored_ot_controller is not None:
            parser.error(
                "Checkpoint uses dynamic OT balancing; pass "
                "--dynamic-ot-weight with the original controller arguments"
            )
        if ot_controller is not None and stored_ot_controller is None:
            parser.error(
                "Checkpoint has no dynamic OT controller state; resume it "
                "without --dynamic-ot-weight or start a new dynamic run"
            )
        if ot_controller is not None:
            try:
                ot_controller.load_state_dict(stored_ot_controller)
            except (KeyError, TypeError, ValueError) as error:
                parser.error(str(error))
        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1
        resume_rng_state = checkpoint.get("rng_state")
        best_metric_value = float(
            checkpoint.get(
                "best_metric_value",
                checkpoint.get("best_mean_loss", float("inf")),
            )
        )
        best_epoch = int(checkpoint.get("best_epoch", completed_epoch))
        stored_rollout_config = checkpoint.get("rollout_config")

        def without_observer(config: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in config.items()
                if key != "observer_checkpoint"
            }

        # The reference observer only adds metrics (the rollout windows and
        # the image-MSE best_rollout.pt are unchanged), so it may be added
        # when resuming a run that did not have it.
        if stored_rollout_config is not None and without_observer(
            stored_rollout_config
        ) != without_observer(rollout_config):
            # Switching the backbone's visual context changes every rollout
            # score, so it is only allowed when the saved epochs are rescored
            # under the new protocol (the rescore rebuilds every best file).
            protocol_only = {
                key: value
                for key, value in without_observer(stored_rollout_config).items()
                if key != "visual_context"
            } == {
                key: value
                for key, value in without_observer(rollout_config).items()
                if key != "visual_context"
            }
            if not (protocol_only and args.rescore_rollout_epochs):
                parser.error(
                    "--resume rollout configuration does not match the checkpoint"
                    + (
                        "; a changed --rollout-visual-context needs "
                        "--rescore-rollout-epochs"
                        if protocol_only
                        else ""
                    )
                )
            print(
                "rollout protocol changed (backbone visual context "
                f"{'on' if args.rollout_visual_context else 'off'}): best "
                "rollouts are rebuilt from the rescored epochs"
            )
        stored_rollout_metric = checkpoint.get("best_rollout_metric")
        if stored_rollout_metric is not None:
            if stored_rollout_metric != rollout_metric_name:
                parser.error(
                    "--resume rollout metric does not match the checkpoint"
                )
            best_rollout_value = float(
                checkpoint.get("best_rollout_value", float("inf"))
            )
            best_rollout_epoch = int(
                checkpoint.get("best_rollout_epoch", 0)
            )
        stored_graph_step = checkpoint.get("best_graph_single_step")
        if stored_graph_step:
            best_graph_single_step = {
                "value": float(stored_graph_step["value"]),
                "epoch": int(stored_graph_step["epoch"]),
            }
        stored_extra = checkpoint.get("best_rollout_by_metric") or {}
        if (
            best_rollout_by_metric
            and stored_rollout_config is not None
            and stored_rollout_config.get("observer_checkpoint")
            == rollout_config.get("observer_checkpoint")
        ):
            for name, state in stored_extra.items():
                if name in best_rollout_by_metric:
                    best_rollout_by_metric[name] = {
                        "value": float(state["value"]),
                        "epoch": int(state["epoch"]),
                    }
            for name, state in (
                checkpoint.get("best_backbone_rollout_by_metric") or {}
            ).items():
                if name in best_backbone_rollout_by_metric:
                    best_backbone_rollout_by_metric[name] = {
                        "value": float(state["value"]),
                        "epoch": int(state["epoch"]),
                    }
        if start_epoch > args.epochs:
            parser.error(
                f"Checkpoint already completed epoch {completed_epoch}, "
                f"but --epochs is only {args.epochs}"
            )
        print(
            f"resumed {args.resume} after epoch={completed_epoch} "
            f"best_value={best_metric_value:.6f} best_epoch={best_epoch}"
        )
        if ot_controller is not None:
            print(
                "resumed dynamic OT controller:",
                f"weight={ot_controller.weight:.6f}",
                f"updates={ot_controller.updates}",
                f"batch_step={ot_controller.batch_step}",
            )
        if args.graph_observer == "explicit":
            print(
                "active graph nodes for GN/OT after resume: "
                f"{model.graph_dynamics_plugin.active_graph_node_count}/"
                f"{model.graph_dynamics_plugin.graph_nodes}"
            )
    elif metrics_path.exists():
        metrics_path.write_text("", encoding="utf-8")
    else:
        metrics_path.touch()

    # A run branched from another run keeps its own copy of every parent
    # checkpoint it depends on, so deleting the parent directory loses
    # nothing.
    copied_parent_checkpoints: list[str] = []

    def keep_local_copy(source: Path, epoch_number: int) -> Path:
        if args.no_epoch_checkpoints:
            return source
        local = checkpoint_dir / f"epoch_{epoch_number:04d}.pt"
        if source.resolve() == local.resolve():
            return local
        if not local.exists():
            atomic_copy(source, local)
            copied_parent_checkpoints.append(str(source))
            print(f"copied parent checkpoint {source} -> {local}")
        return local

    if args.resume is not None and args.resume.resolve().parent != checkpoint_dir.resolve():
        keep_local_copy(args.resume, start_epoch - 1)
    # Score earlier checkpoints (--rescore-rollout-epochs, read from
    # --rescore-checkpoint-dir) and/or the resumed weights
    # (--score-resume-checkpoint) exactly as the training loop now scores an
    # epoch, and rebuild every best-rollout checkpoint from these scores.
    # Plug-in runs score both tracks: graph corrected (correction at full
    # strength) and backbone only. Evaluation only; the resumed weights are
    # restored afterwards and nothing touches the optimizer.
    score_jobs: list[tuple[int, Path, bool]] = []
    if args.rescore_rollout_epochs:
        rescore_dir = args.rescore_checkpoint_dir or checkpoint_dir
        for rescore_epoch in args.rescore_rollout_epochs:
            rescore_checkpoint = rescore_dir / f"epoch_{rescore_epoch:04d}.pt"
            if not rescore_checkpoint.is_file():
                parser.error(f"Missing checkpoint {rescore_checkpoint}")
            score_jobs.append((rescore_epoch, rescore_checkpoint, True))
    if args.score_resume_checkpoint and (start_epoch - 1) not in (
        args.rescore_rollout_epochs or []
    ):
        score_jobs.append((start_epoch - 1, args.resume, False))
    if score_jobs:
        if rollout_loader is None or rollout_observer is None:
            parser.error(
                "--rescore-rollout-epochs / --score-resume-checkpoint need "
                "recursive validation and --rollout-observer-checkpoint"
            )
        resumed_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        best_rollout_value = float("inf")
        best_rollout_epoch = 0
        best_rollout_by_metric = empty_rollout_bests(best_rollout_by_metric)
        best_backbone_rollout_by_metric = empty_rollout_bests(
            best_backbone_rollout_by_metric
        )
        rollout_kwargs = {
            "horizon": args.rollout_horizon,
            "context_frames": args.rollout_context_frames,
            "observer": rollout_observer,
            "window": rollout_window,
            "visual_context": args.rollout_visual_context,
        }
        rescore_path = args.output / "rollout_rescore.jsonl"
        rescore_path.write_text("", encoding="utf-8")
        plugin_node_saved = False
        for score_epoch, source, load in sorted(score_jobs):
            if load:
                source = keep_local_copy(source, score_epoch)
                payload = torch.load(source, map_location=device, weights_only=False)
                model.load_state_dict(payload["model"])
                del payload
            else:
                model.load_state_dict(resumed_state)
            graph_strength = (
                1.0
                if args.plugin_correction
                else (
                    linear_warmup(
                        score_epoch,
                        args.graph_warmup_epochs,
                        args.graph_warmup_delay_epochs,
                    )
                    if use_graph
                    else 0.0
                )
            )
            scored = {
                "graph": validate_rollout(
                    model,
                    rollout_loader,
                    device,
                    use_graph=use_graph,
                    strength=graph_strength,
                    **rollout_kwargs,
                )
            }
            if args.plugin_correction:
                scored["backbone"] = validate_rollout(
                    model,
                    rollout_loader,
                    device,
                    use_graph=use_graph,
                    strength=0.0,
                    **rollout_kwargs,
                )
            with rescore_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "epoch": score_epoch,
                            "source": str(source),
                            "correction_strength": graph_strength,
                            "rollout": scored["graph"],
                            "rollout_backbone": scored.get("backbone"),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            for track, values in scored.items():
                label = (
                    f"graph corrected, strength {graph_strength:.2f}"
                    if track == "graph"
                    else "backbone only"
                )
                print(
                    f"rescored epoch {score_epoch} ({label}): "
                    + " ".join(
                        f"{name}={value:.6f}" for name, value in values.items()
                    )
                )
            if scored["graph"][rollout_metric_name] < best_rollout_value:
                best_rollout_value = scored["graph"][rollout_metric_name]
                best_rollout_epoch = score_epoch
                atomic_copy(source, args.output / "best_rollout.pt")
            for name in update_extra_rollout_bests(
                best_rollout_by_metric,
                scored["graph"],
                horizon=args.rollout_horizon,
                epoch=score_epoch,
            ):
                atomic_copy(source, args.output / f"best_rollout_{name}.pt")
            if "backbone" in scored:
                for name in update_extra_rollout_bests(
                    best_backbone_rollout_by_metric,
                    scored["backbone"],
                    horizon=args.rollout_horizon,
                    epoch=score_epoch,
                ):
                    atomic_copy(
                        source, args.output / f"best_backbone_rollout_{name}.pt"
                    )
            node_epoch = (
                use_graph
                and args.graph_warmup_delay_epochs > 0
                and score_epoch == args.graph_warmup_delay_epochs
            )
            if not plugin_node_saved and (
                node_epoch or (args.plugin_correction and not load)
            ):
                node = (
                    {"plugin": scored["graph"], "backbone": scored["backbone"]}
                    if "backbone" in scored
                    else plugin_node_comparison(
                        model, rollout_loader, device, **rollout_kwargs
                    )
                )
                save_plugin_node(
                    node,
                    epoch=score_epoch,
                    source=source,
                    output=args.output,
                    horizon=args.rollout_horizon,
                )
                plugin_node_saved = True
        model.load_state_dict(resumed_state)
        del resumed_state
        print(
            "best rollouts after rescoring (graph corrected): "
            f"{rollout_metric_name}={best_rollout_value:.6f}"
            f"@epoch{best_rollout_epoch} "
            + " ".join(
                f"{name}_h{args.rollout_horizon}={state['value']:.6f}"
                f"@epoch{state['epoch']}"
                for name, state in best_rollout_by_metric.items()
            )
        )
        if best_backbone_rollout_by_metric:
            print(
                "best rollouts after rescoring (backbone only): "
                + " ".join(
                    f"{name}_h{args.rollout_horizon}={state['value']:.6f}"
                    f"@epoch{state['epoch']}"
                    for name, state in best_backbone_rollout_by_metric.items()
                )
            )
    if resume_rng_state is not None:
        # Continue the interrupted random stream (data order, dropout)
        # rather than restarting it from the seed.
        torch.set_rng_state(resume_rng_state["torch"].cpu())
        if torch.cuda.is_available() and resume_rng_state.get("cuda"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_rng_state["cuda"]]
            )
        print("restored RNG state from the resume checkpoint")
    record_launch(
        args.output,
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "command": sys.argv,
            "args": vars(args),
            "seed": run_seed,
            "seed_explicit": args.seed is not None,
            "deterministic": args.deterministic,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu"
            ),
            **git_state(),
            "model_config": model_config,
            "rollout_config": rollout_config,
            "resumed_from": str(args.resume) if args.resume is not None else None,
            "start_epoch": start_epoch,
            "rng_state_restored": resume_rng_state is not None,
            "copied_parent_checkpoints": copied_parent_checkpoints,
        },
    )
    for epoch in range(start_epoch, args.epochs + 1):
        # graph_scale ramps the graph-only losses (never delayed: they do not
        # reach the backbone); correction_scale ramps the correction applied
        # to the backbone's prediction and stays behind the delay.
        graph_scale = (
            linear_warmup(epoch, args.graph_warmup_epochs)
            if use_graph
            else 0.0
        )
        correction_scale = (
            linear_warmup(
                epoch, args.graph_warmup_epochs, args.graph_warmup_delay_epochs
            )
            if use_graph
            else 0.0
        )
        if use_graph and args.plugin_correction:
            # The delay/ramp only protect the backbone, which a plug-in
            # correction cannot reach.
            correction_scale = 1.0
        ot_scale = (
            linear_warmup(
                epoch, args.ot_warmup_epochs, args.graph_warmup_delay_epochs
            )
            if use_graph
            else 0.0
        )
        model.train()
        model.set_graph_conditioning_strength(correction_scale)
        train_averages = MetricAverages()
        progress = tqdm(loader, desc=f"epoch {epoch}")
        for batch_id, batch in enumerate(progress, start=1):
            batch_ot_weight = (
                ot_controller.weight
                if ot_controller is not None
                else args.ot_weight
            )
            loss, batch_size, diagnostics = compute_batch_loss(
                model,
                batch,
                device,
                use_graph=use_graph,
                graph_scale=graph_scale,
                ot_scale=ot_scale,
                ot_weight=batch_ot_weight,
                ot_variance_weight=args.ot_variance_weight,
                has_graph_targets=dataset.has_graph_targets,
                graph_supervision_weight=args.graph_supervision_weight,
                vae_kl_weight=(
                    0.0 if model.visual_frozen else args.vae_kl_weight
                ),
                vae_reconstruction_weight=(
                    0.0 if model.visual_frozen else 0.5
                ),
                sample_vae=not model.visual_frozen,
                epoch=epoch,
                batch_id=batch_id,
                max_loss=args.max_loss,
                max_activation=args.max_activation,
                motion_weighted_transition=args.motion_weighted_transition_loss,
                graph_config=graph_config,
                visual_config=visual_config,
            )
            optimizer.zero_grad(set_to_none=True)
            # Plug-in: the backbone objective shares no parameter with OT, so
            # the ratio controller has nothing to measure; its weight stays.
            if (
                ot_controller is not None
                and ot_scale > 0.0
                and not args.plugin_correction
            ):
                # Gated on ot_scale (not just "ot_controller is not None"):
                # loss.ot is the RAW Sinkhorn cost, computed every batch
                # regardless of warm-up/delay, so an ungated .step() would
                # keep fitting the gradient-ratio EMA against an untrained,
                # effectively-frozen graph module for the whole
                # --graph-warmup-delay-epochs span and could drift `weight`
                # to an uncalibrated value that then hits at full force the
                # moment ot_scale starts ramping. Skipping .step() here
                # leaves `weight` at its safe initial_weight throughout the
                # delay; its own first-measurement warm-up (see docstring)
                # then starts fresh once ot_scale first becomes positive.
                backbone_objective, _ = split_backbone_and_jointgv_objectives(
                    loss,
                    use_graph=use_graph,
                    graph_scale=graph_scale,
                    ot_scale=ot_scale,
                    ot_weight=batch_ot_weight,
                    ot_variance_weight=args.ot_variance_weight,
                    has_graph_targets=dataset.has_graph_targets,
                    graph_supervision_weight=args.graph_supervision_weight,
                    graph_config=graph_config,
                )
                ot_controller.step(
                    backbone_objective,
                    loss.ot,
                    ot_shared_parameters,
                )
                # Apply an updated coefficient to this same minibatch without
                # performing a second expensive world-model forward pass.
                weight_delta = ot_controller.weight - batch_ot_weight
                if weight_delta != 0.0:
                    objective_delta = (
                        weight_delta * ot_scale
                    ) * loss.ot
                    loss.total = loss.total + objective_delta
                    diagnostics["jointgv_weighted_contribution"] += float(
                        objective_delta.detach().item()
                    )
            if ot_controller is not None:
                diagnostics.update(
                    ot_controller.diagnostics(warmup_scale=ot_scale)
                )
                diagnostics["ot_weighted_contribution"] = float(
                    (
                        loss.ot.detach()
                        * (ot_controller.weight * ot_scale)
                    ).item()
                )
                diagnostics["weighted_ot_fraction"] = abs(
                    diagnostics["ot_weighted_contribution"]
                ) / max(abs(float(loss.total.detach().item())), 1e-12)
            scalars = {**loss_scalars(loss), **diagnostics}
            ensure_finite_loss(
                scalars,
                epoch=epoch,
                batch_id=batch_id,
                max_total=(
                    max(args.max_loss, 1e5)
                    if selected_adapter.transition == "dreamerv3"
                    else args.max_loss
                ),
            )
            loss.total.backward()
            ensure_finite_gradients(
                model,
                epoch=epoch,
                batch_id=batch_id,
            )
            projector_grad_norm = (
                ot_projector_grad_norm(model) if use_graph else None
            )
            if projector_grad_norm is not None:
                if not math.isfinite(projector_grad_norm):
                    raise FloatingPointError(
                        "Non-finite OT projector gradient norm at "
                        f"epoch={epoch} batch={batch_id}"
                    )
                scalars["ot_projector_grad_norm"] = projector_grad_norm
            grad_clip = (
                1000.0
                if selected_adapter.transition == "dreamerv3"
                else 1.0
            )
            if args.separate_graph_grad_clip:
                # Each backbone keeps its own clip (Burchim DreamerV3: 1000);
                # the plug-in's two groups use 1.0 on every backbone.
                backbone_norm = torch.nn.utils.clip_grad_norm_(
                    backbone_clip_parameters,
                    grad_clip,
                    error_if_nonfinite=True,
                )
                graph_norm = torch.nn.utils.clip_grad_norm_(
                    graph_clip_parameters,
                    PLUGIN_GRAD_CLIP,
                    error_if_nonfinite=True,
                )
                correction_norm = torch.nn.utils.clip_grad_norm_(
                    correction_clip_parameters,
                    PLUGIN_GRAD_CLIP,
                    error_if_nonfinite=True,
                )
                # Pre-clip global norm, logged under the historical key so
                # curves stay comparable with single-clip runs.
                grad_norm = torch.sqrt(
                    backbone_norm.detach().square()
                    + graph_norm.detach().square()
                    + correction_norm.detach().square()
                )
                scalars["grad_norm_backbone"] = float(backbone_norm.item())
                scalars["grad_norm_graph"] = float(graph_norm.item())
                scalars["grad_norm_correction"] = float(correction_norm.item())
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip, error_if_nonfinite=True
                )
            grad_norm_value = float(grad_norm.detach().item())
            if not math.isfinite(grad_norm_value):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch={epoch} "
                    f"batch={batch_id}"
                )
            optimizer.step()
            scalars["grad_norm"] = grad_norm_value
            train_averages.update(scalars, batch_size)
            displayed_ot_weight = (
                (
                    ot_controller.weight
                    if ot_controller is not None
                    else args.ot_weight
                )
                if use_graph
                else 0.0
            )
            progress.set_postfix(
                loss=f"{scalars['total']:.4f}",
                transition=f"{scalars['transition']:.4f}",
                ot=f"{scalars['ot']:.4f}",
                ot_w=f"{displayed_ot_weight:.3g}",
                graph_sup=f"{scalars['graph_supervision']:.4f}",
            )

        train_metrics = train_averages.compute()
        validation_metrics = (
            validate_epoch(
                model,
                validation_loader,
                device,
                use_graph=use_graph,
                graph_scale=graph_scale,
                ot_scale=ot_scale,
                ot_weight=(
                    ot_controller.weight
                    if ot_controller is not None
                    else args.ot_weight
                ),
                ot_variance_weight=args.ot_variance_weight,
                has_graph_targets=validation_dataset.has_graph_targets,
                graph_supervision_weight=args.graph_supervision_weight,
                vae_kl_weight=(
                    0.0 if model.visual_frozen else args.vae_kl_weight
                ),
                vae_reconstruction_weight=(
                    0.0 if model.visual_frozen else 0.5
                ),
                epoch=epoch,
                max_loss=args.max_loss,
                max_activation=args.max_activation,
                motion_weighted_transition=args.motion_weighted_transition_loss,
                graph_config=graph_config,
                correction_scale=correction_scale,
                visual_config=visual_config,
            )
            if validation_loader is not None
            and validation_dataset is not None
            else None
        )
        if validation_metrics is not None and ot_controller is not None:
            validation_metrics.update(
                ot_controller.diagnostics(warmup_scale=ot_scale)
            )
        graph_step_improved = False
        if use_graph and validation_metrics is not None:
            corrected_step = validation_metrics["corrected_transition"]
            if corrected_step < best_graph_single_step["value"]:
                best_graph_single_step = {"value": corrected_step, "epoch": epoch}
                graph_step_improved = True
        rollout_metrics = None
        if (
            rollout_loader is not None
            and args.rollout_eval_every > 0
            and epoch >= rollout_eval_start
            and (epoch - rollout_eval_start) % args.rollout_eval_every == 0
        ):
            rollout_metrics = validate_rollout(
                model,
                rollout_loader,
                device,
                horizon=args.rollout_horizon,
                use_graph=use_graph,
                context_frames=args.rollout_context_frames,
                observer=rollout_observer,
                window=rollout_window,
                strength=correction_scale,
                visual_context=args.rollout_visual_context,
            )
        rollout_backbone_metrics = None
        if rollout_metrics is not None and args.plugin_correction:
            rollout_backbone_metrics = validate_rollout(
                model,
                rollout_loader,
                device,
                horizon=args.rollout_horizon,
                use_graph=use_graph,
                context_frames=args.rollout_context_frames,
                observer=rollout_observer,
                window=rollout_window,
                strength=0.0,
                visual_context=args.rollout_visual_context,
            )
        is_best_rollout = False
        improved_rollout_metrics: list[str] = []
        improved_backbone_metrics: list[str] = []
        if rollout_metrics is not None:
            rollout_value = rollout_metrics[rollout_metric_name]
            is_best_rollout = rollout_value < best_rollout_value
            if is_best_rollout:
                best_rollout_value = rollout_value
                best_rollout_epoch = epoch
            improved_rollout_metrics = update_extra_rollout_bests(
                best_rollout_by_metric,
                rollout_metrics,
                horizon=args.rollout_horizon,
                epoch=epoch,
            )
        if rollout_backbone_metrics is not None:
            improved_backbone_metrics = update_extra_rollout_bests(
                best_backbone_rollout_by_metric,
                rollout_backbone_metrics,
                horizon=args.rollout_horizon,
                epoch=epoch,
            )
        # Last epoch before the correction reaches the backbone's training:
        # score and keep the "plugged into an untouched backbone" model.
        plugin_node = None
        if (
            use_graph
            and rollout_loader is not None
            and args.graph_warmup_delay_epochs > 0
            and epoch == args.graph_warmup_delay_epochs
        ):
            plugin_node = plugin_node_comparison(
                model,
                rollout_loader,
                device,
                horizon=args.rollout_horizon,
                context_frames=args.rollout_context_frames,
                observer=rollout_observer,
                window=rollout_window,
                visual_context=args.rollout_visual_context,
            )
        selection_key = (
            "backbone_objective" if ot_controller is not None else "total"
        )
        selection_metrics = (
            validation_metrics
            if validation_metrics is not None
            else train_metrics
        )
        selection_metric = selection_metrics[selection_key]
        selection_name = (
            f"validation_{selection_key}"
            if validation_metrics is not None
            else f"train_{selection_key}"
        )
        is_best = selection_metric < best_metric_value
        if is_best:
            best_metric_value = selection_metric
            best_epoch = epoch

        # Checkpoint metric semantics:
        # - mean_loss: current epoch's average training total loss.
        # - best_mean_loss: compatibility alias for best_metric_value; despite
        #   its legacy name, it stores validation_total when validation exists.
        # - best_epoch: epoch at which best_metric_value was first achieved.
        # - best_metric: validation/train total for fixed OT. Dynamic-OT runs
        #   use backbone_objective so a changing auxiliary coefficient cannot
        #   make early and late validation totals incomparable.
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "mean_loss": train_metrics["total"],
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "best_mean_loss": best_metric_value,
            "best_metric_value": best_metric_value,
            "best_epoch": best_epoch,
            "best_metric": selection_name,
            "rollout_metrics": rollout_metrics,
            "best_rollout_value": best_rollout_value,
            "best_rollout_epoch": best_rollout_epoch,
            "best_rollout_metric": rollout_metric_name,
            "best_rollout_by_metric": best_rollout_by_metric,
            "best_backbone_rollout_by_metric": best_backbone_rollout_by_metric,
            "best_graph_single_step": best_graph_single_step,
            "seed": run_seed,
            # Tensors only, so the checkpoint still loads with weights_only.
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else []
                ),
            },
            "rollout_config": rollout_config,
            "graph_scale": graph_scale,
            "correction_scale": correction_scale,
            "ot_scale": ot_scale,
            "dynamic_ot_controller": (
                ot_controller.state_dict()
                if ot_controller is not None
                else None
            ),
            "current_ot_weight": (
                ot_controller.weight
                if ot_controller is not None
                else args.ot_weight
            ),
            "model_config": model_config,
            "sequence_length": args.sequence_length,
            "checkpoint_version": model.checkpoint_version,
            "graph_conditioning": use_graph,
            "graph_observer_checkpoint": (
                str(args.graph_observer_checkpoint)
                if args.graph_observer_checkpoint is not None
                else None
            ),
            "data_protocol": protocol_info["protocol"],
            "validation_protocol": (
                validation_protocol["protocol"]
                if validation_protocol is not None
                else None
            ),
            "dmc_files": len(training_files),
            "validation_dmc_files": len(validation_files),
            "incomplete_dmc_files": protocol_info["incomplete_dmc_files"],
            "training_config": {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "stride": args.stride,
                "graph_warmup_epochs": args.graph_warmup_epochs,
                "ot_warmup_epochs": args.ot_warmup_epochs,
                "graph_warmup_delay_epochs": args.graph_warmup_delay_epochs,
                "graph_embedding_weight": args.graph_embedding_weight,
                "graph_history": args.graph_history,
                "graph_multistep_horizon": args.graph_multistep_horizon,
                "graph_multistep_weight": args.graph_multistep_weight,
                "rollout_context_frames": args.rollout_context_frames,
                "visual_multistep_horizon": args.visual_multistep_horizon,
                "visual_multistep_context": args.visual_multistep_context,
                "visual_multistep_weight": args.visual_multistep_weight,
                "dino_gradient_checkpointing": args.dino_gradient_checkpointing,
                "seed": run_seed,
                "seed_explicit": args.seed is not None,
                "deterministic": args.deterministic,
                "separate_graph_grad_clip": args.separate_graph_grad_clip,
                "graph_pose_correction": args.graph_pose_correction,
                "graph_pose_sigma": args.graph_pose_sigma,
                "graph_position_weight": args.graph_position_weight,
                "correction_unroll_horizon": args.correction_unroll_horizon,
                "correction_unroll_weight": args.correction_unroll_weight,
                "motion_weighted_transition_loss": (
                    args.motion_weighted_transition_loss
                ),
                "ot_weight": args.ot_weight,
                "dynamic_ot_weight": args.dynamic_ot_weight,
                "ot_target_grad_ratio": args.ot_target_grad_ratio,
                "ot_balance_update_every": args.ot_balance_update_every,
                "ot_balance_ema": args.ot_balance_ema,
                "ot_weight_min": args.ot_weight_min,
                "ot_weight_max": args.ot_weight_max,
                "ot_variance_weight": args.ot_variance_weight,
                "ot_centering": model_config["ot_centering"],
                "max_graph_correction": args.max_graph_correction,
                "graph_correction_gate_init": (
                    args.graph_correction_gate_init
                ),
                "rollout_eval_start": rollout_eval_start,
                "rollout_eval_every": args.rollout_eval_every,
                "rollout_horizon": args.rollout_horizon,
                "rollout_windows_per_episode": (
                    args.rollout_windows_per_episode
                ),
                "graph_supervision_weight": args.graph_supervision_weight,
                "vae_kl_weight": args.vae_kl_weight,
                "visual_checkpoint": (
                    str(args.visual_checkpoint)
                    if args.visual_checkpoint is not None
                    else None
                ),
                "freeze_visual": args.freeze_visual,
            },
        }

        if args.no_epoch_checkpoints:
            saved_path = args.output / "latest.pt"
            atomic_save(checkpoint, saved_path)
        else:
            saved_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            atomic_save(checkpoint, saved_path)
            atomic_copy(saved_path, args.output / "latest.pt")
        if is_best:
            atomic_copy(saved_path, args.output / "best.pt")
        if is_best_rollout:
            atomic_copy(saved_path, args.output / "best_rollout.pt")
        for name in improved_rollout_metrics:
            atomic_copy(saved_path, args.output / f"best_rollout_{name}.pt")
        for name in improved_backbone_metrics:
            atomic_copy(
                saved_path, args.output / f"best_backbone_rollout_{name}.pt"
            )
        if graph_step_improved:
            atomic_copy(saved_path, args.output / "best_graph_single_step.pt")
        if plugin_node is not None:
            save_plugin_node(
                plugin_node,
                epoch=epoch,
                source=saved_path,
                output=args.output,
                horizon=args.rollout_horizon,
            )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "rollout": rollout_metrics,
            "graph_scale": graph_scale,
            "correction_scale": correction_scale,
            "ot_scale": ot_scale,
            "current_ot_weight": (
                ot_controller.weight
                if ot_controller is not None
                else args.ot_weight
            ),
            "best_metric": selection_name,
            "best_metric_value": best_metric_value,
            "best_epoch": best_epoch,
            "best_rollout_metric": rollout_metric_name,
            "best_rollout_value": best_rollout_value,
            "best_rollout_epoch": best_rollout_epoch,
            "best_rollout_by_metric": best_rollout_by_metric,
            "best_graph_single_step": best_graph_single_step,
        }
        if rollout_backbone_metrics is not None:
            record["rollout_backbone"] = rollout_backbone_metrics
            record["best_backbone_rollout_by_metric"] = (
                best_backbone_rollout_by_metric
            )
        if plugin_node is not None:
            record["plugin_node"] = plugin_node
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

        print(format_metrics("train", train_metrics))
        train_ot_line = format_ot_diagnostics("train", train_metrics)
        if train_ot_line is not None:
            print(train_ot_line)
        if validation_metrics is not None:
            print(format_metrics("validation", validation_metrics))
            validation_ot_line = format_ot_diagnostics(
                "validation", validation_metrics
            )
            if validation_ot_line is not None:
                print(validation_ot_line)
        if rollout_metrics is not None:
            label = (
                f"rollout(graph strength {correction_scale:.2f}) "
                if use_graph
                else "rollout "
            )
            print(
                label
                + " ".join(
                    f"{name}={value:.6f}"
                    for name, value in rollout_metrics.items()
                )
            )
        if rollout_backbone_metrics is not None:
            print(
                "rollout(backbone only) "
                + " ".join(
                    f"{name}={value:.6f}"
                    for name, value in rollout_backbone_metrics.items()
                )
            )
        best_save_message = " |--> saved best.pt" if is_best else ""
        displayed_ot_weight = (
            (
                ot_controller.weight
                if ot_controller is not None
                else args.ot_weight
            )
            if use_graph
            else 0.0
        )
        print(
            f"epoch={epoch} graph_scale={graph_scale:.3f} "
            f"correction_scale={correction_scale:.3f} "
            f"ot_scale={ot_scale:.3f} "
            f"ot_weight={displayed_ot_weight:.6f} "
            f"best_metric={selection_name} "
            f"best_value={best_metric_value:.6f} best_epoch={best_epoch}"
            f"{best_save_message}"
        )
        if use_graph and validation_metrics is not None:
            backbone_step = validation_metrics["base_transition"]
            corrected_step = validation_metrics["corrected_transition"]
            if rollout_metrics is not None:
                rollout_note = "rollouts evaluated this epoch"
            elif rollout_loader is None or args.rollout_eval_every <= 0:
                rollout_note = "no rollout evaluation"
            else:
                next_eval = (
                    rollout_eval_start
                    if epoch < rollout_eval_start
                    else rollout_eval_start
                    + args.rollout_eval_every
                    * ((epoch - rollout_eval_start) // args.rollout_eval_every + 1)
                )
                rollout_note = (
                    f"best rollouts unchanged (next eval: epoch {next_eval})"
                )
            print(
                f"epoch {epoch} single-step (validation): "
                f"backbone={backbone_step:.5f} "
                f"graph-corrected={corrected_step:.5f} "
                f"({100.0 * (corrected_step / backbone_step - 1.0):+.1f}%) | "
                f"graph-corrected best={best_graph_single_step['value']:.5f}"
                f"@epoch{best_graph_single_step['epoch']}"
                + (
                    " |--> saved best_graph_single_step.pt"
                    if graph_step_improved
                    else ""
                )
                + f" | {rollout_note}"
            )
        if rollout_metrics is not None:
            rollout_save_message = (
                " |--> saved best_rollout.pt" if is_best_rollout else ""
            )
            print(
                f"rollout_metric={rollout_metric_name} "
                f"best_rollout_value={best_rollout_value:.6f} "
                f"best_rollout_epoch={best_rollout_epoch}"
                f"{rollout_save_message}"
            )
            for name, state in best_rollout_by_metric.items():
                saved = (
                    f" |--> saved best_rollout_{name}.pt"
                    if name in improved_rollout_metrics
                    else ""
                )
                print(
                    f"rollout_metric={name}_h{args.rollout_horizon} "
                    f"best_value={state['value']:.6f} "
                    f"best_epoch={state['epoch']}{saved}"
                )
            for name, state in best_backbone_rollout_by_metric.items():
                saved = (
                    f" |--> saved best_backbone_rollout_{name}.pt"
                    if name in improved_backbone_metrics
                    else ""
                )
                print(
                    f"backbone rollout_metric={name}_h{args.rollout_horizon} "
                    f"best_value={state['value']:.6f} "
                    f"best_epoch={state['epoch']}{saved}"
                )


if __name__ == "__main__":
    main()
