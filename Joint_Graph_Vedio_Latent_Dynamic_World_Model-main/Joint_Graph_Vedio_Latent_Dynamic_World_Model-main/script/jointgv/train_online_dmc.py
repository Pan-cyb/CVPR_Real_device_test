"""Online DMC training: replay + world model + actor + critic.

The DreamerV3 path follows the pinned burchim DMC interaction schedule and
actor--critic objective.  ``--disable-graph-conditioning`` is the paired
graph-off baseline; omitting it recursively evolves JointGV during both world
model fitting and actor imagination.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Iterator

os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Windows" else "egl")

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import trange
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jointgv import JointGraphVideoWorldModel  # noqa: E402
from jointgv.online import (  # noqa: E402
    BURCHIM_DMC_DEFAULTS,
    DreamerBehavior,
    OnlineDMCBatch,
    SlidingWindowReplay,
    lambda_returns,
)

# Reuse the already validated explicit-observer checkpoint loader.
from train import load_graph_observer_checkpoint  # noqa: E402


DREAMER_TRAINING_STAGE_VERSION = "explicit_stages_v3_graph"


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_dreamer_stage_checkpoint(
    model: JointGraphVideoWorldModel,
    path: Path,
    device: torch.device,
    expected_stage: str,
) -> dict[str, Any]:
    """Load model weights for a new Dreamer stage without optimizer state."""
    payload = torch.load(path, map_location=device, weights_only=False)
    source_stage = payload.get("training_stage", payload.get("stage"))
    if source_stage != expected_stage:
        raise RuntimeError(
            f"Stage requires a {expected_stage!r} checkpoint; got "
            f"training_stage={source_stage!r}"
        )
    source_backbone = payload.get("config", {}).get("backbone")
    if source_backbone not in {None, "dreamerv3"}:
        raise RuntimeError(
            "Stage checkpoint backbone is not DreamerV3: "
            f"{source_backbone!r}"
        )
    # Graph was inactive in visual/world source stages. Do not require its
    # observer architecture to match the target stage (proxy -> explicit is a
    # valid Stage-3 transition), and do not import untrained graph weights.
    graph_prefixes = (
        "dreamerv3_core.graph_plugin.",
        "dreamerv3_core.graph_to_deter.",
        "dreamerv3_core.graph_gate_logit",
    )
    source_state = {
        key: value for key, value in payload["model"].items()
        if not key.startswith(graph_prefixes)
    }
    incompatible = model.load_state_dict(source_state, strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(graph_prefixes)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible Dreamer stage checkpoint: "
            f"missing={invalid_missing[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    return payload


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    states = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), states):
            parameter.requires_grad_(requires_grad)


@contextmanager
def isolated_random_state(seed: int) -> Iterator[None]:
    """Run a block under a fixed seed without perturbing the training RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def model_parameters(model: nn.Module, use_graph: bool) -> list[nn.Parameter]:
    if use_graph:
        return list(model.parameters())
    graph_ids = {
        id(parameter)
        for parameter in model.graph_dynamics_plugin.parameters()
    }
    if model.dreamerv3_core is not None:
        graph_ids.update(
            id(parameter)
            for parameter in model.dreamerv3_core.graph_to_deter.parameters()
        )
        graph_ids.add(id(model.dreamerv3_core.graph_gate_logit))
    return [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in graph_ids
    ]


def dreamer_visual_parameters(
    model: JointGraphVideoWorldModel,
) -> list[nn.Parameter]:
    """Exact Stage-1 observation-posterior-decoder parameter set."""
    core = model.dreamerv3_core
    if core is None:
        raise ValueError("Dreamer visual parameters require DreamerV3")
    if core.decoder is None:
        raise ValueError("Online DMC Stage 1 requires the native Dreamer decoder")
    modules = (core.visual, core.rssm.mlp_obs1, core.decoder)
    parameters = [parameter for module in modules for parameter in module.parameters()]
    if not parameters:
        raise RuntimeError("Dreamer Stage 1 has zero trainable parameters")
    return parameters


def dreamer_world_parameters(
    model: JointGraphVideoWorldModel,
) -> list[nn.Parameter]:
    """Exact Stage-2 temporal/reward/continuation parameter set."""
    core = model.dreamerv3_core
    if core is None:
        raise ValueError("Dreamer world parameters require DreamerV3")
    groups: list[nn.Module | nn.Parameter] = [
        core.rssm.weight_init,
        core.rssm.mlp_img1,
        core.rssm.gru,
        core.rssm.mlp_img2,
        core.reward_head,
        core.continuation_head,
    ]
    parameters: list[nn.Parameter] = []
    for group in groups:
        if isinstance(group, nn.Parameter):
            parameters.append(group)
        else:
            parameters.extend(group.parameters())
    if not parameters:
        raise RuntimeError("Dreamer Stage 2 has zero trainable parameters")
    return parameters


def dreamer_graph_parameters(
    model: JointGraphVideoWorldModel,
) -> list[nn.Parameter]:
    """Trainable graph path, excluding a frozen explicit observer."""
    core = model.dreamerv3_core
    if core is None:
        raise ValueError("Dreamer graph parameters require DreamerV3")
    observer_ids = (
        {id(parameter) for parameter in core.graph_builder.parameters()}
        if core.graph_plugin.graph_observer == "explicit"
        else set()
    )
    candidates = [
        *core.graph_plugin.parameters(),
        *core.graph_to_deter.parameters(),
        core.graph_gate_logit,
    ]
    return [parameter for parameter in candidates if id(parameter) not in observer_ids]


def configure_dreamer_training_stage(
    model: JointGraphVideoWorldModel,
    behavior: DreamerBehavior,
    stage: str,
    *,
    use_graph: bool,
) -> list[nn.Parameter]:
    """Apply and audit the Dreamer-only freeze map for an implemented stage."""
    if model.dreamerv3_core is None:
        raise ValueError("Online training supports the DreamerV3 backbone")
    if stage == "joint":
        # Preserve the pre-existing joint-training requires_grad state,
        # including a frozen explicit observer or frozen graph backbone.
        return model_parameters(model, use_graph)
    if stage not in {"visual", "world", "graph"}:
        raise ValueError(f"DreamerV3 training stage is not implemented yet: {stage}")

    model.requires_grad_(False)
    behavior.requires_grad_(False)
    if stage == "visual":
        trainable = dreamer_visual_parameters(model)
    else:
        trainable = dreamer_world_parameters(model)
        if stage == "graph" and use_graph:
            trainable = [*trainable, *dreamer_graph_parameters(model)]
    for parameter in trainable:
        parameter.requires_grad_(True)
    return trainable


def report_training_stage(
    model: JointGraphVideoWorldModel,
    behavior: DreamerBehavior,
    stage: str,
) -> dict[str, dict[str, int]]:
    """Print trainable/frozen counts for every requested subsystem."""
    core = model.dreamerv3_core
    if core is None:
        return {}
    groups: dict[str, nn.Module | list[nn.Parameter]] = {
        "visual encoder": core.visual,
        "RSSM posterior (mlp_obs1)": core.rssm.mlp_obs1,
        "RSSM transition (initial/mlp_img1/gru/mlp_img2)": [
            core.rssm.weight_init,
            *core.rssm.mlp_img1.parameters(),
            *core.rssm.gru.parameters(),
            *core.rssm.mlp_img2.parameters(),
        ],
        "decoder": core.decoder if core.decoder is not None else core.vae_token_decoder,
        "reward head": core.reward_head,
        "continuation head": core.continuation_head,
        "graph module": [
            *core.graph_plugin.parameters(),
            *core.graph_to_deter.parameters(),
            core.graph_gate_logit,
        ],
        "actor": behavior.actor,
        "critic": behavior.critic,
        "slow critic": behavior.slow_critic,
    }
    summary: dict[str, dict[str, int]] = {}
    print(f"training_stage={stage}")
    print("PARAMETER GROUPS:")
    for name, group in groups.items():
        parameters = list(group.parameters()) if isinstance(group, nn.Module) else list(group)
        trainable = sum(p.numel() for p in parameters if p.requires_grad)
        frozen = sum(p.numel() for p in parameters if not p.requires_grad)
        summary[name] = {"trainable": trainable, "frozen": frozen}
        state = "trainable" if trainable else "frozen"
        print(f"  {name:<48} {state:<9} trainable={trainable:,} frozen={frozen:,}")
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_trainable += sum(p.numel() for p in behavior.parameters() if p.requires_grad)
    if total_trainable == 0:
        raise RuntimeError(f"training_stage={stage} has zero trainable parameters")
    return summary


def assert_visual_stage_isolation(
    model: JointGraphVideoWorldModel,
    behavior: DreamerBehavior,
    optimizer_parameters: list[nn.Parameter],
) -> None:
    """Fail loudly if Stage 1 can update anything outside its freeze map."""
    expected = {id(parameter) for parameter in dreamer_visual_parameters(model)}
    actual = {id(parameter) for parameter in optimizer_parameters}
    if actual != expected:
        raise RuntimeError("Dreamer Stage-1 optimizer does not match freeze map")
    named = list(model.named_parameters()) + [
        (f"behavior.{name}", parameter)
        for name, parameter in behavior.named_parameters()
    ]
    unexpected = [
        name for name, parameter in named
        if parameter.requires_grad and id(parameter) not in expected
    ]
    if unexpected:
        raise RuntimeError(
            "Dreamer Stage 1 unexpectedly left parameters trainable: "
            + ", ".join(unexpected[:12])
        )


def assert_world_stage_isolation(
    model: JointGraphVideoWorldModel,
    behavior: DreamerBehavior,
    optimizer_parameters: list[nn.Parameter],
) -> None:
    """Fail loudly unless Stage 2 matches its exact parameter map."""
    expected = {id(parameter) for parameter in dreamer_world_parameters(model)}
    actual = {id(parameter) for parameter in optimizer_parameters}
    if actual != expected:
        raise RuntimeError("Dreamer Stage-2 optimizer does not match freeze map")
    named = list(model.named_parameters()) + [
        (f"behavior.{name}", parameter)
        for name, parameter in behavior.named_parameters()
    ]
    unexpected = [
        name for name, parameter in named
        if parameter.requires_grad and id(parameter) not in expected
    ]
    if unexpected:
        raise RuntimeError(
            "Dreamer Stage 2 unexpectedly left parameters trainable: "
            + ", ".join(unexpected[:12])
        )


def assert_graph_stage_isolation(
    model: JointGraphVideoWorldModel,
    behavior: DreamerBehavior,
    optimizer_parameters: list[nn.Parameter],
    *,
    use_graph: bool,
) -> None:
    """Verify Stage-3 Graph OFF/ON optimizer and frozen observer scope."""
    expected_parameters = dreamer_world_parameters(model)
    if use_graph:
        expected_parameters.extend(dreamer_graph_parameters(model))
    expected = {id(parameter) for parameter in expected_parameters}
    actual = {id(parameter) for parameter in optimizer_parameters}
    if actual != expected:
        raise RuntimeError("Dreamer Stage-3 optimizer does not match freeze map")
    named = list(model.named_parameters()) + [
        (f"behavior.{name}", parameter)
        for name, parameter in behavior.named_parameters()
    ]
    unexpected = [
        name for name, parameter in named
        if parameter.requires_grad and id(parameter) not in expected
    ]
    if unexpected:
        raise RuntimeError(
            "Dreamer Stage 3 unexpectedly left parameters trainable: "
            + ", ".join(unexpected[:12])
        )
    core = model.dreamerv3_core
    assert core is not None
    if core.graph_plugin.graph_observer == "explicit" and any(
        parameter.requires_grad for parameter in core.graph_builder.parameters()
    ):
        raise RuntimeError("Stage-3 explicit graph observer must remain frozen")


def assert_no_frozen_gradients(
    model: JointGraphVideoWorldModel,
    behavior: DreamerBehavior,
    allowed_parameters: list[nn.Parameter],
    stage: str,
) -> None:
    allowed = {id(parameter) for parameter in allowed_parameters}
    named = list(model.named_parameters()) + [
        (f"behavior.{name}", parameter)
        for name, parameter in behavior.named_parameters()
    ]
    violations = [
        name for name, parameter in named
        if id(parameter) not in allowed and parameter.grad is not None
    ]
    if violations:
        raise RuntimeError(
            f"Dreamer {stage} frozen parameters received gradients: "
            + ", ".join(violations[:12])
        )


def warmup_from_start(epoch: int, start_epoch: int, duration: int) -> float:
    if epoch < start_epoch:
        return 0.0
    return min(1.0, (epoch - start_epoch + 1) / max(1, duration))


def nonfinite_gradient_names(
    module: nn.Module,
    *,
    limit: int = 8,
) -> list[str]:
    """Return a short, actionable list of parameters with NaN/Inf gradients."""
    names: list[str] = []
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            names.append(name)
            if len(names) >= limit:
                break
    return names


def gradient_l2_norm(parameters) -> float:
    """Compute a diagnostic norm without modifying or clipping gradients."""
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(squared)


def report_nonfinite_step(
    stage: str,
    epoch: int,
    update: int,
    loss: Tensor,
    module: nn.Module,
    grad_scaler: torch.amp.GradScaler,
) -> None:
    bad_names = nonfinite_gradient_names(module)
    joined = ", ".join(bad_names) if bad_names else "no named gradient found"
    print(
        "WARNING: skipped non-finite",
        f"stage={stage}",
        f"epoch={epoch}",
        f"update={update + 1}",
        f"loss={float(loss.detach()):.6g}",
        f"amp_scale={grad_scaler.get_scale():.6g}",
        f"parameters=[{joined}]",
    )


def preprocess(images: Tensor) -> Tensor:
    return images.float().div(255.0)


def format_metric_value(value: Any) -> str:
    """Format heterogeneous diagnostic values for compact console logging."""
    if value is None:
        return "None"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, Tensor):
        if value.numel() != 1:
            return f"Tensor(shape={tuple(value.shape)})"
        value = value.detach().item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, dict)):
        compact = repr(value)
        return compact if len(compact) <= 120 else compact[:117] + "..."
    return str(value)
def dreamer_online_loss(
    output: dict[str, Any],
    images: Tensor,
    rewards: Tensor,
    dones: Tensor,
    *,
    use_graph: bool,
    graph_scale: float,
    alignment_scale: float,
    alignment_weight: float,
    variance_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    if output["reconstruction_distribution"] is not None:
        reconstruction = -output["reconstruction_distribution"].log_prob(
            images - 0.5
        ).mean()
    else:
        reconstruction = (output["reconstruction"] - images).square().mean()
    reward = -output["reward_distribution"].log_prob(
        rewards.unsqueeze(-1)
    ).mean()
    continuation = -output["continuation_distribution"].log_prob(
        (1.0 - dones).unsqueeze(-1)
    ).mean()
    dynamics = output["dynamics_kl"]
    representation = output["representation_kl"]
    total = reconstruction + reward + continuation + 0.5 * dynamics + 0.1 * representation
    graph = total.new_zeros(())
    alignment = total.new_zeros(())
    variance = total.new_zeros(())
    if use_graph:
        mask = output["transition_mask"]
        graph_values = (
            output["graph_next"] - output["target_graph"]
        ).square().mean(dim=(-2, -1))
        graph = (graph_values * mask).sum() / mask.sum().clamp_min(1.0)
        step_weight = mask.mean(dim=0)
        alignment = (
            output["alignment_by_step"] * step_weight
        ).sum() / step_weight.sum().clamp_min(1.0)
        variance = (
            output["alignment_variance_by_step"] * step_weight
        ).sum() / step_weight.sum().clamp_min(1.0)
        total = total + 0.2 * graph_scale * graph
        total = total + alignment_weight * alignment_scale * alignment
        total = total + variance_weight * alignment_scale * variance
    metrics = {
        "world_total": float(total.detach()),
        "reconstruction": float(reconstruction.detach()),
        "reward": float(reward.detach()),
        "continuation": float(continuation.detach()),
        "dynamics_kl": float(dynamics.detach()),
        "representation_kl": float(representation.detach()),
        "graph": float(graph.detach()),
        "alignment": float(alignment.detach()),
        "alignment_variance": float(variance.detach()),
    }
    return total, metrics


def dreamer_visual_loss(
    output: dict[str, Any], images: Tensor
) -> tuple[Tensor, dict[str, float]]:
    """Stage-1 loss: observation reconstruction and nothing else."""
    flat_images = images.reshape(-1, *images.shape[-3:])
    distribution = output["reconstruction_distribution"]
    if distribution is not None:
        reconstruction_loss = -distribution.log_prob(flat_images - 0.5).mean()
    else:
        reconstruction_loss = (
            output["reconstruction"].reshape_as(images) - images
        ).square().mean()
    reconstruction = output["reconstruction"].reshape_as(images)
    error = reconstruction - images
    return reconstruction_loss, {
        "world_total": float(reconstruction_loss.detach()),
        "reconstruction": float(reconstruction_loss.detach()),
        "reconstruction_mse": float(error.detach().square().mean()),
        "reconstruction_mae": float(error.detach().abs().mean()),
        "reward": 0.0,
        "continuation": 0.0,
        "dynamics_kl": 0.0,
        "representation_kl": 0.0,
        "graph": 0.0,
        "alignment": 0.0,
        "alignment_variance": 0.0,
    }


def dreamer_world_loss(
    output: dict[str, Any],
    images: Tensor,
    rewards: Tensor,
    dones: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Stage-2 temporal objective with reconstruction as monitoring only."""
    if output["reconstruction_distribution"] is not None:
        reconstruction = -output["reconstruction_distribution"].log_prob(
            images - 0.5
        ).mean()
    else:
        reconstruction = (output["reconstruction"] - images).square().mean()
    reward = -output["reward_distribution"].log_prob(
        rewards.unsqueeze(-1)
    ).mean()
    continuation = -output["continuation_distribution"].log_prob(
        (1.0 - dones).unsqueeze(-1)
    ).mean()
    dynamics = output["dynamics_kl"]
    representation = output["representation_kl"]
    # The representation KL has a detached prior and would only optimize the
    # frozen Stage-1 posterior. Keep it visible, but do not include it here.
    total = reward + continuation + 0.5 * dynamics
    return total, {
        "world_total": float(total.detach()),
        "reconstruction": float(reconstruction.detach()),
        "reward": float(reward.detach()),
        "continuation": float(continuation.detach()),
        "dynamics_kl": float(dynamics.detach()),
        "representation_kl_monitor": float(representation.detach()),
        "graph": 0.0,
        "alignment": 0.0,
        "alignment_variance": 0.0,
    }


def dreamer_graph_loss(
    output: dict[str, Any],
    images: Tensor,
    rewards: Tensor,
    dones: Tensor,
    *,
    use_graph: bool,
    graph_scale: float,
    alignment_scale: float,
    alignment_weight: float,
    variance_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    """Stage-2 objective plus the existing Dreamer Graph objectives."""
    total, metrics = dreamer_world_loss(output, images, rewards, dones)
    if not use_graph:
        return total, metrics
    mask = output["transition_mask"]
    graph_values = (
        output["graph_next"] - output["target_graph"]
    ).square().mean(dim=(-2, -1))
    graph = (graph_values * mask).sum() / mask.sum().clamp_min(1.0)
    step_weight = mask.mean(dim=0)
    alignment = (
        output["alignment_by_step"] * step_weight
    ).sum() / step_weight.sum().clamp_min(1.0)
    variance = (
        output["alignment_variance_by_step"] * step_weight
    ).sum() / step_weight.sum().clamp_min(1.0)
    total = total + 0.2 * graph_scale * graph
    total = total + alignment_weight * alignment_scale * alignment
    total = total + variance_weight * alignment_scale * variance
    metrics.update({
        "world_total": float(total.detach()),
        "graph": float(graph.detach()),
        "alignment": float(alignment.detach()),
        "alignment_variance": float(variance.detach()),
    })
    return total, metrics


def fixed_reconstruction_frames(
    replay: SlidingWindowReplay,
    output: Path,
    seed: int,
    count: int = 12,
) -> Tensor:
    """Load or create the immutable uint8 frame set used across epochs."""
    directory = output / "diagnostics" / "reconstruction"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "fixed_frames.pt"
    if path.exists():
        frames = torch.load(path, map_location="cpu", weights_only=True)
    else:
        with isolated_random_state(seed):
            batch = replay.sample(
                max(1, math.ceil(count / replay.sequence_length)),
                torch.device("cpu"),
            )
        candidates = batch["images"].reshape(-1, *batch["images"].shape[-3:])
        indices = torch.linspace(
            0, candidates.shape[0] - 1, steps=min(count, candidates.shape[0])
        ).round().long()
        frames = candidates[indices].to(torch.uint8).contiguous()
        torch.save(frames, path)
    if frames.ndim != 4 or frames.shape[1:] != (3, 64, 64):
        raise RuntimeError(f"Invalid fixed reconstruction frames in {path}")
    return frames


@torch.no_grad()
def save_reconstruction_diagnostic(
    model: JointGraphVideoWorldModel,
    frames: Tensor,
    device: torch.device,
    path: Path,
) -> dict[str, float]:
    """Save a fixed-frame original/reconstruction paired grid."""
    core = model.dreamerv3_core
    if core is None:
        raise ValueError("Dreamer reconstruction diagnostics require DreamerV3")
    was_training = model.training
    model.eval()
    source = preprocess(frames.to(device))
    reconstruction = core.reconstruct_observations(source)["reconstruction"]
    model.train(was_training)
    source_u8 = source.mul(255).round().clamp(0, 255).byte().cpu()
    reconstruction_u8 = reconstruction.mul(255).round().clamp(0, 255).byte().cpu()
    columns = min(4, frames.shape[0])
    rows = math.ceil(frames.shape[0] / columns)
    gap = 2
    canvas = np.full(
        (rows * 64 + (rows - 1) * gap, columns * 128 + (columns - 1) * gap, 3),
        255,
        dtype=np.uint8,
    )
    for index, (original, predicted) in enumerate(zip(source_u8, reconstruction_u8)):
        row, column = divmod(index, columns)
        y = row * (64 + gap)
        x = column * (128 + gap)
        canvas[y : y + 64, x : x + 64] = original.permute(1, 2, 0).numpy()
        canvas[y : y + 64, x + 64 : x + 128] = predicted.permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)
    error = reconstruction.float() - source.float()
    return {
        "fixed_reconstruction_mse": float(error.square().mean()),
        "fixed_reconstruction_mae": float(error.abs().mean()),
    }


def fixed_world_sequences(
    replay: SlidingWindowReplay,
    output: Path,
    seed: int,
    count: int = 4,
    source_path: Path | None = None,
) -> dict[str, Tensor]:
    """Load or create immutable replay probes for Stage-2 diagnostics."""
    directory = output / "diagnostics" / "prediction"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "fixed_sequences.pt"
    if path.exists():
        batch = torch.load(path, map_location="cpu", weights_only=True)
    elif source_path is not None and source_path.exists():
        batch = torch.load(source_path, map_location="cpu", weights_only=True)
        batch = {key: value.cpu().contiguous() for key, value in batch.items()}
        torch.save(batch, path)
        print(f"reused Stage-2 fixed probe sequences: {source_path}")
    else:
        with isolated_random_state(seed):
            sampled = replay.sample(
                count,
                torch.device("cpu"),
                avoid_internal_resets=True,
            )
        batch = {key: value.cpu().contiguous() for key, value in sampled.items()}
        torch.save(batch, path)
    if batch["images"].ndim != 5 or batch["images"].shape[1] < 51:
        raise RuntimeError(
            "Stage-2 fixed sequences require at least 51 replay frames"
        )
    return batch


def _save_prediction_pairs(
    ground_truth: Tensor,
    prediction: Tensor,
    horizon: int,
    path: Path,
) -> None:
    gt = ground_truth.mul(255).round().clamp(0, 255).byte().cpu()
    pred = prediction.mul(255).round().clamp(0, 255).byte().cpu()
    header = 18
    gap = 2
    canvas = np.full(
        (header + len(gt) * 64 + max(0, len(gt) - 1) * gap, 128, 3),
        255,
        dtype=np.uint8,
    )
    for row, (target, estimate) in enumerate(zip(gt, pred)):
        y = header + row * (64 + gap)
        canvas[y : y + 64, :64] = target.permute(1, 2, 0).numpy()
        canvas[y : y + 64, 64:] = estimate.permute(1, 2, 0).numpy()
    image = Image.fromarray(canvas)
    ImageDraw.Draw(image).text(
        (2, 2), f"H={horizon}: GT | prediction", fill=(0, 0, 0)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _save_reward_curve(target: Tensor, prediction: Tensor, path: Path) -> None:
    target_np = target.detach().float().cpu().numpy()
    prediction_np = prediction.detach().float().cpu().numpy()
    width, height, margin = 800, 320, 36
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black")
    values = np.concatenate((target_np, prediction_np))
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-8:
        high = low + 1.0

    def points(values_row: np.ndarray) -> list[tuple[float, float]]:
        denominator = max(1, len(values_row) - 1)
        return [
            (
                margin + index * (width - 2 * margin) / denominator,
                height - margin - (float(value) - low) * (height - 2 * margin) / (high - low),
            )
            for index, value in enumerate(values_row)
        ]

    if len(target_np) > 1:
        draw.line(points(target_np), fill=(30, 100, 220), width=2)
        draw.line(points(prediction_np), fill=(220, 60, 50), width=2)
    draw.text((margin, 8), "GT reward (blue) | predicted reward (red)", fill="black")
    draw.text((2, margin), f"{high:.3g}", fill="black")
    draw.text((2, height - margin - 8), f"{low:.3g}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def save_world_diagnostics(
    model: JointGraphVideoWorldModel,
    fixed: dict[str, Tensor],
    device: torch.device,
    epoch: int,
    output: Path,
    *,
    use_graph: bool = False,
) -> dict[str, Any]:
    """Evaluate fixed teacher-forced and open-loop Stage-2 probes."""
    core = model.dreamerv3_core
    if core is None:
        raise ValueError("World diagnostics require DreamerV3")
    was_training = model.training
    model.eval()
    images = preprocess(fixed["images"].to(device))
    actions = fixed["actions"].to(device)
    rewards = fixed["rewards"].to(device).float()
    firsts = fixed["firsts"].to(device)
    teacher = core.forward_online_sequence(
        images, actions, firsts, use_graph=use_graph
    )
    one_step, _ = core._decode_feature(teacher["prior_features"][:, 1:])
    one_step_mse = (one_step - images[:, 1:]).square().mean()
    rollout = core.open_loop_prediction(
        images[:, 0], actions[:, 1:51], use_graph=use_graph
    )
    horizons = (1, 5, 10, 20, 50)
    metrics: dict[str, Any] = {
        "epoch": epoch,
        "diagnostic_role": "stage",
        "graph_enabled": use_graph,
        "graph_observer": core.graph_plugin.graph_observer if use_graph else "none",
        "one_step_reconstruction_mse": float(one_step_mse),
    }
    prediction_dir = output / "diagnostics" / "prediction"
    for horizon in horizons:
        prediction = rollout["images"][:, horizon - 1]
        target = images[:, horizon]
        metrics[f"horizon_{horizon}_mse"] = float(
            (prediction - target).square().mean()
        )
        _save_prediction_pairs(
            target,
            prediction,
            horizon,
            prediction_dir / f"epoch_{epoch:04d}_h{horizon:02d}.png",
        )

    reward_prediction = teacher["reward_distribution"].mode().squeeze(-1)[:, 1:]
    reward_target = rewards[:, 1:]
    flat_prediction = reward_prediction.reshape(-1).float()
    flat_target = reward_target.reshape(-1).float()
    absolute = (flat_prediction - flat_target).abs()
    squared = (flat_prediction - flat_target).square()
    q10 = torch.quantile(flat_target, 0.10)
    q90 = torch.quantile(flat_target, 0.90)
    low_mask = flat_target <= q10
    positive_mask = flat_target > 0
    top_mask = flat_target >= q90
    centered_target = flat_target - flat_target.mean()
    centered_prediction = flat_prediction - flat_prediction.mean()
    correlation_denominator = (
        centered_target.square().sum().sqrt()
        * centered_prediction.square().sum().sqrt()
    )
    correlation = (
        float((centered_target * centered_prediction).sum() / correlation_denominator)
        if float(correlation_denominator) > 1e-8
        else None
    )
    metrics.update({
        "reward_mae_all": float(absolute.mean()),
        "reward_rmse_all": float(squared.mean().sqrt()),
        "reward_corr_all": correlation,
        "reward_mae_low_reward": float(absolute[low_mask].mean()),
        "reward_mae_positive_reward": (
            float(absolute[positive_mask].mean())
            if bool(positive_mask.any()) else None
        ),
        "reward_mae_top10_reward": float(absolute[top_mask].mean()),
        "reward_low_quantile": float(q10),
        "reward_top10_threshold": float(q90),
        "prior_posterior_kl": float(teacher["dynamics_kl_raw"]),
        "posterior_feature_norm": float(
            teacher["posterior_features"].float().norm(dim=-1).mean()
        ),
        "prior_feature_norm": float(
            teacher["prior_features"].float().norm(dim=-1).mean()
        ),
        "reward_prediction_min": float(flat_prediction.min()),
        "reward_prediction_max": float(flat_prediction.max()),
    })
    if use_graph:
        correction = teacher["graph_correction"].float()
        base_prior_deter = teacher["base_prior_deter"].float()
        correction_rms = correction.square().mean().sqrt()
        prior_rms = base_prior_deter.square().mean().sqrt()
        mask = teacher["transition_mask"]
        graph_values = (
            teacher["graph_next"] - teacher["target_graph"]
        ).square().mean(dim=(-2, -1))
        graph_loss = (graph_values * mask).sum() / mask.sum().clamp_min(1.0)
        step_weight = mask.mean(dim=0)
        alignment = (
            teacher["alignment_by_step"] * step_weight
        ).sum() / step_weight.sum().clamp_min(1.0)
        variance = (
            teacher["alignment_variance_by_step"] * step_weight
        ).sum() / step_weight.sum().clamp_min(1.0)
        role_names = getattr(core.graph_builder, "role_names", None)
        padding_indices = (
            [index for index, name in enumerate(role_names) if name == "padding"]
            if role_names is not None
            else []
        )
        padding_fixed = True
        if padding_indices:
            padding_fixed = bool(
                (
                    teacher["graph_next"][:, :, padding_indices]
                    - teacher["graph"][:, :, padding_indices]
                ).abs().max() <= 1e-8
            )
        metrics.update({
            "graph": float(graph_loss),
            "alignment": float(alignment),
            "alignment_variance": float(variance),
            "graph_correction_rms": float(correction_rms),
            "graph_correction_max": float(correction.abs().max()),
            "graph_correction_ratio_to_prior": float(
                correction_rms / prior_rms.clamp_min(1e-8)
            ),
            "graph_gate": float(torch.sigmoid(core.graph_gate_logit)),
            "graph_conditioning_strength": float(core.conditioning_strength),
            "graph_conditioned_prior_feature_norm": float(
                teacher["prior_features"].float().norm(dim=-1).mean()
            ),
            "active_graph_nodes": core.graph_plugin.active_graph_node_count,
            "total_graph_nodes": core.graph_plugin.graph_nodes,
            "padding_nodes_fixed_zero": padding_fixed,
        })
    else:
        metrics.update({
            "graph": 0.0,
            "alignment": 0.0,
            "alignment_variance": 0.0,
            "graph_correction_rms": 0.0,
            "graph_correction_max": 0.0,
            "graph_correction_ratio_to_prior": 0.0,
            "graph_gate": 0.0,
            "graph_conditioning_strength": 0.0,
            "graph_conditioned_prior_feature_norm": float(
                teacher["prior_features"].float().norm(dim=-1).mean()
            ),
            "active_graph_nodes": 0,
            "total_graph_nodes": core.graph_plugin.graph_nodes,
            "padding_nodes_fixed_zero": True,
        })
    reward_path = (
        output / "diagnostics" / "reward" / f"epoch_{epoch:04d}_reward_curve.png"
    )
    _save_reward_curve(reward_target[0], reward_prediction[0], reward_path)
    model.train(was_training)
    return metrics


def append_world_metrics(
    output: Path, metrics: dict[str, Any]
) -> None:
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    with (diagnostics / "world_metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, allow_nan=False) + "\n")
    csv_path = diagnostics / "world_metrics.csv"
    fieldnames = list(metrics)
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            previous_fieldnames = list(reader.fieldnames or [])
            previous_rows = list(reader)
        if previous_fieldnames != fieldnames:
            fieldnames = list(dict.fromkeys([*previous_fieldnames, *fieldnames]))
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(previous_rows)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


def flatten_state(states: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        key: value.flatten(0, 1).detach()
        for key, value in states.items()
    }


def behavior_losses(
    model: JointGraphVideoWorldModel,
    behavior: DreamerBehavior,
    output: dict[str, Any],
    dones: Tensor,
    *,
    use_graph: bool,
    horizon: int,
    gamma: float,
    lambda_td: float,
    entropy_weight: float,
    slow_regularization: float,
) -> tuple[Tensor, dict[str, Tensor], dict[str, float]]:
    core = model.dreamerv3_core
    if core is None:
        raise ValueError("Online actor--critic supports the DreamerV3 backbone")
    starts = flatten_state(output["posterior_states"])
    start_graph = start_relations = None
    if use_graph:
        start_graph = output["posterior_graph"].flatten(0, 1).detach()
        start_relations = output["posterior_relations"].flatten(0, 1).detach()
    imagined = core.imagine_behavior(
        behavior.actor,
        starts,
        horizon,
        use_graph=use_graph,
        start_graph=start_graph,
        start_relations=start_relations,
    )
    true_first = (1.0 - dones.flatten(0, 1))[:, None, None]
    features = imagined["features"]
    rewards = imagined["rewards"]
    if rewards.ndim == 2:
        rewards = rewards.unsqueeze(-1)
    continuations = imagined["continuations"]
    if continuations.ndim == 2:
        continuations = continuations.unsqueeze(-1)
    continuations = torch.cat((true_first, continuations), dim=1)
    value_dist = behavior.critic(features)
    values = value_dist.mode()
    weights = torch.cumprod(gamma * continuations, dim=1).detach() / gamma
    returns = lambda_returns(
        rewards,
        values[:, 1:],
        gamma * continuations[:, 1:],
        lambda_td,
    )
    offset, scale = behavior.update_return_scale(returns)
    advantage = (
        (returns - offset) / scale
        - (values[:, :-1] - offset) / scale
    ).squeeze(-1)
    policy_dist = behavior.actor(features.detach())
    entropy = policy_dist.entropy()[:, :-1]
    actor_objective = advantage + entropy_weight * entropy
    actor_loss = -(
        actor_objective * weights[:, :-1].squeeze(-1)
    ).mean()

    cache: dict[str, Tensor] = {
        "returns": returns.detach(),
        "weights": weights.detach()[:, :-1],
        "features": features.detach()[:, :-1],
    }
    imagined_actions = imagined["actions"].detach()
    return actor_loss, cache, {
        "actor": float(actor_loss.detach()),
        "return": float(returns.detach().mean()),
        "entropy": float(entropy.detach().mean()),
        "imagined_action_abs_mean": float(imagined_actions.abs().mean()),
        "imagined_action_saturation_fraction": float(
            (imagined_actions.abs() > 0.95).float().mean()
        ),
        "imagined_action_batch_mean_abs": float(
            imagined_actions.mean(dim=0).abs().mean()
        ),
        "imagined_action_population_std": float(
            imagined_actions.std(dim=0, unbiased=False).mean()
        ),
        "imagined_reward_mean": float(rewards.detach().mean()),
        "imagined_reward_max": float(rewards.detach().max()),
        "imagined_reward_above_0_5_fraction": float(
            (rewards.detach() > 0.5).float().mean()
        ),
        "imagined_reward_above_1_0_fraction": float(
            (rewards.detach() > 1.0).float().mean()
        ),
    }


def critic_loss_from_cache(
    behavior: DreamerBehavior,
    cache: dict[str, Tensor],
    slow_regularization: float,
) -> Tensor:
    critic_dist = behavior.critic(cache["features"])
    objective = critic_dist.log_prob(cache["returns"])
    with torch.no_grad():
        slow_target = behavior.slow_critic(cache["features"]).mode()
    objective = objective + slow_regularization * critic_dist.log_prob(
        slow_target
    )
    return -(objective * cache["weights"].squeeze(-1)).mean()


@torch.no_grad()
def policy_actions(
    model: JointGraphVideoWorldModel,
    behavior: DreamerBehavior,
    images: list[np.ndarray],
    device: torch.device,
    *,
    use_graph: bool,
    recurrent: list[dict[str, Tensor] | None],
    previous_actions: Tensor | None,
    graph_states: list[Tensor | None],
    relation_states: list[Tensor | None],
    reset_mask: Tensor | None = None,
    mode: bool = False,
) -> tuple[np.ndarray, list[dict[str, Tensor] | None], Tensor | None, list[Tensor | None], list[Tensor | None]]:
    if model.dreamerv3_core is None:
        raise ValueError("Online actor--critic supports the DreamerV3 backbone")
    image = torch.from_numpy(np.stack(images)).to(device).float().div(255.0)
    batch_state = None
    if recurrent[0] is not None:
        batch_state = {
            key: torch.cat([state[key] for state in recurrent if state is not None], 0)
            for key in recurrent[0]
        }
    batch_graph = None if graph_states[0] is None else torch.cat(
        [value for value in graph_states if value is not None], 0
    )
    batch_relations = None if relation_states[0] is None else torch.cat(
        [value for value in relation_states if value is not None], 0
    )
    state, feature, graph, relations = model.dreamerv3_core.policy_observe(
        image,
        batch_state,
        previous_actions,
        use_graph=use_graph,
        previous_graph=batch_graph,
        previous_relations=batch_relations,
        reset_mask=reset_mask,
    )
    recurrent = [
        {key: value[index : index + 1] for key, value in state.items()}
        for index in range(image.shape[0])
    ]
    graph_states = [
        None if graph is None else graph[index : index + 1]
        for index in range(image.shape[0])
    ]
    relation_states = [
        None if relations is None else relations[index : index + 1]
        for index in range(image.shape[0])
    ]
    distribution = behavior.actor(feature)
    action = distribution.mode() if mode else distribution.sample()
    return (
        action.clamp(-1.0, 1.0).cpu().numpy(),
        recurrent,
        action,
        graph_states,
        relation_states,
    )


@torch.no_grad()
def _evaluate_impl(
    model: JointGraphVideoWorldModel,
    behavior: DreamerBehavior,
    domain: str,
    task: str,
    episodes: int,
    seed: int,
    device: torch.device,
    *,
    use_graph: bool,
    output: Path,
    policy_mode: str = "sample",
) -> dict[str, float]:
    if policy_mode not in {"sample", "mode"}:
        raise ValueError("policy_mode must be 'sample' or 'mode'")
    env = OnlineDMCBatch(domain, task, 1, seed, 64, 2, 0)
    returns: list[float] = []
    evaluated_actions: list[np.ndarray] = []
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    behavior.eval()
    for episode in range(episodes):
        images = env.reset_all()
        recurrent: list[dict[str, Tensor] | None] = [None]
        previous_action = None
        graphs: list[Tensor | None] = [None]
        relations: list[Tensor | None] = [None]
        frames = [images[0].transpose(1, 2, 0)]
        episode_actions: list[np.ndarray] = []
        episode_rewards: list[float] = []
        total = 0.0
        while True:
            actions, recurrent, previous_action, graphs, relations = policy_actions(
                model,
                behavior,
                images,
                device,
                use_graph=use_graph,
                recurrent=recurrent,
                previous_actions=previous_action,
                graph_states=graphs,
                relation_states=relations,
                mode=(policy_mode == "mode"),
            )
            images, rewards, lasts, _ = env.step(actions)
            episode_actions.append(actions[0].copy())
            episode_rewards.append(float(rewards[0]))
            total += float(rewards[0])
            frames.append(images[0].transpose(1, 2, 0))
            if bool(lasts[0]):
                break
        returns.append(total)
        evaluated_actions.append(np.stack(episode_actions))
        episode_payload: dict[str, Any] = {
            "rgb": np.stack(frames),
            "actions": np.stack(episode_actions),
            "rewards": np.asarray(episode_rewards, dtype=np.float32),
            "episode_return": np.float32(total),
        }
        np.savez_compressed(
            output / f"episode_{episode:03d}.npz",
            **episode_payload,
        )
    metrics = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
    }
    if evaluated_actions:
        flat_actions = np.concatenate(evaluated_actions, axis=0)
        metrics.update({
            "action_abs_mean": float(np.abs(flat_actions).mean()),
            "action_saturation_fraction": float(
                (np.abs(flat_actions) > 0.95).mean()
            ),
            "action_std": float(flat_actions.std()),
        })
    summary: dict[str, Any] = dict(metrics)
    summary.update({"policy_mode": policy_mode, "seed": int(seed)})
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return metrics


@torch.no_grad()
def evaluate(
    model: JointGraphVideoWorldModel,
    behavior: DreamerBehavior,
    domain: str,
    task: str,
    episodes: int,
    seed: int,
    device: torch.device,
    *,
    use_graph: bool,
    output: Path,
    policy_mode: str = "sample",
) -> dict[str, float]:
    return _evaluate_impl(
        model,
        behavior,
        domain,
        task,
        episodes,
        seed,
        device,
        use_graph=use_graph,
        output=output,
        policy_mode=policy_mode,
    )


def main() -> None:
    defaults = BURCHIM_DMC_DEFAULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="reacher")
    parser.add_argument("--task", default="easy")
    parser.add_argument("--backbone", choices=("dreamerv3",), default="dreamerv3")
    parser.add_argument(
        "--training-stage",
        choices=("visual", "world", "graph", "joint"),
        default="joint",
        help=(
            "DreamerV3-only staged training. 'joint' preserves the existing "
            "world-model plus actor/critic behavior; Stage 1 'visual' trains "
            "only encoder, RSSM observation posterior, and decoder."
        ),
    )
    parser.add_argument(
        "--stage-checkpoint",
        type=Path,
        help="DreamerV3 source checkpoint used to initialize a new stage.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--resume-replay",
        type=Path,
        help="Explicit replay checkpoint path when branching to a new output.",
    )
    parser.add_argument("--disable-graph-conditioning", action="store_true")
    parser.add_argument("--graph-observer", choices=("proxy", "explicit"), default="proxy")
    parser.add_argument("--graph-observer-checkpoint", type=Path)
    parser.add_argument("--reltr-checkpoint", type=Path, default=Path("Models/reltr/checkpoint0149.pth"))
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--steps-per-epoch", type=int, default=defaults["steps_per_epoch"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--sequence-length", type=int, default=defaults["sequence_length"])
    parser.add_argument("--imagination-horizon", type=int, default=defaults["imagination_horizon"])
    parser.add_argument(
        "--dreamer-behavior-update-period",
        type=int,
        default=1,
        help=(
            "Run the expensive Dreamer actor/critic imagination update once "
            "every N world-model updates. Environment policy decisions and "
            "action_repeat remain unchanged."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=defaults["num_envs"])
    parser.add_argument("--prefill-windows", type=int, default=defaults["prefill_windows"])
    parser.add_argument("--replay-capacity", type=int, default=defaults["replay_capacity"])
    parser.add_argument("--env-step-period", type=int, default=defaults["env_step_period"])
    parser.add_argument("--eval-episodes", type=int, default=defaults["eval_episodes"])
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--graph-warmup-epochs", type=int, default=10)
    parser.add_argument("--alignment-warmup-epochs", type=int, default=20)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--alignment-variance-weight", type=float, default=0.1)
    parser.add_argument("--max-graph-correction", type=float, default=0.25)
    parser.add_argument("--graph-correction-gate-init", type=float, default=0.11920292202211755)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--graph-nodes", type=int, default=8)
    parser.add_argument("--message-steps", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="Permit non-reportable small protocol overrides.")
    args = parser.parse_args()

    if args.dreamer_behavior_update_period < 1:
        parser.error("--dreamer-behavior-update-period must be positive")

    if not args.smoke:
        exact = {
            "batch_size": 16,
            "sequence_length": 64,
            "imagination_horizon": 15,
            "num_envs": 4,
            "env_step_period": 512,
        }
        mismatches = [
            f"--{key.replace('_', '-')}={getattr(args, key)} (expected {value})"
            for key, value in exact.items()
            if getattr(args, key) != value
        ]
        if args.precision != "float16":
            mismatches.append(
                f"--precision={args.precision} (expected float16)"
            )
        if mismatches:
            parser.error(
                "Reportable Burchim DMC protocol mismatch: " + ", ".join(mismatches)
            )
    if args.graph_observer_checkpoint and args.graph_observer != "explicit":
        parser.error("--graph-observer-checkpoint requires --graph-observer explicit")
    if args.resume_replay is not None and args.resume is None:
        parser.error("--resume-replay requires --resume")
    if args.stage_checkpoint is not None and args.resume is not None:
        parser.error("--stage-checkpoint cannot be combined with --resume")
    if args.stage_checkpoint is not None and args.training_stage not in {"world", "graph"}:
        parser.error("--stage-checkpoint is only valid for world/graph stages")
    if (
        args.graph_observer == "explicit"
        and not args.graph_observer_checkpoint
        and not args.resume
        and (
            args.training_stage == "joint"
            or (
                args.training_stage == "graph"
                and not args.disable_graph_conditioning
            )
        )
    ):
        parser.error("Online explicit graph runs require --graph-observer-checkpoint")
    if args.training_stage in {"world", "graph"} and args.stage_checkpoint is None and args.resume is None:
        parser.error(f"--training-stage {args.training_stage} requires --stage-checkpoint or --resume")
    if args.training_stage in {"world", "graph"} and args.sequence_length < 51:
        parser.error(f"--training-stage {args.training_stage} requires --sequence-length >= 51 for H=50 diagnostics")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    env = OnlineDMCBatch(args.domain, args.task, args.num_envs, args.seed, 64, 2, 0)
    config = vars(args).copy()
    config.update({
        "protocol": "burchim_dreamerv3_online_dmc",
        "image_size": 64,
        "action_repeat": 2,
        "camera": 0,
        "actor_grad": "dynamics",
        "gamma": defaults["gamma"],
        "lambda_td": defaults["lambda_td"],
        "pinned_burchim_commit": "a6ee1d86cf3e387694d58a49aa3e81a5b010d8c5",
        "action_dim": env.action_dim,
        "dreamer_training_stage_version": (
            DREAMER_TRAINING_STAGE_VERSION
            if args.backbone == "dreamerv3" else None
        ),
    })
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    (args.output / "config.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    model = JointGraphVideoWorldModel(
        action_dim=env.action_dim,
        backbone=args.backbone,
        node_dim=args.node_dim,
        graph_nodes=args.graph_nodes,
        message_steps=args.message_steps,
        max_graph_correction=args.max_graph_correction,
        graph_correction_gate_init=args.graph_correction_gate_init,
        graph_observer=args.graph_observer,
        graph_backbone="reltr_resnet50",
        graph_backbone_pretrained=False,
        freeze_graph_backbone=True,
        reltr_checkpoint=str(args.reltr_checkpoint),
    ).to(device)
    source_stage_checkpoint: str | None = None
    if args.stage_checkpoint is not None:
        expected_source_stage = (
            "world" if args.training_stage == "graph" else "visual"
        )
        stage_payload = load_dreamer_stage_checkpoint(
            model, args.stage_checkpoint, device, expected_source_stage
        )
        source_stage_checkpoint = str(args.stage_checkpoint)
        print(
            "loaded stage checkpoint:",
            f"source_stage={expected_source_stage}",
            f"path={args.stage_checkpoint}",
        )
    if args.graph_observer_checkpoint and not args.resume:
        load_graph_observer_checkpoint(model, args.graph_observer_checkpoint, device)
    if args.graph_observer == "explicit":
        model.graph_builder.requires_grad_(False)
        model.graph_builder.eval()
    # Stage 1 explicitly excludes all graph inference, losses, and updates.
    # The joint path retains the historical graph-off/on behavior unchanged.
    use_graph = (
        not args.disable_graph_conditioning
        and args.training_stage in {"joint", "graph"}
    )
    # DreamerV3 retains the pinned Burchim actor/value networks and RSSM
    # feature interface.
    behavior: DreamerBehavior = DreamerBehavior(
        env.action_dim, model.dreamerv3_core.feat_dim
    ).to(device)
    staged_world_parameters = configure_dreamer_training_stage(
        model, behavior, args.training_stage, use_graph=use_graph
    )
    parameter_groups = report_training_stage(
        model, behavior, args.training_stage
    )
    replay = SlidingWindowReplay(args.sequence_length, args.replay_capacity)
    world_parameter_list = staged_world_parameters
    if args.training_stage == "visual":
        assert_visual_stage_isolation(
            model, behavior, world_parameter_list
        )
    elif args.training_stage == "world":
        assert_world_stage_isolation(
            model, behavior, world_parameter_list
        )
    elif args.training_stage == "graph":
        assert_graph_stage_isolation(
            model,
            behavior,
            world_parameter_list,
            use_graph=use_graph,
        )
    world_optimizer = torch.optim.Adam(
        world_parameter_list, lr=1e-4, eps=1e-8
    )
    actor_parameters = list(behavior.actor.parameters())
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=3e-5, eps=1e-5)
    critic_optimizer = torch.optim.Adam(
        behavior.critic.parameters(), lr=3e-5, eps=1e-5
    )
    use_amp = device.type == "cuda" and args.precision == "float16"
    grad_scaler = torch.amp.GradScaler(
        device.type, init_scale=32.0, enabled=use_amp
    )
    start_epoch = 1
    global_step = 0
    action_step = 0
    best_return = -math.inf
    best_visual_loss = math.inf
    best_world_score = math.inf
    best_graph_score = math.inf
    images = env.reset_all()
    for index, image in enumerate(images):
        replay.start_with_action_dim(index, image, env.action_dim)
    recurrent: list[dict[str, Tensor] | None] = [None] * args.num_envs
    previous_actions: Tensor | None = None
    graph_states: list[Tensor | None] = [None] * args.num_envs
    relation_states: list[Tensor | None] = [None] * args.num_envs
    policy_reset_mask = torch.ones(args.num_envs, dtype=torch.bool, device=device)

    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        source_stage_checkpoint = payload.get("source_stage_checkpoint")
        saved_stage = payload.get("config", {}).get("training_stage", "joint")
        if args.backbone == "dreamerv3" and saved_stage != args.training_stage:
            raise RuntimeError(
                "Resume training stage mismatch: checkpoint is "
                f"{saved_stage!r}, command requested {args.training_stage!r}. "
                "Cross-stage loading will be added with Stage 2; resume is "
                "currently restricted to the same stage."
            )
        model.load_state_dict(payload["model"])
        world_optimizer.load_state_dict(payload["world_optimizer"])
        behavior.load_state_dict(payload["behavior"])
        actor_optimizer.load_state_dict(payload["actor_optimizer"])
        critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if payload.get("grad_scaler") is not None:
            grad_scaler.load_state_dict(payload["grad_scaler"])
        replay_path = args.resume_replay or (
            args.output / payload.get("replay_path", "replay_latest.pt")
        )
        replay_payload = torch.load(
            replay_path,
            map_location="cpu",
            weights_only=False,
        )
        replay.load_state_dict(replay_payload["replay"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        action_step = int(payload["action_step"])
        best_return = float(payload.get("best_return", -math.inf))
        best_visual_loss = float(payload.get("best_visual_loss", math.inf))
        best_world_score = float(payload.get("best_world_score", math.inf))
        best_graph_score = float(payload.get("best_graph_score", math.inf))
        # Environment simulator state is intentionally not serialized. Mark
        # each freshly reset environment as a new replay episode boundary.
        for index, image in enumerate(images):
            replay.append_reset(index, image, env.action_dim)

    print(
        "online protocol:",
        f"backbone={args.backbone}",
        f"training_stage={args.training_stage}",
        f"graph={'on' if use_graph else 'off'}",
        f"rgb=64 action_repeat=2 envs={args.num_envs} batch={args.batch_size} "
        f"L={args.sequence_length} H={args.imagination_horizon}",
        f"updates_per_epoch={args.steps_per_epoch}",
        "behavior_update_period="
        f"{args.dreamer_behavior_update_period}",
        "planner=actor_critic (CEM disabled)",
    )
    # Prefill remains measured in complete windows.
    while replay.num_windows < args.prefill_windows:
        actions = env.random_actions()
        next_images, rewards, lasts, _ = env.step(actions)
        for index in range(args.num_envs):
            replay.append(index, next_images[index], actions[index], rewards[index], False)
            if lasts[index]:
                replay.finish(index)
                next_images[index] = env.reset_one(index)
                replay.append_reset(index, next_images[index], env.action_dim)
        images = next_images
        action_step += 2 * args.num_envs
    print(
        "prefill complete:",
        "policy=random",
        f"windows={replay.num_windows}",
        f"action_steps={action_step}",
    )
    reconstruction_frames = (
        fixed_reconstruction_frames(
            replay, args.output, args.seed + 91_007
        )
        if model.dreamerv3_core is not None and args.training_stage == "visual"
        else None
    )
    world_probe_sequences = (
        fixed_world_sequences(
            replay,
            args.output,
            args.seed + 92_009,
            source_path=(
                (
                    args.stage_checkpoint.parent.parent
                    if args.stage_checkpoint.parent.name == "checkpoints"
                    else args.stage_checkpoint.parent
                )
                / "diagnostics"
                / "prediction"
                / "fixed_sequences.pt"
                if args.training_stage == "graph"
                and args.stage_checkpoint is not None
                else None
            ),
        )
        if model.dreamerv3_core is not None
        and args.training_stage in {"world", "graph"}
        else None
    )
    if (
        args.training_stage == "graph"
        and world_probe_sequences is not None
        and start_epoch == 1
        and not (args.output / "diagnostics" / "world_metrics.jsonl").exists()
    ):
        with isolated_random_state(args.seed + 93_011):
            source_metrics = save_world_diagnostics(
                model,
                world_probe_sequences,
                device,
                0,
                args.output,
                use_graph=False,
            )
        source_metrics["diagnostic_role"] = "source_stage2_graph_off"
        append_world_metrics(args.output, source_metrics)
        print(
            "stage3 source diagnostics",
            " ".join(
                f"{key}={value:.6f}"
                for key, value in source_metrics.items()
                if isinstance(value, (int, float))
            ),
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        behavior.train()
        if args.graph_observer == "explicit":
            model.graph_builder.eval()
        stage = args.training_stage
        behavior_enabled = stage == "joint"
        graph_scale = (
            min(1.0, epoch / max(1, args.graph_warmup_epochs))
            if use_graph and stage in {"joint", "graph"} else 0.0
        )
        alignment_scale = (
            min(1.0, epoch / max(1, args.alignment_warmup_epochs))
            if use_graph and stage in {"joint", "graph"} else 0.0
        )
        model.set_graph_conditioning_strength(graph_scale)
        sums: dict[str, float] = {}
        behavior_sums: dict[str, float] = {}
        completed_returns: list[float] = []
        successful_updates = 0
        behavior_updates = 0
        skipped_nonfinite_updates = 0
        consecutive_nonfinite_updates = 0
        last_actor_display = 0.0
        last_critic_display = 0.0
        progress = trange(args.steps_per_epoch, desc=f"online epoch {epoch}")
        for update in progress:
            batch = replay.sample(args.batch_size, device)
            batch_images = preprocess(batch["images"])

            output: dict[str, Any] | None = None
            reference_loss = batch_images.new_zeros(())
            world_grad = reference_loss
            optimizer_updated = False
            world_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                if stage == "visual":
                    output = model.dreamerv3_core.reconstruct_observations(
                        batch_images
                    )
                    world_loss, metrics = dreamer_visual_loss(
                        output, batch_images
                    )
                else:
                    output = model.dreamerv3_core.forward_online_sequence(
                        batch_images,
                        batch["actions"],
                        batch["firsts"],
                        use_graph=use_graph,
                    )
                    if stage == "world":
                        world_loss, metrics = dreamer_world_loss(
                            output,
                            batch_images,
                            batch["rewards"],
                            batch["dones"],
                        )
                    elif stage == "graph":
                        world_loss, metrics = dreamer_graph_loss(
                            output,
                            batch_images,
                            batch["rewards"],
                            batch["dones"],
                            use_graph=use_graph,
                            graph_scale=graph_scale,
                            alignment_scale=alignment_scale,
                            alignment_weight=args.alignment_weight,
                            variance_weight=args.alignment_variance_weight,
                        )
                    else:
                        world_loss, metrics = dreamer_online_loss(
                            output,
                            batch_images,
                            batch["rewards"],
                            batch["dones"],
                            use_graph=use_graph,
                            graph_scale=graph_scale,
                            alignment_scale=alignment_scale,
                            alignment_weight=args.alignment_weight,
                            variance_weight=args.alignment_variance_weight,
                        )
            reference_loss = world_loss
            grad_scaler.scale(world_loss).backward()
            grad_scaler.unscale_(world_optimizer)
            if model.dreamerv3_core is not None and stage in {"visual", "world", "graph"}:
                assert_no_frozen_gradients(
                    model,
                    behavior,
                    world_parameter_list,
                    stage,
                )
            if nonfinite_gradient_names(model):
                report_nonfinite_step(
                    "world", epoch, update, world_loss, model, grad_scaler
                )
                world_optimizer.zero_grad(set_to_none=True)
                # GradScaler has already recorded the overflow in unscale_;
                # update once so it lowers the scale, then discard this batch.
                grad_scaler.update()
                skipped_nonfinite_updates += 1
                consecutive_nonfinite_updates += 1
                if consecutive_nonfinite_updates >= 8:
                    raise RuntimeError(
                        "Eight consecutive non-finite world-model updates; "
                        "the failure is persistent rather than a recoverable "
                        "FP16 overflow. Re-run with --precision float32 --smoke "
                        "and inspect the reported parameter names."
                    )
                continue
            world_grad = torch.nn.utils.clip_grad_norm_(
                world_parameter_list,
                1000.0,
                error_if_nonfinite=True,
            )
            grad_scaler.step(world_optimizer)
            optimizer_updated = True

            effective_behavior_update_period = (
                args.dreamer_behavior_update_period if behavior_enabled else 1
            )
            behavior_update_due = (
                not behavior_enabled
                or update % effective_behavior_update_period == 0
            )
            actor_update_enabled = behavior_enabled and behavior_update_due
            behavior_update_completed = False
            if actor_update_enabled:
                if output is None:
                    raise RuntimeError("Actor update requires world-model output")
                actor_optimizer.zero_grad(set_to_none=True)
                with frozen_parameters(model), frozen_parameters(behavior.critic), torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    actor_loss, critic_cache, behavior_metrics = behavior_losses(
                        model,
                        behavior,
                        output,
                        batch["dones"],
                        use_graph=use_graph,
                        horizon=args.imagination_horizon,
                        gamma=defaults["gamma"],
                        lambda_td=defaults["lambda_td"],
                        entropy_weight=defaults["entropy_weight"],
                        slow_regularization=defaults["critic_slow_reg_scale"],
                    )
                    actor_total_loss = actor_loss
                    behavior_metrics["actor_total"] = float(
                        actor_total_loss.detach()
                    )
                # Keep the world model and critic frozen for the actor
                # backward pass, matching the forward pass.
                with frozen_parameters(model), frozen_parameters(
                    behavior.critic
                ):
                    grad_scaler.scale(actor_total_loss).backward()
                grad_scaler.unscale_(actor_optimizer)
                if nonfinite_gradient_names(behavior.actor):
                    report_nonfinite_step(
                        "actor",
                        epoch,
                        update,
                        actor_total_loss,
                        behavior.actor,
                        grad_scaler,
                    )
                    actor_optimizer.zero_grad(set_to_none=True)
                    grad_scaler.update()
                    skipped_nonfinite_updates += 1
                    consecutive_nonfinite_updates += 1
                    if consecutive_nonfinite_updates >= 8:
                        raise RuntimeError(
                            "Eight consecutive non-finite actor updates; retry with "
                            "--precision float32 --smoke and inspect diagnostics."
                        )
                    continue
                torch.nn.utils.clip_grad_norm_(
                    actor_parameters,
                    100.0,
                    error_if_nonfinite=True,
                )
                grad_scaler.step(actor_optimizer)
                optimizer_updated = True

                if behavior_enabled:
                    assert critic_cache is not None
                    critic_optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=use_amp,
                    ):
                        critic_loss = critic_loss_from_cache(
                            behavior,
                            critic_cache,
                            defaults["critic_slow_reg_scale"],
                        )
                    grad_scaler.scale(critic_loss).backward()
                    grad_scaler.unscale_(critic_optimizer)
                    if nonfinite_gradient_names(behavior.critic):
                        report_nonfinite_step(
                            "critic", epoch, update, critic_loss, behavior.critic, grad_scaler
                        )
                        critic_optimizer.zero_grad(set_to_none=True)
                        grad_scaler.update()
                        skipped_nonfinite_updates += 1
                        consecutive_nonfinite_updates += 1
                        if consecutive_nonfinite_updates >= 8:
                            raise RuntimeError(
                                "Eight consecutive non-finite critic updates; retry with "
                                "--precision float32 --smoke and inspect diagnostics."
                            )
                        continue
                    torch.nn.utils.clip_grad_norm_(behavior.critic.parameters(), 100.0, error_if_nonfinite=True)
                    grad_scaler.step(critic_optimizer)
                    optimizer_updated = True
                    behavior.update_slow_critic(0.02)
                else:
                    critic_loss = reference_loss.detach().new_zeros(())
                behavior_update_completed = True
            else:
                behavior_metrics = {
                    "actor": 0.0,
                    "actor_total": 0.0,
                    "return": 0.0,
                    "entropy": 0.0,
                }
                critic_loss = reference_loss.detach().new_zeros(())
            # One scaler update per complete training iteration. Calling this
            # once per optimizer made FP16 scale growth three times too fast.
            if optimizer_updated:
                grad_scaler.update()
            consecutive_nonfinite_updates = 0
            successful_updates += 1

            metrics.update(behavior_metrics)
            metrics["critic"] = float(critic_loss.detach())
            metrics["world_grad_norm"] = float(world_grad)
            metrics["amp_scale"] = float(grad_scaler.get_scale())
            behavior_metric_keys = set(behavior_metrics)
            for key, value in metrics.items():
                if key in behavior_metric_keys or key == "critic":
                    continue
                sums[key] = sums.get(key, 0.0) + float(value)
            if behavior_update_completed:
                behavior_updates += 1
                for key, value in behavior_metrics.items():
                    behavior_sums[key] = behavior_sums.get(key, 0.0) + float(value)
                critic_value = float(critic_loss.detach())
                behavior_sums["critic"] = (
                    behavior_sums.get("critic", 0.0) + critic_value
                )
                last_actor_display = float(behavior_metrics["actor"])
                last_critic_display = critic_value
            global_step += 1

            # Burchim correct_train_ratio: 1024 replay frames / (512*4)
            # = 0.5 synchronous env steps per optimizer batch.
            numerator = args.batch_size * args.sequence_length
            denominator = args.env_step_period * args.num_envs
            before = ((global_step - 1) * numerator) // denominator
            after = (global_step * numerator) // denominator
            for _ in range(int(after - before)):
                if behavior_enabled:
                    actions, recurrent, previous_actions, graph_states, relation_states = policy_actions(
                        model,
                        behavior,
                        images,
                        device,
                        use_graph=use_graph,
                        recurrent=recurrent,
                        previous_actions=previous_actions,
                        graph_states=graph_states,
                        relation_states=relation_states,
                        reset_mask=policy_reset_mask,
                        mode=False,
                    )
                else:
                    actions = env.random_actions()
                next_images, rewards, lasts, completed = env.step(actions)
                for item in completed:
                    completed_returns.append(item["return"])
                collection_reset_mask = lasts.copy()
                for index in range(args.num_envs):
                    # DMC time limits are ``is_last`` resets, not terminal
                    # failures. Pinned burchim stores done=0 for these steps.
                    replay.append(index, next_images[index], actions[index], rewards[index], False)
                    if lasts[index]:
                        replay.finish(index)
                        next_images[index] = env.reset_one(index)
                        replay.append_reset(index, next_images[index], env.action_dim)
                        collection_reset_mask[index] = True
                        if previous_actions is not None:
                            previous_actions[index].zero_()
                policy_reset_mask = torch.from_numpy(collection_reset_mask).to(device)
                images = next_images
                action_step += 2 * args.num_envs
            if (update + 1) % 10 == 0:
                progress.set_postfix(
                    world=f"{metrics['world_total']:.3f}",
                    actor=f"{last_actor_display:.3f}",
                    critic=f"{last_critic_display:.3f}",
                    behavior_updates=behavior_updates,
                    replay=replay.num_windows,
                )

        means = {
            key: value / max(1, successful_updates)
            for key, value in sums.items()
        }
        if behavior_updates:
            means.update(
                {
                    key: value / behavior_updates
                    for key, value in behavior_sums.items()
                }
            )
        else:
            means.update(
                {
                    "actor": 0.0,
                    "actor_total": 0.0,
                    "return": 0.0,
                    "entropy": 0.0,
                    "critic": 0.0,
                }
            )
        means["successful_updates"] = float(successful_updates)
        means["behavior_updates"] = float(behavior_updates)
        means["behavior_update_period"] = float(
            args.dreamer_behavior_update_period if behavior_enabled else 1
        )
        means["skipped_nonfinite_updates"] = float(skipped_nonfinite_updates)
        means["epoch_seconds"] = time.perf_counter() - epoch_started
        means["seconds_per_update"] = (
            means["epoch_seconds"] / max(1, successful_updates)
        )
        means["replay_frames"] = float(replay.total_frames)
        means["replay_windows"] = float(replay.num_windows)
        if device.type == "cuda":
            means["gpu_peak_allocated_gb"] = (
                torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            )
            means["gpu_peak_reserved_gb"] = (
                torch.cuda.max_memory_reserved(device) / (1024.0 ** 3)
            )
        if completed_returns:
            means["train_episode_return"] = float(np.mean(completed_returns))
        if reconstruction_frames is not None:
            diagnostic_path = (
                args.output
                / "diagnostics"
                / "reconstruction"
                / f"epoch_{epoch:04d}.png"
            )
            diagnostic_metrics = save_reconstruction_diagnostic(
                model, reconstruction_frames, device, diagnostic_path
            )
            means.update(diagnostic_metrics)
            print(
                "reconstruction diagnostic",
                f"path={diagnostic_path}",
                " ".join(
                    f"{key}={value:.6f}"
                    for key, value in diagnostic_metrics.items()
                ),
            )
        world_diagnostic_metrics: dict[str, Any] | None = None
        if world_probe_sequences is not None:
            with isolated_random_state(args.seed + 93_011):
                world_diagnostic_metrics = save_world_diagnostics(
                    model,
                    world_probe_sequences,
                    device,
                    epoch,
                    args.output,
                    use_graph=use_graph,
                )
            append_world_metrics(args.output, world_diagnostic_metrics)
            print(
                "world diagnostics",
                " ".join(
                    f"{key}={format_metric_value(value)}"
                    for key, value in world_diagnostic_metrics.items()
                ),
            )
        print("train", " ".join(f"{key}={value:.6f}" for key, value in means.items()))
        checkpoint = {
            "checkpoint_version": 3,
            "training_mode": {
                "visual": "dreamerv3_visual_reconstruction",
                "world": "dreamerv3_world_model",
                "graph": "dreamerv3_graph_world_model",
            }.get(stage, "online_actor_critic"),
            "epoch": epoch,
            "stage": stage,
            "training_stage": stage,
            "global_step": global_step,
            "action_step": action_step,
            "best_return": best_return,
            "best_visual_loss": best_visual_loss,
            "best_world_score": best_world_score,
            "best_graph_score": best_graph_score,
            "graph_enabled": use_graph,
            "graph_observer": args.graph_observer if use_graph else "none",
            "graph_observer_checkpoint": (
                str(args.graph_observer_checkpoint)
                if args.graph_observer_checkpoint is not None else None
            ),
            "actor_training_enabled": behavior_enabled,
            "visual_frozen": stage == "world",
            "transition_trainable": stage in {"world", "graph", "joint"},
            "reward_trainable": stage in {"world", "graph", "joint"},
            "continuation_trainable": stage in {"world", "graph", "joint"},
            "graph_trainable": stage == "graph" and use_graph,
            "parameter_groups": parameter_groups,
            "source_checkpoint": str(args.resume) if args.resume else source_stage_checkpoint,
            "source_stage_checkpoint": source_stage_checkpoint,
            "q_planning_enabled": False,
            "config": serializable,
            "model_config": {
                key: value for key, value in serializable.items()
                if key in {
                    "backbone", "node_dim", "graph_nodes", "message_steps",
                    "max_graph_correction", "graph_correction_gate_init",
                    "graph_observer",
                }
            },
            "model": model.state_dict(),
            "behavior": behavior.state_dict(),
            "world_optimizer": world_optimizer.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "grad_scaler": grad_scaler.state_dict(),
            "replay_path": "replay_latest.pt",
            "train_metrics": means,
            "world_metrics": world_diagnostic_metrics,
        }
        epoch_path = args.output / "checkpoints" / f"epoch_{epoch:04d}.pt"
        atomic_save(
            {"replay": replay.state_dict()},
            args.output / "replay_latest.pt",
        )
        atomic_save(checkpoint, epoch_path)
        if stage == "visual":
            visual_loss = means["fixed_reconstruction_mse"]
            if visual_loss < best_visual_loss:
                best_visual_loss = visual_loss
                checkpoint["best_visual_loss"] = best_visual_loss
                atomic_save(
                    checkpoint, args.output / "checkpoints" / "visual_best.pt"
                )
                atomic_save(checkpoint, args.output / "visual_best.pt")
            atomic_save(
                checkpoint, args.output / "checkpoints" / "visual_latest.pt"
            )
            atomic_save(checkpoint, args.output / "visual_latest.pt")
        elif stage == "world":
            assert world_diagnostic_metrics is not None
            world_score = (
                float(world_diagnostic_metrics["reward_mae_all"])
                + float(world_diagnostic_metrics["horizon_10_mse"])
            )
            checkpoint["world_selection_score"] = world_score
            if world_score < best_world_score:
                best_world_score = world_score
                checkpoint["best_world_score"] = best_world_score
                atomic_save(
                    checkpoint, args.output / "checkpoints" / "world_best.pt"
                )
                atomic_save(checkpoint, args.output / "world_best.pt")
            atomic_save(
                checkpoint, args.output / "checkpoints" / "world_latest.pt"
            )
            atomic_save(checkpoint, args.output / "world_latest.pt")
        elif stage == "graph":
            assert world_diagnostic_metrics is not None
            graph_score = (
                float(world_diagnostic_metrics["horizon_10_mse"])
                + float(world_diagnostic_metrics["reward_mae_top10_reward"])
            )
            checkpoint["graph_selection_score"] = graph_score
            if graph_score < best_graph_score:
                best_graph_score = graph_score
                checkpoint["best_graph_score"] = best_graph_score
                atomic_save(
                    checkpoint, args.output / "checkpoints" / "graph_best.pt"
                )
                atomic_save(checkpoint, args.output / "graph_best.pt")
            atomic_save(
                checkpoint, args.output / "checkpoints" / "graph_latest.pt"
            )
            atomic_save(checkpoint, args.output / "graph_latest.pt")
        if (
            behavior_enabled
            and args.eval_every > 0
            and epoch % args.eval_every == 0
        ):
            metrics_eval = evaluate(
                model,
                behavior,
                args.domain,
                args.task,
                args.eval_episodes,
                args.seed + 100_000 + epoch * args.eval_episodes,
                device,
                use_graph=use_graph,
                output=args.output / "evaluation" / f"epoch_{epoch:04d}",
            )
            print("evaluation", " ".join(f"{key}={value:.6f}" for key, value in metrics_eval.items()))
            if metrics_eval["mean_return"] > best_return:
                best_return = metrics_eval["mean_return"]
                checkpoint["best_return"] = best_return
                checkpoint["eval_metrics"] = metrics_eval
                atomic_save(checkpoint, args.output / "best_return.pt")
                print(f"epoch={epoch} |--> saved best_return.pt")
        checkpoint["best_return"] = best_return
        checkpoint["best_visual_loss"] = best_visual_loss
        checkpoint["best_world_score"] = best_world_score
        checkpoint["best_graph_score"] = best_graph_score
        atomic_save(checkpoint, args.output / "latest.pt")


if __name__ == "__main__":
    main()
