"""Pretrain the RGB-only Explicit Graph Observer with MuJoCo teacher labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv.blocks import ExplicitGraphObserver, VisualFeatures
from jointgv.data import EpisodeSequences
from jointgv.dmc_protocol import (
    EXPLICIT_GRAPH_SCHEMA_VERSION,
    validate_dmc_dataset,
)


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def observer_metrics(
    observer: ExplicitGraphObserver,
    output: dict[str, Tensor],
    targets: dict[str, Tensor],
    identity_weight: float,
    motion_weight: float,
    role_pixel_weights: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    node_state = targets["node_state"]
    node_mask = targets["node_mask"].to(node_state.dtype)
    state = output.get("state")
    if state is None:
        state = observer.decode_state_sequence(
            output["nodes"], output["positions_2d"]
        )
    state_error = F.smooth_l1_loss(state, node_state, reduction="none").mean(-1)
    state_loss = masked_mean(state_error, node_mask)
    position_mae = masked_mean(
        (state[..., :3] - node_state[..., :3]).abs().mean(-1), node_mask
    )
    velocity_mae = masked_mean(
        (state[..., 7:] - node_state[..., 7:]).abs().mean(-1), node_mask
    )

    presence_loss = F.binary_cross_entropy_with_logits(
        output["presence_logits"], node_mask
    )
    presence_accuracy = (
        (output["presence_logits"] >= 0) == (node_mask >= 0.5)
    ).float().mean()

    relation = targets["relation"].long()
    pair_mask = (node_mask[..., :, None] * node_mask[..., None, :]).bool()
    diagonal = torch.eye(
        pair_mask.shape[-1], device=pair_mask.device, dtype=torch.bool
    )
    pair_mask = pair_mask & ~diagonal
    if pair_mask.any():
        relation_loss = F.cross_entropy(
            output["relation_logits"][pair_mask], relation[pair_mask]
        )
        relation_accuracy = (
            output["relation_logits"].argmax(-1)[pair_mask]
            == relation[pair_mask]
        ).float().mean()
    else:
        relation_loss = state_loss.new_zeros(())
        relation_accuracy = state_loss.new_zeros(())

    pixel_loss = state_loss.new_zeros(())
    pixel_mae = state_loss.new_zeros(())
    pixel_motion_loss = state_loss.new_zeros(())
    per_role_pixel: dict[str, Tensor] = {}
    if "node_pixel" in targets:
        visible = targets["node_visible"].to(node_state.dtype)
        pixel_error = (
            output["positions_2d"] - targets["node_pixel"]
        ).abs().mean(-1)
        weighted_visible = (
            visible * role_pixel_weights.view(1, 1, -1)
        )
        # L1 keeps a non-vanishing localization gradient for small but
        # visually important errors; SmoothL1 previously made pixel-motion
        # supervision roughly two orders of magnitude too small.
        pixel_loss = masked_mean(pixel_error, weighted_visible)
        pixel_mae = masked_mean(pixel_error, visible)
        if output["positions_2d"].shape[1] > 1:
            motion_visible = visible[:, 1:] * visible[:, :-1]
            predicted_motion = (
                output["positions_2d"][:, 1:]
                - output["positions_2d"][:, :-1]
            )
            target_motion = (
                targets["node_pixel"][:, 1:]
                - targets["node_pixel"][:, :-1]
            )
            pixel_motion_error = (
                predicted_motion - target_motion
            ).abs().mean(-1)
            pixel_motion_loss = masked_mean(
                pixel_motion_error,
                motion_visible * role_pixel_weights.view(1, 1, -1),
            )
        for index, role_name in enumerate(observer.role_names):
            if role_name == "padding":
                continue
            role_visible = visible[..., index]
            role_key = "".join(
                character if character.isalnum() else "_"
                for character in role_name.lower()
            ).strip("_")
            per_role_pixel[
                f"pixel_mae_role_{index}_{role_key}"
            ] = masked_mean(pixel_error[..., index], role_visible)

    identity_loss = state_loss.new_zeros(())
    identity_accuracy = state_loss.new_zeros(())
    identity_features = output.get("identity_features", output["nodes"])
    if identity_features.shape[1] > 1:
        current_nodes = F.normalize(identity_features[:, 1:], dim=-1)
        previous_nodes = F.normalize(
            identity_features[:, :-1].detach(), dim=-1
        )
        identity_logits = torch.einsum(
            "btid,btjd->btij", current_nodes, previous_nodes
        ) / 0.1
        previous_active = node_mask[:, :-1].bool()
        current_active = node_mask[:, 1:].bool()
        identity_logits = identity_logits.masked_fill(
            ~previous_active[..., None, :], -1e4
        )
        count = identity_logits.shape[-1]
        identity_target = torch.arange(
            count, device=identity_logits.device
        ).view(1, 1, count).expand_as(current_active)
        identity_error = F.cross_entropy(
            identity_logits.reshape(-1, count),
            identity_target.reshape(-1),
            reduction="none",
        ).reshape_as(current_active)
        valid_identity = current_active & previous_active
        identity_loss = masked_mean(
            identity_error, valid_identity.to(identity_error.dtype)
        )
        identity_accuracy = masked_mean(
            (identity_logits.argmax(-1) == identity_target).to(node_state.dtype),
            valid_identity.to(node_state.dtype),
        )

    attention = output["attention"].clamp_min(1e-8)
    attention_entropy = -(
        attention * attention.log()
    ).sum(-1).mean() / math.log(float(attention.shape[-1]))
    normalized_attention = F.normalize(attention, dim=-1)
    overlap = torch.matmul(
        normalized_attention, normalized_attention.transpose(-1, -2)
    )
    off_diagonal = ~diagonal
    attention_overlap = overlap[..., off_diagonal].mean()
    centered_nodes = output["nodes"] - output["nodes"].mean(
        dim=-2, keepdim=True
    )
    node_dispersion = centered_nodes.square().sum(-1).mean()

    total = (
        state_loss
        + 0.25 * presence_loss
        + relation_loss
        + 2.0 * pixel_loss
        + identity_weight * identity_loss
        + motion_weight * pixel_motion_loss
    )
    metrics = {
        "total": total,
        "state": state_loss,
        "position_mae": position_mae,
        "velocity_mae": velocity_mae,
        "presence": presence_loss,
        "presence_accuracy": presence_accuracy,
        "relation": relation_loss,
        "relation_accuracy": relation_accuracy,
        "pixel": pixel_loss,
        "pixel_mae_normalized": pixel_mae,
        "pixel_motion": pixel_motion_loss,
        "identity": identity_loss,
        "identity_accuracy": identity_accuracy,
        "attention_entropy": attention_entropy,
        "attention_overlap": attention_overlap,
        "node_dispersion": node_dispersion,
        "temporal_gate_mean": output["temporal_gate"].mean(),
        "temporal_gate_max": output["temporal_gate"].amax(),
        "temporal_correction_rms": output[
            "temporal_correction"
        ].square().mean().sqrt(),
        "temporal_correction_absmax": output[
            "temporal_correction"
        ].abs().amax(),
    }
    metrics.update(per_role_pixel)
    return total, metrics


def observer_loss(
    observer: ExplicitGraphObserver,
    images: Tensor,
    targets: dict[str, Tensor],
    identity_weight: float,
    motion_weight: float,
    role_pixel_weights: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    output = observer.forward_sequence(images, sample=False)
    return observer_metrics(
        observer,
        output,
        targets,
        identity_weight,
        motion_weight,
        role_pixel_weights,
    )


class Averages:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, Tensor], count: int) -> None:
        self.count += count
        for key, value in metrics.items():
            scalar = float(value.detach().item())
            if not math.isfinite(scalar):
                raise FloatingPointError(f"Non-finite observer metric {key}")
            self.sums[key] = self.sums.get(key, 0.0) + scalar * count

    def compute(self) -> dict[str, float]:
        return {key: value / self.count for key, value in self.sums.items()}


def move_batch(batch, device):
    images, _actions, targets = batch
    return images.to(device), {
        key: value.to(device) for key, value in targets.items()
    }


@torch.no_grad()
def validate(
    observer,
    loader,
    device,
    identity_weight: float,
    motion_weight: float,
    role_pixel_weights: Tensor,
) -> dict[str, float]:
    observer.eval()
    averages = Averages()
    for batch in loader:
        images, targets = move_batch(batch, device)
        _loss, metrics = observer_loss(
            observer,
            images,
            targets,
            identity_weight,
            motion_weight,
            role_pixel_weights,
        )
        averages.update(metrics, images.shape[0])
    return averages.compute()


@torch.no_grad()
def infer_recursive_episode(
    observer: ExplicitGraphObserver,
    frames: np.ndarray,
    device: torch.device,
    frame_batch_size: int,
) -> dict[str, Tensor]:
    """Observe a complete episode without resetting persistent role slots."""
    previous: Tensor | None = None
    collected: dict[str, list[Tensor]] = {
        key: []
        for key in (
            "nodes",
            "node_mu",
            "node_logvar",
            "relation_logits",
            "positions_2d",
            "presence_logits",
            "attention",
            "temporal_gate",
            "identity_features",
            "temporal_correction",
        )
    }
    for start in range(0, len(frames), frame_batch_size):
        images = torch.from_numpy(
            frames[start : start + frame_batch_size].copy()
        ).permute(0, 3, 1, 2).float().to(device) / 255.0
        visual = observer.visual_backbone(images)
        for offset in range(visual.tokens.shape[0]):
            observation = observer._observe_visual(
                VisualFeatures(
                    visual.tokens[offset : offset + 1],
                    visual.positions[offset : offset + 1],
                    visual.dense_tokens[offset : offset + 1],
                    visual.dense_positions[offset : offset + 1],
                ),
                previous,
                sample=False,
            )
            previous = observation.node_mu
            for key in collected:
                collected[key].append(getattr(observation, key))
    output = {
        key: torch.stack(value, dim=1)
        for key, value in collected.items()
    }
    output["state"] = observer.decode_state_sequence(
        output["nodes"], output["positions_2d"]
    )
    return output


@torch.no_grad()
def validate_recursive_episodes(
    observer: ExplicitGraphObserver,
    files: list[Path],
    device: torch.device,
    frame_batch_size: int,
    identity_weight: float,
    motion_weight: float,
    role_pixel_weights: Tensor,
    episode_limit: int,
) -> dict[str, float]:
    """Validate continuous role binding over complete held-out episodes."""
    observer.eval()
    selected = files if episode_limit <= 0 else files[:episode_limit]
    averages = Averages()
    total_frames = 0
    for path in tqdm(selected, desc="recursive validation", leave=False):
        with np.load(path, allow_pickle=False) as episode:
            frames = episode["image"]
            output = infer_recursive_episode(
                observer, frames, device, frame_batch_size
            )
            targets = {
                "node_state": torch.from_numpy(
                    episode["graph_node_state"].copy()
                ).float().to(device)[None],
                "node_mask": torch.from_numpy(
                    episode["graph_node_mask"].copy()
                ).float().to(device)[None],
                "relation": torch.from_numpy(
                    episode["graph_relation"].copy()
                ).long().to(device)[None],
            }
            if {
                "graph_node_pixel", "graph_node_visible"
            }.issubset(episode.files):
                targets["node_pixel"] = torch.from_numpy(
                    episode["graph_node_pixel"].copy()
                ).float().to(device)[None]
                targets["node_visible"] = torch.from_numpy(
                    episode["graph_node_visible"].copy()
                ).float().to(device)[None]
        _loss, metrics = observer_metrics(
            observer,
            output,
            targets,
            identity_weight,
            motion_weight,
            role_pixel_weights,
        )
        count = len(frames)
        total_frames += count
        averages.update(metrics, count)
    result = averages.compute()
    result["episodes"] = float(len(selected))
    result["frames"] = float(total_frames)
    result["long_rollout_score"] = result["total"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--graph-nodes", type=int, default=8)
    parser.add_argument("--relation-count", type=int, default=4)
    parser.add_argument("--temporal-max-correction", type=float, default=0.25)
    parser.add_argument("--max-position-residual", type=float, default=0.25)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--link-pixel-weight", type=float, default=2.0)
    parser.add_argument(
        "--end-effector-pixel-weight", type=float, default=5.0
    )
    parser.add_argument("--grounding-epochs", type=int, default=10)
    parser.add_argument("--temporal-warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--backbone",
        choices=("reltr_resnet50", "debug_conv"),
        default="reltr_resnet50",
    )
    parser.add_argument(
        "--reltr-checkpoint",
        type=Path,
        default=Path("Models/reltr/checkpoint0149.pth"),
    )
    parser.add_argument("--reltr-image-size", type=int, default=224)
    parser.add_argument("--reltr-batch-size", type=int, default=8)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--allow-partial-dmc", action="store_true")
    parser.add_argument("--recursive-validation-every", type=int, default=5)
    parser.add_argument(
        "--recursive-validation-episodes",
        type=int,
        default=0,
        help="Complete validation episodes per check; 0 uses the whole split.",
    )
    parser.add_argument(
        "--recursive-frame-batch-size", type=int, default=64
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.sequence_length < 1:
        parser.error("epochs, batch size, and sequence length must be positive")
    if args.reltr_image_size < 64 or args.reltr_batch_size < 1:
        parser.error("RelTR image size must be >=64 and batch size positive")
    if min(
        args.identity_weight,
        args.motion_weight,
        args.link_pixel_weight,
        args.end_effector_pixel_weight,
    ) < 0:
        parser.error("loss weights must be nonnegative")
    if args.grounding_epochs < 0 or args.temporal_warmup_epochs < 0:
        parser.error("grounding and temporal warm-up epochs must be nonnegative")
    if (
        args.recursive_validation_every < 1
        or args.recursive_validation_episodes < 0
        or args.recursive_frame_batch_size < 1
    ):
        parser.error("recursive validation arguments are invalid")

    validate_dmc_dataset(
        args.data,
        allow_partial=args.allow_partial_dmc,
        require_graph_schema_version=EXPLICIT_GRAPH_SCHEMA_VERSION,
    )
    validate_dmc_dataset(
        args.validation_data,
        allow_partial=args.allow_partial_dmc,
        require_graph_schema_version=EXPLICIT_GRAPH_SCHEMA_VERSION,
    )

    train_dataset = EpisodeSequences(
        args.data,
        sequence_length=args.sequence_length,
        stride=args.stride,
        include_graph_targets=True,
    )
    validation_dataset = EpisodeSequences(
        args.validation_data,
        sequence_length=args.sequence_length,
        stride=args.stride,
        include_graph_targets=True,
    )
    sample = train_dataset[0]
    teacher_state_dim = int(sample[2]["node_state"].shape[-1])
    graph_nodes = int(sample[2]["node_state"].shape[-2])
    with np.load(train_dataset.files[0], allow_pickle=False) as first_episode:
        role_names = (
            [str(value) for value in first_episode["graph_node_name"]]
            if "graph_node_name" in first_episode.files
            else None
        )
    if graph_nodes != args.graph_nodes:
        parser.error(
            f"dataset has {graph_nodes} graph nodes, got --graph-nodes "
            f"{args.graph_nodes}"
        )
    if not train_dataset.has_graph_pixels:
        print(
            "WARNING: legacy data has no camera-aligned graph_node_pixel; "
            "pixel localization supervision is disabled. Recollect with the "
            "updated collect_dmc.py for the Explicit Observer experiment."
        )
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        ),
        "validation": DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observer_config: dict[str, Any] = {
        "node_dim": args.node_dim,
        "nodes": graph_nodes,
        "relation_count": args.relation_count,
        "teacher_state_dim": teacher_state_dim,
        "backbone": args.backbone,
        "pretrained_backbone": not args.no_pretrained_backbone,
        "freeze_backbone": not args.train_backbone,
        "reltr_checkpoint": str(args.reltr_checkpoint),
        "reltr_image_size": args.reltr_image_size,
        "reltr_batch_size": args.reltr_batch_size,
        "temporal_max_correction": args.temporal_max_correction,
        "max_position_residual": args.max_position_residual,
        "role_names": role_names,
    }
    observer = ExplicitGraphObserver(**observer_config).to(device)
    role_pixel_weights = torch.ones(graph_nodes, device=device)
    for index, role_name in enumerate(observer.role_names):
        if role_name.startswith("link_"):
            role_pixel_weights[index] = args.link_pixel_weight
        elif role_name == "end_effector":
            role_pixel_weights[index] = args.end_effector_pixel_weight
    head_parameters = [
        parameter
        for name, parameter in observer.named_parameters()
        if parameter.requires_grad and not name.startswith("visual_backbone.")
    ]
    groups: list[dict[str, Any]] = [
        {"params": head_parameters, "lr": args.lr}
    ]
    backbone_parameters = [
        parameter
        for parameter in observer.visual_backbone.parameters()
        if parameter.requires_grad
    ]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if args.resume is None and any(checkpoint_dir.glob("epoch_*.pt")):
        parser.error(
            f"{args.output} already contains observer checkpoints. Use a new "
            "output directory, or add --resume with latest.pt."
        )
    metrics_path = args.output / "metrics.jsonl"
    start_epoch = 1
    best_value = float("inf")
    best_epoch = 0
    best_long_rollout_value = float("inf")
    best_long_rollout_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if int(checkpoint.get("checkpoint_version", 1)) < 3:
            parser.error(
                "pre-v3 observer checkpoints cannot resume the dense "
                "geom-centric optimizer; start a new output directory"
            )
        observer.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_value = float(checkpoint["best_value"])
        best_epoch = int(checkpoint["best_epoch"])
        best_long_rollout_value = float(
            checkpoint.get("best_long_rollout_value", float("inf"))
        )
        best_long_rollout_epoch = int(
            checkpoint.get("best_long_rollout_epoch", 0)
        )
    else:
        metrics_path.write_text("", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= args.grounding_epochs:
            temporal_strength = 0.0
        elif args.temporal_warmup_epochs == 0:
            temporal_strength = 1.0
        else:
            temporal_strength = min(
                1.0,
                (epoch - args.grounding_epochs)
                / args.temporal_warmup_epochs,
            )
        observer.set_temporal_strength(temporal_strength)
        train_identity_weight = args.identity_weight * temporal_strength
        observer.train()
        averages = Averages()
        progress = tqdm(loaders["train"], desc=f"graph observer epoch {epoch}")
        for batch in progress:
            images, targets = move_batch(batch, device)
            loss, metrics = observer_loss(
                observer,
                images,
                targets,
                train_identity_weight,
                args.motion_weight,
                role_pixel_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                observer.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            metrics["grad_norm"] = gradient_norm
            averages.update(metrics, images.shape[0])
            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                pixel=f"{float(metrics['pixel_mae_normalized'].detach()):.3f}",
                relation=f"{float(metrics['relation_accuracy'].detach()):.3f}",
            )
        train_metrics = averages.compute()
        validation_metrics = validate(
            observer,
            loaders["validation"],
            device,
            args.identity_weight,
            args.motion_weight,
            role_pixel_weights,
        )
        recursive_metrics = None
        if epoch % args.recursive_validation_every == 0:
            recursive_metrics = validate_recursive_episodes(
                observer,
                validation_dataset.files,
                device,
                args.recursive_frame_batch_size,
                args.identity_weight,
                args.motion_weight,
                role_pixel_weights,
                args.recursive_validation_episodes,
            )
        current = validation_metrics["total"]
        is_best = current < best_value
        if is_best:
            best_value = current
            best_epoch = epoch
        is_best_long_rollout = False
        long_rollout_eligible = temporal_strength >= 1.0
        if recursive_metrics is not None and long_rollout_eligible:
            recursive_current = recursive_metrics["long_rollout_score"]
            is_best_long_rollout = (
                recursive_current < best_long_rollout_value
            )
            if is_best_long_rollout:
                best_long_rollout_value = recursive_current
                best_long_rollout_epoch = epoch
        checkpoint = {
            "model": observer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_value": best_value,
            "best_epoch": best_epoch,
            "best_metric": "validation_total",
            "best_long_rollout_value": best_long_rollout_value,
            "best_long_rollout_epoch": best_long_rollout_epoch,
            "best_long_rollout_metric": "recursive_validation_total",
            "long_rollout_eligible": long_rollout_eligible,
            "temporal_strength": temporal_strength,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "recursive_validation_metrics": recursive_metrics,
            "observer_config": observer_config,
            "loss_config": {
                "identity_weight": args.identity_weight,
                "motion_weight": args.motion_weight,
                "link_pixel_weight": args.link_pixel_weight,
                "end_effector_pixel_weight": args.end_effector_pixel_weight,
                "grounding_epochs": args.grounding_epochs,
                "temporal_warmup_epochs": args.temporal_warmup_epochs,
            },
            "checkpoint_version": 3,
        }
        epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(checkpoint, epoch_path)
        shutil.copy2(epoch_path, args.output / "latest.pt")
        if is_best:
            shutil.copy2(epoch_path, args.output / "best.pt")
        if is_best_long_rollout:
            shutil.copy2(epoch_path, args.output / "best_long_rollout.pt")
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "recursive_validation": recursive_metrics,
            "best_value": best_value,
            "best_epoch": best_epoch,
            "best_long_rollout_value": best_long_rollout_value,
            "best_long_rollout_epoch": best_long_rollout_epoch,
            "long_rollout_eligible": long_rollout_eligible,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        suffix = " |--> saved best.pt" if is_best else ""
        print("train", json.dumps(train_metrics, sort_keys=True))
        print("validation", json.dumps(validation_metrics, sort_keys=True))
        if recursive_metrics is not None:
            print(
                "recursive_validation",
                json.dumps(recursive_metrics, sort_keys=True),
            )
        print(
            f"epoch={epoch} best_value={best_value:.6f} "
            f"best_epoch={best_epoch} "
            f"temporal_strength={temporal_strength:.3f}{suffix}"
        )
        if recursive_metrics is not None:
            if not long_rollout_eligible:
                print(
                    "long_rollout checkpoint not eligible until temporal "
                    "warm-up reaches 100%"
                )
                continue
            long_suffix = (
                " |--> saved best_long_rollout.pt"
                if is_best_long_rollout
                else ""
            )
            print(
                "long_rollout "
                f"best_value={best_long_rollout_value:.6f} "
                f"best_epoch={best_long_rollout_epoch}{long_suffix}"
            )


if __name__ == "__main__":
    main()
