"""Final ablation metrics over the whole validation split.

Each ``--model LABEL CHECKPOINT MODE`` is rolled out open-loop exactly like
train.py's recursive validation: the same deterministic windows (one per
episode by default), the visual backbone started from one real frame, and
``--context-frames`` real frames that only seed graph history. MODE is
``graph`` (use_graph=True) or ``base`` (use_graph=False, the backbone alone).
Every predicted step is scored with:

* latent MSE: predicted vs encoded target patch features, observer-free,
* cart-pole only, observer-free: soft pole-mask IoU and pole presence (how
  much of the true pole's coloured area the prediction keeps); unlike pixel
  MSE they do not punish a displaced pole more than a vanished one,
* image MSE and motion-weighted image MSE (evaluate_rollout.py's definition),
* SSIM (Gaussian 11x11 window, sigma 1.5),
* LPIPS (AlexNet; optional, needs ``pip install lpips``),
* joint-position error: one fixed, independently pretrained Explicit Graph
  Observer reads the *predicted* frames, and its node 2D positions are
  compared with the ground-truth ``graph_node_pixel`` over visible nodes, in
  pixels of the frame. The same observer on the real frames gives the
  measurement floor. Every model is measured with this same observer (not
  its own: the baseline has none, and graph-on runs fine-tune theirs).
* for the RM65-B robot cart-pole, hinge and pole-tip pixel errors,
  pole-vector pixel error, wrapped pole-angle MAE, and wrapped pole
  angular-change error. These isolate the passive pole instead of averaging
  it together with the much larger robot body.

Reports per-step curves, h8/h16/h32 prefix means with 95% intervals over
windows, paired differences against ``--reference``, and the fraction of
matched windows on which each model beats that reference. Evaluation only:
reads checkpoints and data, writes results to ``--output``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv import JointGraphVideoWorldModel
from jointgv.blocks.explicit_graph_observer import ExplicitGraphObserver
from jointgv.data import EpisodeSequences
from jointgv.planning import limit_history
from jointgv.rollout_metrics import (
    gaussian_window,
    load_reference_observer,
    rollout_scores,
)


BASE_METRICS = ("image_mse", "motion_mse", "ssim", "lpips", "joint_px")
CARTPOLE_METRICS = (
    "pole_tip_px",
    "hinge_px",
    "pole_vector_px",
    "pole_angle_deg",
    "pole_angle_change_deg",
)
STRUCTURE_METRICS = ("joint_px", *CARTPOLE_METRICS)
# Observer-free cart-pole pole metrics from the distinctive pole colours
# (yellow rod, cyan tip). Pixel MSE double-penalises a sharp but displaced
# pole (missing at the true place AND extra at the predicted one) while a
# prediction that simply loses the pole is penalised once; observer readings
# of a vanished pole fall back to a prior. These measure overlap and how much
# of the pole survives instead.
POLE_COLOUR_METRICS = ("pole_iou", "pole_presence")
HIGHER_IS_BETTER = {"ssim", "pole_iou", "pole_presence"}


def pole_membership(images: Tensor, low: float = 0.15, high: float = 0.5) -> Tensor:
    """Soft 0..1 pole membership per pixel from [..., 3, H, W] RGB in [0, 1]:
    yellowness min(R, G) - B or cyanness min(G, B) - R, ramped between
    ``low`` and ``high``. Black board, blue floor and white/grey arm stay 0."""
    red, green, blue = images.unbind(dim=-3)
    yellow = torch.minimum(red, green) - blue
    cyan = torch.minimum(green, blue) - red
    value = torch.maximum(yellow, cyan)
    return ((value - low) / (high - low)).clamp(0.0, 1.0)


def pole_colour_scores(predicted: Tensor, target: Tensor) -> dict[str, Tensor]:
    """[B, T, 3, H, W] -> per-step soft IoU and presence (pred mass / true
    mass, capped at 1); NaN where the true pole is not in frame."""
    pred = pole_membership(predicted)
    true = pole_membership(target)
    true_mass = true.sum(dim=(-2, -1))
    union = torch.maximum(pred, true).sum(dim=(-2, -1))
    iou = torch.minimum(pred, true).sum(dim=(-2, -1)) / union.clamp_min(1e-6)
    presence = (pred.sum(dim=(-2, -1)) / true_mass.clamp_min(1e-6)).clamp(max=1.0)
    missing = true_mass < 1.0
    nan = torch.full_like(iou, float("nan"))
    return {
        "pole_iou": torch.where(missing, nan, iou),
        "pole_presence": torch.where(missing, nan, presence),
    }
FLOOR_LABEL = "observer on ground truth"


def build_loader(
    root: Path,
    horizon: int,
    context: int,
    windows_per_episode: int,
    batch_size: int,
) -> DataLoader:
    """train.py's rollout windows, but with graph pixel targets kept."""
    dataset = EpisodeSequences(
        root,
        sequence_length=horizon + context,
        stride=horizon + context,
        include_graph_targets=None,
        include_control_targets=True,
        episode_files=sorted(root.glob("*.npz")),
    )
    if not dataset.has_graph_pixels:
        raise SystemExit(
            f"{root} has no graph_node_pixel/graph_node_visible labels"
        )
    counts: dict[int, int] = {}
    selected: list[int] = []
    for item, (file_id, _) in enumerate(dataset.index):
        if windows_per_episode == 0 or counts.get(file_id, 0) < windows_per_episode:
            selected.append(item)
            counts[file_id] = counts.get(file_id, 0) + 1
    return DataLoader(
        Subset(dataset, selected), batch_size=batch_size, shuffle=False
    )


def score(
    predicted: Tensor,
    batch: dict[str, Tensor],
    observer: ExplicitGraphObserver,
    window: Tensor,
    lpips_model: torch.nn.Module | None,
) -> dict[str, Tensor]:
    """All per-step metrics, each [B,H]; definitions shared with train.py."""
    target = batch["target"]
    batch_size, horizon = target.shape[:2]
    predicted = predicted.clamp(0.0, 1.0)
    result = {
        "image_mse": (predicted - target).square().mean(dim=(2, 3, 4)),
        **rollout_scores(
            predicted,
            batch["first"],
            target,
            batch["previous"],
            batch["pixel"],
            batch["visible"],
            observer,
            window,
            first_pixel=batch["first_pixel"],
            first_visible=batch["first_visible"],
        ),
    }
    if lpips_model is not None:
        result["lpips"] = lpips_model(
            predicted.flatten(0, 1) * 2.0 - 1.0, target.flatten(0, 1) * 2.0 - 1.0
        ).view(batch_size, horizon)
    return result


def prepare(batch, device: torch.device, horizon: int, start: int) -> dict[str, Tensor]:
    images, actions, targets = batch
    images = images.to(device)
    actions = actions.to(device)
    return {
        "images": images,
        "actions": actions[:, start : start + horizon],
        "context_actions": actions[:, :start],
        "first": images[:, start],
        "target": images[:, start + 1 : start + horizon + 1],
        "previous": images[:, start : start + horizon],
        "pixel": targets["node_pixel"].to(device)[:, start + 1 : start + horizon + 1],
        "visible": targets["node_visible"].to(device)[:, start + 1 : start + horizon + 1],
        "first_pixel": targets["node_pixel"].to(device)[:, start],
        "first_visible": targets["node_visible"].to(device)[:, start],
    }


def interval(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    half = 1.96 * values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else float("nan")
    return float(values.mean()), float(half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("LABEL", "CHECKPOINT", "MODE"),
        required=True,
        help="Repeatable. MODE is 'graph' (use_graph=True) or 'base' (False).",
    )
    parser.add_argument("--data", type=Path, default=Path("data/dmc_cheetah_run_val_v1"))
    parser.add_argument(
        "--observer-checkpoint",
        type=Path,
        default=Path("runs/cheetah_run_reltr_observer_v1/best.pt"),
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--context-frames", type=int, default=3)
    parser.add_argument(
        "--visual-context",
        action="store_true",
        help=(
            "Also give the visual backbone the --context-frames real frames "
            "(and actions) as history before the first prediction, so the "
            "backbone and the graph start from the same real information. "
            "Off = historical protocol (graph history only, backbone starts "
            "from one frame)."
        ),
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=0,
        help=(
            "DINO-WM only: the predictor attends to the last W frames "
            "(sliding window, every model). 0 = full history up to "
            "dino_max_frames (historical protocol). A W no longer than the "
            "training --sequence-length stays inside the trained frame "
            "positions; the CEM planner uses 4."
        ),
    )
    parser.add_argument(
        "--windows-per-episode",
        type=int,
        default=1,
        help=(
            "Number of deterministic non-overlapping windows per episode. "
            "0 uses every eligible window; 1 is the historical first-window protocol."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device. 'auto' uses CUDA when available.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/eval/ablation"))
    args = parser.parse_args()
    if args.windows_per_episode < 0:
        parser.error("--windows-per-episode must be non-negative")
    labels = [label for label, _, _ in args.model]
    if len(set(labels)) != len(labels):
        parser.error("model labels must be unique")
    for _, _, mode in args.model:
        if mode not in {"graph", "base"}:
            parser.error("MODE must be 'graph' or 'base'")
    if args.reference is not None and args.reference not in labels:
        parser.error("--reference must be one of the model labels")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    loader = build_loader(
        args.data,
        args.horizon,
        args.context_frames,
        args.windows_per_episode,
        args.batch_size,
    )
    observer = load_reference_observer(args.observer_checkpoint, device)
    window = gaussian_window(device)
    lpips_model = None
    if not args.no_lpips:
        try:
            import lpips

            lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except ImportError:
            print("lpips not installed (pip install lpips); skipping LPIPS")
    metrics = [
        name for name in BASE_METRICS if name != "lpips" or lpips_model is not None
    ]
    role_names = set(observer.role_names)
    if {"end_effector", "object_1"}.issubset(role_names):
        metrics.extend(CARTPOLE_METRICS)
        metrics.extend(POLE_COLOUR_METRICS)
    # Observer-free: predicted vs encoded target patch features. Comparable
    # across checkpoints only when they share the frozen encoder and its
    # token width (e.g. native 384-dim DINO tokens); NaN for models whose
    # rollout latent is not a visual token grid (DreamerV3).
    metrics.append("latent_mse")
    start = args.context_frames

    per_window: dict[str, dict[str, list[np.ndarray]]] = {}
    with torch.inference_mode():
        floor_metrics = [name for name in STRUCTURE_METRICS if name in metrics]
        floor: dict[str, list[np.ndarray]] = {name: [] for name in floor_metrics}
        for batch in tqdm(
            loader, desc="observer floor", dynamic_ncols=True
        ):
            data = prepare(batch, device, args.horizon, start)
            floor_scores = rollout_scores(
                data["target"],
                data["first"],
                data["target"],
                data["previous"],
                data["pixel"],
                data["visible"],
                observer,
                window,
                first_pixel=data["first_pixel"],
                first_visible=data["first_visible"],
            )
            for name in floor_metrics:
                floor[name].append(floor_scores[name].float().cpu().numpy())
        per_window[FLOOR_LABEL] = floor
        for label, checkpoint_path, mode in args.model:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = JointGraphVideoWorldModel(**checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            limit_history(model, args.history_window)
            print(
                f"{label}: {checkpoint_path} (epoch {checkpoint.get('epoch', '?')}) "
                f"mode={mode} history_window={args.history_window or 'full'}"
            )
            values: dict[str, list[np.ndarray]] = {name: [] for name in metrics}
            for batch in tqdm(
                loader, desc=f"rollout {label}", dynamic_ncols=True
            ):
                data = prepare(batch, device, args.horizon, start)
                rollout = model.rollout(
                    data["first"],
                    data["actions"],
                    use_graph=mode == "graph",
                    deterministic=True,
                    decode=True,
                    context_images=data["images"][:, :start] if start else None,
                    context_actions=data["context_actions"] if start else None,
                    visual_context=args.visual_context,
                )
                scores = score(rollout["images"], data, observer, window, lpips_model)
                target = data["target"]
                if "pole_iou" in metrics:
                    scores.update(
                        pole_colour_scores(rollout["images"].clamp(0.0, 1.0), target)
                    )
                if model.rollout_latent_matches_visual_tokens:
                    target_tokens = model.encode_visual_tokens(target.flatten(0, 1))
                    target_tokens = target_tokens.reshape(
                        *target.shape[:2], *target_tokens.shape[1:]
                    )
                    scores["latent_mse"] = (
                        (rollout["tokens"] - target_tokens).square().mean(dim=(2, 3))
                    )
                else:
                    scores["latent_mse"] = torch.full(
                        target.shape[:2], float("nan"), device=target.device
                    )
                for name in metrics:
                    values[name].append(scores[name].float().cpu().numpy())
            per_window[label] = values
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

    arrays = {
        label: {name: np.concatenate(chunks, axis=0) for name, chunks in values.items()}
        for label, values in per_window.items()
    }
    windows = next(iter(arrays[labels[0]].values())).shape[0]
    prefixes = sorted({min(8, args.horizon), min(16, args.horizon), args.horizon})
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for label, values in arrays.items():
        summary[label] = {}
        for name, array in values.items():
            for prefix in prefixes:
                mean, half = interval(np.nanmean(array[:, :prefix], axis=1))
                summary[label][f"{name}_h{prefix}"] = {"mean": mean, "ci95": half}
    paired: dict[str, dict[str, dict[str, float]]] = {}
    if args.reference is not None:
        reference = arrays[args.reference]
        for label in labels:
            if label == args.reference:
                continue
            paired[label] = {}
            for name in metrics:
                for prefix in prefixes:
                    ours = np.nanmean(arrays[label][name][:, :prefix], axis=1)
                    base = np.nanmean(reference[name][:, :prefix], axis=1)
                    mean, half = interval(ours - base)
                    reference_mean = float(np.nanmean(base))
                    valid = np.isfinite(ours) & np.isfinite(base)
                    ties = valid & np.isclose(ours, base, rtol=1e-6, atol=1e-9)
                    if name in HIGHER_IS_BETTER:
                        wins = valid & ~ties & (ours > base)
                    else:
                        wins = valid & ~ties & (ours < base)
                    valid_count = int(valid.sum())
                    win_rate = (
                        float(wins.sum() / valid_count)
                        if valid_count
                        else float("nan")
                    )
                    tie_rate = (
                        float(ties.sum() / valid_count)
                        if valid_count
                        else float("nan")
                    )
                    relative_pct = (
                        100.0 * mean / reference_mean
                        if reference_mean != 0.0
                        else float("nan")
                    )
                    relative_ci95_pct = (
                        100.0 * half / abs(reference_mean)
                        if reference_mean != 0.0
                        else float("nan")
                    )
                    paired[label][f"{name}_h{prefix}"] = {
                        "diff": mean,
                        "ci95": half,
                        "relative_pct": relative_pct,
                        "relative_ci95_pct": relative_ci95_pct,
                        "win_rate": win_rate,
                        "tie_rate": tie_rate,
                        "windows": valid_count,
                    }

    args.output.mkdir(parents=True, exist_ok=True)
    curves = {
        label: {name: np.nanmean(array, axis=0).tolist() for name, array in values.items()}
        for label, values in arrays.items()
    }
    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "models": [
                    {"label": label, "checkpoint": path, "mode": mode}
                    for label, path, mode in args.model
                ],
                "windows": windows,
                "horizon": args.horizon,
                "context_frames": args.context_frames,
                "visual_context": args.visual_context,
                "history_window": args.history_window,
                "device": str(device),
                "observer_checkpoint": str(args.observer_checkpoint),
                "win_rate_definition": (
                    "fraction of matched windows whose prefix-mean metric is "
                    "strictly better than the reference; ties are reported separately"
                ),
                "summary": summary,
                "paired_vs_reference": paired,
                "reference": args.reference,
                "curves": curves,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.output / "curves.csv").open("w", encoding="utf-8") as stream:
        columns = [
            (label, name) for label, values in curves.items() for name in values
        ]
        stream.write(
            "step," + ",".join(f"{label}|{name}" for label, name in columns) + "\n"
        )
        for step in range(args.horizon):
            stream.write(
                f"{step + 1},"
                + ",".join(f"{curves[label][name][step]:.8g}" for label, name in columns)
                + "\n"
            )

    print(
        f"\nwindows={windows} horizon={args.horizon} start_frame={start} "
        f"backbone_context={'real frames 0..' + str(start) if args.visual_context else 'start frame only'}"
    )
    for name in metrics:
        better = "higher" if name in HIGHER_IS_BETTER else "lower"
        print(f"\n{name} ({better} is better), mean ± 95% CI over windows")
        rows = labels + ([FLOOR_LABEL] if name in arrays[FLOOR_LABEL] else [])
        for label in rows:
            cells = []
            for prefix in prefixes:
                entry = summary[label][f"{name}_h{prefix}"]
                cells.append(f"h{prefix}={entry['mean']:.6g}±{entry['ci95']:.2g}")
            print(f"  {label:28s} " + "  ".join(cells))
        for label, entries in paired.items():
            cells = []
            for prefix in prefixes:
                entry = entries[f"{name}_h{prefix}"]
                cells.append(
                    f"h{prefix}={entry['relative_pct']:+.1f}%±{entry['relative_ci95_pct']:.1f}, "
                    f"win={100.0 * entry['win_rate']:.1f}%, n={entry['windows']}"
                )
            print(f"  {label + ' vs ' + args.reference:28s} " + "  ".join(cells))

    # Plotting lives in plot_ablation_curves.py: importing matplotlib after
    # torch here loads a second OpenMP runtime on Windows/conda and aborts.
    print(
        f"\nsaved {args.output / 'metrics.json'} and curves.csv; plot with: "
        f"python script/jointgv/viz/plot_ablation_curves.py {args.output}"
    )


if __name__ == "__main__":
    main()
