from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv import JointGraphVideoWorldModel
from jointgv.planning import limit_history


def as_uint8(images: torch.Tensor) -> np.ndarray:
    return (
        images.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare graph-conditioned and backbone-only rollouts."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument(
        "--context-frames",
        type=int,
        default=0,
        help=(
            "Real frames before the rollout start, used only to seed graph "
            "history for models trained with --graph-history > 1. The "
            "rollout (and every compared model) starts at this frame index."
        ),
    )
    parser.add_argument(
        "--compare-checkpoint",
        type=Path,
        default=None,
        help=(
            "A second, independently-trained checkpoint (e.g. a graph-off "
            "baseline) to roll out on the same episode for a direct "
            "side-by-side view alongside --checkpoint's own tracks. Always "
            "rolled out with use_graph=False, since a graph-off checkpoint's "
            "graph pathway was never trained."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/jointgv/rollout_comparison.npz"),
    )
    parser.add_argument(
        "--correction-strength",
        type=float,
        default=1.0,
        help=(
            "Graph correction strength for the graph-conditioned rollout, "
            "e.g. the checkpoint's own training correction_scale (0 inside "
            "the warm-up delay, ramping afterwards). 1.0 = full strength."
        ),
    )
    parser.add_argument(
        "--visual-context",
        action="store_true",
        help=(
            "Also give the visual backbone the --context-frames real frames "
            "(and actions) as history, for every rolled-out model. Off = "
            "historical protocol (backbone starts from one frame)."
        ),
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help=(
            "Frame where the rollout starts (the --context-frames frames "
            "before it are real context). Default 0 = right after the "
            "context at the episode start (historical)."
        ),
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=0,
        help=(
            "DINO-WM only: the predictor attends to the last W frames "
            "(sliding window, every rolled-out model). 0 = full history up "
            "to dino_max_frames (historical protocol). A W no longer than "
            "the training --sequence-length stays inside the trained frame "
            "positions; the CEM planner uses 4."
        ),
    )
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if args.history_window < 0 or args.history_window == 1:
        parser.error("--history-window must be 0 (full) or at least 2")
    if not 0.0 <= args.correction_strength <= 1.0:
        parser.error("--correction-strength must be in [0, 1]")
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    if args.context_frames < 0:
        parser.error("--context-frames must be nonnegative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = JointGraphVideoWorldModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    limit_history(model, args.history_window)
    model.set_graph_conditioning_strength(args.correction_strength)

    # The rollout starts at frame ``start``; the --context-frames frames just
    # before it are the real context (historically start = context_frames).
    start = max(args.start_frame, args.context_frames)
    first_context = start - args.context_frames

    def to_tensor(frames: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(frames.copy())
            .permute(0, 3, 1, 2)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

    with np.load(args.episode, allow_pickle=False) as episode:
        horizon = min(
            args.horizon,
            len(episode["action"]) - start,
            len(episode["image"]) - 1 - start,
        )
        if horizon < 1:
            raise SystemExit(
                f"--start-frame {start} leaves no step to predict in this "
                f"{len(episode['action'])}-step episode"
            )
        initial = to_tensor(episode["image"][start : start + 1])[:, 0]
        context = (
            to_tensor(episode["image"][first_context:start])
            if args.context_frames
            else None
        )
        context_actions = (
            torch.from_numpy(episode["action"][first_context:start].copy())
            .float()
            .unsqueeze(0)
            .to(device)
            if args.context_frames
            else None
        )
        actions = (
            torch.from_numpy(episode["action"][start : start + horizon].copy())
            .float()
            .unsqueeze(0)
            .to(device)
        )
        ground_truth = episode["image"][start + 1 : start + horizon + 1].copy()
        first_frame = episode["image"][start].copy()
        rewards = episode["reward"][start : start + horizon].copy()

    context_kwargs = {
        "context_images": context,
        "context_actions": context_actions,
        "visual_context": args.visual_context,
    }
    with torch.inference_mode():
        graph_rollout = model.rollout(
            initial, actions, use_graph=True, **context_kwargs
        )
        base_rollout = model.rollout(
            initial, actions, use_graph=False, **context_kwargs
        )
    graph_images = as_uint8(graph_rollout["images"][0])
    base_images = as_uint8(base_rollout["images"][0])
    graph_mse = float(
        np.mean(
            (
                graph_images.astype(np.float32) / 255.0
                - ground_truth.astype(np.float32) / 255.0
            )
            ** 2
        )
    )
    base_mse = float(
        np.mean(
            (
                base_images.astype(np.float32) / 255.0
                - ground_truth.astype(np.float32) / 255.0
            )
            ** 2
        )
    )
    # Whole-image MSE dilutes any improvement concentrated on a small moving
    # foreground (e.g. cheetah legs) against a large static background both
    # models already predict trivially well -- the same dilution problem
    # --motion-weighted-transition-loss was built to fix at training time.
    # Reweighting the eval metric by each frame's own ground-truth motion
    # (pixels that actually moved count more) surfaces that same effect here
    # instead of averaging it away.
    gt_float = ground_truth.astype(np.float32)
    previous_frames = np.concatenate(
        (first_frame[None].astype(np.float32), gt_float[:-1]), axis=0
    )
    motion = np.abs(gt_float - previous_frames).mean(axis=-1) + 1.0
    motion_weight = motion / motion.mean(axis=(1, 2), keepdims=True)

    def weighted_mse(predicted: np.ndarray) -> float:
        error = (
            predicted.astype(np.float32) / 255.0
            - ground_truth.astype(np.float32) / 255.0
        ) ** 2
        return float((error.mean(axis=-1) * motion_weight).mean())

    graph_mse_motion_weighted = weighted_mse(graph_images)
    base_mse_motion_weighted = weighted_mse(base_images)
    # 4x-amplified absolute error maps, viewable as extra tracks in
    # view_episode.py -- raw per-pixel differences are usually too faint to
    # read by eye, so scale before clipping to uint8 range.
    graph_prediction_error = np.clip(
        np.abs(
            graph_images.astype(np.float32) - ground_truth.astype(np.float32)
        )
        * 4.0,
        0,
        255,
    ).astype(np.uint8)
    base_prediction_error = np.clip(
        np.abs(
            base_images.astype(np.float32) - ground_truth.astype(np.float32)
        )
        * 4.0,
        0,
        255,
    ).astype(np.uint8)
    # Direct visualization of what the correction changes, rather than two
    # separate error-vs-ground-truth maps a viewer has to mentally subtract:
    # amplified |graph_prediction - base_prediction|, so any correction
    # effect too small to see in the RGB columns still shows up here.
    correction_diff = np.clip(
        np.abs(
            graph_images.astype(np.float32) - base_images.astype(np.float32)
        )
        * 4.0,
        0,
        255,
    ).astype(np.uint8)

    extra_tracks: dict[str, np.ndarray] = {}
    if args.compare_checkpoint is not None:
        compare_checkpoint = torch.load(
            args.compare_checkpoint,
            map_location=device,
            weights_only=False,
        )
        compare_model = JointGraphVideoWorldModel(
            **compare_checkpoint["model_config"]
        ).to(device)
        compare_model.load_state_dict(compare_checkpoint["model"])
        compare_model.eval()
        limit_history(compare_model, args.history_window)
        with torch.inference_mode():
            compare_rollout = compare_model.rollout(
                initial, actions, use_graph=False, **context_kwargs
            )
        compare_images = as_uint8(compare_rollout["images"][0])
        compare_mse = float(
            np.mean(
                (
                    compare_images.astype(np.float32) / 255.0
                    - ground_truth.astype(np.float32) / 255.0
                )
                ** 2
            )
        )
        compare_mse_motion_weighted = weighted_mse(compare_images)
        compare_prediction_error = np.clip(
            np.abs(
                compare_images.astype(np.float32)
                - ground_truth.astype(np.float32)
            )
            * 4.0,
            0,
            255,
        ).astype(np.uint8)
        extra_tracks = {
            "compare_prediction": compare_images,
            "compare_prediction_error": compare_prediction_error,
            "compare_mse": np.asarray(compare_mse, dtype=np.float32),
            "compare_mse_motion_weighted": np.asarray(
                compare_mse_motion_weighted, dtype=np.float32
            ),
            "compare_checkpoint": np.asarray(str(args.compare_checkpoint)),
        }
        print(
            f"compare ({args.compare_checkpoint.name}): "
            f"mse={compare_mse:.6f} "
            f"motion_weighted_mse={compare_mse_motion_weighted:.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ground_truth=ground_truth,
        graph_prediction=graph_images,
        base_prediction=base_images,
        graph_prediction_error=graph_prediction_error,
        base_prediction_error=base_prediction_error,
        correction_diff=correction_diff,
        action=actions[0].cpu().numpy(),
        reward=rewards,
        graph_state=graph_rollout["graphs"][0].cpu().numpy(),
        graph_relation=graph_rollout["relations"][0].argmax(-1).cpu().numpy(),
        graph_mse=np.asarray(graph_mse, dtype=np.float32),
        base_mse=np.asarray(base_mse, dtype=np.float32),
        graph_mse_motion_weighted=np.asarray(
            graph_mse_motion_weighted, dtype=np.float32
        ),
        base_mse_motion_weighted=np.asarray(
            base_mse_motion_weighted, dtype=np.float32
        ),
        source_episode=np.asarray(str(args.episode)),
        checkpoint=np.asarray(str(args.checkpoint)),
        start_frame=np.asarray(start, dtype=np.int64),
        visual_context=np.asarray(args.visual_context),
        history_window=np.asarray(args.history_window, dtype=np.int64),
        **extra_tracks,
    )
    print(f"saved: {args.output}")
    print(f"horizon={horizon} graph_mse={graph_mse:.6f} base_mse={base_mse:.6f}")
    print(
        f"motion-weighted: graph_mse={graph_mse_motion_weighted:.6f} "
        f"base_mse={base_mse_motion_weighted:.6f}"
    )


if __name__ == "__main__":
    main()
