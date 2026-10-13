"""Diagnostic only: does GraphDynamics' own rollout state track ground truth?

During model.rollout(), the graph is observed from the real image exactly
once (frame 0), then GraphDynamics predicts every subsequent graph state
purely from its own previous prediction -- there is no re-observation, so
any of its own error compounds with no external correction. This script
decodes that predicted trajectory into the interpretable MuJoCo teacher
state (via graph_builder.teacher_state, the same head training uses for
graph_supervision) and compares it directly against the dataset's own
ground-truth graph_node_state, frame by frame, so we can see how many
frames the graph's internal state estimate stays accurate for before it
diverges -- and whether that lines up with where correction was visually
observed to stop helping (~frame 21-26).

Does not modify rollout() or any training code; read-only analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv import JointGraphVideoWorldModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GraphDynamics' own rollout trajectory against the "
            "dataset's ground-truth graph_node_state, per frame."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=32)
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = JointGraphVideoWorldModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with np.load(args.episode, allow_pickle=False) as episode:
        if "graph_node_state" not in episode.files:
            raise SystemExit(
                f"{args.episode} has no graph_node_state; use a DMC episode "
                "collected with the explicit graph schema."
            )
        start = args.context_frames
        horizon = min(
            args.horizon,
            len(episode["action"]) - start,
            len(episode["image"]) - 1 - start,
            len(episode["graph_node_state"]) - 1 - start,
        )
        frames = (
            torch.from_numpy(episode["image"][: start + 1].copy())
            .permute(0, 3, 1, 2)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        initial = frames[:, start]
        context = frames[:, :start] if start else None
        actions = (
            torch.from_numpy(episode["action"][start : start + horizon].copy())
            .float()
            .unsqueeze(0)
            .to(device)
        )
        # Ground truth aligned with predictions: graphs[:, t] predicts the
        # state after t+1 actions, i.e. frame t+1 -- same convention
        # evaluate_rollout.py uses for ground_truth images.
        true_state = episode["graph_node_state"][
            start + 1 : start + horizon + 1
        ].copy()
        node_mask = episode["graph_node_mask"][
            start + 1 : start + horizon + 1
        ].copy()

    with torch.inference_mode():
        rollout = model.rollout(
            initial, actions, use_graph=True, context_images=context
        )
        predicted_state = (
            model.graph_builder.teacher_state(rollout["graphs"])[0]
            .cpu()
            .numpy()
        )

    mask = node_mask.astype(np.float32)
    per_frame_error = (
        ((predicted_state - true_state) ** 2).mean(axis=-1) * mask
    ).sum(axis=-1) / mask.sum(axis=-1).clip(min=1.0)

    print(f"episode={args.episode.name} horizon={horizon}")
    print("frame  graph_state_mse_vs_ground_truth")
    for t, value in enumerate(per_frame_error, start=1):
        print(f"{t:>5}  {value:.6f}")

    early = per_frame_error[: min(8, horizon)].mean()
    late = per_frame_error[max(0, horizon - 8) :].mean()
    print()
    print(f"mean over first 8 frames: {early:.6f}")
    print(f"mean over last 8 frames:  {late:.6f}")
    print(f"ratio (late/early): {late / max(early, 1e-8):.2f}x")


if __name__ == "__main__":
    main()
