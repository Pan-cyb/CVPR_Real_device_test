"""Diagnose whether a DMC policy changes its actions with target location."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def target_centroid(frame: np.ndarray) -> np.ndarray:
    """Locate the red Reacher target in one rendered RGB frame."""
    value = frame.astype(np.int16)
    red, green, blue = value[..., 0], value[..., 1], value[..., 2]
    mask = (
        (red > 140)
        & (red - green > 35)
        & (red - blue > 35)
        & (green < 170)
    )
    y, x = np.nonzero(mask)
    if len(x) < 4:
        return np.asarray([np.nan, np.nan], dtype=np.float64)
    height, width = frame.shape[:2]
    return np.asarray(
        [x.mean() / max(1, width - 1), y.mean() / max(1, height - 1)],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = sorted(args.evaluation.glob("episode_*.npz"))
    if len(files) < 2:
        parser.error("Need at least two evaluated episode_*.npz files")

    actions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    returns: list[float] = []
    for path in files:
        with np.load(path, allow_pickle=False) as episode:
            if "actions" not in episode.files:
                parser.error(
                    f"{path} has no actions; re-evaluate with the updated script"
                )
            horizon = min(int(args.horizon), len(episode["actions"]))
            actions.append(np.asarray(episode["actions"][:horizon]))
            targets.append(target_centroid(episode["rgb"][0]))
            returns.append(float(episode["episode_return"]))
    common_horizon = min(item.shape[0] for item in actions)
    action_array = np.stack([item[:common_horizon] for item in actions])
    target_array = np.stack(targets)

    target_distances: list[float] = []
    action_distances: list[float] = []
    for left in range(len(files)):
        for right in range(left + 1, len(files)):
            if np.isfinite(target_array[[left, right]]).all():
                target_distances.append(
                    float(
                        np.sqrt(
                            np.sum(
                                (target_array[left] - target_array[right]) ** 2
                            )
                        )
                    )
                )
                action_distances.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (action_array[left] - action_array[right]) ** 2
                            )
                        )
                    )
                )
    correlation = float("nan")
    if (
        len(target_distances) >= 3
        and np.std(target_distances) > 1e-8
        and np.std(action_distances) > 1e-8
    ):
        target_centered = np.asarray(target_distances) - np.mean(target_distances)
        action_centered = np.asarray(action_distances) - np.mean(action_distances)
        correlation = float(
            np.mean(target_centered * action_centered)
            / np.sqrt(
                np.mean(target_centered ** 2)
                * np.mean(action_centered ** 2)
            )
        )
    metrics = {
        "episodes": len(files),
        "horizon": common_horizon,
        "mean_return": float(np.mean(returns)),
        "first_action_std": action_array[:, 0].std(axis=0).tolist(),
        "mean_pairwise_target_distance": float(np.mean(target_distances)),
        "mean_pairwise_action_rms": float(np.mean(action_distances)),
        "target_action_distance_correlation": correlation,
        "interpretation": (
            "Near-zero action RMS indicates a target-agnostic policy. A "
            "positive target/action distance correlation is expected from a "
            "target-conditioned deterministic policy."
        ),
    }
    destination = args.output or args.evaluation / "target_conditioning.json"
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
