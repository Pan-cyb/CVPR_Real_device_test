"""Diagnostic only: does GraphDynamics' self-fed embedding drift off the
distribution RelTR's frozen teacher_state() decoder was actually trained on?

pretrain_graph_dynamics.py v2's drift check showed single-step loss
improving/stabilizing while long-horizon state-space error kept getting
*worse* with more training -- the opposite of what more training should do.
Leading hypothesis: GraphDynamics accumulates delta additively with nothing
keeping the running embedding within the range RelTR's own observer ever
actually produces; teacher_state() was only ever trained on real, in-
distribution embeddings, so decoding an off-distribution one during a long
self-fed rollout can blow up in an uncontrolled way, inflating state-space
error out of proportion to how far the embedding itself has drifted.

This compares the rolled-out embedding's own norm against real observed
embeddings' norm at the same frames, frame by frame -- a growing gap
directly confirms or rules out the hypothesis before we design a fix.

Reuses pretrain_graph_dynamics.py's classes/functions; does not modify it or
any training code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pretrain_graph_dynamics import (
    HistoryAwareGraphDynamics,
    build_frozen_observer,
    load_episode,
    observe_episode,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GraphDynamics' self-fed embedding norm against real "
            "observed embeddings, frame by frame."
        )
    )
    parser.add_argument("--dynamics-checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument(
        "--reltr-checkpoint", type=Path, default=Path("Models/reltr/checkpoint0149.pth")
    )
    parser.add_argument(
        "--graph-observer-checkpoint",
        type=Path,
        default=Path("runs/cheetah_run_reltr_observer_v1/best.pt"),
    )
    parser.add_argument("--reltr-image-size", type=int, default=224)
    parser.add_argument("--horizon", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.dynamics_checkpoint, map_location=device, weights_only=False)
    config = payload["config"]

    observer = build_frozen_observer(
        args.reltr_checkpoint,
        args.graph_observer_checkpoint,
        config["node_dim"],
        8,
        config["relation_count"],
        13,
        args.reltr_image_size,
        device,
    )
    dynamics = HistoryAwareGraphDynamics(
        node_dim=config["node_dim"],
        action_dim=config["action_dim"],
        relation_count=config["relation_count"],
        history=config["history"],
        steps=config["steps"],
        max_delta=config["max_delta"],
    ).to(device)
    dynamics.load_state_dict(payload["model"])
    dynamics.eval()

    episode = load_episode(args.episode)
    if episode is None:
        raise SystemExit(f"{args.episode} has no graph schema fields")
    episode["embeddings"] = observe_episode(observer, episode["image"], device)
    history = config["history"]
    horizon = min(args.horizon, len(episode["action"]) - history)

    real_nodes = episode["embeddings"]["nodes"]  # [T, N, D], real observed
    real_norm_reference = float(real_nodes.norm(dim=-1).mean())

    window = real_nodes[0:history].unsqueeze(0).to(device)
    relation_logits = episode["embeddings"]["relation_logits"][0:1].to(device)
    true_state = episode["true_state"][history : history + horizon]
    mask = episode["mask"][history : history + horizon]

    print(f"reference: mean norm of real observed embeddings = {real_norm_reference:.4f}")
    print("frame  predicted_embedding_norm  real_embedding_norm  state_mse_vs_truth")
    with torch.inference_mode():
        for step in range(horizon):
            action = torch.from_numpy(
                episode["action"][history - 1 + step : history + step]
            ).to(device)
            next_embedding, _, _ = dynamics(window, action, relation_logits)
            predicted_norm = float(next_embedding.norm(dim=-1).mean())
            real_norm_here = float(
                real_nodes[history + step].norm(dim=-1).mean()
            )
            predicted_state = (
                observer.graph_builder.teacher_state(next_embedding)[0]
                .cpu()
                .numpy()
            )
            error = (
                ((predicted_state - true_state[step]) ** 2).mean(-1) * mask[step]
            ).sum() / max(mask[step].sum(), 1.0)
            print(
                f"{step + 1:>5}  {predicted_norm:>24.4f}  "
                f"{real_norm_here:>20.4f}  {error:>18.6f}"
            )
            window = torch.cat((window[:, 1:], next_embedding.unsqueeze(1)), dim=1)


if __name__ == "__main__":
    main()
