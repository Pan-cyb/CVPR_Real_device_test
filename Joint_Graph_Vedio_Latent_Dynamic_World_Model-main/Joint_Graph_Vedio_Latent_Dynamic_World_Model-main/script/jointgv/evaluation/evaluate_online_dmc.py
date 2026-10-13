"""Evaluate an online actor checkpoint in DMC; no CEM is involved."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jointgv import JointGraphVideoWorldModel  # noqa: E402
from jointgv.online import DreamerBehavior  # noqa: E402
from train_online_dmc import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=200_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--policy-mode", choices=("sample", "mode"), default="sample"
    )
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload.get("training_mode") != "online_actor_critic":
        parser.error(
            "Checkpoint is not an online actor--critic checkpoint. Offline "
            "world-model checkpoints do not contain a trained actor."
        )
    config = payload["config"]
    if config["backbone"] != "dreamerv3":
        parser.error("Only DreamerV3 online checkpoints are supported")
    action_dim = int(config["action_dim"])
    model = JointGraphVideoWorldModel(
        action_dim=action_dim,
        backbone=config["backbone"],
        node_dim=int(config["node_dim"]),
        graph_nodes=int(config["graph_nodes"]),
        message_steps=int(config["message_steps"]),
        max_graph_correction=float(config["max_graph_correction"]),
        graph_correction_gate_init=float(config["graph_correction_gate_init"]),
        graph_observer=config["graph_observer"],
        graph_backbone="reltr_resnet50",
        graph_backbone_pretrained=False,
        freeze_graph_backbone=True,
        reltr_checkpoint=str(config["reltr_checkpoint"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    behavior = DreamerBehavior(action_dim, model.dreamerv3_core.feat_dim).to(device)
    behavior.load_state_dict(payload["behavior"])
    metrics = evaluate(
        model,
        behavior,
        config["domain"],
        config["task"],
        args.episodes,
        args.seed,
        device,
        use_graph=not bool(config["disable_graph_conditioning"]),
        output=args.output,
        policy_mode=args.policy_mode,
    )
    print("actor evaluation", " ".join(
        f"{key}={value:.6f}" for key, value in metrics.items()
    ))


if __name__ == "__main__":
    main()
