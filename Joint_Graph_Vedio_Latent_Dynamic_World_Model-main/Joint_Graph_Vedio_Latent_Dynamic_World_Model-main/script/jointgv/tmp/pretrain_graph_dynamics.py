"""Standalone GraphDynamics pretraining v2: real RelTR embeddings, a short
history/velocity input window, and K-step self-fed training.

v1 (git history) trained a temporary GraphDynamics(node_dim=13) directly on
clean ground-truth graph_node_state, bypassing the real RelTR observer
entirely. That took drift (see check_graph_drift.py) from 92x (untrained)
down to ~2-3x, but could not separate "GraphDynamics needed more training"
from "GraphDynamics needed clean input it will never actually get" -- the
real joint pipeline feeds it noisy, RelTR-derived 64-dim embeddings, not
ground truth.

This version keeps the real, frozen, already-trained RelTR explicit
observer in the loop -- GraphDynamics trains on the exact embeddings it
sees in the real joint pipeline -- and adds a literature-motivated input
change (Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with
Graph Networks", ICML 2020): a short window of finite-difference velocity
features, since most graph-network physics simulators condition on several
past velocities rather than a single instantaneous position, which has no
way to represent motion at all.

Does not touch train.py / losses.py / graph_plugin.py / graph_dynamics.py --
HistoryAwareGraphDynamics is a separate class, local to this script. The
shared GraphDynamics used by the real joint pipeline (DINO-WM and the
DreamerV3 branch) is completely unaffected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv import JointGraphVideoWorldModel
from jointgv.blocks.graph_dynamics import GraphBlock
from train import load_graph_observer_checkpoint


class HistoryAwareGraphDynamics(nn.Module):
    """Like the shared GraphDynamics, but conditions on a short window of
    finite-difference velocities instead of only the current instantaneous
    state -- a single frame cannot represent motion at all.
    """

    def __init__(
        self,
        node_dim: int,
        action_dim: int,
        relation_count: int,
        history: int,
        steps: int = 3,
        max_delta: float = 0.25,
        max_relation_delta: float = 0.5,
    ):
        super().__init__()
        if history < 1:
            raise ValueError("history must be at least 1 (current frame only)")
        self.node_dim = node_dim
        self.history = history
        self.history_encoder = nn.Sequential(
            nn.Linear(node_dim * history, node_dim),
            nn.SiLU(),
        )
        self.max_delta = float(max_delta)
        self.max_relation_delta = float(max_relation_delta)
        self.blocks = nn.ModuleList(
            GraphBlock(node_dim, action_dim, relation_count)
            for _ in range(steps)
        )
        self.delta = nn.Linear(node_dim, node_dim)
        self.relation_delta = nn.Sequential(
            nn.Linear(2 * node_dim + action_dim, 2 * node_dim),
            nn.SiLU(),
            nn.Linear(2 * node_dim, relation_count),
        )
        # Same stability convention as the shared GraphDynamics: start as an
        # exact identity/no-op, let supervision grow bounded increments.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.relation_delta[-1].weight)
        nn.init.zeros_(self.relation_delta[-1].bias)

    def encode_window(self, window: Tensor) -> Tensor:
        """window: [B, history, N, D] (oldest..newest) -> enriched [B, N, D]."""
        batch, history, count, dim = window.shape
        current = window[:, -1]
        velocities = window[:, 1:] - window[:, :-1]  # [B, history-1, N, D]
        velocities = velocities.permute(0, 2, 1, 3).reshape(batch, count, -1)
        features = torch.cat((current, velocities), dim=-1)
        return self.history_encoder(features)

    def forward(
        self, window: Tensor, action: Tensor, relation_logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """window: [B, history, N, D] -> next_nodes, delta, next_relation_logits."""
        nodes = window[:, -1]
        hidden = self.encode_window(window)
        for block in self.blocks:
            hidden = block(hidden, action, relation_logits)
        delta = self.max_delta * torch.tanh(self.delta(hidden))
        next_nodes = nodes + delta
        count = nodes.shape[1]
        sender = hidden[:, :, None, :].expand(-1, -1, count, -1)
        receiver = hidden[:, None, :, :].expand(-1, count, -1, -1)
        edge_action = action[:, None, None, :].expand(-1, count, count, -1)
        relation_delta = self.max_relation_delta * torch.tanh(
            self.relation_delta(
                torch.cat((sender, receiver, edge_action), dim=-1)
            )
        )
        next_relation_logits = relation_logits + relation_delta
        return next_nodes, delta, next_relation_logits


def build_frozen_observer(
    reltr_checkpoint: Path,
    graph_observer_checkpoint: Path,
    node_dim: int,
    graph_nodes: int,
    relation_count: int,
    teacher_state_dim: int,
    reltr_image_size: int,
    device: torch.device,
) -> JointGraphVideoWorldModel:
    """A minimal world model whose only purpose is a working, frozen,
    already-trained explicit graph observer -- backbone=dino_wm with the
    no-download debug encoder, since the visual side is never used here.
    """
    model = JointGraphVideoWorldModel(
        action_dim=6,
        backbone="dino_wm",
        dino_model="debug_conv",
        node_dim=node_dim,
        graph_nodes=graph_nodes,
        relation_count=relation_count,
        teacher_state_dim=teacher_state_dim,
        graph_observer="explicit",
        reltr_checkpoint=str(reltr_checkpoint),
        reltr_image_size=reltr_image_size,
    ).to(device)
    load_graph_observer_checkpoint(model, graph_observer_checkpoint, device)
    for parameter in model.graph_builder.parameters():
        parameter.requires_grad_(False)
    model.graph_builder.eval()
    return model


@torch.no_grad()
def observe_episode(
    model: JointGraphVideoWorldModel,
    images: np.ndarray,
    device: torch.device,
) -> dict[str, Tensor]:
    """Real, frozen-observer embeddings for one full episode -- exactly the
    teacher-forced sequence the joint pipeline's infer_sequence produces,
    cached once per episode since the observer never changes."""
    image_tensor = (
        torch.from_numpy(images)
        .permute(0, 3, 1, 2)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )
    dummy_tokens = image_tensor.new_zeros(image_tensor.shape[0], image_tensor.shape[1], 1, 1)
    observed = model.graph_plugin.infer_sequence(dummy_tokens, image_tensor)
    return {
        "nodes": observed["nodes"][0].cpu(),
        "relation_logits": observed["relation_logits"][0].cpu(),
    }


def load_episode(path: Path) -> dict[str, np.ndarray] | None:
    with np.load(path, allow_pickle=False) as episode:
        if "graph_node_state" not in episode.files:
            return None
        return {
            "image": episode["image"].copy(),
            "true_state": episode["graph_node_state"].astype(np.float32),
            "mask": episode["graph_node_mask"].astype(np.float32),
            "action": episode["action"].astype(np.float32),
        }


def masked_state_loss(
    predicted: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    values = F.smooth_l1_loss(predicted, target, reduction="none").mean(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def sample_batch(
    episodes: list[dict],
    batch_size: int,
    sequence_length: int,
    history: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Each sample needs `history` real frames of context before the first
    predicted step, so windows start at index >= history - 1."""
    node_windows, true_states, masks, relations, actions = [], [], [], [], []
    for _ in range(batch_size):
        episode = episodes[np.random.randint(len(episodes))]
        emb = episode["embeddings"]
        earliest_start = history - 1
        latest_start = len(episode["action"]) - sequence_length
        start = np.random.randint(earliest_start, latest_start + 1)
        node_windows.append(
            emb["nodes"][start - history + 1 : start + sequence_length + 1]
        )
        true_states.append(
            episode["true_state"][start + 1 : start + sequence_length + 1]
        )
        masks.append(episode["mask"][start + 1 : start + sequence_length + 1])
        relations.append(emb["relation_logits"][start])
        actions.append(episode["action"][start : start + sequence_length])
    return {
        "node_window": torch.stack(node_windows).to(device),
        "true_state": torch.from_numpy(np.stack(true_states)).to(device),
        "mask": torch.from_numpy(np.stack(masks)).to(device),
        "relation_logits": torch.stack(relations).to(device),
        "action": torch.from_numpy(np.stack(actions)).to(device),
    }


def masked_embedding_loss(
    predicted: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    """Direct 64-dim embedding-space loss.

    teacher_state() only supervises whatever subspace of the embedding the
    frozen decoder actually reads (13 of node_dim=64 dims) -- the remaining
    dimensions are free to drift with no penalty, since nothing else
    constrains a predicted embedding to resemble one RelTR would ever
    actually produce. This penalizes the full embedding directly against
    the real observed embedding at the same frame, keeping every dimension
    anchored to the real-embedding manifold, not just the decoded ones.
    """
    values = F.smooth_l1_loss(predicted, target, reduction="none").mean(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def single_step_loss(
    dynamics: HistoryAwareGraphDynamics,
    observer: JointGraphVideoWorldModel,
    batch: dict[str, Tensor],
    history: int,
) -> tuple[Tensor, Tensor]:
    """Teacher-forced: every window is real embeddings, target is real
    state and (for the embedding-space term) the real embedding at the
    same frame, which is already present later in the same window."""
    window, action = batch["node_window"], batch["action"]
    steps = action.shape[1]
    state_losses, embedding_losses = [], []
    for step in range(steps):
        current_window = window[:, step : step + history]
        _, delta, _ = dynamics(
            current_window, action[:, step], batch["relation_logits"]
        )
        predicted_embedding = current_window[:, -1] + delta
        real_target_embedding = window[:, step + history]
        predicted_state = observer.graph_builder.teacher_state(
            predicted_embedding
        )
        state_losses.append(
            masked_state_loss(
                predicted_state, batch["true_state"][:, step], batch["mask"][:, step]
            )
        )
        embedding_losses.append(
            masked_embedding_loss(
                predicted_embedding, real_target_embedding, batch["mask"][:, step]
            )
        )
    return torch.stack(state_losses).mean(), torch.stack(embedding_losses).mean()


def multistep_loss(
    dynamics: HistoryAwareGraphDynamics,
    observer: JointGraphVideoWorldModel,
    batch: dict[str, Tensor],
    history: int,
    horizon: int,
) -> tuple[Tensor, Tensor]:
    """K-step self-fed: the window slides forward using the model's own
    predicted embeddings, not real ones, after the first `history` frames.
    `window` (the real, un-mutated sequence) still supplies embedding-space
    targets; only `current_window` (the model's own rolling input) is fed
    back autoregressively."""
    window, action = batch["node_window"], batch["action"]
    horizon = min(horizon, action.shape[1])
    current_window = window[:, :history].clone()
    state_losses, embedding_losses = [], []
    for step in range(horizon):
        next_embedding, delta, _ = dynamics(
            current_window, action[:, step], batch["relation_logits"]
        )
        real_target_embedding = window[:, step + history]
        predicted_state = observer.graph_builder.teacher_state(next_embedding)
        state_losses.append(
            masked_state_loss(
                predicted_state, batch["true_state"][:, step], batch["mask"][:, step]
            )
        )
        embedding_losses.append(
            masked_embedding_loss(
                next_embedding, real_target_embedding, batch["mask"][:, step]
            )
        )
        current_window = torch.cat(
            (current_window[:, 1:], next_embedding.unsqueeze(1)), dim=1
        )
    return torch.stack(state_losses).mean(), torch.stack(embedding_losses).mean()


@torch.no_grad()
def report_drift(
    dynamics: HistoryAwareGraphDynamics,
    observer: JointGraphVideoWorldModel,
    episode: dict,
    history: int,
    horizon: int,
    device: torch.device,
) -> None:
    """Same measurement as check_graph_drift.py: per-frame MSE vs ground
    truth over a long, fully self-fed rollout, seeded with `history` real
    frames instead of check_graph_drift.py's single real frame."""
    emb = episode["embeddings"]
    horizon = min(horizon, len(episode["action"]) - history)
    window = emb["nodes"][0:history].unsqueeze(0).to(device)
    relation_logits = emb["relation_logits"][0:1].to(device)
    true_state = episode["true_state"][history : history + horizon]
    mask = episode["mask"][history : history + horizon]
    errors = []
    for step in range(horizon):
        action = torch.from_numpy(
            episode["action"][history - 1 + step : history + step]
        ).to(device)
        next_embedding, _, _ = dynamics(window, action, relation_logits)
        predicted_state = (
            observer.graph_builder.teacher_state(next_embedding)[0].cpu().numpy()
        )
        error = (
            ((predicted_state - true_state[step]) ** 2).mean(-1) * mask[step]
        ).sum() / max(mask[step].sum(), 1.0)
        errors.append(float(error))
        window = torch.cat((window[:, 1:], next_embedding.unsqueeze(1)), dim=1)
    early = float(np.mean(errors[:8]))
    late = float(np.mean(errors[-8:]))
    print(f"  drift check: frame1={errors[0]:.6f} frame{horizon}={errors[-1]:.6f}")
    print(
        f"  mean first 8={early:.6f} mean last 8={late:.6f} "
        f"ratio={late / max(early, 1e-8):.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain GraphDynamics on real (frozen-observer) embeddings, "
            "with a short history/velocity input window."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reltr-checkpoint", type=Path, default=Path("Models/reltr/checkpoint0149.pth")
    )
    parser.add_argument(
        "--graph-observer-checkpoint",
        type=Path,
        default=Path("runs/cheetah_run_reltr_observer_v1/best.pt"),
    )
    parser.add_argument("--reltr-image-size", type=int, default=224)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--graph-nodes", type=int, default=8)
    parser.add_argument("--relation-count", type=int, default=4)
    parser.add_argument("--teacher-state-dim", type=int, default=13)
    parser.add_argument("--history", type=int, default=4, help="Window length: current frame + (history-1) velocities.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dynamics-steps", type=int, default=3)
    parser.add_argument("--max-delta", type=float, default=0.25)
    parser.add_argument(
        "--multistep-horizon",
        type=int,
        default=0,
        help="K: also train a K-step self-fed unroll each batch. 0 disables it.",
    )
    parser.add_argument("--multistep-weight", type=float, default=1.0)
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for the direct 64-dim embedding-space loss (predicted "
            "vs the real observed embedding at the same frame), on top of "
            "the existing decoded-13-dim-state loss. 0 disables it, "
            "reproducing the previous (state-only) behavior."
        ),
    )
    parser.add_argument("--drift-check-horizon", type=int, default=32)
    args = parser.parse_args()
    if args.multistep_horizon < 0:
        parser.error("--multistep-horizon must be nonnegative")
    if args.history < 1:
        parser.error("--history must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observer = build_frozen_observer(
        args.reltr_checkpoint,
        args.graph_observer_checkpoint,
        args.node_dim,
        args.graph_nodes,
        args.relation_count,
        args.teacher_state_dim,
        args.reltr_image_size,
        device,
    )

    def load_and_embed(directory: Path) -> list[dict]:
        episodes = []
        for path in sorted(directory.glob("*.npz")):
            episode = load_episode(path)
            if episode is None:
                continue
            episode["embeddings"] = observe_episode(observer, episode["image"], device)
            del episode["image"]
            episodes.append(episode)
        return episodes

    train_episodes = load_and_embed(args.data)
    if not train_episodes:
        raise SystemExit(f"No graph-schema episodes found under {args.data}")
    validation_episodes = (
        load_and_embed(args.validation_data)
        if args.validation_data is not None
        else train_episodes
    )
    if not validation_episodes:
        validation_episodes = train_episodes

    dynamics = HistoryAwareGraphDynamics(
        node_dim=args.node_dim,
        action_dim=train_episodes[0]["action"].shape[-1],
        relation_count=args.relation_count,
        history=args.history,
        steps=args.dynamics_steps,
        max_delta=args.max_delta,
    ).to(device)
    optimizer = torch.optim.AdamW(dynamics.parameters(), lr=args.lr)

    print(
        f"train_episodes={len(train_episodes)} "
        f"validation_episodes={len(validation_episodes)} "
        f"history={args.history} multistep_horizon={args.multistep_horizon}"
    )

    for epoch in range(1, args.epochs + 1):
        dynamics.train()
        single_losses, multi_losses = [], []
        single_embed_losses, multi_embed_losses = [], []
        for _ in range(args.steps_per_epoch):
            batch = sample_batch(
                train_episodes,
                args.batch_size,
                args.sequence_length,
                args.history,
                device,
            )
            state_loss, embed_loss = single_step_loss(
                dynamics, observer, batch, args.history
            )
            single_losses.append(float(state_loss.detach()))
            single_embed_losses.append(float(embed_loss.detach()))
            loss = state_loss + args.embedding_weight * embed_loss
            if args.multistep_horizon > 0:
                multi_state, multi_embed = multistep_loss(
                    dynamics, observer, batch, args.history, args.multistep_horizon
                )
                multi_losses.append(float(multi_state.detach()))
                multi_embed_losses.append(float(multi_embed.detach()))
                loss = loss + args.multistep_weight * (
                    multi_state + args.embedding_weight * multi_embed
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
            optimizer.step()
        single_mean = float(np.mean(single_losses))
        single_embed_mean = float(np.mean(single_embed_losses))
        multi_mean = float(np.mean(multi_losses)) if multi_losses else 0.0
        multi_embed_mean = (
            float(np.mean(multi_embed_losses)) if multi_embed_losses else 0.0
        )
        print(
            f"epoch={epoch} single_step_loss={single_mean:.6f} "
            f"single_embed_loss={single_embed_mean:.6f} "
            f"multistep_loss={multi_mean:.6f} "
            f"multistep_embed_loss={multi_embed_mean:.6f}"
        )
        if epoch % 5 == 0 or epoch == args.epochs:
            dynamics.eval()
            report_drift(
                dynamics,
                observer,
                validation_episodes[0],
                args.history,
                args.drift_check_horizon,
                device,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": dynamics.state_dict(),
            "config": {
                "node_dim": args.node_dim,
                "action_dim": train_episodes[0]["action"].shape[-1],
                "relation_count": args.relation_count,
                "history": args.history,
                "steps": args.dynamics_steps,
                "max_delta": args.max_delta,
            },
            "multistep_horizon": args.multistep_horizon,
            "multistep_weight": args.multistep_weight,
            "epochs": args.epochs,
        },
        args.output,
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
