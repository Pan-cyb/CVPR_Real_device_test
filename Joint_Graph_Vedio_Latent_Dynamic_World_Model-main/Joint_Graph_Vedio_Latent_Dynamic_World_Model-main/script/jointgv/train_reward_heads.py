"""Fit one reward head per planning track on that track's own imagination.

In plug-in (P) training the world model's reward head only ever reads the
backbone's own one-step predictions. The graph-corrected tokens a graph
planner imagines are outside its training distribution, so reward-CEM would
compare reward heads rather than dynamics (on RM65-B the head's rollout
reward error was about 5x higher on the graph track, although that track's
tokens are much closer to the real ones).

For every ``--track LABEL CHECKPOINT MODE`` this script freezes the world
model and imagines the training and validation windows open-loop exactly as
the planner does (``LatentRollout``: real context frames and actions,
recorded future actions, the same history window and precision). It then
fits a ``TrackRewardHead`` to the true rewards of those imagined frames,
with early stopping on validation MSE. Every track gets the same windows,
architecture, seed and budget. The world model's own head is scored on the
same imagined frames for reference.

    python script/jointgv/train_reward_heads.py \\
        --track backbone runs/.../best_backbone_rollout_joint_px.pt base \\
        --track graph runs/.../best_rollout_joint_px.pt graph \\
        --amp --output runs/rm65b_pendulum_reward_heads

Then pass ``--reward-head LABEL runs/.../LABEL.pt`` for every ``--model`` of
evaluation/evaluate_planning.py. Reads checkpoints and data; writes the heads
and metrics.json to ``--output``.

MODE ``real`` instead fits one head on the encoded real frames of the same
windows (no imagination), the usual model-based RL reward model, to be shared
by every track (train_cartpole_actor.py). It is then scored on the backbone
and graph imagination of the validation windows: with a fixed scorer the only
difference between the tracks is how realistic their predictions are.

    python script/jointgv/train_reward_heads.py \\
        --track real runs/.../checkpoints/epoch_0050.pt real \\
        --context-frames 1 --horizon 16 --amp --output runs/.../real_head
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jointgv import JointGraphVideoWorldModel  # noqa: E402
from jointgv.data import EpisodeSequences  # noqa: E402
from jointgv.planning import LatentRollout, TrackRewardHead  # noqa: E402


class ImaginedWindows:
    """Imagined tokens of one track: frame 0 is the encoded current frame,
    frames 1..H are predictions; reward[k-1] belongs to frame k."""

    def __init__(self, tokens: Tensor, rewards: Tensor):
        self.tokens = tokens            # [W, H+1, P, D] float16, CPU
        self.rewards = rewards          # [W, H] float32, CPU
        self.windows, frames = tokens.shape[:2]
        self.horizon = frames - 1

    def __len__(self) -> int:
        return self.windows * self.horizon

    def batch(self, index: Tensor, device: torch.device):
        window = index // self.horizon
        step = index % self.horizon + 1
        current = self.tokens[window, step].to(device).float()
        previous = self.tokens[window, step - 1].to(device).float()
        target = self.rewards[window, step - 1].to(device)
        return current, previous, target, step


def window_loader(root: Path, args, stride: int) -> DataLoader:
    dataset = EpisodeSequences(
        root,
        sequence_length=args.context_frames + args.horizon,
        stride=stride,
        include_graph_targets=False,
        include_control_targets=True,
    )
    return DataLoader(dataset, batch_size=args.rollout_batch, shuffle=False)


@torch.no_grad()
def imagine(model, mode: str, loader: DataLoader, args, device, desc: str) -> ImaginedWindows:
    rollout = LatentRollout(
        model,
        use_graph=mode == "graph",
        history_window=args.history_window or None,
        amp=args.amp,
    )
    context, horizon = args.context_frames, args.horizon
    tokens = rewards = None
    filled = 0
    for images, actions, targets in tqdm(loader, desc=desc, dynamic_ncols=True):
        images, actions = images.to(device), actions.to(device)
        rollout.prepare(images[:, : context + 1], actions[:, :context])
        predicted = rollout.rollout(actions[:, context : context + horizon])
        frames = torch.cat((rollout.start_tokens()[:, None], predicted), dim=1)
        if tokens is None:
            count = len(loader.dataset)
            tokens = torch.empty((count, *frames.shape[1:]), dtype=torch.float16)
            rewards = torch.empty((count, horizon))
        batch = frames.shape[0]
        tokens[filled : filled + batch] = frames.half().cpu()
        rewards[filled : filled + batch] = targets["reward"][:, context : context + horizon]
        filled += batch
    return ImaginedWindows(tokens, rewards)


@torch.no_grad()
def encode_real(model, loader: DataLoader, args, device, desc: str) -> ImaginedWindows:
    """The same windows, but frames 1..H are the encoded real frames (MODE
    real): what a reward head fitted on real observations sees."""
    context, horizon = args.context_frames, args.horizon
    tokens = rewards = None
    filled = 0
    for images, _, targets in tqdm(loader, desc=desc, dynamic_ncols=True):
        frames = images[:, context : context + horizon + 1].to(device)
        encoded = model.vae.encode(frames.flatten(0, 1), sample=False)[0]
        encoded = encoded.reshape(*frames.shape[:2], *encoded.shape[1:])
        if tokens is None:
            count = len(loader.dataset)
            tokens = torch.empty((count, *encoded.shape[1:]), dtype=torch.float16)
            rewards = torch.empty((count, horizon))
        batch = encoded.shape[0]
        tokens[filled : filled + batch] = encoded.half().cpu()
        rewards[filled : filled + batch] = targets["reward"][:, context : context + horizon]
        filled += batch
    return ImaginedWindows(tokens, rewards)


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    positives = int(positive.sum())
    negatives = positive.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = np.empty(scores.size)
    ranks[np.argsort(scores, kind="stable")] = np.arange(1, scores.size + 1)
    return float(
        (ranks[positive].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


@torch.no_grad()
def score(predict, data: ImaginedWindows, args, device) -> dict[str, float]:
    """MSE, R^2 and gate AUROC of ``predict(current, previous)`` over all
    imagined validation frames, plus MSE per third of the horizon."""
    predictions, targets, steps = [], [], []
    for start in range(0, len(data), args.batch_size * 4):
        index = torch.arange(start, min(start + args.batch_size * 4, len(data)))
        current, previous, target, step = data.batch(index, device)
        predictions.append(predict(current, previous).float().cpu())
        targets.append(target.cpu())
        steps.append(step)
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    step = torch.cat(steps).numpy()
    error = (prediction - target) ** 2
    variance = float(target.var())
    result = {
        "mse": float(error.mean()),
        "r2": 1.0 - float(error.mean()) / variance if variance > 0 else float("nan"),
        "auroc": auroc(prediction, target > args.positive_threshold),
    }
    thirds = np.array_split(np.arange(1, data.horizon + 1), 3)
    for part in thirds:
        mask = (step >= part[0]) & (step <= part[-1])
        result[f"mse_h{part[0]}-{part[-1]}"] = float(error[mask].mean())
    return result


def fit(train: ImaginedWindows, validation: ImaginedWindows, args, device, label: str):
    torch.manual_seed(args.seed)
    token_count, token_dim = train.tokens.shape[-2:]
    grid = int(round(math.sqrt(token_count)))
    if grid * grid != token_count:
        raise SystemExit(f"{token_count} tokens are not a square patch grid")
    head = TrackRewardHead(token_dim, grid, args.hidden, args.dropout).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    sample = torch.randperm(len(train), generator=generator)[: args.stat_samples]
    current, previous, _, _ = train.batch(sample, device)
    features = head.features(current, previous).flatten(0, -2)
    head.input_mean.copy_(features.mean(0))
    head.input_std.copy_(features.std(0).clamp_min(1e-4))
    del current, previous, features

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_mse, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(train), generator=generator)
        total, seen = 0.0, 0
        bar = tqdm(
            range(0, len(order), args.batch_size),
            desc=f"{label} epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
        )
        for start in bar:
            current, previous, target, _ = train.batch(
                order[start : start + args.batch_size], device
            )
            loss = torch.nn.functional.mse_loss(head(current, previous), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * target.shape[0]
            seen += target.shape[0]
            bar.set_postfix(loss=f"{total / seen:.4f}")
        head.eval()
        val = score(head, validation, args, device)["mse"]
        improved = val < best_mse
        if improved:
            best_mse, best_state, best_epoch, stale = val, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            stale += 1
        print(
            f"{label} epoch {epoch:3d}  train_mse={total / seen:.4f}  "
            f"val_mse={val:.4f}" + ("  *best" if improved else "")
        )
        if stale >= args.patience:
            print(f"{label}: no improvement for {args.patience} epochs, stopping")
            break
    head.load_state_dict(best_state)
    return head.eval(), best_epoch, epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--track",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "CHECKPOINT", "MODE"),
        help="Repeatable; same LABEL/CHECKPOINT/MODE as evaluate_planning.py --model.",
    )
    parser.add_argument("--data", type=Path, default=Path("data/rm65b_pendulum_train_v2"))
    parser.add_argument(
        "--validation-data", type=Path, default=Path("data/rm65b_pendulum_val_v2")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/rm65b_pendulum_reward_heads"))
    # Imagination: must match the planner (evaluate_planning.py defaults).
    parser.add_argument("--context-frames", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=24, help="Planning horizon H.")
    parser.add_argument("--history-window", type=int, default=4, help="0 = full history.")
    parser.add_argument("--amp", action="store_true", help="fp16 predictor, as evaluate_planning --amp.")
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--validation-stride", type=int, default=8)
    parser.add_argument("--rollout-batch", type=int, default=64)
    # Head fitting.
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--stat-samples", type=int, default=4096)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=1.0,
        help="Reward above this counts as gate open for AUROC (max = action repeat).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    labels = [label for label, _, _ in args.track]
    if len(set(labels)) != len(labels):
        parser.error("track labels must be unique")
    for _, _, mode in args.track:
        if mode not in {"graph", "base", "real"}:
            parser.error("MODE must be 'graph', 'base' or 'real'")
    if args.history_window and args.history_window < 2:
        parser.error("--history-window must be 0 (full) or at least 2")

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    train_loader = window_loader(args.data, args, args.train_stride)
    validation_loader = window_loader(args.validation_data, args, args.validation_stride)
    print(
        f"{len(train_loader.dataset)} training / {len(validation_loader.dataset)} "
        f"validation windows: {args.context_frames} context frames + "
        f"{args.horizon} imagined steps, history window "
        f"{args.history_window or 'full'}, amp={'on' if args.amp else 'off'}"
    )

    models: dict[str, JointGraphVideoWorldModel] = {}
    epochs: dict[str, object] = {}
    summary: dict[str, dict] = {}
    for label, checkpoint_path, mode in args.track:
        if checkpoint_path not in models:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = JointGraphVideoWorldModel(**checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["model"])
            models[checkpoint_path] = model.eval()
            epochs[checkpoint_path] = checkpoint.get("epoch", "?")
            del checkpoint
        model = models[checkpoint_path]
        print(f"\n== {label}: {checkpoint_path} (epoch {epochs[checkpoint_path]}) mode={mode}")
        if mode == "real":
            train = encode_real(model, train_loader, args, device, f"{label} encode train")
            validation = encode_real(model, validation_loader, args, device, f"{label} encode val")
        else:
            train = imagine(model, mode, train_loader, args, device, f"{label} imagine train")
            validation = imagine(model, mode, validation_loader, args, device, f"{label} imagine val")

        def original(current: Tensor, previous: Tensor) -> Tensor:
            return model.reward_head(current.mean(dim=-2)).squeeze(-1)

        own = score(original, validation, args, device)
        print(
            f"{label}: world model's own head on these {'real' if mode == 'real' else 'imagined'} "
            f"val frames: mse={own['mse']:.4f} r2={own['r2']:.3f} auroc={own['auroc']:.3f}"
        )
        head, best_epoch, last_epoch = fit(train, validation, args, device, label)
        refit = score(head, validation, args, device)
        scores = {"own_head": own, "refit_head": refit}
        if mode == "real":
            # The shared scorer on each planning track's imagination of the
            # same validation windows: how well each track's predictions can
            # be read by a reward model of the real world.
            train = None        # free the training windows before imagining
            for track_mode in ("base", "graph"):
                imagined = imagine(
                    model, track_mode, validation_loader, args, device,
                    f"{label} imagine val ({track_mode})",
                )
                scores[f"on_{track_mode}"] = score(head, imagined, args, device)
                del imagined
                values = scores[f"on_{track_mode}"]
                print(
                    f"{label} head on {track_mode} imagination: mse={values['mse']:.4f} "
                    f"r2={values['r2']:.3f} auroc={values['auroc']:.3f}"
                )
        track = {
            "label": label,
            "mode": mode,
            "checkpoint": str(checkpoint_path),
            "checkpoint_resolved": str(Path(checkpoint_path).resolve()),
            "checkpoint_epoch": epochs[checkpoint_path],
            "context_frames": args.context_frames,
            "horizon": args.horizon,
            "history_window": args.history_window or None,
            "amp": args.amp,
            "data": str(args.data),
            "validation_data": str(args.validation_data),
            "best_epoch": best_epoch,
            "epochs_run": last_epoch,
            "validation": scores,
        }
        path = args.output / f"{label}.pt"
        torch.save(
            {"config": head.config, "state_dict": head.state_dict(), "track": track},
            path,
        )
        print(
            f"{label}: refit head mse={refit['mse']:.4f} r2={refit['r2']:.3f} "
            f"auroc={refit['auroc']:.3f} (best epoch {best_epoch}) |--> saved {path}"
        )
        summary[label] = track
        del train, validation, head
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        "\nvalidation frames (lower mse / higher r2, auroc is better): imagined by the "
        "track's mode; a real head's own/refit rows are on real frames, its on_base / "
        "on_graph rows on each track's imagination"
    )
    thirds = [key for key in next(iter(summary.values()))["validation"]["refit_head"] if key.startswith("mse_h")]
    header = ["track", "head", "mse", "r2", "auroc", *thirds]
    print("  ".join(name.rjust(10) for name in header))
    for label, track in summary.items():
        rows = [("own" if key == "own_head" else "refit" if key == "refit_head" else key, values)
                for key, values in track["validation"].items()]
        for name, values in rows:
            cells = [label, name] + [
                f"{values[key]:.4f}" if key != "r2" and key != "auroc" else f"{values[key]:.3f}"
                for key in ("mse", "r2", "auroc", *thirds)
            ]
            print("  ".join(cell.rjust(10) for cell in cells))
    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "config": {
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(args).items()
                },
                "tracks": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {args.output / 'metrics.json'}")


if __name__ == "__main__":
    main()
