"""Graph-only (G) ablation: the plug-in branch alone used as the world model.

Each ``--model LABEL CHECKPOINT MODE`` of a plug-in (P) run is rolled out on
the same test windows as evaluate_ablation.py: the ``--context-frames`` real
frames before the start and the start frame, then ``--horizon`` open-loop
steps driven by the recorded actions. Both visual tracks get the real context
frames (evaluate_ablation.py --visual-context); DINO-WM predictors slide over
the last ``--history-window`` frames (3 = the official num_hist).

MODE
  base       the visual backbone alone (graph correction off)
  graph      the graph-corrected visual rollout (P)
  graphonly  the plug-in branch alone (G): the checkpoint's own observer reads
             the context and start frames into its graph history, GraphDynamics
             rolls the graph forward on its own predictions with the recorded
             actions (it never reads the backbone), and the position head
             decodes every node's 2D position. No frame is produced.
             Two graph baselines come with it, from the same observed graph:
             copy-last (last positions held) and constant-velocity (last
             displacement extrapolated).

Per-step metrics:
  joint_px    joint-position error in pixels of the frame (evaluate_ablation's
              definition). base/graph: the fixed --observer-checkpoint reads
              their predicted frames ('observer on ground truth' is that
              measurement floor). graphonly: read from its predicted positions
              ('<label> observer on real frames' is the same checkpoint's
              observer on the real future frames, G's own floor).
  image_mse / motion_mse / ssim   visual tracks only (G produces no frames).
  reward_mse  squared error of the predicted reward. Every head is fitted on
              REAL data of its own modality, so none has seen imagination:
                base/graph, DINO-WM: a TrackRewardHead fitted here on encoded
                  real frames (current + change, as train_reward_heads.py
                  --track real); DreamerV3: the model's own reward head, which
                  Dreamer trains on posterior (real-frame) features.
                graphonly: an MLP fitted here on the checkpoint's observer
                  graphs of real frames (current + change, active nodes).
              Each head's validation R^2 on real data is printed: it bounds
              what its track can predict at all. 'reward copy-last' holds the
              last observed reward. Heads are fitted on --train-data with
              early stopping on --validation-data.
For roles with a left/right pair (walker), windows whose observed start graph
has a left/right part exchanged are also reported separately for graphonly.

Evaluation only: reads checkpoints and data, writes --output.

    python script/jointgv/evaluation/evaluate_graph_only.py \\
        --model backbone RUN/best_backbone_rollout_joint_px.pt base \\
        --model P RUN/best_rollout_joint_px.pt graph \\
        --model G RUN/best_rollout_joint_px.pt graphonly \\
        --reference backbone --data data/dmc_walker_walk_test_v1 \\
        --train-data data/dmc_walker_walk_train_v1 \\
        --validation-data data/dmc_walker_walk_val_v1 \\
        --observer-checkpoint runs/walker_walk_reltr_observer_v1/best_long_rollout.pt \\
        --output runs/eval/walker_graph_only
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jointgv import JointGraphVideoWorldModel  # noqa: E402
from jointgv.data import EpisodeSequences  # noqa: E402
from jointgv.planning import TrackRewardHead, limit_history  # noqa: E402
from jointgv.rollout_metrics import (  # noqa: E402
    _joint_error_from_positions,
    gaussian_window,
    load_reference_observer,
)
from evaluate_ablation import build_loader, interval, prepare, score  # noqa: E402

VISUAL_METRICS = ("joint_px", "image_mse", "motion_mse", "ssim", "reward_mse")
HIGHER_IS_BETTER = {"ssim"}
FLOOR_LABEL = "observer on ground truth"


class GraphRewardHead(nn.Module):
    """Reward of one graph from its active nodes and their change since the
    previous graph; per-channel input standardisation as TrackRewardHead."""

    def __init__(self, node_dim: int, nodes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        channels = 2 * node_dim
        self.register_buffer("input_mean", torch.zeros(channels))
        self.register_buffer("input_std", torch.ones(channels))
        self.net = nn.Sequential(
            nn.Linear(nodes * channels, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )

    def features(self, current: Tensor, previous: Tensor) -> Tensor:
        return torch.cat((current, current - previous), dim=-1).float()

    def forward(self, current: Tensor, previous: Tensor) -> Tensor:
        x = (self.features(current, previous) - self.input_mean) / self.input_std
        return self.net(x.flatten(-2)).squeeze(-1)


class RealWindows:
    """Real-frame windows of one modality: states [W, L+1, ...] (fp16, CPU),
    rewards [W, L]; reward[k-1] belongs to state k (previous state k-1)."""

    def __init__(self, states: Tensor, rewards: Tensor):
        self.states, self.rewards = states, rewards
        self.length = rewards.shape[1]

    def __len__(self) -> int:
        return self.states.shape[0] * self.length

    def batch(self, index: Tensor, device: torch.device):
        window, step = index // self.length, index % self.length + 1
        return (self.states[window, step].to(device).float(),
                self.states[window, step - 1].to(device).float(),
                self.rewards[window, step - 1].to(device))


def active_indices(model) -> list[int]:
    names = model.graph_dynamics_plugin.builder.role_names
    return [index for index, name in enumerate(names) if name != "padding"]


def observe_graphs(model, images: Tensor) -> dict[str, Tensor]:
    """The checkpoint's own observer over [B, T, 3, H, W] frames (its temporal
    memory starts at the first frame)."""
    plugin = model.graph_dynamics_plugin
    tokens = None
    if not plugin.builder.requires_images:
        tokens = model.encode_visual_tokens(images.flatten(0, 1))
        tokens = tokens.reshape(*images.shape[:2], *tokens.shape[1:])
    return plugin.infer_sequence(tokens, images)


@torch.no_grad()
def collect_real(model, root: Path, kind: str, args, device, desc: str) -> RealWindows:
    """Encoded real frames ('visual', DINO-WM tokens) or observer graphs
    ('graph', active nodes) of non-overlapping windows, with their rewards."""
    dataset = EpisodeSequences(root, sequence_length=args.head_window, stride=args.head_window,
                               include_graph_targets=False, include_control_targets=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    active = active_indices(model)
    states, rewards = [], []
    for images, _, targets in tqdm(loader, desc=desc, dynamic_ncols=True):
        images = images.to(device)
        if kind == "visual":
            encoded = model.encode_visual_tokens(images.flatten(0, 1))
            encoded = encoded.reshape(*images.shape[:2], *encoded.shape[1:])
        else:
            encoded = observe_graphs(model, images)["nodes"][:, :, active]
        states.append(encoded.half().cpu())
        rewards.append(targets["reward"].float())
    return RealWindows(torch.cat(states), torch.cat(rewards))


def evaluate_head(head, data: RealWindows, args, device) -> dict[str, float]:
    predictions, targets = [], []
    with torch.no_grad():
        for start in range(0, len(data), args.head_batch * 4):
            current, previous, target = data.batch(
                torch.arange(start, min(start + args.head_batch * 4, len(data))), device)
            predictions.append(head(current, previous).float().cpu())
            targets.append(target.cpu())
    prediction, target = torch.cat(predictions), torch.cat(targets)
    mse = float((prediction - target).square().mean())
    variance = float(target.var())
    return {"mse": mse, "r2": 1.0 - mse / variance if variance > 0 else float("nan")}


def fit_head(head, train: RealWindows, validation: RealWindows, args, device, label: str):
    """train_reward_heads.py's recipe: standardise inputs, AdamW, MSE, early
    stopping on validation MSE."""
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    head = head.to(device)
    sample = torch.randperm(len(train), generator=generator)[:4096]
    current, previous, _ = train.batch(sample, device)
    features = head.features(current, previous).flatten(0, -2)
    head.input_mean.copy_(features.mean(0))
    head.input_std.copy_(features.std(0).clamp_min(1e-4))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=1e-2)
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(1, args.head_epochs + 1):
        head.train()
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(order), args.head_batch):
            current, previous, target = train.batch(order[start : start + args.head_batch], device)
            loss = nn.functional.mse_loss(head(current, previous), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        val = evaluate_head(head, validation, args, device)
        improved = val["mse"] < best
        if improved:
            best, best_state, stale = val["mse"], copy.deepcopy(head.state_dict()), 0
        else:
            stale += 1
        print(f"  {label} epoch {epoch:2d}: val reward MSE {val['mse']:.4f}  R^2 {val['r2']:.3f}"
              + ("  *best" if improved else ""), flush=True)
        if stale >= args.head_patience:
            break
    head.load_state_dict(best_state)
    return head.eval()


def swapped_at_start(names: list[str], positions: Tensor, pixel: Tensor, visible: Tensor) -> Tensor:
    """[B] True where a left/right part of the start graph fits the labels at
    least 1 px better exchanged (view_graph_observer.py's definition)."""
    flags = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    scale = (64 - 1) / 2.0
    for part in ("thigh", "leg", "foot", "shin"):
        if f"right_{part}" not in names or f"left_{part}" not in names:
            continue
        pair = [names.index(f"right_{part}"), names.index(f"left_{part}")]
        seen = visible[:, pair].bool().all(-1)
        direct = (positions[:, pair] - pixel[:, pair]).norm(dim=-1).mean(-1) * scale
        crossed = (positions[:, pair[::-1]] - pixel[:, pair]).norm(dim=-1).mean(-1) * scale
        flags |= seen & (crossed + 1.0 < direct)
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", nargs=3, action="append", required=True,
                        metavar=("LABEL", "CHECKPOINT", "MODE"), help="MODE: base, graph or graphonly.")
    parser.add_argument("--reference", required=True, help="Label every track is paired against.")
    parser.add_argument("--data", type=Path, required=True, help="Test split.")
    parser.add_argument("--train-data", type=Path, required=True, help="Real data the reward heads are fitted on.")
    parser.add_argument("--validation-data", type=Path, required=True, help="Early stopping of the reward heads.")
    parser.add_argument("--observer-checkpoint", type=Path, required=True,
                        help="Fixed reference observer that reads the visual tracks' frames.")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--context-frames", type=int, default=3)
    parser.add_argument("--history-window", type=int, default=3,
                        help="DINO-WM only: predictor window (3 = official num_hist; 0 = full history).")
    parser.add_argument("--windows-per-episode", type=int, default=0, help="0 = every eligible window.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-window", type=int, default=32, help="Real-window length for fitting heads.")
    parser.add_argument("--head-epochs", type=int, default=60)
    parser.add_argument("--head-patience", type=int, default=8)
    parser.add_argument("--head-batch", type=int, default=256)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = [label for label, _, _ in args.model]
    if len(set(labels)) != len(labels) or args.reference not in labels:
        parser.error("labels must be unique and --reference one of them")
    if any(mode not in {"base", "graph", "graphonly"} for _, _, mode in args.model):
        parser.error("MODE must be base, graph or graphonly")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is not visible in this terminal")
    device = torch.device(args.device)
    start = args.context_frames

    loader = build_loader(args.data, args.horizon, start, args.windows_per_episode, args.batch_size)
    with np.load(sorted(args.data.glob("*.npz"))[0], allow_pickle=False) as first_episode:
        names = [str(name) for name in first_episode["graph_node_name"]]
    observer = load_reference_observer(args.observer_checkpoint, device)
    ssim_window = gaussian_window(device)

    models: dict[str, JointGraphVideoWorldModel] = {}
    heads: dict[tuple[str, str], nn.Module] = {}
    head_quality: dict[str, dict[str, float]] = {}

    def load(path: str) -> JointGraphVideoWorldModel:
        if path not in models:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = JointGraphVideoWorldModel(**checkpoint["model_config"]).to(device).eval()
            model.load_state_dict(checkpoint["model"])
            if model.backbone_name == "dino_wm" and args.history_window:
                limit_history(model, args.history_window)
            if tuple(model.graph_dynamics_plugin.builder.role_names) != tuple(names):
                raise SystemExit(f"{path}: role order {model.graph_dynamics_plugin.builder.role_names} "
                                 f"differs from the data's {names}")
            print(f"{path}: epoch {checkpoint.get('epoch')}, backbone {model.backbone_name}", flush=True)
            models[path] = model
        return models[path]

    def head_for(path: str, kind: str) -> nn.Module:
        """A reward head fitted on real data of this checkpoint's modality."""
        if (path, kind) in heads:
            return heads[(path, kind)]
        model = load(path)
        name = f"{kind} head of {Path(path).parent.name}/{Path(path).name}"
        print(f"\nfitting the {name} on real data", flush=True)
        train = collect_real(model, args.train_data, kind, args, device, f"{kind} train")
        validation = collect_real(model, args.validation_data, kind, args, device, f"{kind} validation")
        if kind == "visual":
            token_count, token_dim = train.states.shape[-2:]
            grid = int(round(token_count ** 0.5))
            head = TrackRewardHead(token_dim, grid)
        else:
            head = GraphRewardHead(train.states.shape[-1], train.states.shape[-2])
        head = fit_head(head, train, validation, args, device, kind)
        head_quality[name] = evaluate_head(head, validation, args, device)
        print(f"{name}: validation R^2 on real data {head_quality[name]['r2']:.3f}", flush=True)
        heads[(path, kind)] = head
        del train, validation
        return head

    for label, path, mode in args.model:
        model = load(path)
        if mode == "graphonly":
            if start < model.graph_dynamics_plugin.graph_history - 1:
                parser.error("graphonly needs --context-frames >= graph_history - 1")
            head_for(path, "graph")
        elif model.dreamerv3_core is None:
            head_for(path, "visual")

    per_window: dict[str, dict[str, list[np.ndarray]]] = {}

    def add(label: str, name: str, value: Tensor) -> None:
        per_window.setdefault(label, {}).setdefault(name, []).append(value.float().cpu().numpy())

    swap_flags: dict[str, list[np.ndarray]] = {}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="test windows", dynamic_ncols=True):
            data = prepare(batch, device, args.horizon, start)
            rewards = batch[2]["reward"].to(device)
            true_reward = rewards[:, start : start + args.horizon]
            floor = score(data["target"], data, observer, ssim_window, None)
            add(FLOOR_LABEL, "joint_px", floor["joint_px"])
            add("reward copy-last", "reward_mse",
                (rewards[:, start - 1 : start] - true_reward).square())
            for label, path, mode in args.model:
                model = load(path)
                if mode in {"base", "graph"}:
                    model.set_graph_conditioning_strength(1.0)
                    rollout = model.rollout(
                        data["first"], data["actions"], use_graph=mode == "graph", deterministic=True,
                        decode=True, context_images=data["images"][:, :start],
                        context_actions=data["context_actions"], visual_context=True,
                    )
                    scores = score(rollout["images"], data, observer, ssim_window, None)
                    for name in ("joint_px", "image_mse", "motion_mse", "ssim"):
                        add(label, name, scores[name])
                    if model.dreamerv3_core is not None:
                        predicted_reward = rollout["rewards"]
                    else:
                        head = head_for(path, "visual")
                        current = rollout["tokens"]
                        previous = torch.cat((model.encode_visual_tokens(data["first"])[:, None],
                                              current[:, :-1]), dim=1)
                        predicted_reward = head(current, previous)
                    add(label, "reward_mse", (predicted_reward - true_reward).square())
                    continue
                # graphonly: observer -> GraphDynamics on its own predictions -> positions.
                plugin = model.graph_dynamics_plugin
                history = plugin.graph_history
                active = active_indices(model)
                seed = data["images"][:, start - history + 1 : start + 1]
                observed = observe_graphs(model, seed)
                window = observed["nodes"][:, -history:]
                relations = observed["relation_logits"][:, -1]
                graphs = []
                for step in range(args.horizon):
                    next_graph, relations = plugin.predict_next_graph(window, data["actions"][:, step], relations)
                    graphs.append(next_graph)
                    window = torch.cat((window[:, 1:], next_graph[:, None]), dim=1)
                graphs = torch.stack(graphs, dim=1)
                positions = plugin.node_position(graphs)
                add(label, "joint_px", _joint_error_from_positions(positions, data["pixel"], data["visible"], 64))
                head = head_for(path, "graph")
                start_graph = observed["nodes"][:, -1]
                previous = torch.cat((start_graph[:, None], graphs[:, :-1]), dim=1)
                add(label, "reward_mse",
                    (head(graphs[:, :, active], previous[:, :, active]) - true_reward).square())
                # Baselines from the same observed graph.
                last = plugin.node_position(observed["nodes"][:, -1])
                before = plugin.node_position(observed["nodes"][:, -2])
                steps = torch.arange(1, args.horizon + 1, device=device, dtype=last.dtype)
                copy_last = last[:, None].expand_as(positions)
                velocity = last[:, None] + (last - before)[:, None] * steps[None, :, None, None]
                add(f"{label}: copy-last", "joint_px",
                    _joint_error_from_positions(copy_last, data["pixel"], data["visible"], 64))
                add(f"{label}: constant-velocity", "joint_px",
                    _joint_error_from_positions(velocity.clamp(-1, 1), data["pixel"], data["visible"], 64))
                # Floors: the same observer and head on the real future frames.
                real = observe_graphs(model, torch.cat((seed, data["target"]), dim=1))["nodes"]
                real_future, real_previous = real[:, history:], real[:, history - 1 : -1]
                add(f"{label} observer on real frames", "joint_px",
                    _joint_error_from_positions(plugin.node_position(real_future), data["pixel"], data["visible"], 64))
                add(f"{label} head on real graphs", "reward_mse",
                    (head(real_future[:, :, active], real_previous[:, :, active]) - true_reward).square())
                if any(name.startswith("left_") for name in names):
                    swap_flags.setdefault(label, []).append(
                        swapped_at_start(names, plugin.node_position(start_graph), data["first_pixel"],
                                         data["first_visible"]).cpu().numpy())
            for path, model in models.items():
                if model.dreamerv3_core is None and (path, "visual") in heads:
                    head = heads[(path, "visual")]
                    current = model.encode_visual_tokens(data["target"].flatten(0, 1))
                    current = current.reshape(*data["target"].shape[:2], *current.shape[1:])
                    previous = torch.cat((model.encode_visual_tokens(data["first"])[:, None], current[:, :-1]), 1)
                    add(f"visual head of {Path(path).name} on real frames", "reward_mse",
                        (head(current, previous) - true_reward).square())

    arrays = {label: {name: np.concatenate(chunks) for name, chunks in values.items()}
              for label, values in per_window.items()}
    prefixes = sorted({min(8, args.horizon), min(16, args.horizon), args.horizon})
    summary, paired = {}, {}
    for label, values in arrays.items():
        summary[label] = {}
        for name, array in values.items():
            for prefix in prefixes:
                mean, half = interval(np.nanmean(array[:, :prefix], axis=1))
                summary[label][f"{name}_h{prefix}"] = {"mean": mean, "ci95": half}
    reference = arrays[args.reference]
    for label, values in arrays.items():
        if label == args.reference:
            continue
        for name, array in values.items():
            if name not in reference or array.shape[0] != reference[name].shape[0]:
                continue
            for prefix in prefixes:
                ours = np.nanmean(array[:, :prefix], axis=1)
                base = np.nanmean(reference[name][:, :prefix], axis=1)
                valid = np.isfinite(ours) & np.isfinite(base)
                better = ours > base if name in HIGHER_IS_BETTER else ours < base
                base_mean = float(np.nanmean(base))
                diff, half = interval(ours - base)
                paired.setdefault(label, {})[f"{name}_h{prefix}"] = {
                    "relative_pct": 100.0 * diff / base_mean if base_mean else float("nan"),
                    "relative_ci95_pct": 100.0 * half / abs(base_mean) if base_mean else float("nan"),
                    "win_rate": float((valid & better).sum() / max(valid.sum(), 1)),
                    "windows": int(valid.sum()),
                }
    swap_summary = {}
    for label, chunks in swap_flags.items():
        flags = np.concatenate(chunks)
        joint = np.nanmean(arrays[label]["joint_px"], axis=1)
        swap_summary[label] = {
            "swapped_windows": int(flags.sum()), "windows": int(flags.size),
            "joint_px_swapped": float(np.nanmean(joint[flags])) if flags.any() else float("nan"),
            "joint_px_not_swapped": float(np.nanmean(joint[~flags])) if (~flags).any() else float("nan"),
        }
        for other in labels:
            if other != label and "joint_px" in arrays[other]:
                other_joint = np.nanmean(arrays[other]["joint_px"], axis=1)
                swap_summary[label][f"{other}_joint_px_swapped"] = (
                    float(np.nanmean(other_joint[flags])) if flags.any() else float("nan"))
                swap_summary[label][f"{other}_joint_px_not_swapped"] = (
                    float(np.nanmean(other_joint[~flags])) if (~flags).any() else float("nan"))

    args.output.mkdir(parents=True, exist_ok=True)
    curves = {label: {name: np.nanmean(array, axis=0).tolist() for name, array in values.items()}
              for label, values in arrays.items()}
    (args.output / "metrics.json").write_text(json.dumps({
        "models": [{"label": label, "checkpoint": path, "mode": mode} for label, path, mode in args.model],
        "windows": int(reference["joint_px"].shape[0]), "horizon": args.horizon,
        "context_frames": start, "visual_context": True, "history_window": args.history_window,
        "observer_checkpoint": str(args.observer_checkpoint), "reference": args.reference,
        "reward_head_quality_on_real_validation": head_quality, "summary": summary,
        "paired_vs_reference": paired, "start_swap_split": swap_summary, "curves": curves,
    }, indent=2), encoding="utf-8")

    print(f"\nwindows={reference['joint_px'].shape[0]} horizon={args.horizon} context={start}+1 real frames "
          f"for every track; DINO-WM history window {args.history_window or 'full'}")
    print("\nreward heads, validation on REAL data (what each modality can predict at all):")
    for name, quality in head_quality.items():
        print(f"  {name:60s} MSE {quality['mse']:.4f}  R^2 {quality['r2']:.3f}")
    for name in ("joint_px", "reward_mse", "image_mse", "motion_mse", "ssim"):
        rows = [label for label in arrays if name in arrays[label]]
        if not rows:
            continue
        better = "higher" if name in HIGHER_IS_BETTER else "lower"
        print(f"\n{name} ({better} is better), mean ± 95% CI over windows")
        for label in rows:
            cells = "  ".join(f"h{p}={summary[label][f'{name}_h{p}']['mean']:.5g}"
                              f"±{summary[label][f'{name}_h{p}']['ci95']:.2g}" for p in prefixes)
            print(f"  {label:44s} {cells}")
        for label in rows:
            entry = paired.get(label, {})
            compared = label in labels or ": " in label or label == "reward copy-last"
            if not compared or f"{name}_h{prefixes[-1]}" not in entry:
                continue
            cells = "  ".join(f"h{p}={entry[f'{name}_h{p}']['relative_pct']:+.1f}%"
                              f"±{entry[f'{name}_h{p}']['relative_ci95_pct']:.1f}, "
                              f"win={100 * entry[f'{name}_h{p}']['win_rate']:.0f}%" for p in prefixes)
            print(f"  {label + ' vs ' + args.reference:44s} {cells}")
    for label, entry in swap_summary.items():
        print(f"\n{label}: {entry['swapped_windows']}/{entry['windows']} windows start from an observed graph "
              f"with left/right exchanged; mean joint_px over the horizon, swapped / not swapped:")
        print(f"  {label:20s} {entry['joint_px_swapped']:.3f} / {entry['joint_px_not_swapped']:.3f}")
        for other in labels:
            if f"{other}_joint_px_swapped" in entry:
                print(f"  {other:20s} {entry[f'{other}_joint_px_swapped']:.3f} / "
                      f"{entry[f'{other}_joint_px_not_swapped']:.3f}")
    print(f"\nsaved {args.output / 'metrics.json'}")


if __name__ == "__main__":
    main()
