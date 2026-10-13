"""Watch the explicit graph observer on recorded episodes, one window each.

Every window plays one recorded episode (collect_dmc.py .npz, current graph
schema): left the MuJoCo ground-truth graph drawn on the frame, right the
graph the observer reads from the RGB frames alone (causally, frame by
frame, with its temporal memory, as in training). The title names the
episode, how it was driven (online actor + noise, or random actions) and
the mean node error; the lines underneath show the per-frame node errors
in pixels of the 64x64 frame, the worst node and the relation accuracy,
and flag frames where the observer has the left and right legs exchanged
(walker's legs look identical, so only its temporal memory tells them
apart). The same summary is printed to the terminal. Close a window to open
the next. Space pauses, left/right step. --no-window only prints the
summaries, e.g. for a whole split folder, with totals per driver.

    python script/jointgv/viz/view_graph_observer.py \\
        --checkpoint runs/walker_walk_reltr_observer_v1/best_long_rollout.pt \\
        data/dmc_walker_walk_val_v1/walker_walk_00000.npz \\
        data/dmc_walker_walk_val_v1/walker_walk_00016.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jointgv.blocks import ExplicitGraphObserver  # noqa: E402
from jointgv.dmc_protocol import EXPLICIT_GRAPH_SCHEMA_VERSION  # noqa: E402
from visualize_graph_observer import COLORS, infer, overlay_graph  # noqa: E402


def load_observer(path: Path, device: torch.device) -> ExplicitGraphObserver:
    # CPU first: the training checkpoint also holds the Adam states.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("checkpoint_version", 1)) < 3:
        raise SystemExit(f"{path}: needs a v3 observer checkpoint")
    config = dict(checkpoint["observer_config"])
    config["pretrained_backbone"] = False
    observer = ExplicitGraphObserver(**config)
    observer.load_state_dict(checkpoint["model"])
    observer.set_temporal_strength(float(checkpoint.get("temporal_strength", 1.0)))
    print(
        f"observer {path}: epoch {checkpoint.get('epoch')}, "
        f"temporal strength {float(checkpoint.get('temporal_strength', 0.0)):.2f}",
        flush=True,
    )
    return observer.to(device).eval()


def load_episode(path: Path, max_frames: int) -> dict:
    with np.load(path, allow_pickle=False) as episode:
        schema = int(episode["graph_schema_version"].item()) if "graph_schema_version" in episode.files else -1
        if schema != EXPLICIT_GRAPH_SCHEMA_VERSION:
            raise SystemExit(f"{path}: graph schema v{schema}, expected v{EXPLICIT_GRAPH_SCHEMA_VERSION}; recollect it")
        end = len(episode["image"]) if max_frames <= 0 else max_frames
        policy = str(episode["policy_checkpoint"]) if "policy_checkpoint" in episode.files else ""
        noise = float(episode["policy_noise"]) if "policy_noise" in episode.files else 0.0
        return {
            "frames": episode["image"][:end],
            "pixel": episode["graph_node_pixel"][:end],
            "visible": episode["graph_node_visible"][:end].astype(bool),
            "mask": episode["graph_node_mask"][:end].astype(bool),
            "relation": episode["graph_relation"][:end],
            "names": [str(name) for name in episode["graph_node_name"]],
            "driver": f"online actor + noise {noise:g}" if policy else "random actions",
        }


def score(episode: dict, prediction: dict) -> dict:
    """Node errors in pixels of the recorded frame, and relation accuracy."""
    size = episode["frames"].shape[1]
    distance = np.sqrt(np.square(prediction["positions_2d"] - episode["pixel"]).sum(-1)) * 0.5 * (size - 1)
    node_error = np.where(episode["visible"], distance, np.nan)
    pairs = episode["mask"][:, :, None] & episode["mask"][:, None, :]
    pairs &= ~np.eye(pairs.shape[-1], dtype=bool)
    predicted_relation = prediction["relation_logits"].argmax(-1)
    correct = (predicted_relation == episode["relation"]) & pairs
    relation = correct.sum((1, 2)) / np.maximum(pairs.sum((1, 2)), 1)
    # Left/right swap per part (thigh, leg, foot): the predicted right_* and
    # left_* nodes fit the ground truth at least 1 px better exchanged.
    names = episode["names"]
    swapped_parts = np.zeros((len(node_error), 3), dtype=bool)
    for column, part in enumerate(("thigh", "leg", "foot")):
        if f"right_{part}" not in names or f"left_{part}" not in names:
            continue
        pair = [names.index(f"right_{part}"), names.index(f"left_{part}")]
        direct = np.nanmean(node_error[:, pair], 1)
        crossed = np.sqrt(np.square(prediction["positions_2d"][:, pair[::-1]] - episode["pixel"][:, pair]).sum(-1))
        crossed = np.nanmean(np.where(episode["visible"][:, pair], crossed * 0.5 * (size - 1), np.nan), 1)
        swapped_parts[:, column] = crossed + 1.0 < direct
    return {"node_error": node_error, "relation": relation, "swapped": swapped_parts.any(1),
            "swapped_parts": swapped_parts,
            "positions_2d": prediction["positions_2d"],
            "presence": 1.0 / (1.0 + np.exp(-prediction["presence_logits"])),
            "predicted_relation": predicted_relation}


def summary_line(names: list[str], scores: dict) -> str:
    error = scores["node_error"]
    roles = [index for index, name in enumerate(names) if name != "padding" and np.isfinite(error[:, index]).any()]
    per_node = "  ".join(f"{names[index]} {np.nanmean(error[:, index]):.2f}" for index in roles)
    worst = np.nanmax(error[:, roles], axis=1)
    swapped = scores["swapped"]
    runs = np.flatnonzero(np.diff(np.concatenate(([0], swapped.astype(int), [0]))) == 1)
    return (
        f"mean node error {np.nanmean(error[:, roles]):.2f} px (95th pct of the worst node per frame "
        f"{np.percentile(worst, 95):.2f} px) | relations {100 * scores['relation'].mean():.1f}% | "
        f"left/right swapped {swapped.sum()}/{len(swapped)} frames in {len(runs)} runs "
        f"(thighs {scores['swapped_parts'][:, 0].sum()}, legs {scores['swapped_parts'][:, 1].sum()}, "
        f"feet {scores['swapped_parts'][:, 2].sum()})"
        f"{' (from frame 0)' if swapped[:1].any() else ''}"
        + (f", runs start at {', '.join(str(start) for start in runs[:8])}{' ...' if len(runs) > 8 else ''}" if len(runs) else "")
        + f"\n    per node px: {per_node}"
    )


def panel(episode: dict, scores: dict, index: int, scale: int) -> np.ndarray:
    """Ground-truth graph | observer graph, both drawn on the frame."""
    frame = episode["frames"][index]
    truth = overlay_graph(frame, episode["pixel"][index], episode["visible"][index].astype(float),
                          episode["relation"][index], "MuJoCo graph (ground truth)", scale)
    observed = overlay_graph(frame, scores["positions_2d"][index], scores["presence"][index],
                             scores["predicted_relation"][index], "Observer graph (from RGB)", scale)
    gap = np.full((truth.height, 4, 3), 21, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate((np.asarray(truth), gap, np.asarray(observed)), 1))


def play(title: str, episode: dict, scores: dict, scale: int, fps: float) -> None:
    import tkinter as tk

    names, frames = episode["names"], episode["frames"]
    roles = [index for index, name in enumerate(names) if name != "padding"]
    cache: dict[int, np.ndarray] = {}

    def compose(index: int) -> np.ndarray:
        if index not in cache:
            cache[index] = panel(episode, scores, index, scale)
        return cache[index]

    first = compose(0)
    height, width = first.shape[:2]
    root = tk.Tk()
    root.title(title)
    tk.Label(root, text=title, font=("Segoe UI", 12, "bold")).pack(fill="x")
    legend = tk.Frame(root, background="#0f172a")
    legend.pack(fill="x")
    for index in roles:
        tk.Label(legend, text=f"{index}={names[index]}", fg=COLORS[index % len(COLORS)], bg="#0f172a",
                 font=("Consolas", 9, "bold")).pack(side="left", padx=3)
    tk.Label(legend, text="edges: blue kinematic, yellow proximity, pink contact", fg="#94a3b8", bg="#0f172a",
             font=("Consolas", 9)).pack(side="right", padx=3)
    canvas = tk.Canvas(root, width=width, height=height, background="#15171a", highlightthickness=0)
    canvas.pack()
    status, nodes = tk.StringVar(), tk.StringVar()
    tk.Label(root, textvariable=status, anchor="w", font=("Consolas", 10)).pack(fill="x")
    tk.Label(root, textvariable=nodes, anchor="w", justify="left", font=("Consolas", 9),
             wraplength=width).pack(fill="x")
    tk.Label(root, text="space pause, left/right step, close for next", anchor="w", fg="#6b7280",
             font=("Consolas", 8)).pack(fill="x")
    state = {"frame": 0, "paused": False, "photo": None}

    def render() -> None:
        index = state["frame"]
        frame = compose(index)
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + frame.tobytes()
        state["photo"] = tk.PhotoImage(data=ppm, format="PPM")
        canvas.delete("all")
        canvas.create_image(0, 0, image=state["photo"], anchor="nw")
        error = scores["node_error"][index]
        shown = [role for role in roles if np.isfinite(error[role])]
        worst = max(shown, key=lambda role: error[role]) if shown else None
        status.set(
            f"frame {index:3d}/{len(frames) - 1}  mean {np.nanmean(error[shown]) if shown else float('nan'):.2f} px  "
            + (f"worst {names[worst]} {error[worst]:.2f} px  " if worst is not None else "")
            + f"relations {100 * scores['relation'][index]:.0f}%"
            + ("   LEFT/RIGHT SWAPPED: " + ", ".join(
                part for part, hit in zip(("thighs", "legs", "feet"), scores["swapped_parts"][index]) if hit)
               if scores["swapped"][index] else "")
            + ("   [paused]" if state["paused"] else "")
        )
        nodes.set("  ".join(f"{role}:{error[role]:.2f}" if np.isfinite(error[role]) else f"{role}:hidden"
                            for role in roles) + "   (node error, px of the 64x64 frame)")

    def tick() -> None:
        if not state["paused"]:
            state["frame"] = (state["frame"] + 1) % len(frames)
            render()
        root.after(max(1, round(1000.0 / fps)), tick)

    def toggle(_event=None) -> None:
        state["paused"] = not state["paused"]
        render()

    def step_by(delta: int) -> None:
        state["paused"] = True
        state["frame"] = (state["frame"] + delta) % len(frames)
        render()

    root.bind("<space>", toggle)
    root.bind("<Left>", lambda _e: step_by(-1))
    root.bind("<Right>", lambda _e: step_by(1))
    render()
    root.after(max(1, round(1000.0 / fps)), tick)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("episodes", type=Path, nargs="+",
                        help="Recorded .npz episodes (or folders of them), shown in this order.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Observer checkpoint (pretrain_graph_observer.py).")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = the whole episode.")
    parser.add_argument("--scale", type=int, default=4, help="64 px frame -> 256 px per panel.")
    parser.add_argument("--fps", type=float, default=20.0, help="20 = real time for walker (0.025 s x action repeat 2).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-window", action="store_true", help="Only print each episode's summary (e.g. a whole split).")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is not visible in this terminal; open a fresh one or pass --device cpu")
    device = torch.device(args.device)
    observer = load_observer(args.checkpoint, device)
    episodes = [file for path in args.episodes for file in (sorted(path.glob("*.npz")) if path.is_dir() else [path])]
    totals: dict[str, list[float]] = {}
    for number, path in enumerate(episodes, 1):
        episode = load_episode(path, args.max_frames)
        prediction = infer(observer, episode["frames"], device, args.batch_size)
        scores = score(episode, prediction)
        error = scores["node_error"]
        roles = [index for index, name in enumerate(episode["names"]) if name != "padding"]
        title = (
            f"[{number}/{len(episodes)}] {path.name} ({episode['driver']}): "
            f"mean node error {np.nanmean(error[:, roles]):.2f} px, relations {100 * scores['relation'].mean():.1f}%"
        )
        print(f"{title}\n    {summary_line(episode['names'], scores)}", flush=True)
        group = totals.setdefault(episode["driver"].split(" +")[0], [0, 0, 0, 0, 0.0])
        group[0] += 1
        group[1] += int(scores["swapped"].sum())
        group[2] += len(scores["swapped"])
        group[3] += int(scores["swapped"][:1].any())
        group[4] += float(np.nanmean(error[:, roles]))
        if not args.no_window:
            play(title, episode, scores, args.scale, args.fps)
    for driver, (count, swapped, frames, at_start, error_sum) in totals.items():
        print(f"== {driver}: {count} episodes, mean node error {error_sum / count:.2f} px, left/right swapped "
              f"{100 * swapped / frames:.1f}% of frames, {at_start}/{count} episodes swapped from frame 0")


if __name__ == "__main__":
    main()
