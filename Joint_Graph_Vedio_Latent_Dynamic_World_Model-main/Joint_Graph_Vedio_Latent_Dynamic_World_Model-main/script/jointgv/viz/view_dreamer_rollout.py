"""Watch a DreamerV3 plug-in (P) model imagine, backbone vs graph, one window each.

Every window is one start frame of a recorded episode (collect_dmc.py .npz):
the real future, the backbone's imagination (graph correction strength 0)
and the graph-corrected imagination (strength 1), side by side, from the
same real frames and driven by the same recorded actions. After the start
no real frame is seen again: every shown step is the model's own open-loop
prediction, step 1 to --horizon.

Both tracks see the same real history: the RSSM filters the --context-frames
frames before the start and the start frame through its posterior (the
evaluation protocol's visual context), and the same frames seed the graph
history. --no-visual-context reproduces train.py's rollout validation
instead, where the RSSM starts from the start frame alone and only the graph
sees the earlier frames.

The line underneath shows the step and, per track, the image MSE and the MSE
on the robot (orange pixels of the real frame) at that step. The terminal
prints both at steps 8, 16 and 32 for every window. Close a window to open
the next. Space pauses, left/right step.

    python script/jointgv/viz/view_dreamer_rollout.py \\
        --checkpoint runs/walker_walk_dreamerv3_P_plugin/latest.pt \\
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
from jointgv import JointGraphVideoWorldModel  # noqa: E402


def load_model(path: Path, device: torch.device) -> JointGraphVideoWorldModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" not in payload:
        raise SystemExit(f"{path}: not a train.py checkpoint")
    model = JointGraphVideoWorldModel(**payload["model_config"])
    if model.backbone_name != "dreamerv3" or not model.dreamerv3_core.global_correction:
        raise SystemExit(f"{path}: needs a DreamerV3 plug-in (--plugin-correction) checkpoint")
    model.load_state_dict(payload["model"])
    print(f"model {path} (epoch {payload.get('epoch')}, correction scale {payload.get('correction_scale')})", flush=True)
    return model.to(device).eval()


@torch.no_grad()
def imagine(model, images: np.ndarray, actions: np.ndarray, starts: list[int], context: int,
            horizon: int, visual_context: bool, device: torch.device) -> dict[str, np.ndarray]:
    """Decoded open-loop imagination per track, uint8 [start, step, H, W, 3]."""
    frames = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0
    acts = torch.from_numpy(actions).to(device).float()
    initial = torch.stack([frames[s] for s in starts])
    history = torch.stack([frames[s - context : s] for s in starts]) if context else None
    history_actions = torch.stack([acts[s - context : s] for s in starts]) if context else None
    future_actions = torch.stack([acts[s : s + horizon] for s in starts])
    result = {}
    for track, strength in (("backbone", 0.0), ("graph", 1.0)):
        model.set_graph_conditioning_strength(strength)
        prediction = model.dreamerv3_core.rollout(
            initial, future_actions, use_graph=True, decode=True, deterministic=True,
            context_images=history, context_actions=history_actions, visual_context=visual_context,
        )
        result[track] = (prediction["images"].permute(0, 1, 3, 4, 2).clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    return result


def errors(real: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-step image MSE and MSE on the robot (orange pixels of the real frame)."""
    a = real.astype(np.float32) / 255.0
    b = predicted.astype(np.float32) / 255.0
    error = np.square(a - b).mean(-1)
    robot = (real[..., 0].astype(np.int16) - real[..., 2].astype(np.int16)) > 40
    robot_error = (error * robot).sum((1, 2)) / np.maximum(robot.sum((1, 2)), 1)
    return error.mean((1, 2)), robot_error


def panel(frames: list[np.ndarray], scale: int) -> np.ndarray:
    gap = np.full((frames[0].shape[0], 1, 3), 21, dtype=np.uint8)
    row = [frames[0]]
    for frame in frames[1:]:
        row += [gap, frame]
    return np.ascontiguousarray(np.concatenate(row, 1).repeat(scale, 0).repeat(scale, 1))


def play(title: str, real: np.ndarray, tracks: dict[str, np.ndarray], scores: dict, scale: int, fps: float) -> None:
    import tkinter as tk

    steps = len(real)
    first = panel([real[0], tracks["backbone"][0], tracks["graph"][0]], scale)
    height, width = first.shape[:2]
    root = tk.Tk()
    root.title(title)
    tk.Label(root, text=title, font=("Segoe UI", 12, "bold"), wraplength=width).pack(fill="x")
    heading = tk.Frame(root)
    heading.pack(fill="x")
    for text in ("real future", "backbone imagination", "graph imagination"):
        tk.Label(heading, text=text, width=1, font=("Segoe UI", 10)).pack(side="left", expand=True, fill="x")
    canvas = tk.Canvas(root, width=width, height=height, background="#15171a", highlightthickness=0)
    canvas.pack()
    status = tk.StringVar()
    tk.Label(root, textvariable=status, anchor="w", font=("Consolas", 10)).pack(fill="x")
    tk.Label(root, text="every step after the start is imagined open-loop with the recorded actions   |   "
             "space pause, left/right step, close for next", anchor="w", fg="#6b7280",
             font=("Consolas", 8), wraplength=width).pack(fill="x")
    state = {"step": 0, "paused": False, "photo": None}

    def render() -> None:
        index = state["step"]
        frame = panel([real[index], tracks["backbone"][index], tracks["graph"][index]], scale)
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + frame.tobytes()
        state["photo"] = tk.PhotoImage(data=ppm, format="PPM")
        canvas.delete("all")
        canvas.create_image(0, 0, image=state["photo"], anchor="nw")
        b_mse, b_robot = scores["backbone"]
        g_mse, g_robot = scores["graph"]
        status.set(
            f"step {index + 1:2d}/{steps}  backbone MSE {b_mse[index]:.4f} robot {b_robot[index]:.4f}   "
            f"graph MSE {g_mse[index]:.4f} robot {g_robot[index]:.4f}" + ("   [paused]" if state["paused"] else "")
        )

    def tick() -> None:
        if not state["paused"]:
            state["step"] = (state["step"] + 1) % steps
            render()
        root.after(max(1, round(1000.0 / fps)), tick)

    def toggle(_event=None) -> None:
        state["paused"] = not state["paused"]
        render()

    def step_by(delta: int) -> None:
        state["paused"] = True
        state["step"] = (state["step"] + delta) % steps
        render()

    root.bind("<space>", toggle)
    root.bind("<Left>", lambda _e: step_by(-1))
    root.bind("<Right>", lambda _e: step_by(1))
    render()
    root.after(max(1, round(1000.0 / fps)), tick)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("episodes", type=Path, nargs="+", help="Recorded .npz episodes, shown in this order.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="train.py DreamerV3 --plugin-correction checkpoint.")
    parser.add_argument("--starts", type=int, nargs="+", default=[60, 220, 380], help="Start frames per episode.")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--context-frames", type=int, default=3, help="Real frames before the start (training used 3).")
    parser.add_argument("--no-visual-context", action="store_true",
                        help="train.py rollout-validation protocol: the RSSM sees only the start frame.")
    parser.add_argument("--scale", type=int, default=3, help="64 px frame -> 192 px per panel.")
    parser.add_argument("--fps", type=float, default=10.0, help="20 = real time for walker; slower to follow the blur.")
    parser.add_argument("--no-window", action="store_true", help="Only print the per-window errors.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is not visible in this terminal; open a fresh one or pass --device cpu")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    protocol = "start frame only for the RSSM" if args.no_visual_context else f"{args.context_frames}+1 real frames for both tracks"
    print(f"protocol: {protocol}, horizon {args.horizon}", flush=True)
    windows = []
    for path in args.episodes:
        with np.load(path, allow_pickle=False) as episode:
            images, actions = episode["image"], episode["action"]
            driver = (f"online actor + noise {float(episode['policy_noise']):g}"
                      if "policy_checkpoint" in episode.files and str(episode["policy_checkpoint"]) else "random actions")
        starts = [s for s in args.starts if args.context_frames <= s and s + args.horizon < len(images)]
        if not starts:
            print(f"{path.name}: no start fits context {args.context_frames} + horizon {args.horizon}", flush=True)
            continue
        tracks = imagine(model, images, actions, starts, args.context_frames, args.horizon,
                         not args.no_visual_context, device)
        for row, start in enumerate(starts):
            real = images[start + 1 : start + args.horizon + 1]
            per_track = {name: (frames[row], errors(real, frames[row])) for name, frames in tracks.items()}
            windows.append((path, driver, start, real, per_track))
    marks = [step for step in (8, 16, 32, 64) if step <= args.horizon]
    for number, (path, driver, start, real, per_track) in enumerate(windows, 1):
        title = f"[{number}/{len(windows)}] {path.name} ({driver}), start frame {start} ({protocol})"
        lines = []
        for name, (_, (mse, robot)) in per_track.items():
            lines.append(f"    {name:8s} image MSE " + "  ".join(f"h{m} {mse[m - 1]:.4f}" for m in marks)
                         + "  | robot MSE " + "  ".join(f"h{m} {robot[m - 1]:.4f}" for m in marks))
        print(title + "\n" + "\n".join(lines), flush=True)
        if not args.no_window:
            play(title, real, {name: frames for name, (frames, _) in per_track.items()},
                 {name: score for name, (_, score) in per_track.items()}, args.scale, args.fps)


if __name__ == "__main__":
    main()
