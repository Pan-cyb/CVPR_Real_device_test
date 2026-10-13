"""Watch a DreamerV3 model reconstruct recorded episodes, one window each.

Is the blur on moving limbs already in Dreamer's decoder, before any
imagination? Every window plays one recorded episode (collect_dmc.py .npz)
side by side: the real 64x64 frame, the decoder's reconstruction of it, and
the absolute error x4. The reconstruction is the posterior Dreamer itself
filters during the episode: the RSSM steps with the recorded actions and
corrects with every real frame (stochastic state at its mode, graph off),
then the decoder renders that state. No frame is imagined.

The line underneath shows, per frame, the image MSE, the MSE on the robot
(orange pixels of the real frame) and on moving pixels (real frame changed
since the previous one), and the robot sharpness: mean image-gradient
magnitude on the robot, reconstruction / real (1.00 = as sharp as the real
frame, lower = blurred). The same per-episode means are printed to the
terminal. Close a window to open the next. Space pauses, left/right step.

Takes online checkpoints (train_online_dmc.py, e.g. best_return.pt) and
offline train.py DreamerV3 checkpoints.

    python script/jointgv/viz/view_dreamer_reconstruction.py \\
        --checkpoint runs/walker_walk_dreamerv3_graphoff_online100_seed0/best_return.pt \\
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
    if payload.get("training_mode") == "online_actor_critic":
        config = payload["config"]
        model = JointGraphVideoWorldModel(
            action_dim=int(config["action_dim"]),
            backbone=config["backbone"],
            node_dim=int(config["node_dim"]),
            graph_nodes=int(config["graph_nodes"]),
            message_steps=int(config["message_steps"]),
            max_graph_correction=float(config["max_graph_correction"]),
            graph_correction_gate_init=float(config["graph_correction_gate_init"]),
            graph_observer=config["graph_observer"],
            graph_backbone_pretrained=False,
            reltr_checkpoint=str(config["reltr_checkpoint"]),
        )
        kind = f"online {config['domain']} {config['task']}, epoch {payload.get('epoch')}, return {payload.get('best_return', float('nan')):.0f}"
    else:
        model = JointGraphVideoWorldModel(**payload["model_config"])
        kind = f"train.py, epoch {payload.get('epoch')}"
    if model.backbone_name != "dreamerv3":
        raise SystemExit(f"{path}: backbone {model.backbone_name}, this viewer is for DreamerV3")
    model.load_state_dict(payload["model"])
    print(f"model {path} ({kind})", flush=True)
    return model.to(device).eval()


@torch.no_grad()
def reconstruct(model: JointGraphVideoWorldModel, images: np.ndarray, actions: np.ndarray,
                device: torch.device) -> np.ndarray:
    """Decoded filtered posterior of every frame; action[t] leads frame t -> t+1."""
    core = model.dreamerv3_core
    frames = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0
    _, embeds, _, _ = core._encode(frames, sample=False)
    acts = torch.from_numpy(actions).to(device).float()
    state = core._start_posterior(embeds[:1], deterministic=True)
    features = [core.rssm.get_feat(state)]
    for t in range(1, len(frames)):
        prior = core._image_step(state, acts[t - 1 : t], deterministic=True)
        state = core._observe_step(prior["deter"], embeds[t : t + 1], deterministic=True)
        features.append(core.rssm.get_feat(state))
    decoded, _ = core._decode_feature(torch.cat(features))
    return (decoded.permute(0, 2, 3, 1).clamp(0, 1) * 255.0).round().byte().cpu().numpy()


def gradient(image: np.ndarray) -> np.ndarray:
    gray = image.astype(np.float32).mean(-1) / 255.0
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(np.diff(gray, axis=1))
    gy[1:, :] = np.abs(np.diff(gray, axis=0))
    return gx + gy


def measure(real: np.ndarray, recon: np.ndarray) -> dict[str, np.ndarray]:
    a = real.astype(np.float32) / 255.0
    b = recon.astype(np.float32) / 255.0
    error = np.square(a - b).mean(-1)
    # Walker/cheetah bodies are orange (material "self"); the floor and sky are blue/grey.
    robot = (real[..., 0].astype(np.int16) - real[..., 2].astype(np.int16)) > 40
    moving = np.zeros_like(robot)
    moving[1:] = np.abs(a[1:] - a[:-1]).mean(-1) > 0.05
    real_grad, recon_grad = gradient(real), gradient(recon)

    def masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        count = mask.sum((1, 2))
        return np.where(count > 0, (values * mask).sum((1, 2)) / np.maximum(count, 1), np.nan)

    return {
        "mse": error.mean((1, 2)),
        "robot_mse": masked(error, robot),
        "moving_mse": masked(error, moving),
        "sharpness": masked(recon_grad, robot) / np.maximum(masked(real_grad, robot), 1e-6),
    }


def panel(real: np.ndarray, recon: np.ndarray, scale: int) -> np.ndarray:
    error = np.clip(np.abs(real.astype(np.int16) - recon.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    gap = np.full((real.shape[0], 1, 3), 21, dtype=np.uint8)
    row = np.concatenate((real, gap, recon, gap, error), 1)
    return np.ascontiguousarray(row.repeat(scale, 0).repeat(scale, 1))


def play(title: str, real: np.ndarray, recon: np.ndarray, metrics: dict, scale: int, fps: float) -> None:
    import tkinter as tk

    first = panel(real[0], recon[0], scale)
    height, width = first.shape[:2]
    root = tk.Tk()
    root.title(title)
    tk.Label(root, text=title, font=("Segoe UI", 12, "bold")).pack(fill="x")
    heading = tk.Frame(root)
    heading.pack(fill="x")
    for text in ("real frame", "Dreamer reconstruction", "|error| x4"):
        tk.Label(heading, text=text, width=1, font=("Segoe UI", 10)).pack(side="left", expand=True, fill="x")
    canvas = tk.Canvas(root, width=width, height=height, background="#15171a", highlightthickness=0)
    canvas.pack()
    status = tk.StringVar()
    tk.Label(root, textvariable=status, anchor="w", font=("Consolas", 10)).pack(fill="x")
    tk.Label(root, text="robot sharpness = gradient on the robot, reconstruction / real (1.00 = as sharp)   |   "
             "space pause, left/right step, close for next", anchor="w", fg="#6b7280",
             font=("Consolas", 8), wraplength=width).pack(fill="x")
    state = {"frame": 0, "paused": False, "photo": None}

    def render() -> None:
        index = state["frame"]
        frame = panel(real[index], recon[index], scale)
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + frame.tobytes()
        state["photo"] = tk.PhotoImage(data=ppm, format="PPM")
        canvas.delete("all")
        canvas.create_image(0, 0, image=state["photo"], anchor="nw")
        status.set(
            f"frame {index:3d}/{len(real) - 1}  MSE {metrics['mse'][index]:.4f}  "
            f"robot {metrics['robot_mse'][index]:.4f}  moving {metrics['moving_mse'][index]:.4f}  "
            f"sharpness {metrics['sharpness'][index]:.2f}" + ("   [paused]" if state["paused"] else "")
        )

    def tick() -> None:
        if not state["paused"]:
            state["frame"] = (state["frame"] + 1) % len(real)
            render()
        root.after(max(1, round(1000.0 / fps)), tick)

    def toggle(_event=None) -> None:
        state["paused"] = not state["paused"]
        render()

    def step_by(delta: int) -> None:
        state["paused"] = True
        state["frame"] = (state["frame"] + delta) % len(real)
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
    parser.add_argument("--checkpoint", type=Path, required=True, help="DreamerV3 checkpoint (online or train.py).")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = the whole episode.")
    parser.add_argument("--scale", type=int, default=3, help="64 px frame -> 192 px per panel.")
    parser.add_argument("--fps", type=float, default=20.0, help="20 = real time for walker (0.025 s x action repeat 2).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is not visible in this terminal; open a fresh one or pass --device cpu")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    for number, path in enumerate(args.episodes, 1):
        with np.load(path, allow_pickle=False) as episode:
            end = len(episode["image"]) if args.max_frames <= 0 else args.max_frames
            real = episode["image"][:end]
            actions = episode["action"][: end - 1]
            driver = (f"online actor + noise {float(episode['policy_noise']):g}"
                      if "policy_checkpoint" in episode.files and str(episode["policy_checkpoint"]) else "random actions")
        recon = reconstruct(model, real, actions, device)
        metrics = measure(real, recon)
        title = (
            f"[{number}/{len(args.episodes)}] {path.name} ({driver}): reconstruction MSE "
            f"{np.nanmean(metrics['mse']):.4f}, robot sharpness {np.nanmean(metrics['sharpness']):.2f}"
        )
        print(
            f"{title}\n    robot MSE {np.nanmean(metrics['robot_mse']):.4f}   moving-pixel MSE "
            f"{np.nanmean(metrics['moving_mse']):.4f}   sharpness 5th-95th pct "
            f"{np.nanpercentile(metrics['sharpness'], 5):.2f}-{np.nanpercentile(metrics['sharpness'], 95):.2f}",
            flush=True,
        )
        play(title, real, recon, metrics, args.scale, args.fps)


if __name__ == "__main__":
    main()
