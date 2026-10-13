from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def describe(path: Path, episode: np.lib.npyio.NpzFile) -> None:
    print(f"file: {path}")
    for key in episode.files:
        value = episode[key]
        print(f"  {key}: shape={value.shape} dtype={value.dtype}")


def image_tracks(episode: np.lib.npyio.NpzFile) -> list[tuple[str, np.ndarray]]:
    if "frames" in episode.files and "actor_rgb" in episode.files:
        # visualize_dino_decoder.py: encode/decode reconstruction.
        candidates = (
            ("Ground truth", "frames"),
            ("Decoder reconstruction", "actor_rgb"),
        )
    else:
        candidates = (
            ("Ground truth", "ground_truth"),
            ("Graph corrected", "graph_prediction"),
            ("Graph error x4", "graph_prediction_error"),
            ("Backbone (graph off)", "base_prediction"),
            ("Backbone error x4", "base_prediction_error"),
            ("Correction diff x4", "correction_diff"),
            ("Compare model", "compare_prediction"),
            ("Compare error x4", "compare_prediction_error"),
            ("Collected RGB", "image"),
            ("Online actor RGB", "rgb"),
        )
    tracks = []
    for label, key in candidates:
        if key not in episode.files:
            continue
        frames = episode[key]
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"{key} must have shape [T,H,W,3], got {frames.shape}")
        if frames.dtype != np.uint8:
            frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        tracks.append((label, frames))
    if not tracks:
        raise ValueError("No displayable RGB sequence was found")
    length = min(len(frames) for _, frames in tracks)
    return [(label, frames[:length]) for label, frames in tracks]


def play_rgb(
    path: Path,
    episode: np.lib.npyio.NpzFile,
    fps: float,
    scale: int,
    loop: bool,
    start_frame: int,
    frame_limit: int,
) -> None:
    try:
        import tkinter as tk
    except ImportError as error:
        raise RuntimeError("Tk is required for RGB playback") from error

    tracks = image_tracks(episode)
    available = len(tracks[0][1])
    if start_frame >= available:
        raise ValueError(
            f"--start-frame {start_frame} is outside the {available}-frame episode"
        )
    stop_frame = available if frame_limit == 0 else min(
        available, start_frame + frame_limit
    )
    tracks = [(label, frames[start_frame:stop_frame]) for label, frames in tracks]
    frame_count = len(tracks[0][1])
    height, width = tracks[0][1].shape[1:3]
    root = tk.Tk()
    root.title(f"JointGV episode viewer — {path.name}")
    canvas = tk.Canvas(
        root,
        width=len(tracks) * width * scale,
        height=height * scale + 52,
        background="#15171a",
        highlightthickness=0,
    )
    canvas.pack()
    status = tk.StringVar()
    tk.Label(
        root,
        textvariable=status,
        anchor="w",
        font=("Consolas", 10),
    ).pack(fill="x")

    state = {"frame": 0, "paused": False, "photos": []}
    interval_ms = max(1, round(1000.0 / fps))
    reward_key = (
        "rewards" if "rewards" in episode.files
        else "reward" if "reward" in episode.files
        else None
    )

    def render() -> None:
        canvas.delete("all")
        photos = []
        frame_id = state["frame"]
        for track_id, (label, frames) in enumerate(tracks):
            frame = np.ascontiguousarray(frames[frame_id])
            ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + frame.tobytes()
            photo = tk.PhotoImage(data=ppm, format="PPM").zoom(scale, scale)
            photos.append(photo)
            x = track_id * width * scale
            canvas.create_text(
                x + width * scale / 2,
                18,
                text=label,
                fill="white",
                font=("Segoe UI", 11, "bold"),
            )
            canvas.create_image(x, 36, image=photo, anchor="nw")
        state["photos"] = photos
        source_frame_id = start_frame + frame_id
        transition_id = source_frame_id - 1
        reward = ""
        if reward_key is not None and 0 <= transition_id < len(episode[reward_key]):
            rewards = episode[reward_key]
            reward = (
                f"  reward={float(rewards[transition_id]):.4f}"
                f"  cumulative={float(rewards[: transition_id + 1].sum()):.2f}"
            )
            if (
                "soft_predicted_rewards" in episode.files
                and "hard_predicted_rewards" in episode.files
                and transition_id < len(episode["soft_predicted_rewards"])
                and transition_id < len(episode["hard_predicted_rewards"])
            ):
                reward += (
                    f"  predicted_soft/hard="
                    f"{float(episode['soft_predicted_rewards'][transition_id]):.4f}/"
                    f"{float(episode['hard_predicted_rewards'][transition_id]):.4f}"
                )
            elif (
                "predicted_rewards" in episode.files
                and transition_id < len(episode["predicted_rewards"])
            ):
                reward += (
                    f"  predicted={float(episode['predicted_rewards'][transition_id]):.4f}"
                )
        distance = ""
        if (
            0 <= transition_id
            and "true_distance" in episode.files
            and "soft_predicted_distance" in episode.files
            and "hard_predicted_distance" in episode.files
            and transition_id < len(episode["true_distance"])
            and transition_id < len(episode["soft_predicted_distance"])
            and transition_id < len(episode["hard_predicted_distance"])
        ):
            distance = (
                f"  distance=true/soft/hard="
                f"{float(episode['true_distance'][transition_id]):.3f}/"
                f"{float(episode['soft_predicted_distance'][transition_id]):.3f}/"
                f"{float(episode['hard_predicted_distance'][transition_id]):.3f}"
            )
        elif (
            0 <= transition_id
            and "true_distance" in episode.files
            and "predicted_distance" in episode.files
            and transition_id < len(episode["true_distance"])
            and transition_id < len(episode["predicted_distance"])
        ):
            distance = (
                f"  distance={float(episode['true_distance'][transition_id]):.3f}/"
                f"{float(episode['predicted_distance'][transition_id]):.3f}"
            )
        action = ""
        if "actions" in episode.files and 0 <= transition_id < len(episode["actions"]):
            values = np.asarray(episode["actions"][transition_id]).reshape(-1)
            action = "  action=[" + ", ".join(
                f"{float(value):+.3f}" for value in values
            ) + "]"
        status.set(
            f"window frame {frame_id + 1}/{frame_count}  "
            f"episode frame {source_frame_id + 1}/{available}"
            f"{reward}{distance}{action}  "
            "Space: pause  ←/→: step  Esc: close"
        )

    def tick() -> None:
        if not state["paused"]:
            if state["frame"] + 1 < frame_count:
                state["frame"] += 1
            elif loop:
                state["frame"] = 0
            else:
                state["paused"] = True
            render()
        root.after(interval_ms, tick)

    def toggle(_event=None) -> None:
        state["paused"] = not state["paused"]

    def step(delta: int) -> None:
        state["paused"] = True
        state["frame"] = max(0, min(frame_count - 1, state["frame"] + delta))
        render()

    root.bind("<space>", toggle)
    root.bind("<Left>", lambda _event: step(-1))
    root.bind("<Right>", lambda _event: step(1))
    root.bind("<Escape>", lambda _event: root.destroy())
    render()
    root.after(interval_ms, tick)
    root.mainloop()


def play_native(
    episode: np.lib.npyio.NpzFile,
    fps: float,
    loop: bool,
) -> None:
    required = {"domain", "task", "episode_seed", "physics_state"}
    missing = required.difference(episode.files)
    if missing:
        raise ValueError(
            "Native replay requires a newly collected episode with: "
            + ", ".join(sorted(missing))
        )
    import mujoco
    import mujoco.viewer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jointgv.envs import load_environment

    domain = str(episode["domain"].item())
    task = str(episode["task"].item())
    seed = int(episode["episode_seed"].item())
    states = episode["physics_state"]
    env = load_environment(domain, task, task_kwargs={"random": seed})
    # DMC randomizes task-level model properties (for example the Reacher
    # target geom position) during reset. Physics state only stores qpos/qvel
    # and other mjData values, so reset must happen before replaying states.
    env.reset()
    delay = 1.0 / fps
    frame_id = 0
    with mujoco.viewer.launch_passive(
        env.physics.model.ptr,
        env.physics.data.ptr,
    ) as viewer:
        if "camera" in episode.files:
            camera_id = int(episode["camera"].item())
            if 0 <= camera_id < int(env.physics.model.ncam):
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id
        while viewer.is_running():
            start = time.perf_counter()
            env.physics.set_state(states[frame_id])
            env.physics.forward()
            viewer.sync()
            frame_id += 1
            if frame_id >= len(states):
                if loop:
                    frame_id = 0
                else:
                    break
            time.sleep(max(0.0, delay - (time.perf_counter() - start)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play collected MuJoCo RGB or JointGV rollout comparisons."
    )
    parser.add_argument("episode", type=Path)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument(
        "--start-frame", type=int, default=0, help="First stored RGB frame to show."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Maximum RGB frames to show; 0 shows the rest of the episode.",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--native",
        action="store_true",
        help="Replay saved physics states in an interactive MuJoCo window.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print stored arrays without opening a window.",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.scale < 1:
        parser.error("--scale must be at least 1")
    if args.start_frame < 0 or args.frames < 0:
        parser.error("--start-frame and --frames must be non-negative")
    with np.load(args.episode, allow_pickle=False) as episode:
        if args.info:
            describe(args.episode, episode)
        elif args.native:
            play_native(episode, args.fps, args.loop)
        else:
            play_rgb(
                args.episode,
                episode,
                args.fps,
                args.scale,
                args.loop,
                args.start_frame,
                args.frames,
            )


if __name__ == "__main__":
    main()
