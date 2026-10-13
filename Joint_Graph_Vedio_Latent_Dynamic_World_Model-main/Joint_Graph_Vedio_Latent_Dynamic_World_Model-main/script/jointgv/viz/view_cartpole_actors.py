"""Watch cart-pole actors balance in the real simulator, one window each.

Every window plays one episode of one actor (train_cartpole_actor.py) in the
RM65-B MuJoCo cart-pole, from the control protocol's start (lean uniform in
+/-7 deg, at rest) on a test seed. Each seed is shown for every actor in
turn (e.g. graph, then base), so they start from the same state. The actor
acts from the 64x64 camera frame exactly as in its evaluation; the window
shows a larger render of the same camera at real-time speed. The title says
which actor, which seed and how the episode ended; the line underneath
shows time, lean, pivot offset from the rail centre and the command. Close
a window to open the next. Space pauses, left/right step.

    python script/jointgv/viz/view_cartpole_actors.py \\
        --actor graph runs/rm65b_cartpole_actor_v4ft_graph/actor_best.pt \\
        --actor base runs/rm65b_cartpole_actor_v4ft_base/actor_best.pt
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Windows" else "egl")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv import JointGraphVideoWorldModel  # noqa: E402
from jointgv.dmc_protocol import resolve_camera  # noqa: E402
from jointgv.envs import load_environment  # noqa: E402
from jointgv.envs.rm65b_cartpole import CART_CENTER, CONTROL_TIMESTEP, hinge_task_kwargs  # noqa: E402
from train_cartpole_actor import Actor, encode  # noqa: E402


def load_actors(specs: list[list[str]], device: torch.device) -> list[tuple[str, Actor, object, str]]:
    """(label, actor, world model whose encoder it reads, hinge it was
    trained for; "damper" for actors saved before hinges were recorded)."""
    models: dict[str, JointGraphVideoWorldModel] = {}
    actors = []
    for label, path in specs:
        saved = torch.load(path, map_location=device, weights_only=False)
        actor = Actor(saved["token_dim"], saved["grid"], saved["action_dim"]).to(device)
        actor.load_state_dict(saved["actor"])
        world_model = str(saved["checkpoint"])
        if world_model not in models:
            checkpoint = torch.load(world_model, map_location=device, weights_only=False)
            model = JointGraphVideoWorldModel(**checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["model"])
            models[world_model] = model.eval()
            del checkpoint
        hinge = saved.get("hinge") or "damper"
        print(f"{label}: {path} ({saved['mode']} track, update {saved.get('update')}, hinge {hinge}, WM {world_model})")
        actors.append((label, actor.eval(), models[world_model], hinge))
    return actors


@torch.no_grad()
def run_episode(actor: Actor, model, seed: int, args, device: torch.device, hinge: str = "damper") -> dict:
    """One evaluation episode; frames are the large view, one per agent step."""
    camera = resolve_camera("rm65b", -1)
    env = load_environment("rm65b", "cartpole", task_kwargs={"random": seed, **hinge_task_kwargs(hinge)})
    env.reset()
    physics, task = env.physics, env.task

    def snapshot(command: float, guard: bool) -> None:
        frames.append(np.ascontiguousarray(physics.render(args.view_size, args.view_size, camera_id=camera)))
        info.append((
            len(info) * args.action_repeat * CONTROL_TIMESTEP,
            math.degrees(physics.pole_angle()),
            100.0 * (physics.cart_position() - CART_CENTER),
            command,
            guard,
        ))

    frames: list[np.ndarray] = []
    info: list[tuple] = []
    start_lean = math.degrees(physics.pole_angle())
    current = encode(model, physics.render(args.size, args.size, camera_id=camera), device)
    previous = current
    snapshot(0.0, False)
    fell = False
    for _ in range(args.steps):
        action = actor.act(current, previous)[0].cpu().numpy()
        guards = task.unsafe_commands
        for _ in range(args.action_repeat):
            if env.step(action).last():
                fell = True
                break
        snapshot(float(action[0]), task.unsafe_commands > guards)
        if fell:
            break
        previous, current = current, encode(
            model, physics.render(args.size, args.size, camera_id=camera), device
        )
    seconds = (len(info) - 1) * args.action_repeat * CONTROL_TIMESTEP
    if fell:
        result, success = f"fell at {seconds:.2f} s", False
    elif task.unsafe_commands:
        result, success = f"upright {seconds:.1f} s, but {task.unsafe_commands} rail-guard commands", False
    else:
        result, success = f"balanced the full {seconds:.1f} s", True
    return {"frames": frames, "info": info, "result": result, "success": success,
            "start_lean": start_lean}


def play(title: str, header: str, episode: dict, fps: float) -> None:
    import tkinter as tk

    frames, info = episode["frames"], episode["info"]
    height, width = frames[0].shape[:2]
    root = tk.Tk()
    root.title(title)
    tk.Label(root, text=title, font=("Segoe UI", 12, "bold"),
             fg="#1b7f3b" if episode["success"] else "#b3261e").pack(fill="x")
    tk.Label(root, text=header, font=("Consolas", 9)).pack(fill="x")
    canvas = tk.Canvas(root, width=width, height=height, background="#15171a", highlightthickness=0)
    canvas.pack()
    status = tk.StringVar()
    tk.Label(root, textvariable=status, anchor="w", font=("Consolas", 10)).pack(fill="x")
    state = {"frame": 0, "paused": False, "photo": None}

    def render() -> None:
        index = state["frame"]
        frame = frames[index]
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + frame.tobytes()
        state["photo"] = tk.PhotoImage(data=ppm, format="PPM")
        canvas.delete("all")
        canvas.create_image(0, 0, image=state["photo"], anchor="nw")
        seconds, lean, offset, command, guard = info[index]
        status.set(
            f"t {seconds:4.2f} s  lean {lean:+6.1f} deg  pivot {offset:+5.1f} cm  "
            f"command {command:+.2f}" + ("  RAIL GUARD" if guard else "")
            + ("   [paused]" if state["paused"] else "")
        )

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
    parser.add_argument(
        "--actor", nargs=2, action="append", required=True, metavar=("LABEL", "ACTOR_PT"),
        help="Repeatable; each seed is shown for every actor in this order.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of seeds.")
    parser.add_argument("--seed", type=int, default=42000, help="First seed (42000 = the actor test seeds).")
    parser.add_argument("--steps", type=int, default=150, help="150 x 0.04 s = 6 s, as in the evaluation.")
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--size", type=int, default=64, help="Frame size the actor sees.")
    parser.add_argument("--view-size", type=int, default=320, help="Size of the frame shown.")
    parser.add_argument("--fps", type=float, default=25.0, help="25 = real time.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is not visible in this terminal; open a fresh one or pass --device cpu")
    device = torch.device(args.device)
    actors = load_actors(args.actor, device)
    total = args.episodes * len(actors)
    index = 0
    for seed in range(args.seed, args.seed + args.episodes):
        for label, actor, model, hinge in actors:
            index += 1
            episode = run_episode(actor, model, seed, args, device, hinge)
            title = f"[{index}/{total}] {label} actor ({hinge} hinge), seed {seed}: {episode['result']}"
            header = (
                f"start: lean {episode['start_lean']:+.1f} deg at rest   |   success = 6 s upright "
                "without a rail-guard command   |   space pause, left/right step, close for next"
            )
            print(title, flush=True)
            play(title, header, episode, args.fps)


if __name__ == "__main__":
    main()
