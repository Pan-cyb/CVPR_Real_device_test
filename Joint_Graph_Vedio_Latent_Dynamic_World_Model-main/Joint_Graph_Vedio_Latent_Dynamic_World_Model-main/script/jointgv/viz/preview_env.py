"""Preview an environment under smooth random exploration.

Renders the stored 64x64 frames (what the world model sees) next to a
high-resolution view of the same camera, and prints motion statistics.

    python script/jointgv/viz/preview_env.py --domain rm65b --task pendulum \
        --steps 250 --output runs/env_preview/rm65b
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Windows" else "egl")

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv.dmc_protocol import resolve_camera  # noqa: E402
from jointgv.envs import load_environment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="rm65b")
    parser.add_argument("--task", default="pendulum")
    parser.add_argument("--steps", type=int, default=250, help="stored steps")
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument(
        "--random-hold-steps",
        type=float,
        default=8.0,
        help="Mean number of stored steps a random action is held.",
    )
    parser.add_argument(
        "--uniform-random-actions",
        action="store_true",
        help="Ignore the task's own exploration policy, if it has one.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--preview-size", type=int, default=320)
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--output", type=Path, default=Path("runs/env_preview/rm65b"))
    args = parser.parse_args()

    env = load_environment(args.domain, args.task, task_kwargs={"random": args.seed})
    rng = np.random.default_rng(args.seed)
    camera = resolve_camera(args.domain, -1)
    spec = env.action_spec()
    env.reset()
    physics = env.physics
    frames: list[Image.Image] = []
    pendulum, bob, rewards, tracking = [], [], [], []
    action = rng.uniform(spec.minimum, spec.maximum)

    def snapshot() -> None:
        small = physics.render(args.size, args.size, camera_id=camera)
        large = physics.render(args.preview_size, args.preview_size, camera_id=camera)
        upscaled = Image.fromarray(small).resize(
            (args.preview_size, args.preview_size), Image.NEAREST
        )
        canvas = Image.new("RGB", (2 * args.preview_size, args.preview_size + 18))
        canvas.paste(upscaled, (0, 18))
        canvas.paste(Image.fromarray(large), (args.preview_size, 18))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 2), f"{args.size}x{args.size} (training input)", fill=(255, 255, 255))
        draw.text((args.preview_size + 4, 2), f"t={len(frames)}", fill=(255, 255, 255))
        frames.append(canvas)
        if hasattr(physics, "pendulum_angle"):
            pendulum.append(physics.pendulum_angle())
            bob.append(physics.bob_position())
            tracking.append(
                np.abs(physics.named.data.ctrl[:] - physics.arm_angles()).max()
            )

    explore = (
        None if args.uniform_random_actions
        else getattr(env.task, "exploration_action", None)
    )
    snapshot()
    for _ in range(args.steps):
        if explore is not None:
            action = explore(physics, rng, args.action_repeat)
        elif rng.random() < 1.0 / max(1.0, args.random_hold_steps):
            action = rng.uniform(spec.minimum, spec.maximum)
        for _ in range(args.action_repeat):
            timestep = env.step(action)
            rewards.append(float(timestep.reward or 0.0))
            if timestep.last():
                break
        snapshot()
        if timestep.last():
            break

    args.output.mkdir(parents=True, exist_ok=True)
    gif = args.output / "preview.gif"
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000.0 / args.fps)),
        loop=0,
    )
    picks = np.linspace(0, len(frames) - 1, 8).round().astype(int)
    strip = Image.new("RGB", (frames[0].width * 2, frames[0].height * 4))
    for slot, index in enumerate(picks):
        strip.paste(frames[index], ((slot % 2) * frames[0].width, (slot // 2) * frames[0].height))
    strip.save(args.output / "preview_strip.png")
    print(f"saved {gif} ({len(frames)} frames) and preview_strip.png")
    if pendulum:
        angles = np.degrees(np.asarray(pendulum))
        positions = np.asarray(bob)
        passive_name = "pole" if args.task == "cartpole" else "pendulum"
        print(
            f"{passive_name} angle from task vertical (deg): "
            f"min={angles.min():.1f} max={angles.max():.1f} std={angles.std():.1f}"
        )
        print(
            f"{passive_name} tip x range (m): "
            f"{positions[:, 0].min():.2f}..{positions[:, 0].max():.2f}, "
            f"z range: {positions[:, 2].min():.2f}..{positions[:, 2].max():.2f}"
        )
        print(
            "servo tracking error (deg): "
            f"mean={np.degrees(np.mean(tracking)):.2f} "
            f"max={np.degrees(np.max(tracking)):.2f}"
        )
        task = env.task
        print(
            "unsafe commands held:",
            getattr(task, "unsafe_commands", "n/a"),
            f"of {len(rewards)} control steps; mean reward {np.mean(rewards):.3f}",
        )


if __name__ == "__main__":
    main()
