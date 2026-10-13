"""Interactive MuJoCo view of an environment, and passive-joint release test.

    # Arm holds its pose; drag the pendulum with the mouse (double-click the
    # bob, then Ctrl + right-drag) and watch it swing.
    python script/jointgv/viz/view_env.py --domain rm65b

    # Arm moves with the task's random point-to-point exploration.
    python script/jointgv/viz/view_env.py --domain rm65b --explore --episodes 1

    # RM65-B robot cart-pole: one fixed-length balance episode.
    python script/jointgv/viz/view_env.py --domain rm65b --task cartpole \
        --explore --episodes 1

    # Frozen set-up for photos: no physics stepping, pole upright (or at
    # --release DEG), 3/4 overview; ']' / '[' cycle to the task camera.
    python script/jointgv/viz/view_env.py --domain rm65b --task cartpole --still

    # Headless release test: arm fixed, pendulum released from 30 degrees;
    # prints the period and the amplitude decay per cycle. Run the same test
    # on the printed pendulum to calibrate hinge damping and friction.
    python script/jointgv/viz/view_env.py --domain rm65b --swing-test 30
"""

from __future__ import annotations

import argparse
import math
import os
import platform
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Windows" else "egl")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv.envs import load_environment  # noqa: E402


def release(env, degrees: float) -> None:
    """Hold the arm and set the passive link ``degrees`` from its vertical."""
    # Not physics.reset_context(): that first resets every joint to zero.
    physics = env.physics
    if hasattr(physics, "set_pole_angle"):
        physics.set_pole_angle(math.radians(degrees))
    else:
        arm = physics.arm_angles()
        physics.named.data.qpos["pendulum"] = -arm.sum() + math.radians(degrees)
        physics.named.data.qvel[:] = 0.0
        physics.forward()


def swing_test(env, degrees: float, seconds: float) -> None:
    env.reset()
    release(env, degrees)
    physics = env.physics
    dt = env.control_timestep()
    angles, times = [], []
    for step in range(int(seconds / dt)):
        timestep = env.step(np.zeros(env.action_spec().shape))
        angles.append(physics.pendulum_angle())
        times.append((step + 1) * dt)
        if timestep.last():
            # The episode time limit ends the test (the next step would reset).
            break
    angles = np.degrees(np.asarray(angles))
    times = np.asarray(times)
    # Positive peaks (local maxima above zero) give period and decay.
    peaks = [
        index for index in range(1, len(angles) - 1)
        if angles[index] > 0 and angles[index - 1] < angles[index] >= angles[index + 1]
    ]
    if len(peaks) < 3:
        print("not enough swings to measure")
        return
    peak_times, peak_values = times[peaks], angles[peaks]
    period = float(np.mean(np.diff(peak_times)))
    # Only swings above 2 degrees: below that, servo micro-motion of the
    # holding arm dominates the peak detection.
    valid = peak_values[:-1] > 2.0
    ratios = (peak_values[1:] / peak_values[:-1])[valid]
    print(f"release from {degrees:.1f} deg, arm fixed, {times[-1]:.0f} s simulated")
    print(f"period: {period:.3f} s ({1.0 / period:.3f} Hz)")
    print(
        "amplitude per cycle: "
        f"{(1.0 - float(np.median(ratios))) * 100.0:.2f}% decay (median)"
    )
    print(
        "peaks (deg): "
        + " ".join(f"{value:.1f}" for value in peak_values[:12])
        + (" ..." if len(peak_values) > 12 else "")
    )
    half = np.flatnonzero(peak_values < peak_values[0] / 2.0)
    if len(half):
        print(f"amplitude halves after {peak_times[half[0]]:.1f} s")
    else:
        print(f"amplitude still above half after {seconds:.0f} s")


def interactive(
    env,
    explore: bool,
    seed: int,
    degrees: float | None,
    episodes: int,
    still: bool = False,
) -> None:
    import mujoco.viewer

    env.reset()
    if degrees is not None or still:
        release(env, degrees or 0.0)
    physics = env.physics
    if still:
        # Show the set-up without stepping physics, so nothing falls while
        # the view is arranged; start from a 3/4 overview of arm and pole.
        with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = (0.12, 0.0, 0.32)
                viewer.cam.distance = 1.6
                viewer.cam.azimuth = 65.0
                viewer.cam.elevation = -18.0
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 30.0)
        return
    rng = np.random.default_rng(seed)
    explore_action = getattr(env.task, "exploration_action", None) if explore else None
    zeros = np.zeros(env.action_spec().shape)
    action = zeros
    step = 0
    completed = 0
    with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            if explore_action is not None and step % 2 == 0:
                # Frames are stored every second control step (action repeat 2).
                action = explore_action(physics, rng, 2)
            timestep = env.step(action if explore_action is not None else zeros)
            step += 1
            viewer.sync()
            if timestep.last():
                completed += 1
                if episodes and completed >= episodes:
                    break
                env.reset()
                step = 0
            remaining = env.control_timestep() - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="rm65b")
    parser.add_argument("--task", default="pendulum")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument(
        "--cartpole-speed",
        type=float,
        default=None,
        metavar="MPS",
        help=(
            "Fix the rm65b/cartpole preview controller's horizontal speed "
            "ceiling in m/s; omit it to use stratified speed sampling."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Interactive episodes before closing (0 = loop until window closes).",
    )
    parser.add_argument(
        "--release",
        type=float,
        default=None,
        help="Interactive view: start the pendulum this many degrees from vertical.",
    )
    parser.add_argument(
        "--swing-test",
        type=float,
        default=None,
        metavar="DEG",
        help="Run the headless release test from DEG degrees instead.",
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument(
        "--still",
        action="store_true",
        help="Show the set-up frozen (no physics stepping), e.g. for photos.",
    )
    args = parser.parse_args()
    if args.episodes < 0:
        parser.error("--episodes must be non-negative")
    task_kwargs = {"random": args.seed}
    if args.cartpole_speed is not None:
        if (args.domain, args.task) != ("rm65b", "cartpole"):
            parser.error("--cartpole-speed is only valid for rm65b/cartpole")
        task_kwargs["exploration_speed"] = args.cartpole_speed
    env = load_environment(args.domain, args.task, task_kwargs=task_kwargs)
    if args.swing_test is not None:
        swing_test(env, args.swing_test, args.seconds)
    else:
        interactive(env, args.explore, args.seed, args.release, args.episodes, args.still)


if __name__ == "__main__":
    main()
