"""Privileged LQR upper bound for the RM65-B robot cart-pole.

Answers one question before any data collection or training: can the robot,
with its real action limits (0.45 m/s pivot speed, 6 m/s^2 acceleration,
22 cm rail) and its position servos, recover the pole from the sampled
initial lean at all? It is an environment acceptance test, not a reportable
visual controller.

The controller is a noise-free discrete-time LQR on the complete simulator
state (arm joint angles and velocities, pole angle and rate, and the commanded
pivot target and velocity), at the same 25 Hz action cadence as the learned
controllers (``--action-repeat 2``). The model is linearised numerically
around the upright equilibrium on a dedicated environment instance, with the
hinge's Coulomb friction removed only there because it is not differentiable
at rest. Every control step reads the true state and outputs one clipped
command; the evaluated environments are unchanged.

Q penalises the pole angle and rate and the pivot's offset from the rail
centre and its velocity, each normalised by a typical scale. R and the
rail-centring scale are chosen from a small grid (``--tune-r``,
``--tune-position-scale``) on separate tuning seeds (``--tune-seed``). The
chosen setting is then evaluated on ``--episodes`` evaluation seeds.
``--seed 33000`` reuses the seeds of the earlier ``collect_dmc.py``
privileged-heuristic debug run (same initial angles at the same range).

Result on 2026-10-02 at +/-10 deg (seed 33000): 35/50. Every start up to
7.8 deg survived; every failure ran the pivot into the rail end (peak
excursion ~1.3 cm per degree of initial lean). The task range was therefore
locked at +/-7 deg (``DEFAULT_INITIAL_ANGLE_RANGE_DEG``), the default here.

The script also prints a controller-independent necessary condition. With
pivot speed limited to v_max and the unstable pole mode growing at rate
lambda, a pole released at rest can only be recovered when
|theta0| < lambda * v_max / g (small-angle, ignoring acceleration limits,
servo lag and rail length, so the true limit is lower).

Success means surviving all ``--steps`` agent steps with no workspace-guard
activation, the same definition evaluate_cartpole_planning.py uses.

    python script/jointgv/evaluation/evaluate_cartpole_lqr.py \\
        --episodes 100 --seed 35000 \\
        --output runs/eval/rm65b_cartpole_lqr_uniform7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Windows" else "egl")

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jointgv.dmc_protocol import resolve_camera  # noqa: E402
from jointgv.envs import load_environment  # noqa: E402
from jointgv.envs.rm65b_cartpole import (  # noqa: E402
    CART_CENTER,
    CART_RANGE,
    CONTROL_TIMESTEP,
    DEFAULT_INITIAL_ANGLE_RANGE_DEG,
    MAX_CART_ACCEL,
    MAX_CART_SPEED,
)
# Shared with utils/collect_dmc.py --cartpole-policy lqr. Its linear algebra
# runs in torch (numpy BLAS aborts with an OpenMP clash once torch is loaded).
from jointgv.envs.rm65b_cartpole_lqr import (  # noqa: E402
    Linearisation,
    closed_loop_radius,
    get_state,
    lqr_command,
    lqr_gain,
)

DOMAIN, TASK = "rm65b", "cartpole"
GRAVITY = 9.81


def run_episode(seed: int, gain: np.ndarray, lin: Linearisation, args, record: bool):
    env = load_environment(
        DOMAIN,
        TASK,
        task_kwargs={"random": seed, "initial_angle_range_deg": args.initial_angle_deg},
    )
    timestep = env.reset()
    physics, task = env.physics, env.task
    camera = resolve_camera(DOMAIN, -1)
    sampled_angle = float(physics.pole_angle())
    frames = [physics.render(args.gif_size, args.gif_size, camera_id=camera)] if record else []
    done = timestep.last()

    def env_step(command: float) -> float:
        nonlocal done
        reward = 0.0
        for _ in range(args.action_repeat):
            step = env.step(np.asarray([command]))
            reward += float(step.reward or 0.0)
            if step.last():
                done = True
                break
        if record:
            frames.append(physics.render(args.gif_size, args.gif_size, camera_id=camera))
        return reward

    # Optional zero-action steps before control starts, e.g. the context
    # frames a visual controller needs (not part of the scored interval).
    for _ in range(args.start_delay_steps):
        env_step(0.0)
        if done:
            break
    start_angle = float(physics.pole_angle())
    task.unsafe_commands = 0
    angles, offsets, commands, rewards = [], [], [], []
    compute_seconds = 0.0
    steps = 0
    while not done and steps < args.steps:
        started = time.perf_counter()
        error = get_state(physics, task) - lin.equilibrium
        command = lqr_command(gain, error)
        compute_seconds += time.perf_counter() - started
        rewards.append(env_step(command))
        steps += 1
        angles.append(float(physics.pole_angle()))
        offsets.append(float(physics.cart_position() - CART_CENTER))
        commands.append(command)
    survived = steps == args.steps and not done
    angle_array = np.abs(np.asarray(angles)) if angles else np.asarray([abs(start_angle)])
    result = {
        "seed": seed,
        "sampled_initial_angle_deg": math.degrees(sampled_angle),
        "control_start_angle_deg": math.degrees(start_angle),
        "success": bool(survived and task.unsafe_commands == 0),
        "survived": bool(survived),
        "survival_time_s": steps * args.action_repeat * CONTROL_TIMESTEP,
        "unsafe_commands": int(task.unsafe_commands),
        "mean_upright_reward": float(np.sum(rewards) / max(steps * args.action_repeat, 1)),
        "rms_angle_deg": math.degrees(float(np.sqrt(np.mean(angle_array**2)))),
        "max_angle_deg": math.degrees(float(angle_array.max())),
        "max_pivot_offset_cm": 100.0 * float(np.max(np.abs(offsets))) if offsets else 0.0,
        "saturated_fraction": float(np.mean(np.abs(commands) >= 0.999)) if commands else 0.0,
        "controller_us_per_step": 1e6 * compute_seconds / max(steps, 1),
    }
    return result, frames


def summarise(results: list[dict]) -> dict:
    success = np.asarray([item["success"] for item in results], dtype=float)
    return {
        "episodes": len(results),
        "success_rate": float(success.mean()),
        "survival_time_s": float(np.mean([r["survival_time_s"] for r in results])),
        "mean_upright_reward": float(np.mean([r["mean_upright_reward"] for r in results])),
        "rms_angle_deg": float(np.mean([r["rms_angle_deg"] for r in results])),
        "unsafe_episode_rate": float(np.mean([r["unsafe_commands"] > 0 for r in results])),
        "max_pivot_offset_cm": float(np.mean([r["max_pivot_offset_cm"] for r in results])),
    }


def write_gif(path: Path, frames: list[np.ndarray], result: dict, args) -> None:
    step_seconds = args.action_repeat * CONTROL_TIMESTEP
    images = []
    for index, frame in enumerate(frames):
        canvas = Image.new("RGB", (args.gif_size, args.gif_size + 30), (20, 20, 20))
        canvas.paste(Image.fromarray(frame), (0, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (4, 2),
            f"LQR  theta0={result['sampled_initial_angle_deg']:+.1f}deg",
            fill=(255, 255, 255),
        )
        draw.text(
            (4, 16),
            f"t={index * step_seconds:.2f}s  "
            + ("survived" if result["success"] else "failed"),
            fill=(90, 230, 90) if result["success"] else (240, 120, 120),
        )
        images.append(canvas)
    images[0].save(
        path, save_all=True, append_images=images[1:],
        duration=max(1, round(1000.0 * step_seconds)), loop=0,
    )


def angle_bins(results: list[dict], limit: float) -> list[tuple[str, int, int]]:
    edges = np.linspace(0.0, limit, 6)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = [
            r for r in results
            if low <= abs(r["sampled_initial_angle_deg"]) < high
            or (high == edges[-1] and abs(r["sampled_initial_angle_deg"]) == high)
        ]
        rows.append((f"{low:4.1f}-{high:4.1f}", sum(r["success"] for r in inside), len(inside)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--initial-angle-deg", type=float, default=DEFAULT_INITIAL_ANGLE_RANGE_DEG,
                        help="Initial lean is uniform in [-X, +X] deg; default = locked task range.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33000)
    parser.add_argument("--steps", type=int, default=150, help="Agent steps (150 x 0.04 s = 6 s).")
    parser.add_argument("--action-repeat", type=int, default=2, help="2 = 25 Hz, as the learned controllers.")
    parser.add_argument("--start-delay-steps", type=int, default=0,
                        help="Zero-action agent steps before control (3 = old visual context warm-up).")
    parser.add_argument("--tune-r", type=float, nargs="+",
                        default=[0.01, 1.0, 100.0, 10000.0],
                        help="Action weights tried (R <= ~1 is the cheap-control limit here).")
    parser.add_argument("--tune-position-scale", type=float, nargs="+",
                        default=[0.03, 0.05, 0.08, 0.12],
                        help="Rail-centring scales tried (m); smaller = stronger pull to centre.")
    parser.add_argument("--tune-episodes", type=int, default=20)
    parser.add_argument("--tune-seed", type=int, default=34000)
    parser.add_argument("--angle-scale-deg", type=float, default=3.0)
    parser.add_argument("--rate-scale", type=float, default=0.5, help="rad/s")
    parser.add_argument("--velocity-scale", type=float, default=0.3, help="m/s")
    parser.add_argument("--eps-state", type=float, default=1e-5)
    parser.add_argument("--eps-action", type=float, default=1e-4)
    parser.add_argument("--gif-episodes", type=int, default=3)
    parser.add_argument("--gif-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("runs/eval/rm65b_cartpole_lqr"))
    args = parser.parse_args()
    if not 0.0 < args.initial_angle_deg < 90.0:
        parser.error("--initial-angle-deg must be in (0, 90)")
    if min(args.episodes, args.steps, args.action_repeat, args.tune_episodes) < 1:
        parser.error("--episodes, --steps, --action-repeat and --tune-episodes must be positive")
    if args.start_delay_steps < 0 or min(args.tune_r + args.tune_position_scale) <= 0:
        parser.error("--start-delay-steps must be >= 0 and tuning values positive")

    # Dedicated instance for linearisation only.
    env = load_environment(DOMAIN, TASK, task_kwargs={"random": 0})
    env.reset()
    dt = args.action_repeat * CONTROL_TIMESTEP
    lin = Linearisation(env.physics, args.action_repeat, args.eps_state, args.eps_action)
    rate = lin.unstable_rate(dt)
    bound_deg = math.degrees(math.asin(min(1.0, rate * MAX_CART_SPEED / GRAVITY)))
    print(
        f"linearised {lin.A.shape[0]}-state model at {1.0 / dt:.1f} Hz "
        f"(equilibrium drift {lin.drift:.1e}); unstable pole mode grows at "
        f"{rate:.2f} 1/s (time constant {1000.0 / rate:.0f} ms)"
    )
    print(
        f"limits: pivot speed {MAX_CART_SPEED} m/s, accel {MAX_CART_ACCEL} m/s^2, "
        f"rail {100 * (CART_RANGE[1] - CART_RANGE[0]):.0f} cm (+/-"
        f"{100 * (CART_RANGE[1] - CART_CENTER):.0f} cm)"
    )
    print(
        f"necessary condition (any controller, pole released at rest): "
        f"|theta0| < asin(lambda*v_max/g) = {bound_deg:.1f} deg; requested "
        f"+/-{args.initial_angle_deg:g} deg"
        + ("  -> outside the bound, some starts are unrecoverable"
           if args.initial_angle_deg >= bound_deg else "")
    )

    # 1. Pick (R, rail-centring scale) on separate tuning seeds.
    print(f"\ntuning on {args.tune_episodes} seeds from {args.tune_seed} "
          f"(not the evaluation seeds):")
    tuning = []
    for position_scale in args.tune_position_scale:
        scales = np.asarray([
            math.radians(args.angle_scale_deg), args.rate_scale,
            position_scale, args.velocity_scale,
        ])
        for r_weight in args.tune_r:
            label = f"R={r_weight:<7g} pos_scale={position_scale:<5g}"
            try:
                gain = lqr_gain(lin, scales, r_weight)
            except (RuntimeError, ValueError) as error:
                print(f"  {label}: DARE failed ({error})")
                continue
            closed = closed_loop_radius(lin, gain)
            results = [
                run_episode(args.tune_seed + index, gain, lin, args, record=False)[0]
                for index in range(args.tune_episodes)
            ]
            summary = summarise(results)
            tuning.append(
                {"r": r_weight, "position_scale": position_scale, "gain": gain,
                 "summary": summary, "closed": float(closed)}
            )
            print(
                f"  {label} success {summary['success_rate']:5.0%}  "
                f"survive {summary['survival_time_s']:.2f}s  upright "
                f"{summary['mean_upright_reward']:.3f}  unsafe "
                f"{summary['unsafe_episode_rate']:4.0%}  closed-loop |eig|max {closed:.3f}"
            )
    if not tuning:
        raise SystemExit("no setting produced a stabilising LQR")
    chosen = max(
        tuning,
        key=lambda item: (item["summary"]["success_rate"], item["summary"]["mean_upright_reward"]),
    )
    r_weight, gain, closed = chosen["r"], chosen["gain"], chosen["closed"]
    print(f"chosen R={r_weight:g}, rail-centring scale {chosen['position_scale']:g} m")

    # 2. Evaluate the chosen controller.
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gifs").mkdir(exist_ok=True)
    print(f"\nevaluating {args.episodes} episodes from seed {args.seed}, "
          f"initial lean uniform in +/-{args.initial_angle_deg:g} deg, "
          f"{args.steps} steps ({args.steps * dt:.1f} s), start delay "
          f"{args.start_delay_steps} steps:")
    results = []
    for index in range(args.episodes):
        seed = args.seed + index
        record = index < args.gif_episodes
        result, frames = run_episode(seed, gain, lin, args, record)
        results.append(result)
        print(
            f"  ep {index + 1:3d}/{args.episodes} seed {seed}: theta0="
            f"{result['sampled_initial_angle_deg']:+6.2f}deg  "
            f"{'OK  ' if result['success'] else 'FAIL'} survive="
            f"{result['survival_time_s']:.2f}s rms={result['rms_angle_deg']:5.2f}deg "
            f"max_offset={result['max_pivot_offset_cm']:4.1f}cm "
            f"saturated={result['saturated_fraction']:4.0%} unsafe={result['unsafe_commands']}"
        )
        if record:
            write_gif(args.output / "gifs" / f"episode_{seed}.gif", frames, result, args)

    summary = summarise(results)
    print(
        f"\nLQR (R={r_weight:g}, rail scale {chosen['position_scale']:g} m): success "
        f"{sum(r['success'] for r in results)}/"
        f"{len(results)} ({summary['success_rate']:.0%}), mean survival "
        f"{summary['survival_time_s']:.2f}s, upright {summary['mean_upright_reward']:.3f}, "
        f"unsafe episodes {summary['unsafe_episode_rate']:.0%}, "
        f"controller {np.mean([r['controller_us_per_step'] for r in results]):.0f} us/step"
    )
    print("success by |initial lean| (deg):")
    for label, wins, count in angle_bins(results, args.initial_angle_deg):
        print(f"  {label}: {wins}/{count}")
    print(f"(necessary-condition bound: {bound_deg:.1f} deg)")

    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "unstable_rate_per_s": rate,
                "necessary_bound_deg": bound_deg,
                "chosen_r": r_weight,
                "chosen_position_scale": chosen["position_scale"],
                "closed_loop_spectral_radius": closed,
                "gain": gain.tolist(),
                "equilibrium": lin.equilibrium.tolist(),
                "tuning": [
                    {"r": item["r"], "position_scale": item["position_scale"],
                     "closed_loop_spectral_radius": item["closed"], **item["summary"]}
                    for item in tuning
                ],
                "summary": summary,
                "episodes": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"saved {args.output / 'metrics.json'}, episodes.csv and gifs/")


if __name__ == "__main__":
    main()
