"""CEM-MPC evaluation for the RM65-B robot cart-pole task.

The task has one scalar objective: keep the passive pole upright.  A trial is
successful only when it survives the complete fixed-duration evaluation with
no workspace-guard activation.  Random, zero-action, true-simulator Oracle
CEM, backbone reward-CEM and graph-corrected reward-CEM share episode seeds,
action cadence and budgets.

The default initial lean is the environment's locked task range (uniform
+/-7 deg). The v2 model was collected at +/-3 deg, so pass
``--initial-angle-deg 3`` for its in-distribution evaluation. The script
reports both control quality and whether each learned CEM replan meets its
real-time budget.
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
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jointgv import JointGraphVideoWorldModel  # noqa: E402
from jointgv.dmc_protocol import resolve_camera  # noqa: E402
from jointgv.envs import load_environment  # noqa: E402
from jointgv.envs.rm65b_cartpole import (  # noqa: E402
    CART_RANGE,
    CONTROL_TIMESTEP,
    DEFAULT_INITIAL_ANGLE_RANGE_DEG,
    FALL_ANGLE,
    MAX_CART_ACCEL,
    MAX_CART_SPEED,
    PHYSICS_TIMESTEP,
)
from jointgv.planning import (  # noqa: E402
    CEMConfig,
    CEMPlanner,
    LatentRollout,
    load_track_reward_head,
    predicted_reward_cost,
)


DOMAIN, TASK = "rm65b", "cartpole"
SUBSTEPS = round(CONTROL_TIMESTEP / PHYSICS_TIMESTEP)


def unsafe_sequence_mask(
    sequences: torch.Tensor,
    control_state: tuple[float, float],
    action_repeat: int,
) -> torch.Tensor:
    """Exact scalar cart-command guard for candidate action sequences."""
    initial_target, initial_velocity = control_state
    unsafe = np.zeros(sequences.shape[0], dtype=bool)
    for index, sequence in enumerate(sequences.detach().cpu().numpy()):
        target = float(initial_target)
        velocity = float(initial_velocity)
        for action in sequence:
            command = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
            for _ in range(action_repeat):
                desired = command * MAX_CART_SPEED
                velocity += float(
                    np.clip(
                        desired - velocity,
                        -MAX_CART_ACCEL * CONTROL_TIMESTEP,
                        MAX_CART_ACCEL * CONTROL_TIMESTEP,
                    )
                )
                candidate = target + velocity * CONTROL_TIMESTEP
                bounded = float(np.clip(candidate, *CART_RANGE))
                if bounded != candidate:
                    unsafe[index] = True
                    velocity = 0.0
                target = bounded
    return torch.as_tensor(unsafe, device=sequences.device)


class Episode:
    """One deterministic cart-pole episode and controller observation history."""

    def __init__(self, seed: int, args: argparse.Namespace):
        self.seed = seed
        self.args = args
        self.env = load_environment(
            DOMAIN,
            TASK,
            task_kwargs={
                "random": seed,
                "initial_angle_range_deg": args.initial_angle_deg,
            },
        )
        timestep = self.env.reset()
        self.physics, self.task = self.env.physics, self.env.task
        self.action_dim = int(self.env.action_spec().shape[0])
        self.camera = resolve_camera(DOMAIN, -1)
        self.frames: list[np.ndarray] = [self.render()]
        self.actions: list[np.ndarray] = []
        self.done = bool(timestep.last())
        self.physics_steps = 0
        for _ in range(args.context_frames):
            _, repeats = self.step(np.zeros(self.action_dim))
            if repeats != args.action_repeat or self.done:
                raise RuntimeError("cart-pole failed during the real-frame context warm-up")
        # Context is shared evidence, not part of the scored control interval.
        self.physics_steps = 0

    def render(self, size: int | None = None) -> np.ndarray:
        size = self.args.size if size is None else size
        return self.physics.render(size, size, camera_id=self.camera)

    def step(self, action: np.ndarray) -> tuple[float, int]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        reward = 0.0
        repeats = 0
        for _ in range(self.args.action_repeat):
            timestep = self.env.step(action)
            reward += float(timestep.reward or 0.0)
            repeats += 1
            self.physics_steps += 1
            if timestep.last():
                self.done = True
                break
        self.actions.append(action.astype(np.float32))
        self.frames.append(self.render())
        return reward, repeats


class ZeroController:
    name = "zero"

    def reset(self, episode: Episode) -> None:
        del episode

    def plan(self, episode: Episode) -> np.ndarray:
        return np.zeros((episode.args.replan_every, episode.action_dim), dtype=np.float32)


class RandomController:
    name = "random"

    def reset(self, episode: Episode) -> None:
        self.rng = np.random.default_rng(episode.seed + 7_000_000)

    def plan(self, episode: Episode) -> np.ndarray:
        return self.rng.uniform(
            -1.0, 1.0, (episode.args.replan_every, episode.action_dim)
        ).astype(np.float32)


class OracleController:
    """CEM using true MuJoCo dynamics and the task's single upright reward."""

    name = "oracle"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = CEMConfig(
            horizon=args.horizon,
            candidates=args.oracle_candidates,
            elites=args.oracle_elites,
            iterations=args.oracle_iterations,
            init_std=args.init_std,
            min_std=args.min_std,
            cost_steps=args.horizon,
        )

    def reset(self, episode: Episode) -> None:
        self.planner = CEMPlanner(
            episode.action_dim,
            self.config,
            torch.device("cpu"),
            seed=episode.seed,
        )
        self.simulator = episode.physics.copy(share_model=True)

    def cost(self, episode: Episode, sequences: torch.Tensor) -> torch.Tensor:
        args, task, simulator = self.args, episode.task, self.simulator
        state = episode.physics.get_state().copy()
        control_state = task.control_state()
        real_unsafe = task.unsafe_commands
        costs = np.zeros(sequences.shape[0], dtype=np.float64)
        for index, sequence in enumerate(sequences.numpy()):
            simulator.set_state(state)
            simulator.forward()
            task.set_control_state(control_state)
            task.unsafe_commands = real_unsafe
            total = 0.0
            weight = 1.0
            candidate_unsafe = False
            failed = False
            for action in sequence:
                repeated_reward = 0.0
                for _ in range(args.action_repeat):
                    task.before_step(action, simulator)
                    candidate_unsafe |= task.unsafe_commands > real_unsafe
                    simulator.step(SUBSTEPS)
                    angle = abs(float(simulator.pole_angle()))
                    repeated_reward += float(np.clip(1.0 - angle / FALL_ANGLE, 0.0, 1.0))
                    if angle >= FALL_ANGLE:
                        failed = True
                        break
                total -= weight * repeated_reward
                weight *= args.discount
                if failed:
                    break
            if candidate_unsafe:
                total += args.unsafe_penalty
            costs[index] = total
        task.set_control_state(control_state)
        task.unsafe_commands = real_unsafe
        return torch.from_numpy(costs)

    def plan(self, episode: Episode) -> np.ndarray:
        plan = self.planner.plan(lambda sequences: self.cost(episode, sequences))
        self.planner.shift(self.args.replan_every)
        return plan[: self.args.replan_every].numpy()


class WorldModelController:
    """Reward-CEM on backbone-only or graph-corrected DINO-WM imagination."""

    def __init__(self, name, model, mode, args, device, reward_head):
        self.name = name
        self.args = args
        self.device = device
        self.rollout = LatentRollout(
            model,
            use_graph=mode == "graph",
            history_window=args.history_window or None,
            amp=args.amp,
            reward_head=reward_head,
        )
        self.config = CEMConfig(
            horizon=args.horizon,
            candidates=args.candidates,
            elites=args.elites,
            iterations=args.iterations,
            init_std=args.init_std,
            min_std=args.min_std,
            cost_steps=args.horizon,
        )

    def images(self, frames: list[np.ndarray]) -> torch.Tensor:
        array = np.stack(frames).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(0, 3, 1, 2).to(self.device)

    def reset(self, episode: Episode) -> None:
        self.planner = CEMPlanner(
            episode.action_dim, self.config, self.device, seed=episode.seed
        )

    def plan(self, episode: Episode) -> np.ndarray:
        context = self.args.context_frames
        actions = episode.actions[-context:]
        self.rollout.prepare(
            self.images(episode.frames[-(context + 1) :]),
            torch.from_numpy(np.stack(actions)).to(self.device),
        )
        reward_cost = predicted_reward_cost(
            self.rollout,
            discount=self.args.discount,
            chunk_size=self.args.chunk_size,
            reward_min=0.0,
            reward_max=float(self.args.action_repeat),
        )

        def constrained_cost(sequences: torch.Tensor) -> torch.Tensor:
            costs = reward_cost(sequences)
            unsafe = unsafe_sequence_mask(
                sequences, episode.task.control_state(), self.args.action_repeat
            )
            return costs + unsafe.to(costs.dtype) * self.args.unsafe_penalty

        plan = self.planner.plan(constrained_cost)
        self.planner.shift(self.args.replan_every)
        return plan[: self.args.replan_every].float().cpu().numpy()


def run_episode(controller, episode: Episode, args, record_gif: bool, desc: str):
    controller.reset(episode)
    initial_angle = abs(float(episode.physics.pole_angle()))
    rewards: list[float] = []
    angles: list[float] = []
    rates: list[float] = []
    actions: list[float] = []
    plan_seconds: list[float] = []
    views: list[np.ndarray] = []
    if record_gif:
        views.append(episode.render(args.gif_size))

    plan = None
    bar = tqdm(range(args.steps), desc=desc, leave=False, dynamic_ncols=True)
    for step in bar:
        if step % args.replan_every == 0:
            started = time.perf_counter()
            plan = controller.plan(episode)
            plan_seconds.append(time.perf_counter() - started)
        action = plan[step % args.replan_every]
        reward, _ = episode.step(action)
        rewards.append(reward)
        actions.append(float(np.asarray(action).reshape(-1)[0]))
        angles.append(abs(float(episode.physics.pole_angle())))
        rates.append(abs(float(episode.physics.pole_angular_velocity())))
        if record_gif:
            views.append(episode.render(args.gif_size))
        bar.set_postfix(
            angle=f"{math.degrees(angles[-1]):.1f}deg",
            reward=f"{reward / args.action_repeat:.3f}",
            plan=f"{np.mean(plan_seconds):.3f}s",
            alive="no" if episode.done else "yes",
        )
        if episode.done:
            break
    bar.close()

    dt = CONTROL_TIMESTEP * args.action_repeat
    planning_budget = args.replan_every * dt
    plan_array = np.asarray(plan_seconds, dtype=np.float64)
    survived = len(rewards) == args.steps and not episode.done
    physics_steps = max(episode.physics_steps, 1)
    return {
        "success": bool(survived and episode.task.unsafe_commands == 0),
        "survived_full_horizon": bool(survived),
        "initial_angle_deg": math.degrees(initial_angle),
        "survival_time_s": float(episode.physics_steps * CONTROL_TIMESTEP),
        "return": float(np.sum(rewards)),
        "mean_upright_reward": float(np.sum(rewards) / physics_steps),
        "rms_angle_deg": math.degrees(float(np.sqrt(np.mean(np.square(angles))))),
        "max_angle_deg": math.degrees(float(np.max(angles))),
        "final_angle_deg": math.degrees(float(angles[-1])),
        "rms_angular_rate_rad_s": float(np.sqrt(np.mean(np.square(rates)))),
        "max_angular_rate_rad_s": float(np.max(rates)),
        "action_rms": float(np.sqrt(np.mean(np.square(actions)))),
        "unsafe_commands": int(episode.task.unsafe_commands),
        "planning_budget_s": planning_budget,
        "plan_seconds": float(plan_array.mean()),
        "plan_seconds_p95": float(np.percentile(plan_array, 95)),
        "realtime_plan_rate": float(np.mean(plan_array <= planning_budget)),
        "reward_curve": rewards,
        "angle_curve": angles,
        "angular_rate_curve": rates,
        "action_curve": actions,
        "plan_seconds_curve": plan_seconds,
    }, views


def write_gif(path: Path, columns, results, args) -> None:
    size = args.gif_size
    step_seconds = CONTROL_TIMESTEP * args.action_repeat
    header = 36
    length = max(len(frames) for frames in columns.values())
    canvases = []
    for index in range(length):
        canvas = Image.new("RGB", (len(columns) * size, size + header), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for slot, (label, frames) in enumerate(columns.items()):
            frame = frames[min(index, len(frames) - 1)]
            canvas.paste(Image.fromarray(frame), (slot * size, header))
            result = results[label]
            draw.text((slot * size + 4, 2), label, fill=(255, 255, 255))
            draw.text(
                (slot * size + 4, 18),
                f"t={index * step_seconds:.2f}s  "
                + ("survived" if result["success"] else "failed"),
                fill=(90, 230, 90) if result["success"] else (240, 120, 120),
            )
        canvases.append(canvas)
    canvases[0].save(
        path,
        save_all=True,
        append_images=canvases[1:],
        duration=max(1, round(1000.0 / args.fps)),
        loop=0,
    )


def summarise(results: list[dict]) -> dict[str, float]:
    success = np.asarray([item["success"] for item in results], dtype=float)
    probability = float(success.mean())
    half = 1.96 * math.sqrt(
        max(probability * (1.0 - probability), 1e-12) / len(results)
    )
    keys = (
        "survival_time_s",
        "mean_upright_reward",
        "rms_angle_deg",
        "max_angle_deg",
        "rms_angular_rate_rad_s",
        "action_rms",
        "plan_seconds",
        "plan_seconds_p95",
        "realtime_plan_rate",
    )
    summary = {key: float(np.mean([item[key] for item in results])) for key in keys}
    summary.update(
        episodes=len(results),
        success_rate=probability,
        success_ci95=float(half) if len(results) > 1 else float("nan"),
        unsafe_episode_rate=float(
            np.mean([item["unsafe_commands"] > 0 for item in results])
        ),
    )
    return summary


def print_table(summaries: dict[str, dict[str, float]]) -> None:
    columns = (
        ("success", "success_rate", "{:.0%}"),
        ("survive_s", "survival_time_s", "{:.2f}"),
        ("upright", "mean_upright_reward", "{:.3f}"),
        ("rms_deg", "rms_angle_deg", "{:.1f}"),
        ("max_deg", "max_angle_deg", "{:.1f}"),
        ("plan_s", "plan_seconds", "{:.3f}"),
        ("rt_plan", "realtime_plan_rate", "{:.0%}"),
        ("unsafe", "unsafe_episode_rate", "{:.0%}"),
    )
    width = max(len(label) for label in summaries) + 2
    print("".ljust(width) + " ".join(title.rjust(10) for title, _, _ in columns))
    for label, summary in summaries.items():
        cells = [fmt.format(summary[key]).rjust(10) for _, key, fmt in columns]
        print(label.ljust(width) + " ".join(cells) + f"   (n={summary['episodes']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        default=[],
        metavar=("LABEL", "CHECKPOINT", "MODE"),
        help="Repeatable. MODE is graph or base.",
    )
    parser.add_argument(
        "--reward-head",
        nargs=2,
        action="append",
        default=[],
        metavar=("LABEL", "PATH"),
        help="Track-specific head from train_reward_heads.py; one per model.",
    )
    parser.add_argument("--no-zero", action="store_true")
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--context-frames", type=int, default=3)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument(
        "--initial-angle-deg",
        type=float,
        default=DEFAULT_INITIAL_ANGLE_RANGE_DEG,
        help="Uniform initial lean range (deg); default = locked task range. v2 model: 3.",
    )
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--oracle-candidates", type=int, default=64)
    parser.add_argument("--oracle-elites", type=int, default=8)
    parser.add_argument("--oracle-iterations", type=int, default=3)
    parser.add_argument("--init-std", type=float, default=0.5)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--unsafe-penalty", type=float, default=100.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gif-episodes", type=int, default=2)
    parser.add_argument("--gif-size", type=int, default=192)
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = [label for label, _, _ in args.model]
    if len(set(labels)) != len(labels):
        parser.error("model labels must be unique")
    for label, _, mode in args.model:
        if mode not in {"graph", "base"}:
            parser.error("MODE must be graph or base")
        if label in {"zero", "random", "oracle"}:
            parser.error(f"{label!r} is a reserved controller label")
    head_paths = dict(args.reward_head)
    if len(head_paths) != len(args.reward_head) or set(head_paths) != set(labels):
        parser.error("give exactly one unique --reward-head for every --model")
    if not 1 <= args.replan_every <= args.horizon:
        parser.error("need 1 <= --replan-every <= --horizon")
    if not 1 <= args.elites <= args.candidates:
        parser.error("need 1 <= --elites <= --candidates")
    if not 1 <= args.oracle_elites <= args.oracle_candidates:
        parser.error("need 1 <= --oracle-elites <= --oracle-candidates")
    if args.history_window and args.history_window < 2:
        parser.error("--history-window must be 0 or at least 2")
    if args.episodes < 1 or args.steps < 1 or args.context_frames < 1:
        parser.error("--episodes, --steps and --context-frames must be positive")
    if not 0.0 < args.initial_angle_deg < 90.0:
        parser.error("--initial-angle-deg must be in (0, 90)")
    if args.action_repeat != 2:
        parser.error(
            "the v2 world model was trained at --action-repeat 2; recollect and "
            "retrain before evaluating a different temporal protocol"
        )
    if not 0.0 < args.discount <= 1.0:
        parser.error("--discount must be in (0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"--output is not empty: {args.output}; use a fresh directory")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    controllers = []
    if not args.no_zero:
        controllers.append(ZeroController())
    if not args.no_random:
        controllers.append(RandomController())
    if not args.no_oracle:
        controllers.append(OracleController(args))

    models: dict[str, JointGraphVideoWorldModel] = {}
    epochs: dict[str, object] = {}
    for label, checkpoint_path, mode in args.model:
        if checkpoint_path not in models:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = JointGraphVideoWorldModel(**checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["model"])
            models[checkpoint_path] = model.eval()
            epochs[checkpoint_path] = checkpoint.get("epoch", "?")
            print(f"loaded {checkpoint_path} (epoch {epochs[checkpoint_path]})")
            del checkpoint
        reward_head, track = load_track_reward_head(head_paths[label], device)
        problems = []
        if track["mode"] != mode:
            problems.append(f"head fitted for mode {track['mode']}, not {mode}")
        # The relative path keeps heads valid after the checkout moves.
        fitted_on = {Path(track["checkpoint_resolved"]), Path(track["checkpoint"]).resolve()}
        if Path(checkpoint_path).resolve() not in fitted_on:
            problems.append(f"head fitted on {track['checkpoint']}")
        elif track["checkpoint_epoch"] != epochs[checkpoint_path]:
            problems.append(
                f"head fitted on checkpoint epoch {track['checkpoint_epoch']}, "
                f"current file is epoch {epochs[checkpoint_path]}"
            )
        if track["history_window"] != (args.history_window or None):
            problems.append(f"head history window is {track['history_window']}")
        if track["context_frames"] != args.context_frames:
            problems.append(f"head context is {track['context_frames']}")
        if track["horizon"] < args.horizon:
            problems.append(f"head horizon is only {track['horizon']}")
        if problems:
            parser.error(f"--reward-head {label}: " + "; ".join(problems))
        if track["amp"] != args.amp:
            print(f"WARNING: {label} head amp={track['amp']}, evaluation amp={args.amp}")
        controllers.append(
            WorldModelController(
                label, models[checkpoint_path], mode, args, device, reward_head
            )
        )
        print(f"  {label}: mode={mode}, reward_head={head_paths[label]}")
    if not controllers:
        parser.error("nothing to evaluate")

    dt = CONTROL_TIMESTEP * args.action_repeat
    budget = args.replan_every * dt
    print(
        f"protocol: {args.episodes} episodes/controller, {args.steps}x{dt:.3f}s "
        f"({args.steps * dt:.2f}s), initial +/-{args.initial_angle_deg:g}deg; "
        f"CEM H={args.horizon}, learned N={args.candidates}/{args.iterations}iters, "
        f"oracle N={args.oracle_candidates}/{args.oracle_iterations}iters; "
        f"replan every {args.replan_every} steps => real-time budget {budget:.3f}s"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gifs").mkdir(exist_ok=True)
    all_results: dict[str, list[dict]] = {controller.name: [] for controller in controllers}
    csv_rows = []
    for episode_index in range(args.episodes):
        seed = args.seed + episode_index
        record_gif = episode_index < args.gif_episodes
        columns = {}
        episode_results = {}
        for controller in controllers:
            episode = Episode(seed, args)
            result, frames = run_episode(
                controller,
                episode,
                args,
                record_gif,
                f"episode {episode_index + 1}/{args.episodes} {controller.name}",
            )
            result["seed"] = seed
            all_results[controller.name].append(result)
            episode_results[controller.name] = result
            csv_rows.append(
                {
                    "controller": controller.name,
                    **{key: value for key, value in result.items() if not key.endswith("_curve")},
                }
            )
            if record_gif:
                columns[controller.name] = frames
            print(
                f"episode={episode_index + 1}/{args.episodes} seed={seed} "
                f"controller={controller.name} success={result['success']} "
                f"survival={result['survival_time_s']:.2f}s "
                f"upright={result['mean_upright_reward']:.3f} "
                f"rms_angle={result['rms_angle_deg']:.2f}deg "
                f"plan={result['plan_seconds']:.3f}s "
                f"realtime={result['realtime_plan_rate']:.0%}"
            )
        if record_gif:
            write_gif(
                args.output / "gifs" / f"episode_{episode_index:03d}.gif",
                columns,
                episode_results,
                args,
            )

    summaries = {label: summarise(results) for label, results in all_results.items()}
    print("\nCart-pole control summary")
    print_table(summaries)
    serial_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "config": serial_args,
                "timing": {
                    "model_step_hz": 1.0 / dt,
                    "replan_hz": 1.0 / budget,
                    "replan_budget_s": budget,
                    "horizon_seconds": args.horizon * dt,
                },
                "summary": summaries,
                "episodes": all_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"saved {args.output / 'metrics.json'}, episodes.csv and gifs/")


if __name__ == "__main__":
    main()
