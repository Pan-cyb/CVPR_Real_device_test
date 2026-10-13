"""Reward-CEM evaluation for visible-target RM65-B anti-sway transport.

Every episode contains a matte black board and one visible red target. The
robot must move the bracket tip / pendulum hinge (the TCP) into that disk,
bring the passive yellow pendulum close to vertical and nearly motionless,
stop the TCP, and hold all conditions together. There is no hidden goal pose
and no synthetic goal image: any safe inverse-kinematics solution is valid.

Controllers share seeds, observations and action budget. Learned controllers
share CEM settings; the slower true-simulator oracle has an explicit separate
candidate budget:

* ``random`` is the lower bound.
* each ``--model LABEL CHECKPOINT MODE`` uses a learned reward head on an
  imagined DINO-WM rollout; MODE is ``base`` or ``graph``. With
  ``--reward-head LABEL PATH`` (from train_reward_heads.py, one per model)
  each track is scored by a head fitted to its own imagination; without it,
  every track uses the world model's head, which P training fits to the
  backbone's predictions only.
* ``oracle`` uses the same hard task gate on true MuJoCo rollouts.

Official success is conjunctive and non-compensatory. Distance, pendulum
angle, pendulum angular rate and TCP speed must all pass their thresholds for
``--hold-steps`` consecutive outer steps, before a fixed deadline. Any
workspace-guard activation disqualifies the episode. Results are ranked
lexicographically: success rate first, certified completion time among
successful episodes second, then swing diagnostics. They are never collapsed
into one weighted score.

    python script/jointgv/evaluation/evaluate_planning.py \
        --model backbone runs/.../best_backbone_rollout_joint_px.pt base \
        --model graph runs/.../best_rollout_joint_px.pt graph \
        --reward-head backbone runs/rm65b_pendulum_reward_heads/backbone.pt \
        --reward-head graph runs/rm65b_pendulum_reward_heads/graph.pt \
        --output runs/eval/rm65b_planning

Evaluation only: reads checkpoints and writes metrics, CSV and GIFs.
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
from jointgv.envs.rm65b_pendulum import (  # noqa: E402
    CONTROL_TIMESTEP,
    MAX_CONTROL_ACCEL,
    MAX_CONTROL_SPEED,
    PHYSICS_TIMESTEP,
    SWING_ANGLE_TOLERANCE,
    SWING_RATE_TOLERANCE,
    TARGET_RADIUS,
    TCP_SPEED_TOLERANCE,
    arm_target_is_safe,
)
from jointgv.planning import (  # noqa: E402
    CEMConfig,
    CEMPlanner,
    LatentRollout,
    load_track_reward_head,
    predicted_reward_cost,
)

DOMAIN, TASK = "rm65b", "pendulum"
SUBSTEPS = round(CONTROL_TIMESTEP / PHYSICS_TIMESTEP)


def unsafe_sequence_mask(
    sequences: torch.Tensor,
    control_state: tuple[np.ndarray, np.ndarray],
    action_repeat: int,
) -> torch.Tensor:
    """Exact command-integrator guard for every CEM candidate sequence."""
    initial_target, initial_velocity = control_state
    unsafe = np.zeros(sequences.shape[0], dtype=bool)
    for index, sequence in enumerate(sequences.detach().cpu().numpy()):
        target = initial_target.copy()
        velocity = initial_velocity.copy()
        for action in sequence:
            for _ in range(action_repeat):
                desired = np.clip(action, -1.0, 1.0) * MAX_CONTROL_SPEED
                velocity = np.clip(
                    desired,
                    velocity - MAX_CONTROL_ACCEL * CONTROL_TIMESTEP,
                    velocity + MAX_CONTROL_ACCEL * CONTROL_TIMESTEP,
                )
                candidate = target + velocity * CONTROL_TIMESTEP
                if arm_target_is_safe(candidate):
                    target = candidate
                else:
                    unsafe[index] = True
                    velocity = np.zeros_like(velocity)
    return torch.as_tensor(unsafe, device=sequences.device)


class Episode:
    """One deterministic visible-target episode and controller history."""

    def __init__(self, seed: int, args: argparse.Namespace):
        self.seed = seed
        self.args = args
        self.env = load_environment(DOMAIN, TASK, task_kwargs={"random": seed})
        self.env.reset()
        self.physics, self.task = self.env.physics, self.env.task
        self.action_dim = int(self.env.action_spec().shape[0])
        self.camera = resolve_camera(DOMAIN, -1)
        self.target_position = self.physics.target_position().copy()
        self.frames: list[np.ndarray] = [self.render()]
        self.actions: list[np.ndarray] = []
        self.tcp_path_length = 0.0
        self._last_tcp_speed = float(self.physics.tcp_speed())
        for _ in range(args.context_frames):
            self.step(np.zeros(self.action_dim))
        # Context is an observation warm-up, not part of the 6 s task clock.
        self.tcp_path_length = 0.0

    def render(self, size: int | None = None) -> np.ndarray:
        size = self.args.size if size is None else size
        return self.physics.render(size, size, camera_id=self.camera)

    def step(self, action: np.ndarray) -> None:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        previous_tcp = self.physics.tcp_position()[[0, 2]].copy()
        for _ in range(self.args.action_repeat):
            timestep = self.env.step(action)
            if timestep.last():
                raise RuntimeError("episode hit the environment time limit")
        current_tcp = self.physics.tcp_position()[[0, 2]].copy()
        self.tcp_path_length += float(np.linalg.norm(current_tcp - previous_tcp))
        self._last_tcp_speed = float(self.physics.tcp_speed())
        self.actions.append(action.astype(np.float32))
        self.frames.append(self.render())

    def distance(self) -> float:
        return float(self.physics.target_distance())

    def swing_angle(self) -> float:
        return abs(float(self.physics.pendulum_angle()))

    def swing_rate(self) -> float:
        return abs(float(self.physics.swing_rate()))

    def tcp_speed(self) -> float:
        return self._last_tcp_speed


class RandomController:
    name = "random"

    def reset(self, episode: Episode) -> None:
        self.rng = np.random.default_rng(episode.seed + 7_000_000)

    def plan(self, episode: Episode) -> np.ndarray:
        return self.rng.uniform(
            -1.0, 1.0, (episode.args.replan_every, episode.action_dim)
        )


class OracleController:
    """CEM on true MuJoCo state with a bottleneck, hard-gated objective."""

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
        control = task.control_state()
        real_unsafe = task.unsafe_commands
        costs = np.zeros(sequences.shape[0], dtype=np.float64)

        for index, sequence in enumerate(sequences.numpy()):
            simulator.set_state(state)
            simulator.forward()
            task.set_control_state(control)
            task.unsafe_commands = real_unsafe
            total = 0.0
            weight = 1.0
            candidate_unsafe = False
            for action in sequence:
                repeated_reward = 0.0
                for _ in range(args.action_repeat):
                    task.before_step(action, simulator)
                    candidate_unsafe |= task.unsafe_commands > real_unsafe
                    simulator.step(SUBSTEPS)
                    repeated_reward += float(task.get_reward(simulator))
                total -= weight * repeated_reward
                weight *= args.discount

            if candidate_unsafe:
                total += args.unsafe_penalty
            costs[index] = total

        task.set_control_state(control)
        task.unsafe_commands = real_unsafe
        return torch.from_numpy(costs)

    def plan(self, episode: Episode) -> np.ndarray:
        plan = self.planner.plan(lambda sequences: self.cost(episode, sequences))
        self.planner.shift(self.args.replan_every)
        return plan[: self.args.replan_every].numpy()


class WorldModelController:
    """CEM-MPC using the learned task-reward head, without a goal image."""

    def __init__(
        self,
        name: str,
        model,
        mode: str,
        args,
        device: torch.device,
        reward_head=None,
    ):
        self.name = name
        self.model = model
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

    @torch.no_grad()
    def reset(self, episode: Episode) -> None:
        self.planner = CEMPlanner(
            episode.action_dim, self.config, self.device, seed=episode.seed
        )

    def plan(self, episode: Episode) -> np.ndarray:
        context = self.args.context_frames
        actions = episode.actions[len(episode.actions) - context :]
        self.rollout.prepare(
            self.images(episode.frames[-(context + 1) :]),
            torch.from_numpy(np.stack(actions)).to(self.device)
            if actions
            else torch.zeros(0, episode.action_dim, device=self.device),
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
                sequences,
                episode.task.control_state(),
                self.args.action_repeat,
            )
            return costs + unsafe.to(costs.dtype) * self.args.unsafe_penalty

        plan = self.planner.plan(constrained_cost)
        self.planner.shift(self.args.replan_every)
        return plan[: self.args.replan_every].float().cpu().numpy()


def run_episode(controller, episode: Episode, args, record_gif: bool, desc: str):
    controller.reset(episode)
    distances: list[float] = []
    angles: list[float] = []
    rates: list[float] = []
    tcp_speeds: list[float] = []
    plan_seconds: list[float] = []
    views: list[np.ndarray] = []
    if record_gif:
        views.append(episode.render(args.gif_size))

    first_entry_step = None
    certified_step = None
    successful_hold_start = None
    hold = 0
    plan = None
    angle_limit = math.radians(args.swing_angle_threshold_deg)
    bar = tqdm(range(args.steps), desc=desc, leave=False, dynamic_ncols=True)
    for step in bar:
        if step % args.replan_every == 0:
            started = time.perf_counter()
            plan = controller.plan(episode)
            plan_seconds.append(time.perf_counter() - started)
        episode.step(plan[step % args.replan_every])

        distances.append(episode.distance())
        angles.append(episode.swing_angle())
        rates.append(episode.swing_rate())
        tcp_speeds.append(episode.tcp_speed())
        inside = distances[-1] <= args.success_radius
        if inside and first_entry_step is None:
            first_entry_step = step
        gate = (
            inside
            and angles[-1] <= angle_limit
            and rates[-1] <= args.swing_rate_threshold
            and tcp_speeds[-1] <= args.tcp_speed_threshold
            and episode.task.unsafe_commands == 0
        )
        hold = hold + 1 if gate else 0
        if certified_step is None and hold >= args.hold_steps:
            certified_step = step
            successful_hold_start = step + 1 - args.hold_steps

        if record_gif:
            views.append(episode.render(args.gif_size))
        bar.set_postfix(
            d=f"{distances[-1]:.3f}",
            ang=f"{math.degrees(angles[-1]):.1f}deg",
            rate=f"{rates[-1]:.2f}",
            plan=f"{np.mean(plan_seconds):.2f}s",
            ok="yes" if certified_step is not None else "no",
        )
    bar.close()

    dt = CONTROL_TIMESTEP * args.action_repeat
    success = certified_step is not None and episode.task.unsafe_commands == 0
    first_entry_time = (
        (first_entry_step + 1) * dt if first_entry_step is not None else None
    )
    hold_start_time = (
        (successful_hold_start + 1) * dt
        if success and successful_hold_start is not None
        else None
    )
    certified_time = (certified_step + 1) * dt if success else None
    entry_to_success = (
        certified_time - first_entry_time
        if certified_time is not None and first_entry_time is not None
        else None
    )
    tail = slice(max(0, len(distances) - args.hold_steps), len(distances))
    return {
        "success": success,
        "first_entry_time_s": first_entry_time,
        "hold_start_time_s": hold_start_time,
        "time_to_success_s": certified_time,
        "entry_to_success_s": entry_to_success,
        "settled_at_end": hold >= args.hold_steps and episode.task.unsafe_commands == 0,
        "final_distance_m": distances[-1],
        "min_distance_m": float(np.min(distances)),
        "rms_swing_angle_deg": math.degrees(float(np.sqrt(np.mean(np.square(angles))))),
        "max_swing_angle_deg": math.degrees(float(np.max(angles))),
        "final_swing_angle_deg": math.degrees(float(np.mean(angles[tail]))),
        "rms_swing_rate_rad_s": float(np.sqrt(np.mean(np.square(rates)))),
        "final_swing_rate_rad_s": float(np.mean(rates[tail])),
        "final_tcp_speed_m_s": float(np.mean(tcp_speeds[tail])),
        "tcp_path_length_m": float(episode.tcp_path_length),
        "plan_seconds": float(np.mean(plan_seconds)),
        "unsafe_commands": int(episode.task.unsafe_commands),
        "distance_curve": distances,
        "angle_curve": angles,
        "swing_rate_curve": rates,
        "tcp_speed_curve": tcp_speeds,
    }, views


def write_gif(
    path: Path,
    columns: dict[str, list[np.ndarray]],
    results: dict[str, dict],
    args: argparse.Namespace,
) -> None:
    """Write controller columns; every frame already contains the red target."""
    size = args.gif_size
    step_seconds = CONTROL_TIMESTEP * args.action_repeat
    header = 34
    length = max(len(frames) for frames in columns.values())
    canvases = []
    for index in range(length):
        canvas = Image.new("RGB", (len(columns) * size, size + header), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for slot, (label, frames) in enumerate(columns.items()):
            frame = frames[min(index, len(frames) - 1)]
            canvas.paste(Image.fromarray(frame), (slot * size, header))
            result = results[label]
            done = (
                result["time_to_success_s"] is not None
                and index * step_seconds >= result["time_to_success_s"]
            )
            draw.text((slot * size + 4, 2), label, fill=(255, 255, 255))
            draw.text(
                (slot * size + 4, 17),
                f"t={index * step_seconds:.2f}s" + ("  success" if done else ""),
                fill=(90, 230, 90) if done else (200, 200, 200),
            )
        canvases.append(canvas)
    canvases[0].save(
        path,
        save_all=True,
        append_images=canvases[1:],
        duration=max(1, round(1000.0 / args.fps)),
        loop=0,
    )


def _mean_defined(results: list[dict], key: str, successful_only: bool = False) -> float:
    values = [
        result[key]
        for result in results
        if result[key] is not None and (not successful_only or result["success"])
    ]
    return float(np.mean(values)) if values else float("nan")


def summarise(results: list[dict]) -> dict[str, float]:
    success = np.asarray([result["success"] for result in results], dtype=float)
    probability = float(success.mean())
    half = 1.96 * math.sqrt(
        max(probability * (1.0 - probability), 1e-12) / len(results)
    )
    return {
        "episodes": len(results),
        "success_rate": probability,
        "success_ci95": float(half) if len(results) > 1 else float("nan"),
        "settled_rate": float(np.mean([r["settled_at_end"] for r in results])),
        "first_entry_time_s": _mean_defined(results, "first_entry_time_s"),
        "time_to_success_s": _mean_defined(
            results, "time_to_success_s", successful_only=True
        ),
        "entry_to_success_s": _mean_defined(
            results, "entry_to_success_s", successful_only=True
        ),
        "final_distance_m": float(np.mean([r["final_distance_m"] for r in results])),
        "rms_swing_angle_deg": float(
            np.mean([r["rms_swing_angle_deg"] for r in results])
        ),
        "max_swing_angle_deg": float(
            np.mean([r["max_swing_angle_deg"] for r in results])
        ),
        "rms_swing_rate_rad_s": float(
            np.mean([r["rms_swing_rate_rad_s"] for r in results])
        ),
        "unsafe_episode_rate": float(
            np.mean([r["unsafe_commands"] > 0 for r in results])
        ),
        "plan_seconds": float(np.mean([r["plan_seconds"] for r in results])),
    }


def print_table(summaries: dict[str, dict[str, float]]) -> None:
    columns = (
        ("success", "success_rate", "{:.0%}"),
        ("t_enter", "first_entry_time_s", "{:.2f}"),
        ("t_success", "time_to_success_s", "{:.2f}"),
        ("settle", "entry_to_success_s", "{:.2f}"),
        ("final_d", "final_distance_m", "{:.3f}"),
        ("rms_ang", "rms_swing_angle_deg", "{:.1f}"),
        ("rms_rate", "rms_swing_rate_rad_s", "{:.2f}"),
        ("unsafe", "unsafe_episode_rate", "{:.0%}"),
        ("plan", "plan_seconds", "{:.2f}"),
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
        help="Repeatable. MODE is 'graph' (correction on) or 'base' (backbone alone).",
    )
    parser.add_argument(
        "--reward-head",
        nargs=2,
        action="append",
        default=[],
        metavar=("LABEL", "PATH"),
        help=(
            "Repeatable: the train_reward_heads.py head fitted on this "
            "--model's own track. Give one for every --model or for none."
        ),
    )
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument(
        "--steps", type=int, default=150, help="Outer steps; 150 x 0.04 s = 6 s."
    )
    parser.add_argument("--context-frames", type=int, default=3)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--success-radius", type=float, default=TARGET_RADIUS)
    parser.add_argument(
        "--swing-angle-threshold-deg",
        type=float,
        default=math.degrees(SWING_ANGLE_TOLERANCE),
    )
    parser.add_argument(
        "--swing-rate-threshold", type=float, default=SWING_RATE_TOLERANCE
    )
    parser.add_argument(
        "--tcp-speed-threshold", type=float, default=TCP_SPEED_TOLERANCE
    )
    parser.add_argument("--hold-steps", type=int, default=12)
    parser.add_argument(
        "--allow-protocol-override",
        action="store_true",
        help="Allow diagnostics that no longer match the trained reward.",
    )
    # CEM-MPC.
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--init-std", type=float, default=0.5)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--history-window",
        type=int,
        default=4,
        help="Frames the predictor attends to during planning (0 = full history).",
    )
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--oracle-candidates", type=int, default=64)
    parser.add_argument("--oracle-elites", type=int, default=8)
    parser.add_argument("--oracle-iterations", type=int, default=3)
    parser.add_argument("--unsafe-penalty", type=float, default=100.0)
    # Output.
    parser.add_argument("--gif-episodes", type=int, default=3)
    parser.add_argument("--gif-size", type=int, default=192)
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/eval/rm65b_planning"))
    args = parser.parse_args()

    labels = [label for label, _, _ in args.model]
    for label, _, mode in args.model:
        if mode not in {"graph", "base"}:
            parser.error("MODE must be 'graph' or 'base'")
        if label in {"random", "oracle"}:
            parser.error("'random' and 'oracle' are reserved labels")
    if len(set(labels)) != len(labels):
        parser.error("model labels must be unique")
    head_paths = dict(args.reward_head)
    if len(head_paths) != len(args.reward_head) or not set(head_paths) <= set(labels):
        parser.error("--reward-head labels must be unique --model labels")
    if head_paths and set(head_paths) != set(labels):
        parser.error(
            "give a --reward-head for every --model or for none, so all "
            "tracks are scored the same way"
        )
    if not 1 <= args.replan_every <= args.horizon:
        parser.error("need 1 <= --replan-every <= --horizon")
    if args.history_window and args.history_window < 2:
        parser.error("--history-window must be 0 (full) or at least 2")
    if args.episodes < 1 or args.context_frames < 1:
        parser.error("--episodes and --context-frames must be positive")
    if args.steps < args.hold_steps:
        parser.error("--steps must be at least --hold-steps")
    if min(
        args.success_radius,
        args.swing_angle_threshold_deg,
        args.swing_rate_threshold,
        args.tcp_speed_threshold,
        args.discount,
        args.unsafe_penalty,
    ) <= 0:
        parser.error("task thresholds, --discount and --unsafe-penalty must be positive")
    if args.discount > 1.0:
        parser.error("--discount must be at most 1")
    protocol_mismatch = (
        args.action_repeat != 2
        or args.steps != 150
        or args.hold_steps != 12
        or not math.isclose(args.success_radius, TARGET_RADIUS)
        or not math.isclose(
            args.swing_angle_threshold_deg,
            math.degrees(SWING_ANGLE_TOLERANCE),
        )
        or not math.isclose(args.swing_rate_threshold, SWING_RATE_TOLERANCE)
        or not math.isclose(args.tcp_speed_threshold, TCP_SPEED_TOLERANCE)
    )
    if protocol_mismatch and not args.allow_protocol_override:
        parser.error(
            "task thresholds, action repeat and 6 s/0.48 s timing must match "
            "the trained reward; add --allow-protocol-override only for a "
            "clearly labelled diagnostic run"
        )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    controllers = []
    if not args.no_random:
        controllers.append(RandomController())
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
        reward_head = None
        if label in head_paths:
            reward_head, track = load_track_reward_head(head_paths[label], device)
            problems = []
            if track["mode"] != mode:
                problems.append(f"fitted for mode {track['mode']}, not {mode}")
            # The relative path keeps heads valid after the checkout moves.
            fitted_on = {Path(track["checkpoint_resolved"]), Path(track["checkpoint"]).resolve()}
            if Path(checkpoint_path).resolve() not in fitted_on:
                problems.append(f"fitted on {track['checkpoint']}")
            elif track["checkpoint_epoch"] != epochs[checkpoint_path]:
                # best_*.pt files are overwritten while training runs.
                problems.append(
                    f"fitted on epoch {track['checkpoint_epoch']} of this file, "
                    f"now epoch {epochs[checkpoint_path]}; refit the heads"
                )
            if track["history_window"] != (args.history_window or None):
                problems.append(f"fitted with history window {track['history_window']}")
            if track["context_frames"] != args.context_frames:
                problems.append(f"fitted with {track['context_frames']} context frames")
            if track["horizon"] < args.horizon:
                problems.append(f"fitted on {track['horizon']} imagined steps only")
            if problems:
                parser.error(f"--reward-head {label}: " + "; ".join(problems))
            if track["amp"] != args.amp:
                print(
                    f"WARNING: {label} head fitted with amp={track['amp']}, "
                    f"planning with amp={args.amp}"
                )
        controllers.append(
            WorldModelController(
                label, models[checkpoint_path], mode, args, device, reward_head
            )
        )
        print(
            f"  {label}: mode={mode}, reward head="
            + (head_paths[label] if label in head_paths else "world model's own")
        )
    if not args.no_oracle:
        controllers.append(OracleController(args))
    if not controllers:
        parser.error("nothing to evaluate")

    dt = CONTROL_TIMESTEP * args.action_repeat
    print(
        f"{args.episodes} episodes x {len(controllers)} controllers, "
        f"deadline={args.steps * dt:.2f}s; CEM H={args.horizon} "
        f"N={args.candidates} iters={args.iterations}; success requires "
        f"TCP d<={args.success_radius:.3f}m, angle<="
        f"{args.swing_angle_threshold_deg:.1f}deg, rate<="
        f"{args.swing_rate_threshold:.2f}rad/s, TCP speed<="
        f"{args.tcp_speed_threshold:.3f}m/s for {args.hold_steps} steps "
        f"({args.hold_steps * dt:.2f}s), with zero unsafe commands"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gifs").mkdir(exist_ok=True)
    results: dict[str, list[dict]] = {
        controller.name: [] for controller in controllers
    }
    rows = []
    for index in range(args.episodes):
        seed = args.seed + index
        record_gif = index < args.gif_episodes
        columns: dict[str, list[np.ndarray]] = {}
        episode_results: dict[str, dict] = {}
        for controller in controllers:
            episode = Episode(seed, args)
            result, views = run_episode(
                controller,
                episode,
                args,
                record_gif,
                desc=f"ep {index + 1}/{args.episodes} {controller.name}",
            )
            result["seed"] = seed
            result["target_position"] = episode.target_position.tolist()
            results[controller.name].append(result)
            episode_results[controller.name] = result
            columns[controller.name] = views
            rows.append(
                {
                    "controller": controller.name,
                    "seed": seed,
                    **{
                        key: value
                        for key, value in result.items()
                        if not key.endswith("_curve") and key != "target_position"
                    },
                }
            )

        line = " | ".join(
            f"{name} {'OK ' if result['success'] else 'no '}"
            f"d={result['final_distance_m']:.3f} "
            f"a={result['final_swing_angle_deg']:.1f}deg "
            f"w={result['final_swing_rate_rad_s']:.2f}"
            for name, result in episode_results.items()
        )
        print(f"ep {index + 1:3d}/{args.episodes} seed {seed}: {line}")
        running = "  ".join(
            f"{name} {sum(r['success'] for r in values)}/{len(values)}"
            for name, values in results.items()
        )
        print(f"          success so far: {running}")
        if record_gif:
            path = args.output / "gifs" / f"episode_{seed}.gif"
            write_gif(path, columns, episode_results, args)
            print(f"          saved {path}")

    summaries = {name: summarise(values) for name, values in results.items()}
    print()
    print_table(summaries)
    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "ranking": [
                    "success_rate",
                    "time_to_success_s among successful episodes",
                    "swing diagnostics",
                ],
                "config": {
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(args).items()
                },
                "summary": summaries,
                "episodes": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.output / 'metrics.json'}, episodes.csv and gifs/")


if __name__ == "__main__":
    main()
