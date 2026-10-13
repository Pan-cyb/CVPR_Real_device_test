"""Privileged full-state LQR for the RM65-B robot cart-pole.

Used for the controllability check (evaluation/evaluate_cartpole_lqr.py) and
as an optional data-collection policy (utils/collect_dmc.py
``--cartpole-policy lqr``). It reads the true simulator state, so it is never
a reported visual controller.

The model is linearised numerically around the upright equilibrium at the
agent cadence (``action_repeat`` control steps per action), with the hinge's
Coulomb friction removed in the instance used for linearisation because it is
not differentiable at rest.

Linear algebra runs in float64 torch, not numpy: importing jointgv loads
torch's OpenMP runtime, and numpy's BLAS/LAPACK would then load a second one
and abort the process on the project's Windows environment.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .rm65b_cartpole import (
    CART_CENTER,
    CART_RANGE,
    CONTROL_TIMESTEP,
    MAX_CART_ACCEL,
    PHYSICS_TIMESTEP,
    CartPoleBalanceTask,
    arm_pose_for_cart_x,
    hinge_task_kwargs,
    load,
)

SUBSTEPS = round(CONTROL_TIMESTEP / PHYSICS_TIMESTEP)
# Typical scales normalising Q: pole angle (rad), pole rate (rad/s), pivot
# offset from the rail centre (m), commanded pivot velocity (m/s).
DEFAULT_SCALES = (math.radians(3.0), 0.5, 0.05, 0.3)
# Smallest peak rail excursion in the +/-7 deg tuning grid (2026-10-02).
DEFAULT_R = 1.0


def get_state(physics, task) -> np.ndarray:
    target, velocity = task.control_state()
    return np.concatenate(
        (physics.data.qpos, physics.data.qvel, [target, velocity])
    ).astype(np.float64)


def set_state(physics, task, state: np.ndarray) -> None:
    nq, nv = physics.model.nq, physics.model.nv
    physics.data.qpos[:] = state[:nq]
    physics.data.qvel[:] = state[nq : nq + nv]
    task.set_control_state((state[-2], state[-1]))
    physics.forward()


def advance(physics, task, command: float, repeat: int) -> None:
    """One agent step, exactly as control.Environment.step does it."""
    for _ in range(repeat):
        task.before_step(np.asarray([command]), physics)
        physics.step(SUBSTEPS)


def spectral_radius(matrix: np.ndarray) -> float:
    return float(torch.linalg.eigvals(torch.from_numpy(matrix)).abs().max())


class Linearisation:
    """Finite-difference x' = A x + B u around the upright equilibrium.

    State x = [qpos (J2, J3, J5, pole), qvel, pivot target, pivot velocity].
    ``physics`` must be a dedicated instance: its friction is zeroed.
    """

    def __init__(self, physics, repeat: int, eps_state: float = 1e-5,
                 eps_action: float = 1e-4):
        self.physics = physics
        self.physics.model.dof_frictionloss[:] = 0.0
        self.task = CartPoleBalanceTask(random=0)
        self.repeat = repeat
        nq, nv = physics.model.nq, physics.model.nv
        self.nq, self.nv = nq, nv
        pose = arm_pose_for_cart_x(CART_CENTER)
        equilibrium = np.zeros(nq + nv + 2)
        equilibrium[:3] = pose
        equilibrium[3] = -float(pose.sum())       # pole straight up
        equilibrium[-2] = CART_CENTER
        # The servos sag slightly under the pole's weight: settle the arm with
        # the pole held exactly upright and at rest.
        for _ in range(200):
            equilibrium = self._step(equilibrium, 0.0)
            equilibrium[3] = -float(equilibrium[:3].sum())
            equilibrium[nq + 3] = -float(equilibrium[nq : nq + 3].sum())
        self.equilibrium = equilibrium
        self.sag = equilibrium[:3] - pose
        size = equilibrium.size
        self.A = np.zeros((size, size))
        for column in range(size):
            delta = np.zeros(size)
            delta[column] = eps_state
            self.A[:, column] = (
                self._step(equilibrium + delta, 0.0)
                - self._step(equilibrium - delta, 0.0)
            ) / (2.0 * eps_state)
        self.B = (
            (self._step(equilibrium, eps_action) - self._step(equilibrium, -eps_action))
            / (2.0 * eps_action)
        )[:, None]
        self.drift = float(np.abs(self._step(equilibrium, 0.0) - equilibrium).max())

    def _step(self, state: np.ndarray, command: float) -> np.ndarray:
        set_state(self.physics, self.task, state)
        advance(self.physics, self.task, command, self.repeat)
        return get_state(self.physics, self.task)

    def output_matrix(self) -> np.ndarray:
        """Rows: pole world angle, pole world rate, pivot target, commanded
        pivot velocity (all linear in the state)."""
        nq, nv = self.nq, self.nv
        C = np.zeros((4, self.equilibrium.size))
        C[0, :nq] = 1.0               # world angle = arm pitch sum + pole
        C[1, nq : nq + nv] = 1.0
        C[2, -2] = 1.0
        C[3, -1] = 1.0
        return C

    def unstable_rate(self, dt: float) -> float:
        """Growth rate (1/s) of the unstable pole mode with zero command."""
        return float(math.log(spectral_radius(self.A)) / dt)

    def reference_state(self, x: float, velocity: float = 0.0) -> np.ndarray:
        """Upright state with the pivot at ``x`` moving at ``velocity``."""
        nq = self.nq
        state = self.equilibrium.copy()
        state[:3] = arm_pose_for_cart_x(x) + self.sag
        state[3] = -float(state[:3].sum())
        step = 1e-4
        low = max(CART_RANGE[0], x - step)
        high = min(CART_RANGE[1], x + step)
        joint_rate = (arm_pose_for_cart_x(high) - arm_pose_for_cart_x(low)) / (high - low)
        state[nq : nq + 3] = joint_rate * velocity
        state[nq + 3] = -float(state[nq : nq + 3].sum())
        state[-2] = x
        state[-1] = velocity
        return state


def lqr_gain(
    lin: Linearisation,
    scales,
    r_weight: float,
    iterations: int = 200000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Discrete-time LQR gain K (u = -K x) by Riccati fixed-point iteration."""
    A = torch.from_numpy(lin.A)
    B = torch.from_numpy(lin.B)
    C = torch.from_numpy(lin.output_matrix())
    W = torch.diag(torch.from_numpy(1.0 / np.square(np.asarray(scales, dtype=np.float64))))
    Q = C.T @ W @ C + 1e-8 * torch.eye(A.shape[0], dtype=torch.float64)
    R = torch.tensor([[r_weight]], dtype=torch.float64)
    P = Q.clone()
    for _ in range(iterations):
        gain = torch.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        updated = Q + A.T @ P @ (A - B @ gain)
        updated = 0.5 * (updated + updated.T)
        if float((updated - P).abs().max()) <= tolerance * max(1.0, float(P.abs().max())):
            P = updated
            break
        P = updated
    else:
        raise ValueError("Riccati iteration did not converge")
    return torch.linalg.solve(R + B.T @ P @ B, B.T @ P @ A).numpy()


def closed_loop_radius(lin: Linearisation, gain: np.ndarray) -> float:
    return spectral_radius(
        (torch.from_numpy(lin.A) - torch.from_numpy(lin.B) @ torch.from_numpy(gain)).numpy()
    )


def lqr_command(gain: np.ndarray, error: np.ndarray) -> float:
    """Clipped u = -K e (elementwise sum: no numpy BLAS, see module doc)."""
    return float(np.clip(-float((gain[0] * error).sum()), -1.0, 1.0))


class CartPoleLQR:
    """Full-state LQR with an optional moving pivot reference."""

    def __init__(self, action_repeat: int, r_weight: float = DEFAULT_R,
                 scales=DEFAULT_SCALES, hinge: str | None = None):
        # Dedicated linearisation instance with the controlled plant's hinge
        # (rm65b_cartpole.hinge_parameters; None = the default).
        env = load(task_kwargs={"random": 0, **hinge_task_kwargs(hinge)})
        env.reset()
        self.lin = Linearisation(env.physics, action_repeat)
        self.gain = lqr_gain(self.lin, scales, r_weight)
        self.r_weight = float(r_weight)
        self.scales = tuple(float(value) for value in scales)

    def command(self, physics, task, reference: float = CART_CENTER,
                reference_velocity: float = 0.0) -> float:
        target = (
            self.lin.equilibrium
            if reference == CART_CENTER and reference_velocity == 0.0
            else self.lin.reference_state(reference, reference_velocity)
        )
        return lqr_command(self.gain, get_state(physics, task) - target)


def keep_off_rail_ends(
    command: float,
    task,
    action_repeat: int,
    rng: np.random.Generator,
    margin: float = 0.02,
) -> float:
    """Collection helper: never drive the pivot into a rail end-stop.

    A saturated correction otherwise reaches the end of the 22 cm rail within
    a few frames; the workspace guard then zeroes the velocity and the arm
    stands still while the pole falls, which is not what a failing
    correction looks like. If braking now at the acceleration limit would
    still stop within ``margin`` of an end, a command towards that end is
    replaced by one of random size (30-100 % of full speed) back towards the
    rail, so the arm keeps working near the end instead of freezing there.
    """
    target, velocity = task.control_state()
    horizon = action_repeat * CONTROL_TIMESTEP
    stop = target + velocity * horizon + velocity * abs(velocity) / (2.0 * MAX_CART_ACCEL)
    if stop >= CART_RANGE[1] - margin and command > 0.0:
        return -float(rng.uniform(0.3, 1.0))
    if stop <= CART_RANGE[0] + margin and command < 0.0:
        return float(rng.uniform(0.3, 1.0))
    return float(command)


class ImperfectCorrector:
    """Collection helper: a casual horizontal correction that usually fails.

    It reacts to the pole like someone trying to balance it: it moves the
    pivot towards the lean with a lead on the lean rate and a weak pull back
    to the rail centre, but every episode draws its own gain (often too weak
    or too strong), lead, reaction delay (0-4 steps), speed ceiling and
    smooth command noise. The arm therefore keeps chasing the pole, over- and
    under-shooting, until it falls (occasionally it holds it for a while).
    Rail ends are avoided with ``keep_off_rail_ends``.
    """

    def __init__(self, action_repeat: int):
        self.action_repeat = action_repeat
        self.dt = action_repeat * CONTROL_TIMESTEP

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        # The privileged LQR uses about 14 per rad on the lean; draw around it.
        self.gain = float(rng.uniform(3.0, 25.0))          # command per rad
        self.lead = float(rng.uniform(0.0, 0.25))          # s
        self.centring = float(rng.uniform(0.0, 20.0))      # command per m
        self.delay = int(rng.integers(0, 5))               # agent steps
        self.ceiling = float(rng.uniform(0.4, 1.0))        # share of full speed
        self.noise_std = float(rng.uniform(0.05, 0.4))
        self.noise_decay = math.exp(-self.dt / 0.3)
        self.noise = 0.0
        self.history: list[tuple[float, float, float]] = []

    def parameters(self) -> dict[str, float]:
        return {
            "gain": self.gain, "lead": self.lead, "centring": self.centring,
            "delay": float(self.delay), "ceiling": self.ceiling,
            "noise_std": self.noise_std,
        }

    def action(self, physics, task) -> np.ndarray:
        self.history.append(
            (physics.pole_angle(), physics.pole_angular_velocity(), physics.cart_position())
        )
        angle, rate, x = self.history[max(0, len(self.history) - 1 - self.delay)]
        command = self.gain * (angle + self.lead * rate) - self.centring * (x - CART_CENTER)
        command = float(np.clip(command, -self.ceiling, self.ceiling))
        self.noise = self.noise_decay * self.noise + self.noise_std * math.sqrt(
            1.0 - self.noise_decay**2
        ) * float(self.rng.normal())
        command = float(np.clip(command + self.noise, -1.0, 1.0))
        command = keep_off_rail_ends(command, task, self.action_repeat, self.rng)
        return np.asarray([command], dtype=np.float32)


class LQRCollectionPolicy:
    """Data-coverage policy: LQR tracking a slowly moving random pivot
    reference, plus Ornstein-Uhlenbeck command noise.

    The extra pole motion comes from the robot's own recorded actions, so a
    world model can predict it from the action sequence (unlike unobserved
    pushes). With ``reference_amplitude=0`` and ``noise_std=0`` it is the
    pure LQR, which recovers within about 0.5 s and then holds the pole
    almost still. The first reference segment is always the rail centre, so
    the initial recovery is undisturbed.
    """

    def __init__(self, controller: CartPoleLQR, action_repeat: int,
                 reference_amplitude: float = 0.0,
                 segment_seconds: tuple[float, float] = (0.8, 2.0),
                 noise_std: float = 0.0, noise_tau: float = 0.3,
                 avoid_rail_ends: bool = False):
        if reference_amplitude < 0 or noise_std < 0:
            raise ValueError("reference amplitude and noise must be >= 0")
        limit = min(CART_RANGE[1] - CART_CENTER, CART_CENTER - CART_RANGE[0])
        if reference_amplitude >= limit:
            raise ValueError(f"reference amplitude must stay below {limit:.3f} m")
        if not 0 < segment_seconds[0] <= segment_seconds[1]:
            raise ValueError("segment seconds must satisfy 0 < min <= max")
        self.controller = controller
        self.action_repeat = action_repeat
        self.avoid_rail_ends = avoid_rail_ends
        self.dt = action_repeat * CONTROL_TIMESTEP
        self.amplitude = float(reference_amplitude)
        self.segment_seconds = segment_seconds
        self.noise_std = float(noise_std)
        self.noise_decay = math.exp(-self.dt / noise_tau)

    def reset(self, rng: np.random.Generator, noise_std: float | None = None) -> None:
        """New episode; ``noise_std`` overrides the command-noise level."""
        if noise_std is not None:
            if noise_std < 0:
                raise ValueError("noise must be >= 0")
            self.noise_std = float(noise_std)
        self.rng = rng
        self.start = self.goal = CART_CENTER
        self.elapsed = 0.0
        self.duration = float(rng.uniform(*self.segment_seconds))
        self.noise = 0.0

    def _reference(self) -> tuple[float, float]:
        if self.elapsed >= self.duration:
            self.start = self.goal
            self.goal = CART_CENTER + float(
                self.rng.uniform(-self.amplitude, self.amplitude)
            )
            self.elapsed = 0.0
            self.duration = float(self.rng.uniform(*self.segment_seconds))
        phase = self.elapsed / self.duration
        # Cosine blend: zero reference velocity at both segment ends.
        blend = 0.5 - 0.5 * math.cos(math.pi * phase)
        rate = 0.5 * math.pi * math.sin(math.pi * phase) / self.duration
        reference = self.start + (self.goal - self.start) * blend
        velocity = (self.goal - self.start) * rate
        return reference, velocity

    def action(self, physics, task) -> np.ndarray:
        reference, velocity = self._reference()
        command = self.controller.command(physics, task, reference, velocity)
        if self.noise_std > 0:
            self.noise = self.noise_decay * self.noise + self.noise_std * math.sqrt(
                1.0 - self.noise_decay**2
            ) * float(self.rng.normal())
            command += self.noise
        self.elapsed += self.dt
        command = float(np.clip(command, -1.0, 1.0))
        if self.avoid_rail_ends:
            command = keep_off_rail_ends(command, task, self.action_repeat, self.rng)
        return np.asarray([command], dtype=np.float32)
