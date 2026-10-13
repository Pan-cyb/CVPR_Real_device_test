"""Planar RM65-B robot cart-pole balance task.

This is the robot-arm counterpart of DMC Cartpole Balance, not a wrapper
around DMC's slider-cart model.  The RM65-B moves a side-mounted passive hinge
along a horizontal rail by coordinated J2/J3/J5 motion.  A long, conspicuous
pole starts close to the upward vertical and must be kept upright.

The task deliberately has one scalar objective only: upright angle. Workspace
limits are physical guards and reaching 90 degrees is a terminal failure, not
an additional reward term. The matte-black board and side camera are shared
with the earlier RM65-B pendulum pre-study so simulation and the intended
physical setup have the same visual background.
"""

from __future__ import annotations

import collections
import math

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_env import specs

from .rm65b_pendulum import (
    BOARD_HALF_THICKNESS,
    BOARD_HALF_WIDTH,
    BOARD_Y,
    BRACKET_LENGTH,
    FOREARM,
    JOINT2_RANGE,
    JOINT3_RANGE,
    JOINT5_RANGE,
    PHYSICS_TIMESTEP,
    SHOULDER_HEIGHT,
    UPPER_ARM,
    WRIST_TO_FLANGE,
)

ARM_JOINTS = ("joint2", "joint3", "joint5")

# A horizontal Cartesian rail makes the first version a clean balance task.
# The visible robot still executes the motion through its full planar chain.
CART_HEIGHT = 0.42
CART_RANGE = (0.08, 0.30)
CART_CENTER = sum(CART_RANGE) / 2.0
TOOL_PITCH = math.radians(150.0)
FLANGE_TO_HINGE = WRIST_TO_FLANGE + BRACKET_LENGTH

# Passive printed pole: large enough to remain legible after 64x64 rendering.
HINGE_SIDE_OFFSET = 0.100
POLE_LENGTH = 0.300
TIP_RADIUS = 0.035
TIP_MASS = 0.120
ROD_WIDTH = 0.018
ROD_THICKNESS = 0.008
ROD_MASS = 0.040
# Deliberately damped printed hinge.  The value is high enough to remove the
# unrealistically abrupt free fall seen with the first 0.002 N m s/rad draft,
# but there is no spring or hard stop that can hold the pole upright by itself.
HINGE_DAMPING = 0.015
HINGE_FRICTION = 0.0002
# Candidate real hinges, (viscous damping N m s/rad, dry friction N m), chosen
# per environment with task_kwargs hinge_damping / hinge_friction (see
# hinge_task_kwargs). "damper" is the default above, reproduced by a
# TRD-CG5D-A-220 silicone gear damper driven 19:16 (about 0.0148).
# "bearing606" is a plain 606ZZ ball bearing: an estimate until measured.
HINGE_PRESETS = {
    "damper": (HINGE_DAMPING, HINGE_FRICTION),
    "bearing606": (0.0005, 0.001),
}


def hinge_parameters(spec: str | None) -> tuple[float, float]:
    """'damper', 'bearing606' or 'DAMPING,FRICTION' -> (N m s/rad, N m);
    None = the module's current HINGE_DAMPING / HINGE_FRICTION."""
    if spec is None:
        return HINGE_DAMPING, HINGE_FRICTION
    if spec in HINGE_PRESETS:
        return HINGE_PRESETS[spec]
    try:
        damping, friction = (float(value) for value in spec.split(","))
    except ValueError:
        raise ValueError(
            f"hinge {spec!r}: use {', '.join(HINGE_PRESETS)} or DAMPING,FRICTION"
        ) from None
    if damping < 0 or friction < 0:
        raise ValueError("hinge damping and friction must be non-negative")
    return damping, friction


def hinge_task_kwargs(spec: str | None) -> dict[str, float]:
    """task_kwargs entries for a hinge; None adds none (module default)."""
    if spec is None:
        return {}
    damping, friction = hinge_parameters(spec)
    return {"hinge_damping": damping, "hinge_friction": friction}

CONTROL_TIMESTEP = 0.02
TIME_LIMIT = 20.0
MAX_CART_SPEED = 0.45
MAX_CART_ACCEL = 6.0
# Stratified magnitudes used only by the privileged preview/data-coverage
# controller.  The environment action itself remains continuous in [-1, 1].
EXPLORATION_SPEED_LEVELS = np.asarray(
    [0.10, 0.18, 0.27, 0.36, 0.45], dtype=np.float64
)
# Initial lean, uniform in [-X, +X] deg; balance, not swing-up. Locked at 7 deg
# (2026-10-02): a privileged full-state LQR (evaluation/evaluate_cartpole_lqr.py)
# recovered every start up to 7.8 deg, and every failure beyond it ran the
# pivot into the end of the 22 cm rail (peak excursion ~1.3 cm per degree).
# The pivot-speed limit alone would allow ~14 deg; +/-30 deg is unrecoverable.
DEFAULT_INITIAL_ANGLE_RANGE_DEG = 7.0

# Side camera zoomed onto the robot and pole (2026-10-02; the pre-study
# camera at (0, -1.55, 0.48) is kept by the pendulum environment). The pole is
# ~26 px long in the 64x64 frames instead of ~16 px. Arm, base and the pole up
# to +/-90 deg at the rail centre stay in frame; only a pole lying horizontal
# outward at either rail end (an episode's terminal frame) leaves it. The
# board is taller than the pendulum's so it still fills the background.
CAMERA_POSITION = (0.19, -1.00, 0.41)
BOARD_CENTER_Z = 0.45
BOARD_HALF_HEIGHT = 0.62
FALL_ANGLE = math.pi / 2.0


def arm_pose_for_cart_x(cart_x: float) -> np.ndarray:
    """Closed-form planar IK for a fixed-height, fixed-pitch hinge path."""
    if not CART_RANGE[0] - 1e-9 <= cart_x <= CART_RANGE[1] + 1e-9:
        raise ValueError(f"cart x={cart_x:.4f} is outside {CART_RANGE}")

    wrist_x = cart_x - FLANGE_TO_HINGE * math.sin(TOOL_PITCH)
    wrist_z = CART_HEIGHT - FLANGE_TO_HINGE * math.cos(TOOL_PITCH)
    vertical = wrist_z - SHOULDER_HEIGHT
    radius_sq = wrist_x * wrist_x + vertical * vertical
    cosine_elbow = (
        radius_sq - UPPER_ARM * UPPER_ARM - FOREARM * FOREARM
    ) / (2.0 * UPPER_ARM * FOREARM)
    if abs(cosine_elbow) > 1.0 + 1e-9:
        raise ValueError(f"cart x={cart_x:.4f} is outside the arm workspace")
    cosine_elbow = float(np.clip(cosine_elbow, -1.0, 1.0))
    joint3 = math.acos(cosine_elbow)
    joint2 = math.atan2(wrist_x, vertical) - math.atan2(
        FOREARM * math.sin(joint3), UPPER_ARM + FOREARM * math.cos(joint3)
    )
    joint5 = TOOL_PITCH - joint2 - joint3
    pose = np.asarray([joint2, joint3, joint5], dtype=np.float64)
    for value, bounds, name in zip(
        pose, (JOINT2_RANGE, JOINT3_RANGE, JOINT5_RANGE), ARM_JOINTS
    ):
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(
                f"IK produced unsafe {name}={math.degrees(value):.1f} deg"
            )
    return pose


def make_model_xml(
    hinge_damping: float | None = None, hinge_friction: float | None = None
) -> str:
    """None takes the module's HINGE_DAMPING / HINGE_FRICTION at call time."""
    damping = HINGE_DAMPING if hinge_damping is None else float(hinge_damping)
    friction = HINGE_FRICTION if hinge_friction is None else float(hinge_friction)
    flange = WRIST_TO_FLANGE
    hinge = FLANGE_TO_HINGE
    rod_half = (POLE_LENGTH - TIP_RADIUS) / 2.0
    return f"""
<mujoco model="rm65b_cartpole">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{PHYSICS_TIMESTEP}" integrator="implicitfast"/>
  <visual>
    <global offwidth="512" offheight="512"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="300" height="300" mark="edge" markrgb="0.2 0.3 0.4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
    <material name="robot" rgba="0.82 0.82 0.80 1"/>
    <material name="joint" rgba="0.25 0.25 0.27 1"/>
    <material name="printed" rgba="0.95 0.95 0.95 1"/>
    <material name="rod" rgba="0.98 0.75 0.08 1"/>
    <material name="tip" rgba="0.05 0.90 0.95 1" specular="0.1" shininess="0.1"/>
    <material name="board" rgba="0.025 0.030 0.035 1" specular="0" shininess="0" reflectance="0"/>
  </asset>
  <default>
    <!-- The side offset keeps the pole physically clear of the arm.  Pole-arm
         contacts are disabled so neither simulation nor hardware can prop the
         pole against a link as a balance shortcut. -->
    <geom contype="0" conaffinity="0"/>
    <joint type="hinge" axis="0 1 0"/>
  </default>
  <worldbody>
    <camera name="side" pos="{CAMERA_POSITION[0]} {CAMERA_POSITION[1]} {CAMERA_POSITION[2]}" xyaxes="1 0 0 0 0 1" fovy="45"/>
    <light name="top" pos="0.4 -1.0 2.0" dir="-0.1 0.35 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid"/>
    <geom name="target_board" type="box"
          pos="0 {BOARD_Y} {BOARD_CENTER_Z}"
          size="{BOARD_HALF_WIDTH} {BOARD_HALF_THICKNESS} {BOARD_HALF_HEIGHT}"
          material="board"/>
    <body name="base">
      <geom name="base_plate" type="cylinder" pos="0 0 0.025" size="0.075 0.025" material="joint" mass="0.25"/>
      <geom name="base_column" type="cylinder" pos="0 0 0.12" size="0.0535 0.095" material="robot" mass="1.19"/>
      <geom name="shoulder_housing" type="cylinder" fromto="0 -0.060 {SHOULDER_HEIGHT} 0 0.060 {SHOULDER_HEIGHT}" size="0.065" material="joint" mass="0.01"/>
      <body name="upper_arm" pos="0 0 {SHOULDER_HEIGHT}" gravcomp="1">
        <joint name="joint2" range="{JOINT2_RANGE[0]} {JOINT2_RANGE[1]}" armature="0.10"/>
        <geom name="upper_arm_beam" type="capsule" fromto="0 0 0 0 0 {UPPER_ARM}" size="0.045" material="robot" mass="0.864"/>
        <geom name="joint2_cap" type="cylinder" fromto="0 -0.052 0 0 0.052 0" size="0.060" material="joint" mass="0.01"/>
        <body name="forearm" pos="0 0 {UPPER_ARM}" gravcomp="1">
          <joint name="joint3" range="{JOINT3_RANGE[0]} {JOINT3_RANGE[1]}" armature="0.08"/>
          <geom name="forearm_tube" type="capsule" fromto="0 0 0 0 0 {FOREARM}" size="0.040" material="robot" mass="0.529"/>
          <geom name="joint3_cap" type="cylinder" fromto="0 -0.047 0 0 0.047 0" size="0.052" material="joint" mass="0.01"/>
          <body name="wrist" pos="0 0 {FOREARM}" gravcomp="1">
            <joint name="joint5" range="{JOINT5_RANGE[0]} {JOINT5_RANGE[1]}" armature="0.03"/>
            <geom name="joint5_cap" type="cylinder" fromto="0 -0.043 0 0 0.043 0" size="0.047" material="joint" mass="0.01"/>
            <geom name="wrist_housing" type="capsule" fromto="0 0 0 0 0 {flange - 0.01}" size="0.035" material="robot" mass="0.284"/>
            <geom name="flange" type="cylinder" fromto="0 0 {flange - 0.01} 0 0 {flange}" size="0.025" material="joint" mass="0.05"/>
            <geom name="bracket" type="box" pos="0 {-HINGE_SIDE_OFFSET / 2} {(flange + hinge) / 2}" size="0.012 {HINGE_SIDE_OFFSET / 2 + 0.01} {BRACKET_LENGTH / 2}" material="printed" mass="0.15"/>
            <body name="pole" pos="0 {-HINGE_SIDE_OFFSET} {hinge}">
              <joint name="pole" limited="false" damping="{damping}" frictionloss="{friction}" armature="1e-5"/>
              <geom name="pole_rod" type="box" pos="0 0 {rod_half}" size="{ROD_WIDTH / 2} {ROD_THICKNESS / 2} {rod_half}" material="rod" mass="{ROD_MASS}"/>
              <geom name="pole_tip" type="sphere" pos="0 0 {POLE_LENGTH}" size="{TIP_RADIUS}" material="tip" mass="{TIP_MASS}"/>
              <geom name="hinge_hub" type="cylinder" fromto="0 -0.008 0 0 0.008 0" size="0.018" material="printed" mass="0.02"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="joint2" joint="joint2" kp="2500" dampratio="1" ctrlrange="{JOINT2_RANGE[0]} {JOINT2_RANGE[1]}"/>
    <position name="joint3" joint="joint3" kp="1800" dampratio="1" ctrlrange="{JOINT3_RANGE[0]} {JOINT3_RANGE[1]}"/>
    <position name="joint5" joint="joint5" kp="2500" dampratio="1" ctrlrange="{JOINT5_RANGE[0]} {JOINT5_RANGE[1]}"/>
  </actuator>
</mujoco>
"""


class Physics(mujoco.Physics):
    def arm_angles(self) -> np.ndarray:
        return self.named.data.qpos[list(ARM_JOINTS)].copy()

    def pole_angle(self) -> float:
        """Signed pole error from world-frame upward vertical."""
        angle = float(self.arm_angles().sum() + self.named.data.qpos["pole"])
        return math.atan2(math.sin(angle), math.cos(angle))

    def pendulum_angle(self) -> float:
        """Compatibility alias used by the generic environment preview."""
        return self.pole_angle()

    def pole_angular_velocity(self) -> float:
        joints = list(ARM_JOINTS) + ["pole"]
        return float(self.named.data.qvel[joints].sum())

    def pivot_position(self) -> np.ndarray:
        return self.named.data.xpos["pole"].copy()

    def cart_position(self) -> float:
        return float(self.pivot_position()[0])

    def tip_position(self) -> np.ndarray:
        return self.named.data.geom_xpos["pole_tip"].copy()

    def bob_position(self) -> np.ndarray:
        """Compatibility alias used by the generic environment preview."""
        return self.tip_position()

    def set_pole_angle(self, angle: float) -> None:
        self.named.data.qpos["pole"] = float(angle) - float(self.arm_angles().sum())
        self.named.data.qvel[:] = 0.0
        self.forward()


class CartPoleBalanceTask(base.Task):
    """Keep the side-mounted passive pole upright for the full episode."""

    def __init__(
        self,
        random=None,
        exploration_speed: float | None = None,
        initial_angle_range_deg: float = DEFAULT_INITIAL_ANGLE_RANGE_DEG,
        initial_angular_velocity: float = 0.0,
    ):
        super().__init__(random=random)
        if not 0.0 <= initial_angular_velocity <= 20.0:
            raise ValueError("initial_angular_velocity must be in [0, 20] rad/s")
        # World-model data uses wide initial angles AND rates; control
        # evaluation keeps the locked +/-7 deg range at rest (rate 0).
        self._initial_rate_range = float(initial_angular_velocity)
        if exploration_speed is not None and not 0.0 < exploration_speed <= MAX_CART_SPEED:
            raise ValueError(
                f"exploration_speed must be in (0, {MAX_CART_SPEED}], "
                f"got {exploration_speed}"
            )
        if not 0.0 < initial_angle_range_deg < 90.0:
            raise ValueError("initial_angle_range_deg must be in (0, 90)")
        self._fixed_explore_speed = (
            None if exploration_speed is None else float(exploration_speed)
        )
        self._initial_angle_range = math.radians(float(initial_angle_range_deg))
        self._cart_target = CART_CENTER
        self._cart_velocity = 0.0
        self._explore_speed = (
            float(EXPLORATION_SPEED_LEVELS[-1])
            if self._fixed_explore_speed is None
            else self._fixed_explore_speed
        )
        self.unsafe_commands = 0

    def action_spec(self, physics):
        del physics
        return specs.BoundedArray(
            shape=(1,),
            dtype=np.float64,
            minimum=-1.0,
            maximum=1.0,
            name="horizontal_pivot_velocity",
        )

    def initialize_episode(self, physics):
        # Controlled initialization: every episode starts from the same robot
        # pose and zero velocity.  Only the pole's initial lean is sampled.
        self._cart_target = CART_CENTER
        self._cart_velocity = 0.0
        self._explore_speed = (
            float(self.random.choice(EXPLORATION_SPEED_LEVELS))
            if self._fixed_explore_speed is None
            else self._fixed_explore_speed
        )
        pose = arm_pose_for_cart_x(self._cart_target)
        physics.named.data.qpos[list(ARM_JOINTS)] = pose
        lean = self.random.uniform(-self._initial_angle_range, self._initial_angle_range)
        physics.named.data.qpos["pole"] = -float(pose.sum()) + lean
        physics.named.data.qvel[:] = 0.0
        if self._initial_rate_range > 0.0:
            # Drawn only when enabled, so rate-0 episodes keep their
            # historical random stream. The pole is already falling: the
            # rate always points the way it leans (an upward "flick" against
            # the lean has no visible cause in the first frame). The arm is
            # at rest, so the hinge rate is also the world-frame rate.
            magnitude = self.random.uniform(0.0, self._initial_rate_range)
            physics.named.data.qvel["pole"] = math.copysign(magnitude, lean)
        physics.named.data.ctrl[list(ARM_JOINTS)] = pose
        self.unsafe_commands = 0
        super().initialize_episode(physics)
        physics.forward()

    def control_state(self) -> tuple[float, float]:
        return self._cart_target, self._cart_velocity

    def set_control_state(self, state: tuple[float, float]) -> None:
        self._cart_target = float(state[0])
        self._cart_velocity = float(state[1])

    @property
    def exploration_speed(self) -> float:
        """Fixed horizontal correction-speed ceiling for this episode."""
        return self._explore_speed

    @property
    def initial_angle_range_deg(self) -> float:
        return math.degrees(self._initial_angle_range)

    @property
    def initial_angular_velocity(self) -> float:
        """Initial pole rate magnitude is uniform in [0, this] rad/s, always
        in the direction the pole leans (already falling)."""
        return self._initial_rate_range

    def exploration_action(
        self, physics, rng: np.random.Generator, repeat: int
    ) -> np.ndarray:
        """Noisy bang-bang-style stabiliser for the visual setting preview.

        This privileged controller is only a data-coverage/preview helper.  It
        is not a reportable policy and does not change the environment reward.
        Full train/validation collection should later mix stabilised,
        perturbed and failed episodes explicitly.
        """
        del repeat
        angle = physics.pole_angle()
        angular_velocity = physics.pole_angular_velocity()
        centered = physics.cart_position() - CART_CENTER
        # Lead compensation reacts to the predicted near-future lean rather
        # than waiting for the current angle to become visibly large.
        predicted_angle = angle + 0.22 * angular_velocity
        balance_signal = 18.0 * predicted_angle - 3.0 * centered
        response = math.tanh(balance_signal)
        # Once the lean is visible, do not collapse every correction back to
        # an imperceptibly small command; the sampled speed level still bounds
        # the motion and the continuous response supplies intermediate values.
        if abs(predicted_angle) >= math.radians(0.5):
            response = math.copysign(max(abs(response), 0.35), response)
        desired_velocity = self._explore_speed * response
        desired_velocity += float(rng.normal(0.0, 0.005))
        return np.asarray(
            [np.clip(desired_velocity / MAX_CART_SPEED, -1.0, 1.0)],
            dtype=np.float64,
        )

    def before_step(self, action, physics):
        command = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        desired_velocity = command * MAX_CART_SPEED
        velocity_change = float(
            np.clip(
                desired_velocity - self._cart_velocity,
                -MAX_CART_ACCEL * CONTROL_TIMESTEP,
                MAX_CART_ACCEL * CONTROL_TIMESTEP,
            )
        )
        velocity = self._cart_velocity + velocity_change
        candidate = self._cart_target + velocity * CONTROL_TIMESTEP
        bounded = float(np.clip(candidate, *CART_RANGE))
        if bounded != candidate:
            self.unsafe_commands += 1
            velocity = 0.0
        self._cart_target = bounded
        self._cart_velocity = velocity
        physics.named.data.ctrl[list(ARM_JOINTS)] = arm_pose_for_cart_x(bounded)

    def get_observation(self, physics):
        observation = collections.OrderedDict()
        observation["arm_angles"] = physics.arm_angles()
        observation["arm_velocities"] = physics.named.data.qvel[
            list(ARM_JOINTS)
        ].copy()
        observation["pivot_x"] = np.asarray([physics.cart_position()])
        observation["pole_angle"] = np.asarray([physics.pole_angle()])
        observation["pole_velocity"] = np.asarray(
            [physics.pole_angular_velocity()]
        )
        return observation

    def get_reward(self, physics):
        # The only task objective: normalised (90 deg - absolute angle).
        angle = abs(physics.pole_angle())
        return float(np.clip(1.0 - angle / FALL_ANGLE, 0.0, 1.0))

    def get_termination(self, physics):
        """Terminate immediately when the pole reaches the horizontal."""
        if abs(physics.pole_angle()) >= FALL_ANGLE:
            return 0.0
        return None

def load(task: str = "cartpole", task_kwargs=None, environment_kwargs=None):
    if task != "cartpole":
        raise ValueError(f"Unknown RM65-B cart-pole task: {task}")
    task_kwargs = dict(task_kwargs or {})
    physics = Physics.from_xml_string(
        make_model_xml(task_kwargs.get("hinge_damping"), task_kwargs.get("hinge_friction"))
    )
    return control.Environment(
        physics,
        CartPoleBalanceTask(
            random=task_kwargs.get("random"),
            exploration_speed=task_kwargs.get("exploration_speed"),
            initial_angle_range_deg=task_kwargs.get(
                "initial_angle_range_deg", DEFAULT_INITIAL_ANGLE_RANGE_DEG
            ),
            initial_angular_velocity=task_kwargs.get("initial_angular_velocity", 0.0),
        ),
        time_limit=float(task_kwargs.get("time_limit", TIME_LIMIT)),
        control_timestep=CONTROL_TIMESTEP,
        **(environment_kwargs or {}),
    )
