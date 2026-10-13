"""Planar RealMan RM65-B carrying a passive 3D-printed pendulum (MuJoCo).

Side-view simulation of the real-robot task. The policy controls pitch joints
2, 3 and 5; joints 1, 4 and 6 are locked at zero. A world-frame guard confines
the final wrist link to a 120-degree sector centred on vertically down, so the
policy can use the wrist without turning it upward or wedging the bob against
the arm. The arm and pendulum remain in the camera's x-z plane.

Arm geometry follows RealMan's official RM65-B modified-DH parameters and
URDF: the shoulder is 240.5 mm above the base, the upper arm is 256 mm, the
forearm is 210 mm, and the final wrist-to-flange offset is 144 mm. The three
pitch joints are collinear in the side view. Links use conservative primitive
collision shapes, which remain legible at the 64x64 world-model input
resolution.

The pendulum hangs from a printed bracket 60 mm beyond the flange on a
printed sleeve hinge on a pin, parallel to the pitch axes (no bearing): a
14 x 6 mm PLA rod and a 60 mm printed sphere with ballast, 0.15 m from hinge
to bob centre, about 0.14 kg in total (period about 0.78 s). A side-mounted
bracket separates the pendulum from the arm's collision plane. The hinge is
unrestricted, so gravity can keep the rod vertical as the wrist moves without
a rotating local joint limit trapping it against a stop.

The camera also sees a matte black calibration board behind the arm and one
large red magnetic-style target disk.  The disk is sampled once per episode
from a reachable tool-centre-point (TCP) position and then remains fixed.  The
task is Reacher-like: move the printed bracket tip / pendulum hinge into that
visible disk in the board's x-z plane, while bringing the passive pendulum to
rest.  The disk is never a hidden simulator-only goal.

Actions are three normalised joint-velocity commands, integrated into J2/J3/J5
position targets like streamed joint commands through the RealMan controller.
A hinge workspace box and the downward-facing tool sector keep the pendulum
above the floor and away from the base. These are conservative experiment
limits, not replacements for the real controller's collision and joint-limit
protection.
"""

from __future__ import annotations

import collections
import math

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.utils import rewards
from dm_env import specs

# RealMan RM65-B planar kinematics (m), from the official MDH model and URDF.
SHOULDER_HEIGHT = 0.2405
UPPER_ARM = 0.256
FOREARM = 0.210
WRIST_TO_FLANGE = 0.144
# Official hardware joint ranges (rad): J2 +-130 deg, J3 +-135 deg, J5 +-128 deg.
JOINT2_RANGE = (math.radians(-130.0), math.radians(130.0))
JOINT3_RANGE = (math.radians(-135.0), math.radians(135.0))
JOINT5_RANGE = (math.radians(-128.0), math.radians(128.0))

# 3D-printed pendulum.
BRACKET_LENGTH = 0.060        # flange to hinge along the tool axis
HINGE_SIDE_OFFSET = 0.100     # side mount towards camera keeps bob off arm
PENDULUM_LENGTH = 0.150       # hinge to bob centre, scaled for the RM65-B
BOB_RADIUS = 0.030
BOB_MASS = 0.120
ROD_WIDTH = 0.014             # in the swing plane
ROD_THICKNESS = 0.006
ROD_MASS = 0.018
# PLA sleeve on a ~8 mm pin: Coulomb friction mu * N * r_pin with mu ~0.35
# and N ~ the pendulum weight (1.35 N) gives about 0.002 N m; air drag is
# the small viscous part. Calibrate both with the release test
# (viz/view_env.py --swing-test) on the printed part.
HINGE_DAMPING = 0.001         # N m s / rad
HINGE_FRICTION = 0.003        # N m

# Control.
PHYSICS_TIMESTEP = 0.002
CONTROL_TIMESTEP = 0.02       # 50 Hz; action repeat 2 gives 25 Hz frames
TIME_LIMIT = 20.0             # 1000 control steps, 500 stored at repeat 2
MAX_CONTROL_SPEED = np.asarray([1.20, 1.50, 1.80])  # rad/s for J2, J3, J5
# The executed joint velocity follows the command with bounded acceleration,
# like the controller's motion planner; step commands would otherwise kick
# the pendulum over the top.
MAX_CONTROL_ACCEL = np.asarray([6.0, 8.0, 12.0])  # rad/s^2; tune on hardware
# Conservative study envelope inside the official hardware ranges. The hinge
# (not merely the wrist centre) must stay in this side-view workspace.
SAFE_JOINT2 = (math.radians(-40.0), math.radians(40.0))
SAFE_JOINT3 = (math.radians(20.0), math.radians(120.0))
SAFE_JOINT5 = (math.radians(-125.0), math.radians(125.0))
WORKSPACE_X = (-0.40, 0.40)
WORKSPACE_Z = (0.22, 0.52)
TOOL_PITCH = math.pi
TOOL_PITCH_RANGE = (math.radians(120.0), math.radians(240.0))

# The real study uses a board parallel to the arm's side-view motion plane.
# It is behind the robot relative to the camera, hence y > 0.  Success is a
# 2-D x-z alignment: the physical board and the TCP deliberately have a safe
# depth separation and are not supposed to touch.
CAMERA_POSITION = (0.0, -1.55, 0.48)
BOARD_Y = 0.18
BOARD_CENTER_Z = 0.38
# The 1.50 m x 1.00 m board covers the camera-projected envelope of the
# guarded TCP workspace plus the complete 150 mm pendulum and 30 mm bob.
BOARD_HALF_WIDTH = 0.75
BOARD_HALF_HEIGHT = 0.50
BOARD_HALF_THICKNESS = 0.010
TARGET_PLANE_Y = BOARD_Y - BOARD_HALF_THICKNESS - 0.006
TARGET_RADIUS = 0.05          # 100 mm diameter, legible at 64 x 64
MIN_TARGET_DISTANCE = 0.15   # reject trivial episode starts
DEFAULT_TARGET_POSITION = (0.27, TARGET_PLANE_Y, 0.36)

SWING_ANGLE_TOLERANCE = math.radians(5.0)
SWING_RATE_TOLERANCE = 0.25
TCP_SPEED_TOLERANCE = 0.03
# Keep dense guidance deliberately subordinate to the conjunctive task gate.
# A weighted sum of distance, angle, rate and terminal speed would let one easy
# objective pay for violating another one. Here the smooth term is a soft AND
# (minimum), and it accounts for only one percent of the per-step reward.
REWARD_SHAPING_WEIGHT = 0.01

ARM_JOINTS = ("joint2", "joint3", "joint5")
CONTROLLED_JOINTS = ARM_JOINTS


def hinge_position(joint2: float, joint3: float, joint5: float) -> tuple[float, float]:
    """Planar forward kinematics of the pendulum hinge, x and z."""
    elbow_x = UPPER_ARM * math.sin(joint2)
    elbow_z = SHOULDER_HEIGHT + UPPER_ARM * math.cos(joint2)
    forearm_pitch = joint2 + joint3
    wrist_x = elbow_x + FOREARM * math.sin(forearm_pitch)
    wrist_z = elbow_z + FOREARM * math.cos(forearm_pitch)
    tool_pitch = forearm_pitch + joint5
    tool_length = WRIST_TO_FLANGE + BRACKET_LENGTH
    return (
        wrist_x + tool_length * math.sin(tool_pitch),
        wrist_z + tool_length * math.cos(tool_pitch),
    )


def project_hinge_to_target_plane(hinge_xz: np.ndarray) -> np.ndarray:
    """Project a hinge point onto the board along the side-camera ray."""
    camera = np.asarray(CAMERA_POSITION, dtype=np.float64)
    hinge = np.asarray(
        [hinge_xz[0], -HINGE_SIDE_OFFSET, hinge_xz[1]], dtype=np.float64
    )
    scale = (TARGET_PLANE_Y - camera[1]) / (hinge[1] - camera[1])
    projected = camera + scale * (hinge - camera)
    return projected[[0, 2]]


def target_fits_on_board(target_xz: np.ndarray) -> bool:
    """Keep the full red disk inside the visible board with a small margin."""
    margin = TARGET_RADIUS + 0.01
    return (
        abs(float(target_xz[0])) <= BOARD_HALF_WIDTH - margin
        and BOARD_CENTER_Z - BOARD_HALF_HEIGHT + margin
        <= float(target_xz[1])
        <= BOARD_CENTER_Z + BOARD_HALF_HEIGHT - margin
    )


def arm_target_is_safe(target: np.ndarray) -> bool:
    joint2, joint3, joint5 = (float(value) for value in target)
    x, z = hinge_position(joint2, joint3, joint5)
    tool = joint2 + joint3 + joint5
    return (
        SAFE_JOINT2[0] <= joint2 <= SAFE_JOINT2[1]
        and SAFE_JOINT3[0] <= joint3 <= SAFE_JOINT3[1]
        and SAFE_JOINT5[0] <= joint5 <= SAFE_JOINT5[1]
        and WORKSPACE_X[0] <= x <= WORKSPACE_X[1]
        and WORKSPACE_Z[0] <= z <= WORKSPACE_Z[1]
        and TOOL_PITCH_RANGE[0] <= tool <= TOOL_PITCH_RANGE[1]
    )


def joint_path_is_safe(start: np.ndarray, goal: np.ndarray, samples: int = 33) -> bool:
    """Conservative endpoint-to-endpoint guard for target sampling."""
    return all(
        arm_target_is_safe(start + fraction * (goal - start))
        for fraction in np.linspace(0.0, 1.0, samples)
    )


def make_model_xml() -> str:
    flange = WRIST_TO_FLANGE
    hinge = WRIST_TO_FLANGE + BRACKET_LENGTH
    rod_half = (PENDULUM_LENGTH - BOB_RADIUS) / 2.0
    return f"""
<mujoco model="rm65b_pendulum">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{PHYSICS_TIMESTEP}" integrator="implicitfast">
    <!-- Contacts: pendulum rod/bob (contype 3) hit the arm links
         (conaffinity 2) and the floor (1). Arm links have contype 0, so
         they never collide with each other; parent filtering is off so
         the pendulum also hits its own wrist. -->
    <flag filterparent="disable"/>
  </option>
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
    <material name="rod" rgba="0.95 0.72 0.10 1"/>
    <material name="board" rgba="0.025 0.030 0.035 1" specular="0" shininess="0" reflectance="0"/>
    <material name="target" rgba="0.90 0.035 0.025 1" specular="0" shininess="0" reflectance="0"/>
  </asset>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint type="hinge" axis="0 1 0"/>
  </default>
  <worldbody>
    <camera name="side" pos="{CAMERA_POSITION[0]} {CAMERA_POSITION[1]} {CAMERA_POSITION[2]}" xyaxes="1 0 0 0 0 1" fovy="45"/>
    <light name="top" pos="0.4 -1.0 2.0" dir="-0.1 0.35 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid" contype="1" conaffinity="1"/>
    <!-- The board and target are ordinary visible, non-colliding geoms.
         The target is part of every 64 x 64 model observation, just as the
         magnetic disk is visible to the real side camera. -->
    <geom name="target_board" type="box"
          pos="0 {BOARD_Y} {BOARD_CENTER_Z}"
          size="{BOARD_HALF_WIDTH} {BOARD_HALF_THICKNESS} {BOARD_HALF_HEIGHT}"
          material="board"/>
    <geom name="target" type="ellipsoid"
          pos="{DEFAULT_TARGET_POSITION[0]} {DEFAULT_TARGET_POSITION[1]} {DEFAULT_TARGET_POSITION[2]}"
          size="{TARGET_RADIUS} 0.005 {TARGET_RADIUS}"
          material="target"/>
    <body name="base">
      <geom name="base_plate" type="cylinder" pos="0 0 0.025" size="0.075 0.025" material="joint" mass="0.25"/>
      <geom name="base_column" conaffinity="2" type="cylinder" pos="0 0 0.12" size="0.0535 0.095" material="robot" mass="1.19"/>
      <geom name="shoulder_housing" conaffinity="2" type="cylinder" fromto="0 -0.060 {SHOULDER_HEIGHT} 0 0.060 {SHOULDER_HEIGHT}" size="0.065" material="joint" mass="0.01"/>
      <body name="upper_arm" pos="0 0 {SHOULDER_HEIGHT}" gravcomp="1">
        <joint name="joint2" range="{JOINT2_RANGE[0]} {JOINT2_RANGE[1]}" armature="0.10"/>
        <geom name="upper_arm_beam" conaffinity="2" type="capsule" fromto="0 0 0 0 0 {UPPER_ARM}" size="0.045" material="robot" mass="0.864"/>
        <geom name="joint2_cap" conaffinity="2" type="cylinder" fromto="0 -0.052 0 0 0.052 0" size="0.060" material="joint" mass="0.01"/>
        <body name="forearm" pos="0 0 {UPPER_ARM}" gravcomp="1">
          <joint name="joint3" range="{JOINT3_RANGE[0]} {JOINT3_RANGE[1]}" armature="0.08"/>
          <geom name="forearm_tube" conaffinity="2" type="capsule" fromto="0 0 0 0 0 {FOREARM}" size="0.040" material="robot" mass="0.529"/>
          <geom name="joint3_cap" conaffinity="2" type="cylinder" fromto="0 -0.047 0 0 0.047 0" size="0.052" material="joint" mass="0.01"/>
          <body name="wrist" pos="0 0 {FOREARM}" gravcomp="1">
            <joint name="joint5" range="{JOINT5_RANGE[0]} {JOINT5_RANGE[1]}" armature="0.03"/>
            <geom name="joint5_cap" conaffinity="2" type="cylinder" fromto="0 -0.043 0 0 0.043 0" size="0.047" material="joint" mass="0.01"/>
            <geom name="wrist_housing" conaffinity="2" type="capsule" fromto="0 0 0 0 0 {flange - 0.01}" size="0.035" material="robot" mass="0.284"/>
            <geom name="flange" conaffinity="2" type="cylinder" fromto="0 0 {flange - 0.01} 0 0 {flange}" size="0.025" material="joint" mass="0.05"/>
            <geom name="bracket" type="box" pos="0 {-HINGE_SIDE_OFFSET / 2} {(flange + hinge) / 2}" size="0.012 {HINGE_SIDE_OFFSET / 2 + 0.01} {BRACKET_LENGTH / 2}" material="printed" mass="0.15"/>
            <body name="pendulum" pos="0 {-HINGE_SIDE_OFFSET} {hinge}">
              <joint name="pendulum" limited="false" damping="{HINGE_DAMPING}" frictionloss="{HINGE_FRICTION}" armature="1e-5"/>
              <geom name="bob" type="sphere" pos="0 0 {-PENDULUM_LENGTH}" size="{BOB_RADIUS}" material="rod" mass="{BOB_MASS}" contype="3" conaffinity="1" solref="0.004 1"/>
              <geom name="rod" type="box" pos="0 0 {-rod_half}" size="{ROD_WIDTH / 2} {ROD_THICKNESS / 2} {rod_half}" material="rod" mass="{ROD_MASS}" contype="3" conaffinity="1" solref="0.004 1"/>
              <geom name="hinge_hub" type="cylinder" fromto="0 -0.006 0 0 0.006 0" size="0.015" material="printed" mass="0.02"/>
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

    def pendulum_angle(self) -> float:
        """Pendulum angle from the downward vertical (world frame)."""
        angle = float(
            self.arm_angles().sum() + self.named.data.qpos["pendulum"]
        )
        return math.atan2(math.sin(angle), math.cos(angle))

    def bob_position(self) -> np.ndarray:
        return self.named.data.geom_xpos["bob"].copy()

    def tcp_position(self) -> np.ndarray:
        """World position of the bracket tip / pendulum hinge."""
        return self.named.data.xpos["pendulum"].copy()

    def target_position(self) -> np.ndarray:
        """World position of the visible red disk centre."""
        return self.named.data.geom_xpos["target"].copy()

    def target_tcp_position(self) -> np.ndarray:
        """TCP-plane point whose camera ray meets the visible disk centre."""
        camera = np.asarray(CAMERA_POSITION, dtype=np.float64)
        target = self.target_position()
        scale = (-HINGE_SIDE_OFFSET - camera[1]) / (target[1] - camera[1])
        return camera + scale * (target - camera)

    def target_distance(self) -> float:
        """Camera-ray projected TCP distance in the board's x-z plane."""
        camera = np.asarray(CAMERA_POSITION, dtype=np.float64)
        tcp = self.tcp_position()
        scale = (TARGET_PLANE_Y - camera[1]) / (tcp[1] - camera[1])
        projected_tcp = camera + scale * (tcp - camera)
        delta = projected_tcp[[0, 2]] - self.target_position()[[0, 2]]
        return float(np.linalg.norm(delta))

    def tcp_speed(self) -> float:
        """Analytic planar TCP speed from arm joint rates (m/s)."""
        angles = self.arm_angles()
        rates = self.named.data.qvel[list(ARM_JOINTS)].copy()
        epsilon = 1e-6
        jacobian = np.empty((2, len(ARM_JOINTS)), dtype=np.float64)
        for column in range(len(ARM_JOINTS)):
            plus, minus = angles.copy(), angles.copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            jacobian[:, column] = (
                np.asarray(hinge_position(*plus))
                - np.asarray(hinge_position(*minus))
            ) / (2.0 * epsilon)
        return float(np.linalg.norm(jacobian @ rates))

    def swing_rate(self) -> float:
        """World-frame angular velocity of the pendulum (rad/s)."""
        joints = list(ARM_JOINTS) + ["pendulum"]
        return float(self.named.data.qvel[joints].sum())

    def goal_state(self, pose: np.ndarray) -> np.ndarray:
        """Physics state with the arm at ``pose`` and the pendulum hanging still."""
        saved = self.get_state().copy()
        self.named.data.qpos[list(ARM_JOINTS)] = pose
        self.named.data.qpos["pendulum"] = -float(np.sum(pose))
        self.named.data.qvel[:] = 0.0
        self.forward()
        goal = self.get_state().copy()
        self.set_state(saved)
        self.forward()
        return goal

    def render_state(
        self, state: np.ndarray, size: int, camera_id: int = 0, scene_option=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render ``state`` without disturbing the current state.

        Returns the image and the bob position in that state.
        """
        saved = self.get_state().copy()
        self.set_state(state)
        self.forward()
        image = self.render(size, size, camera_id=camera_id, scene_option=scene_option)
        bob = self.bob_position()
        self.set_state(saved)
        self.forward()
        return image, bob


class PendulumTask(base.Task):
    """Rapid visible-target reaching with a passive swinging payload.

    The TCP must enter the red disk in the board plane.  The main reward is a
    hard conjunction: target, angle, angular-rate and terminal TCP-speed
    tolerances must all hold on the same step. A small minimum-of-qualities
    term supplies dense CEM guidance without allowing one objective to
    compensate for another.
    Terminal success is evaluated separately with a continuous hold window
    and deadline; repeated gated reward therefore favours earlier settling.
    """

    def __init__(self, random=None):
        super().__init__(random=random)
        self._target = np.zeros(len(ARM_JOINTS))
        self._velocity = np.zeros(len(CONTROLLED_JOINTS))
        self.unsafe_commands = 0
        self._goal: np.ndarray | None = None
        self._speed = 1.0
        self._dwell = 0
        self._free_steps = 0
        self._free_action = np.zeros(len(CONTROLLED_JOINTS))
        self._last_explore_target: np.ndarray | None = None
        self._board_target_pose: np.ndarray | None = None
        self._target_visit_due = True

    def action_spec(self, physics):
        return specs.BoundedArray(
            shape=(len(CONTROLLED_JOINTS),),
            dtype=np.float64,
            minimum=-1.0,
            maximum=1.0,
            name="joint_velocity",
        )

    def initialize_episode(self, physics):
        while True:
            start = np.asarray([
                self.random.uniform(*SAFE_JOINT2),
                self.random.uniform(*SAFE_JOINT3),
                self.random.uniform(*SAFE_JOINT5),
            ])
            if arm_target_is_safe(start):
                break
        physics.named.data.qpos[list(ARM_JOINTS)] = start
        # Hang the pendulum near vertical, with a small random swing.
        physics.named.data.qpos["pendulum"] = -start.sum() + self.random.uniform(
            -0.3, 0.3
        )
        physics.named.data.qvel[:] = 0.0

        start_xz = project_hinge_to_target_plane(
            np.asarray(hinge_position(*start))
        )
        while True:
            board_target_pose = self._random_safe_pose(self.random)
            target_xz = project_hinge_to_target_plane(
                np.asarray(hinge_position(*board_target_pose))
            )
            if (
                target_fits_on_board(target_xz)
                and np.linalg.norm(target_xz - start_xz) >= MIN_TARGET_DISTANCE
                and joint_path_is_safe(start, board_target_pose)
            ):
                break
        physics.named.model.geom_pos["target"] = np.asarray(
            [target_xz[0], TARGET_PLANE_Y, target_xz[1]], dtype=np.float64
        )
        self._board_target_pose = board_target_pose

        self._target = start.copy()
        self._velocity = np.zeros(len(CONTROLLED_JOINTS))
        physics.named.data.ctrl[list(ARM_JOINTS)] = start
        self.unsafe_commands = 0
        self._goal = None
        self._dwell = 0
        self._free_steps = 0
        self._last_explore_target = None
        self._target_visit_due = True
        super().initialize_episode(physics)
        physics.forward()

    def control_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Commanded position target and velocity (for simulate-ahead planners)."""
        return self._target.copy(), self._velocity.copy()

    def set_control_state(self, state: tuple[np.ndarray, np.ndarray]) -> None:
        self._target, self._velocity = state[0].copy(), state[1].copy()

    def sample_goal_pose(self, rng: np.random.Generator) -> np.ndarray:
        """Return a reachable pose helper; it is not the task definition."""
        return self._random_safe_pose(rng)

    def target_pose(self) -> np.ndarray:
        """Pose used only to place the reachable visual target on reset."""
        if self._board_target_pose is None:
            raise RuntimeError("reset the environment before reading target_pose")
        return self._board_target_pose.copy()

    def _random_safe_pose(self, rng: np.random.Generator) -> np.ndarray:
        while True:
            pose = np.asarray([
                rng.uniform(*SAFE_JOINT2),
                rng.uniform(*SAFE_JOINT3),
                rng.uniform(*SAFE_JOINT5),
            ])
            if arm_target_is_safe(pose):
                return pose

    def exploration_action(
        self, physics, rng: np.random.Generator, repeat: int
    ) -> np.ndarray:
        """Random point-to-point exploration, the way the real arm is driven.

        Moves to random safe poses with trapezoidal joint profiles (random
        cruise speed, 30-100 % of the limit, braking at the acceleration
        limit), dwells for a random time so the pendulum swings freely, and
        occasionally holds a random velocity instead. Returns the normalised
        joint-velocity command for the next ``repeat`` control steps.
        """
        del physics
        stuck = (
            self._goal is not None
            and self._last_explore_target is not None
            and np.array_equal(self._last_explore_target, self._target)
            and self._dwell == 0
        )
        self._last_explore_target = self._target.copy()
        if stuck:
            # The guard stopped the arm (a joint-space path crossed the
            # workspace boundary between two safe poses).
            self._goal = None
        if self._free_steps > 0:
            self._free_steps -= 1
            return self._free_action.copy()
        if self._dwell > 0:
            self._dwell -= 1
            return np.zeros(len(CONTROLLED_JOINTS))
        horizon = CONTROL_TIMESTEP * repeat
        reached = (
            self._goal is not None
            and np.all(np.abs(self._goal - self._target) < 2e-3)
            and np.all(np.abs(self._velocity) < 0.05)
        )
        if self._goal is None or reached:
            reached_board_target = (
                reached
                and self._board_target_pose is not None
                and np.all(np.abs(self._goal - self._board_target_pose) < 2e-3)
            )
            if reached_board_target:
                # Give the passive load 2--5 seconds to settle so reward-head
                # training sees genuine gated positives, not only approaches.
                self._dwell = int(rng.integers(50, 126))
                self._goal = None
                return np.zeros(len(CONTROLLED_JOINTS))
            if reached and rng.random() < 0.5:
                self._dwell = int(rng.integers(2, 25))
                self._goal = None
                return np.zeros(len(CONTROLLED_JOINTS))
            if not self._target_visit_due and rng.random() < 0.15:
                self._free_steps = int(rng.integers(3, 15))
                self._free_action = rng.uniform(
                    -1.0, 1.0, len(CONTROLLED_JOINTS)
                )
                self._goal = None
                return self._free_action.copy()
            # Deliberately visit the visible target often enough that offline
            # data contain positive reward and stable-arrival examples.  The
            # policy still receives only RGB; this is collection behaviour.
            choose_target = self._board_target_pose is not None and (
                self._target_visit_due or rng.random() < 0.35
            )
            if choose_target:
                self._goal = self._board_target_pose.copy()
                self._target_visit_due = False
            else:
                self._goal = self._random_safe_pose(rng)
            self._speed = float(rng.uniform(0.3, 1.0))
        delta = self._goal - self._target
        speed = np.minimum.reduce([
            self._speed * MAX_CONTROL_SPEED,
            np.sqrt(2.0 * MAX_CONTROL_ACCEL * np.abs(delta)),
            np.abs(delta) / horizon,
        ])
        action = np.sign(delta) * speed / MAX_CONTROL_SPEED
        # Straight joint-space paths between safe poses can leave the
        # workspace box; brake and pick a new goal instead.
        candidate = self._target + action * MAX_CONTROL_SPEED * horizon
        if not arm_target_is_safe(candidate):
            self._goal = None
            return np.zeros(len(CONTROLLED_JOINTS))
        return action

    def before_step(self, action, physics):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        change = np.clip(
            action * MAX_CONTROL_SPEED - self._velocity,
            -MAX_CONTROL_ACCEL * CONTROL_TIMESTEP,
            MAX_CONTROL_ACCEL * CONTROL_TIMESTEP,
        )
        velocity = self._velocity + change
        candidate = self._target + velocity * CONTROL_TIMESTEP
        # Robot-side guard: all policy joints stop before the hinge workspace
        # or the 120-degree downward-facing tool sector can be violated.
        if arm_target_is_safe(candidate):
            self._target, self._velocity = candidate, velocity
        else:
            self.unsafe_commands += 1
            self._velocity = np.zeros(len(CONTROLLED_JOINTS))
        physics.named.data.ctrl[list(ARM_JOINTS)] = self._target

    def get_observation(self, physics):
        observation = collections.OrderedDict()
        observation["arm_angles"] = physics.arm_angles()
        observation["arm_velocities"] = physics.named.data.qvel[
            list(ARM_JOINTS)
        ].copy()
        observation["pendulum_angle"] = np.asarray([physics.pendulum_angle()])
        observation["pendulum_velocity"] = physics.named.data.qvel[
            ["pendulum"]
        ].copy()
        return observation

    def get_reward(self, physics):
        if self.unsafe_commands > 0:
            return 0.0
        distance = physics.target_distance()
        swing_angle = abs(physics.pendulum_angle())
        swing_rate = abs(physics.swing_rate())
        tcp_speed = physics.tcp_speed()
        reach_quality = rewards.tolerance(
            distance,
            bounds=(0.0, TARGET_RADIUS),
            margin=0.20,
            sigmoid="long_tail",
            value_at_margin=0.01,
        )
        angle_quality = rewards.tolerance(
            swing_angle,
            bounds=(0.0, SWING_ANGLE_TOLERANCE),
            margin=math.radians(45.0),
            sigmoid="long_tail",
            value_at_margin=0.01,
        )
        rate_quality = rewards.tolerance(
            swing_rate,
            bounds=(0.0, SWING_RATE_TOLERANCE),
            margin=2.0,
            sigmoid="long_tail",
            value_at_margin=0.01,
        )
        speed_quality = rewards.tolerance(
            tcp_speed,
            bounds=(0.0, TCP_SPEED_TOLERANCE),
            margin=0.30,
            sigmoid="long_tail",
            value_at_margin=0.01,
        )
        # Terminal speed should guide braking near the disk, not reward slow
        # creeping throughout transport. It enters the bottleneck gradually
        # between 2R and R and is fully active inside the target.
        braking_blend = float(
            np.clip((2.0 * TARGET_RADIUS - distance) / TARGET_RADIUS, 0.0, 1.0)
        )
        braking_quality = 1.0 - braking_blend * (1.0 - speed_quality)
        soft_and = min(
            reach_quality, angle_quality, rate_quality, braking_quality
        )
        hard_and = float(
            distance <= TARGET_RADIUS
            and swing_angle <= SWING_ANGLE_TOLERANCE
            and swing_rate <= SWING_RATE_TOLERANCE
            and tcp_speed <= TCP_SPEED_TOLERANCE
        )
        return float(
            (1.0 - REWARD_SHAPING_WEIGHT) * hard_and
            + REWARD_SHAPING_WEIGHT * soft_and
        )


def load(task: str = "pendulum", task_kwargs=None, environment_kwargs=None):
    if task != "pendulum":
        raise ValueError(f"Unknown rm65b task: {task}")
    task_kwargs = dict(task_kwargs or {})
    physics = Physics.from_xml_string(make_model_xml())
    return control.Environment(
        physics,
        PendulumTask(random=task_kwargs.get("random")),
        time_limit=TIME_LIMIT,
        control_timestep=CONTROL_TIMESTEP,
        **(environment_kwargs or {}),
    )
