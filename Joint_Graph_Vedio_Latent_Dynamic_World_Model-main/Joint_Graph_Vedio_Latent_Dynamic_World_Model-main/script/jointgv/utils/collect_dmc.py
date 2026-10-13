from __future__ import annotations

import argparse
from importlib.metadata import version
import os
import platform
from pathlib import Path
import sys

# GLFW is the supported local Windows backend; EGL is preferable on headless
# Linux machines. Users can still override this before launching the script.
os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Windows" else "egl")

import numpy as np
import mujoco
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv.envs import load_environment
from jointgv.envs.rm65b_cartpole import (
    CONTROL_TIMESTEP as CONTROL_TIMESTEP_CARTPOLE,
    DEFAULT_INITIAL_ANGLE_RANGE_DEG,
    EXPLORATION_SPEED_LEVELS,
    hinge_parameters,
    hinge_task_kwargs,
)
from jointgv.dmc_protocol import (
    DREAMERV3_DMC_VISION,
    EXPLICIT_GRAPH_COORDINATE_FRAME,
    EXPLICIT_GRAPH_SCHEMA_VERSION,
    resolve_camera,
    validate_dreamerv3_vision,
)


def build_graph_schema(
    physics, domain: str, graph_nodes: int
) -> list[tuple[str, int, str]]:
    """Return stable role-ordered physical entities for graph supervision."""
    body_ids = list(range(1, int(physics.model.nbody)))
    if domain == "reacher":
        # DMC Reacher's target is a world-attached geom rather than a body.
        # Include an explicit base and target so identities match the intended
        # base/link_1/link_2/target physical schema.
        schema: list[tuple[str, int, str]] = [("body", 0, "base")]
        for index, body_id in enumerate(body_ids, start=1):
            body_name = str(
                physics.model.id2name(body_id, "body") or ""
            ).lower()
            role = "end_effector" if "finger" in body_name else f"link_{index}"
            schema.append(("body", body_id, role))
        target_geom = next(
            (
                geom_id
                for geom_id in range(int(physics.model.ngeom))
                if "target" in str(
                    physics.model.id2name(geom_id, "geom") or ""
                ).lower()
            ),
            None,
        )
        if target_geom is not None:
            schema.append(("geom", int(target_geom), "target"))
    elif domain == "rm65b":
        body_by_name = {
            str(physics.model.id2name(body_id, "body")): body_id
            for body_id in body_ids
        }
        geom_by_name = {
            str(physics.model.id2name(geom_id, "geom")): geom_id
            for geom_id in range(int(physics.model.ngeom))
        }
        arm_schema = [
            ("body", body_by_name["base"], "base"),
            ("body", body_by_name["upper_arm"], "link_1"),
            ("body", body_by_name["forearm"], "link_2"),
            ("body", body_by_name["wrist"], "link_3"),
        ]
        if "pole" in body_by_name:
            # Robot cart-pole: the pole body is represented at the hinge hub,
            # with the bright tip as a separate rigidly connected node.
            schema = arm_schema + [
                ("body", body_by_name["pole"], "end_effector"),
                ("geom", geom_by_name["pole_tip"], "object_1"),
            ]
        else:
            # Legacy visible-target pendulum transport.  The TCP is the hinge,
            # while the bob and static red disk remain separate graph nodes.
            schema = arm_schema + [
                ("body", body_by_name["pendulum"], "end_effector"),
                ("geom", geom_by_name["bob"], "object_1"),
                ("geom", geom_by_name["target"], "target"),
            ]
    else:
        schema = []
        for body_id in body_ids:
            name = physics.model.id2name(body_id, "body") or f"body_{body_id}"
            schema.append(("body", body_id, str(name)))
    if len(schema) > graph_nodes:
        raise ValueError(
            f"Graph schema has {len(schema)} physical nodes but --graph-nodes "
            f"is only {graph_nodes}"
        )
    return schema


def _reference_geom_ids(
    physics, kind: str, object_id: int, role_name: str | None = None
) -> np.ndarray:
    if kind == "geom":
        return np.asarray([object_id], dtype=np.int64)
    geom_body = np.asarray(physics.model.geom_bodyid)
    geom_ids = np.flatnonzero(geom_body == object_id).astype(np.int64)
    if role_name == "end_effector":
        hinge_ids = [
            geom_id
            for geom_id in geom_ids
            if str(
                physics.model.id2name(int(geom_id), "geom") or ""
            ).lower() == "hinge_hub"
        ]
        if hinge_ids:
            return np.asarray(hinge_ids, dtype=np.int64)
    if role_name == "base":
        root_ids = [
            geom_id
            for geom_id in geom_ids
            if str(physics.model.id2name(int(geom_id), "geom") or "").lower()
            == "root"
        ]
        if root_ids:
            return np.asarray(root_ids, dtype=np.int64)
    return geom_ids


def _representative_geom_id(
    physics, kind: str, object_id: int, role_name: str | None = None
) -> int:
    """Choose the same visible rigid geometry for 3D and 2D supervision."""
    geom_ids = _reference_geom_ids(physics, kind, object_id, role_name)
    if not len(geom_ids):
        raise ValueError(
            f"Graph role {role_name!r} has no visible MuJoCo geom"
        )
    return int(geom_ids[0])


def graph_snapshot(
    physics,
    schema: list[tuple[str, int, str]],
    graph_nodes: int,
    proximity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create privileged labels; these are never model inputs at inference."""
    node_state = np.zeros((graph_nodes, 13), dtype=np.float32)
    node_mask = np.zeros(graph_nodes, dtype=np.float32)
    relation = np.zeros((graph_nodes, graph_nodes), dtype=np.int64)
    body_to_slot = {
        object_id: slot
        for slot, (kind, object_id, _name) in enumerate(schema)
        if kind == "body"
    }

    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    for slot, (kind, object_id, name) in enumerate(schema):
        node_mask[slot] = 1.0
        geom_id = _representative_geom_id(
            physics, kind, object_id, name
        )
        position = np.asarray(physics.data.geom_xpos[geom_id])
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            quaternion, np.asarray(physics.data.geom_xmat[geom_id])
        )
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            physics.model.ptr,
            physics.data.ptr,
            geom_type,
            geom_id,
            velocity,
            0,
        )
        node_state[slot] = np.concatenate(
            (position, quaternion, velocity)
        ).astype(np.float32)

    # Relation ids: 0 none, 1 kinematic, 2 proximity, 3 contact.
    parent_ids = np.asarray(physics.model.body_parentid)
    for child_id, child_slot in body_to_slot.items():
        parent_id = int(parent_ids[child_id])
        if parent_id in body_to_slot:
            parent_slot = body_to_slot[parent_id]
            relation[child_slot, parent_slot] = 1
            relation[parent_slot, child_slot] = 1

    # A geom used as its own semantic node is rigidly attached to its owning
    # body.  This connects the RM65-B bob to the pendulum-hinge/TCP node while
    # leaving the world-attached target unconnected.
    geom_body_ids = np.asarray(physics.model.geom_bodyid)
    for slot, (kind, object_id, _name) in enumerate(schema):
        if kind != "geom":
            continue
        owner_id = int(geom_body_ids[object_id])
        if owner_id in body_to_slot:
            owner_slot = body_to_slot[owner_id]
            relation[slot, owner_slot] = 1
            relation[owner_slot, slot] = 1

    positions = node_state[:, :3]
    for first in range(len(schema)):
        for second in range(first + 1, len(schema)):
            if np.linalg.norm(positions[first] - positions[second]) < proximity_threshold:
                if relation[first, second] == 0:
                    relation[first, second] = 2
                    relation[second, first] = 2

    geom_to_slots: dict[int, set[int]] = {}
    for slot, (kind, object_id, name) in enumerate(schema):
        for geom_id in _reference_geom_ids(physics, kind, object_id, name):
            geom_to_slots.setdefault(int(geom_id), set()).add(slot)
    for contact in physics.data.contact[: int(physics.data.ncon)]:
        first_slots = geom_to_slots.get(int(contact.geom1), set())
        second_slots = geom_to_slots.get(int(contact.geom2), set())
        for first in first_slots:
            for second in second_slots:
                if first != second:
                    relation[first, second] = 3
                    relation[second, first] = 3
    return node_state, node_mask, relation


def graph_pixel_snapshot(
    physics,
    schema: list[tuple[str, int, str]],
    graph_nodes: int,
    size: int,
    camera: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact camera-aligned MuJoCo centroids from segmentation rendering."""
    pixels = np.zeros((graph_nodes, 2), dtype=np.float32)
    visible = np.zeros(graph_nodes, dtype=np.float32)
    segmentation = physics.render(
        size, size, camera_id=camera, segmentation=True
    )
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    for slot, (kind, object_id, name) in enumerate(schema):
        geom_ids = np.asarray(
            [_representative_geom_id(physics, kind, object_id, name)],
            dtype=np.int64,
        )
        # dm_control documents channels as (object type, object id).  The
        # alternate order is accepted for renderer-version portability.
        mask_a = (segmentation[..., 0] == geom_type) & np.isin(
            segmentation[..., 1], geom_ids
        )
        mask_b = (segmentation[..., 1] == geom_type) & np.isin(
            segmentation[..., 0], geom_ids
        )
        mask = mask_a if mask_a.sum() >= mask_b.sum() else mask_b
        rows, columns = np.nonzero(mask)
        if len(rows):
            visible[slot] = 1.0
            pixels[slot, 0] = 2.0 * float(columns.mean()) / max(1, size - 1) - 1.0
            pixels[slot, 1] = 2.0 * float(rows.mean()) / max(1, size - 1) - 1.0
    return pixels, visible


class OnlineAgent:
    """A trained online DreamerV3 actor driving collection.

    Loads a ``train_online_dmc.py`` actor--critic checkpoint and acts from
    the same 64x64 camera-0 frames it was trained on (walker/cheetah use
    camera 0 in both pipelines). The executed action (after any added
    noise) is what updates its recurrent state.
    """

    def __init__(self, path: Path, device: torch.device) -> None:
        from jointgv import JointGraphVideoWorldModel
        from jointgv.online import DreamerBehavior
        from train_online_dmc import policy_actions

        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("training_mode") != "online_actor_critic":
            raise ValueError(f"{path} is not an online actor--critic checkpoint")
        config = payload["config"]
        if config["backbone"] != "dreamerv3":
            raise ValueError("--policy-checkpoint supports DreamerV3 agents only")
        self.domain = str(config["domain"])
        self.task = str(config["task"])
        self.use_graph = not bool(config["disable_graph_conditioning"])
        self.device = device
        self._policy_actions = policy_actions
        action_dim = int(config["action_dim"])
        self.model = JointGraphVideoWorldModel(
            action_dim=action_dim,
            backbone="dreamerv3",
            node_dim=int(config["node_dim"]),
            graph_nodes=int(config["graph_nodes"]),
            message_steps=int(config["message_steps"]),
            max_graph_correction=float(config["max_graph_correction"]),
            graph_correction_gate_init=float(config["graph_correction_gate_init"]),
            graph_observer=config["graph_observer"],
            graph_backbone_pretrained=False,
            reltr_checkpoint=str(config["reltr_checkpoint"]),
        ).to(device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.behavior = DreamerBehavior(
            action_dim, self.model.dreamerv3_core.feat_dim
        ).to(device)
        self.behavior.load_state_dict(payload["behavior"])
        self.behavior.eval()
        self.reset()

    def reset(self) -> None:
        self.recurrent: list = [None]
        self.previous: torch.Tensor | None = None
        self.graphs: list = [None]
        self.relations: list = [None]

    @torch.no_grad()
    def propose(self, image: np.ndarray) -> np.ndarray:
        """Sampled policy action for the current HWC frame."""
        actions, self.recurrent, _, self.graphs, self.relations = (
            self._policy_actions(
                self.model,
                self.behavior,
                [image.transpose(2, 0, 1).copy()],
                self.device,
                use_graph=self.use_graph,
                recurrent=self.recurrent,
                previous_actions=self.previous,
                graph_states=self.graphs,
                relation_states=self.relations,
            )
        )
        return np.asarray(actions[0], dtype=np.float32)

    def executed(self, action: np.ndarray) -> None:
        self.previous = torch.as_tensor(
            action[None], dtype=torch.float32, device=self.device
        )


class CartPoleActorPolicy:
    """train_cartpole_actor.py actors on the RM65-B cart-pole.

    Each actor acts from the 64x64 camera frame and its change from the
    previous frame, both encoded by the frozen DINO encoder of the world
    model it was trained in, exactly as in its evaluation. Actors that share
    a world model share one loaded copy.
    """

    def __init__(self, paths: list[Path], device: torch.device) -> None:
        from jointgv import JointGraphVideoWorldModel
        from train_cartpole_actor import Actor

        self.device = device
        self.actors: list[tuple] = []
        self.labels: list[str] = []
        models: dict[str, JointGraphVideoWorldModel] = {}
        for path in paths:
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
            self.actors.append((actor.eval(), models[world_model]))
            self.labels.append(f"{saved['mode']} ({path})")
        self.select(0)

    def select(self, index: int) -> None:
        self.actor, self.model = self.actors[index]
        self.previous: torch.Tensor | None = None

    @torch.no_grad()
    def action(self, image: np.ndarray) -> np.ndarray:
        """Deterministic action for the current HWC frame (call every step:
        it also tracks the previous frame)."""
        frame = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
        frame = frame.permute(2, 0, 1).float()[None] / 255.0
        current = self.model.vae.encode(frame, sample=False)[0]
        previous = current if self.previous is None else self.previous
        self.previous = current
        return self.actor.act(current, previous)[0].cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="reacher")
    parser.add_argument("--task", default="easy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--cartpole-initial-angle-deg",
        type=float,
        default=DEFAULT_INITIAL_ANGLE_RANGE_DEG,
        help=(
            "RM65-B cart-pole only: sample the initial pole angle uniformly "
            "over +/- this many degrees."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Maximum stored agent steps; 0 runs to the natural DMC ending.",
    )
    parser.add_argument(
        "--allow-partial-episode",
        action="store_true",
        help="Acknowledge that --steps > 0 creates non-reportable smoke data.",
    )
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument(
        "--camera",
        type=int,
        default=-1,
        help="Official mapping: -1 (Reacher resolves to fixed camera 0).",
    )
    parser.add_argument("--graph-nodes", type=int, default=8)
    parser.add_argument("--proximity-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/dmc_reacher"))
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=None,
        help=(
            "Online DreamerV3 actor checkpoint (train_online_dmc.py, e.g. "
            "best_return.pt) that drives the first --policy-fraction of the "
            "episodes; the rest use uniform random actions. Default: all "
            "random, as before."
        ),
    )
    parser.add_argument("--policy-fraction", type=float, default=1.0)
    parser.add_argument(
        "--policy-noise",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to policy actions (then clipped).",
    )
    parser.add_argument(
        "--uniform-random-actions",
        action="store_true",
        help=(
            "Random episodes use i.i.d. uniform actions even when the task "
            "defines its own exploration (rm65b: random point-to-point "
            "moves). DMC tasks always use uniform actions."
        ),
    )
    parser.add_argument(
        "--cartpole-policy",
        choices=("explore", "lqr", "actor"),
        default="explore",
        help=(
            "RM65-B cart-pole only. 'explore' = the task's noisy heuristic "
            "stabiliser (previous behaviour). 'lqr' = privileged full-state "
            "LQR (jointgv/envs/rm65b_cartpole_lqr.py), optionally tracking a "
            "slowly moving random pivot reference with command noise. "
            "'actor' = train_cartpole_actor.py actors (--cartpole-actors) "
            "acting from the camera frames, with Gaussian action noise."
        ),
    )
    parser.add_argument(
        "--cartpole-actors",
        type=Path,
        nargs="+",
        default=None,
        help="Actor only: actor_best.pt files; episode i uses actor i mod count.",
    )
    parser.add_argument(
        "--cartpole-actor-noise",
        type=float,
        nargs=2,
        default=(0.0, 0.3),
        metavar=("LOW", "HIGH"),
        help="Actor only: per-episode std, uniform in [LOW, HIGH], of Gaussian noise on each action.",
    )
    parser.add_argument(
        "--hold-action-rate",
        type=float,
        default=0.0,
        help=(
            "RM65-B cart-pole lqr/actor only: rate (per second) of held-action "
            "excursions. At a uniformly random time the controller is "
            "overridden by one command, uniform in [-1, 1], held for a "
            "duration uniform in --hold-action-seconds (rail ends still "
            "avoided), then control returns. Shows the world model what "
            "sustained wrong or idle commands do near upright. 0 = off."
        ),
    )
    parser.add_argument(
        "--hold-action-seconds",
        type=float,
        nargs=2,
        default=(0.1, 0.6),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--cartpole-hinge",
        default=None,
        help=(
            "RM65-B cart-pole only: pole hinge, 'damper' (the default, viscous "
            "0.015), 'bearing606' (606ZZ ball bearing) or DAMPING,FRICTION in "
            "N m s/rad,N m. Used by the environment and the LQR alike."
        ),
    )
    parser.add_argument(
        "--cartpole-time-limit",
        type=float,
        default=None,
        help=(
            "RM65-B cart-pole only: episode time limit in seconds (default: the "
            "task's 20 s). A natural, complete ending (is_last, discount 1), "
            "unlike --steps, which marks the episode as a truncated smoke run."
        ),
    )
    parser.add_argument(
        "--file-tag",
        default="",
        help=(
            "Inserted into the episode file names (<domain>_<task>_<tag>_00000.npz) "
            "so a new collection can share a directory with an older one."
        ),
    )
    parser.add_argument(
        "--lqr-reference-cm",
        type=float,
        default=0.0,
        help="LQR only: random pivot reference within +/- this many cm of the rail centre (0 = hold centre).",
    )
    parser.add_argument(
        "--lqr-reference-seconds",
        type=float,
        nargs=2,
        default=(0.8, 2.0),
        metavar=("MIN", "MAX"),
        help="LQR only: duration range of each smooth reference move.",
    )
    parser.add_argument(
        "--lqr-action-noise",
        type=float,
        nargs="+",
        default=[0.0],
        metavar="STD",
        help=(
            "LQR only: std of Ornstein-Uhlenbeck noise (tau 0.3 s) on the "
            "normalised command. Two values LOW HIGH draw one level per "
            "episode, uniformly."
        ),
    )
    parser.add_argument(
        "--lqr-full-fraction",
        type=float,
        default=1.0,
        help=(
            "LQR only: share of episodes the LQR drives to the end (drawn per "
            "episode). The others hand over after --lqr-handoff-seconds to "
            "a casual imperfect correction (random gain, lead, delay, speed "
            "ceiling and noise; jointgv/envs/rm65b_cartpole_lqr.py), which usually "
            "fails. 1.0 = always LQR (pilot behaviour)."
        ),
    )
    parser.add_argument(
        "--lqr-handoff-seconds",
        type=float,
        default=2.0,
        help="LQR only: hand-over time uniform in [0, this] s for non-full episodes (0 = no LQR help).",
    )
    parser.add_argument(
        "--cartpole-min-steps",
        type=int,
        default=0,
        help=(
            "RM65-B cart-pole only: when the pole reaches horizontal before "
            "this many stored steps, keep simulating below horizontal (same "
            "controller, no termination) until the episode has this many, so "
            "short failures still form windows. Padded steps are flagged in "
            "'past_horizontal' and the crossing step is 'fall_step'. 0 = off "
            "(end at horizontal). 35 = 3 context + 32 predicted steps."
        ),
    )
    parser.add_argument(
        "--cartpole-initial-velocity",
        type=float,
        default=0.0,
        help=(
            "RM65-B cart-pole only: initial pole angular speed uniform in "
            "[0, this] rad/s, always in the direction the pole leans (it is "
            "already falling). 0 = start at rest, the control protocol."
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.policy_fraction <= 1.0 or args.policy_noise < 0:
        parser.error("--policy-fraction must be in [0, 1], --policy-noise >= 0")
    if not 0.0 < args.cartpole_initial_angle_deg < 90.0:
        parser.error("--cartpole-initial-angle-deg must be in (0, 90)")
    if not 0.0 <= args.cartpole_initial_velocity <= 20.0:
        parser.error("--cartpole-initial-velocity must be in [0, 20] rad/s")
    if len(args.lqr_action_noise) not in (1, 2) or min(args.lqr_action_noise) < 0:
        parser.error("--lqr-action-noise takes STD or LOW HIGH, all >= 0")
    noise_low, noise_high = min(args.lqr_action_noise), max(args.lqr_action_noise)
    if args.cartpole_min_steps < 0:
        parser.error("--cartpole-min-steps must be >= 0")
    if not 0.0 <= args.lqr_full_fraction <= 1.0 or args.lqr_handoff_seconds < 0:
        parser.error("--lqr-full-fraction must be in [0, 1] and --lqr-handoff-seconds >= 0")
    if args.steps < 0:
        parser.error("--steps must be nonnegative (0 means natural ending)")
    if args.steps > 0 and not args.allow_partial_episode:
        parser.error(
            "--steps > 0 truncates the official episode; add "
            "--allow-partial-episode only for a smoke test"
        )
    try:
        validate_dreamerv3_vision(
            args.size, args.action_repeat, args.camera
        )
    except ValueError as error:
        parser.error(str(error))
    is_cartpole = (args.domain, args.task) == ("rm65b", "cartpole")
    if args.cartpole_time_limit is not None and (not is_cartpole or args.cartpole_time_limit <= 0):
        parser.error("--cartpole-time-limit needs rm65b cartpole and a positive value")
    if args.cartpole_hinge is not None:
        if not is_cartpole:
            parser.error("--cartpole-hinge needs rm65b cartpole")
        try:
            hinge = hinge_parameters(args.cartpole_hinge)
        except ValueError as error:
            parser.error(str(error))
        print(f"cart-pole hinge {args.cartpole_hinge}: damping {hinge[0]:g} N m s/rad, friction {hinge[1]:g} N m")
    if args.hold_action_rate < 0 or not 0.0 < args.hold_action_seconds[0] <= args.hold_action_seconds[1]:
        parser.error("--hold-action-rate must be >= 0 and --hold-action-seconds 0 < MIN <= MAX")
    if args.hold_action_rate > 0 and args.cartpole_policy not in ("lqr", "actor"):
        parser.error("--hold-action-rate needs --cartpole-policy lqr or actor")
    if not 0.0 <= args.cartpole_actor_noise[0] <= args.cartpole_actor_noise[1]:
        parser.error("--cartpole-actor-noise takes 0 <= LOW <= HIGH")
    cartpole_actor = None
    if args.cartpole_policy == "actor":
        if not is_cartpole or not args.cartpole_actors:
            parser.error("--cartpole-policy actor requires rm65b cartpole and --cartpole-actors")
        if args.policy_checkpoint is not None or args.uniform_random_actions:
            parser.error("--cartpole-policy actor excludes --policy-checkpoint and --uniform-random-actions")
        cartpole_actor = CartPoleActorPolicy(
            args.cartpole_actors,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        print(
            f"cart-pole actors on {cartpole_actor.device}: "
            + ", ".join(cartpole_actor.labels)
            + f"; action noise std uniform in [{args.cartpole_actor_noise[0]:g}, "
            f"{args.cartpole_actor_noise[1]:g}] per episode"
        )
    if is_cartpole:
        from jointgv.envs.rm65b_cartpole_lqr import keep_off_rail_ends

    step_seconds = CONTROL_TIMESTEP_CARTPOLE * args.action_repeat
    hold_probability = args.hold_action_rate * step_seconds
    if args.hold_action_rate > 0:
        print(
            f"held-action excursions: {args.hold_action_rate:g}/s, "
            f"{args.hold_action_seconds[0]:g}-{args.hold_action_seconds[1]:g} s, "
            "command uniform in [-1, 1]"
        )
    lqr_policy = None
    if args.cartpole_policy == "lqr":
        if not is_cartpole:
            parser.error("--cartpole-policy lqr requires --domain rm65b --task cartpole")
        if args.policy_checkpoint is not None or args.uniform_random_actions:
            parser.error("--cartpole-policy lqr excludes --policy-checkpoint and --uniform-random-actions")
        from jointgv.envs.rm65b_cartpole_lqr import (
            CartPoleLQR,
            ImperfectCorrector,
            LQRCollectionPolicy,
        )

        try:
            lqr_policy = LQRCollectionPolicy(
                CartPoleLQR(args.action_repeat, hinge=args.cartpole_hinge),
                args.action_repeat,
                reference_amplitude=args.lqr_reference_cm / 100.0,
                segment_seconds=tuple(args.lqr_reference_seconds),
                noise_std=noise_low,
                avoid_rail_ends=True,
            )
            corrector = ImperfectCorrector(args.action_repeat)
        except ValueError as error:
            parser.error(str(error))
        print(
            f"cart-pole LQR collection: reference +/-{args.lqr_reference_cm:g} cm, "
            f"moves of {args.lqr_reference_seconds[0]:g}-{args.lqr_reference_seconds[1]:g} s, "
            "command noise "
            + (f"{noise_low:g}" if noise_low == noise_high
               else f"uniform per episode in [{noise_low:g}, {noise_high:g}]")
        )
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    render_camera = resolve_camera(args.domain, args.camera)
    agent = None
    policy_episodes = 0
    if args.policy_checkpoint is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(args.seed)
        agent = OnlineAgent(args.policy_checkpoint, device)
        if (agent.domain, agent.task) != (args.domain, args.task):
            parser.error(
                f"policy was trained on {agent.domain}/{agent.task}, not "
                f"{args.domain}/{args.task}"
            )
        policy_episodes = round(args.policy_fraction * args.episodes)
        print(
            f"policy {args.policy_checkpoint}: episodes 0-{policy_episodes - 1} "
            f"(noise std {args.policy_noise}), the rest random"
        )

    for episode_id in range(args.episodes):
        task_kwargs = {"random": args.seed + episode_id}
        if (args.domain, args.task) == ("rm65b", "cartpole"):
            # Stratify across episodes instead of changing the controller gain
            # during a recovery.  Consecutive blocks of five episodes contain
            # every speed exactly once; split seeds still remain disjoint.
            task_kwargs["exploration_speed"] = float(
                EXPLORATION_SPEED_LEVELS[episode_id % len(EXPLORATION_SPEED_LEVELS)]
            )
            task_kwargs["initial_angle_range_deg"] = float(
                args.cartpole_initial_angle_deg
            )
            task_kwargs["initial_angular_velocity"] = float(
                args.cartpole_initial_velocity
            )
            if args.cartpole_time_limit is not None:
                task_kwargs["time_limit"] = float(args.cartpole_time_limit)
            task_kwargs.update(hinge_task_kwargs(args.cartpole_hinge))
        env = load_environment(
            args.domain, args.task, task_kwargs=task_kwargs
        )
        timestep = env.reset()
        use_policy = episode_id < policy_episodes
        if use_policy:
            agent.reset()
        explore = (
            None
            if use_policy or args.uniform_random_actions
            else getattr(env.task, "exploration_action", None)
        )
        episode_noise = 0.0
        if lqr_policy is not None:
            episode_noise = (
                noise_low if noise_low == noise_high
                else float(rng.uniform(noise_low, noise_high))
            )
            lqr_policy.reset(rng, noise_std=episode_noise)
            # LQR until handoff_step, then the task's noisy correction
            # heuristic (usually fails); -1 = LQR for the whole episode.
            handoff_step = (
                -1 if rng.random() < args.lqr_full_fraction
                else int(
                    rng.uniform(0.0, args.lqr_handoff_seconds)
                    / (CONTROL_TIMESTEP_CARTPOLE * args.action_repeat)
                )
            )
            # After the hand-over: a casual correction with random gain,
            # lead, delay, speed ceiling and noise that usually fails
            # (replaces the task's bang-bang preview heuristic here).
            corrector.reset(rng)
            explore = None
        actor_index, actor_noise = -1, 0.0
        if cartpole_actor is not None:
            actor_index = episode_id % len(cartpole_actor.actors)
            cartpole_actor.select(actor_index)
            actor_noise = float(rng.uniform(*args.cartpole_actor_noise))
            explore = None
        hold_left, hold_value, holds = 0, 0.0, 0
        # 0 = LQR, 1 = imperfect corrector, 2 = held random command, 3 = actor
        controller_ids: list[int] = []
        padding, fall_step = False, -1
        past_horizontal: list[bool] = []
        substeps = round(CONTROL_TIMESTEP_CARTPOLE / float(env.physics.model.opt.timestep))
        images, actions, rewards, discounts = [], [], [], []
        qpos, qvel, physics_state, geom_xpos = [], [], [], []
        graph_node_state, graph_node_mask, graph_relation = [], [], []
        graph_node_pixel, graph_node_visible = [], []
        graph_schema = build_graph_schema(
            env.physics, args.domain, args.graph_nodes
        )
        is_last, is_terminal = [], []
        environment_steps = 0
        while args.steps == 0 or len(actions) < args.steps:
            images.append(
                env.physics.render(
                    args.size, args.size, camera_id=render_camera
                )
            )
            qpos.append(env.physics.data.qpos.copy())
            qvel.append(env.physics.data.qvel.copy())
            physics_state.append(env.physics.get_state().copy())
            geom_xpos.append(env.physics.data.geom_xpos.copy())
            state_label, mask_label, relation_label = graph_snapshot(
                env.physics,
                graph_schema,
                args.graph_nodes,
                args.proximity_threshold,
            )
            pixel_label, visible_label = graph_pixel_snapshot(
                env.physics,
                graph_schema,
                args.graph_nodes,
                args.size,
                render_camera,
            )
            graph_node_state.append(state_label)
            graph_node_mask.append(mask_label)
            graph_relation.append(relation_label)
            graph_node_pixel.append(pixel_label)
            graph_node_visible.append(visible_label)
            spec = env.action_spec()
            if use_policy:
                action = agent.propose(images[-1])
                if args.policy_noise > 0:
                    action = action + rng.normal(
                        0.0, args.policy_noise, action.shape
                    ).astype(np.float32)
                action = np.clip(action, spec.minimum, spec.maximum).astype(
                    np.float32
                )
                agent.executed(action)
            elif lqr_policy is not None:
                if handoff_step < 0 or len(actions) < handoff_step:
                    action = lqr_policy.action(env.physics, env.task)
                    controller_ids.append(0)
                else:
                    action = corrector.action(env.physics, env.task)
                    controller_ids.append(1)
            elif cartpole_actor is not None:
                action = cartpole_actor.action(images[-1])
                if actor_noise > 0:
                    action = action + rng.normal(0.0, actor_noise, action.shape).astype(np.float32)
                action = np.clip(action, spec.minimum, spec.maximum).astype(np.float32)
                controller_ids.append(3)
            elif explore is not None:
                action = np.asarray(
                    explore(env.physics, rng, args.action_repeat), dtype=np.float32
                )
            else:
                action = rng.uniform(spec.minimum, spec.maximum).astype(np.float32)
            if hold_probability > 0:
                # The controller still ran this step (its internal state
                # stays current); a held command overrides what it sent.
                if hold_left == 0 and rng.random() < hold_probability:
                    hold_left = max(1, round(rng.uniform(*args.hold_action_seconds) / step_seconds))
                    hold_value = float(rng.uniform(-1.0, 1.0))
                    holds += 1
                if hold_left > 0:
                    hold_left -= 1
                    action = np.asarray(
                        [keep_off_rail_ends(hold_value, env.task, args.action_repeat, rng)],
                        dtype=np.float32,
                    )
                    controller_ids[-1] = 2
            repeated_reward = 0.0
            repeated_discount = 1.0
            for _ in range(args.action_repeat):
                if padding:
                    # Below horizontal: the environment has terminated (its
                    # next step() would reset), so advance it by hand exactly
                    # as control.Environment.step does, without termination.
                    # The episode has already failed: reward stays 0 even if
                    # the pole swings back above horizontal.
                    env.task.before_step(action, env.physics)
                    env.physics.step(substeps)
                    env.task.after_step(env.physics)
                    environment_steps += 1
                    continue
                timestep = env.step(action)
                environment_steps += 1
                repeated_reward += float(timestep.reward or 0.0)
                repeated_discount *= float(
                    1.0 if timestep.discount is None else timestep.discount
                )
                if timestep.last():
                    if (
                        is_cartpole
                        and repeated_discount == 0.0
                        and len(actions) + 1 < args.cartpole_min_steps
                    ):
                        padding, fall_step = True, len(actions)
                        repeated_discount = 1.0
                        continue
                    break
            padded_end = padding and len(actions) + 1 >= args.cartpole_min_steps
            if padded_end:
                repeated_discount = 0.0      # the episode still ends in a fall
            actions.append(action)
            rewards.append(repeated_reward)
            discounts.append(repeated_discount)
            past_horizontal.append(padding)
            last = padded_end or (timestep.last() and not padding)
            is_last.append(bool(last))
            is_terminal.append(bool(last and repeated_discount == 0.0))
            if last and not padding and repeated_discount == 0.0:
                fall_step = len(actions) - 1      # fell without needing padding
            if last:
                break
        images.append(
            env.physics.render(
                args.size, args.size, camera_id=render_camera
            )
        )
        qpos.append(env.physics.data.qpos.copy())
        qvel.append(env.physics.data.qvel.copy())
        physics_state.append(env.physics.get_state().copy())
        geom_xpos.append(env.physics.data.geom_xpos.copy())
        state_label, mask_label, relation_label = graph_snapshot(
            env.physics,
            graph_schema,
            args.graph_nodes,
            args.proximity_threshold,
        )
        pixel_label, visible_label = graph_pixel_snapshot(
            env.physics,
            graph_schema,
            args.graph_nodes,
            args.size,
            render_camera,
        )
        graph_node_state.append(state_label)
        graph_node_mask.append(mask_label)
        graph_relation.append(relation_label)
        graph_node_pixel.append(pixel_label)
        graph_node_visible.append(visible_label)
        tag = f"{args.file_tag}_" if args.file_tag else ""
        path = args.output / f"{args.domain}_{args.task}_{tag}{episode_id:05d}.npz"
        np.savez_compressed(
            path,
            image=np.asarray(images, dtype=np.uint8),
            action=np.asarray(actions, dtype=np.float32),
            reward=np.asarray(rewards, dtype=np.float32),
            discount=np.asarray(discounts, dtype=np.float32),
            is_last=np.asarray(is_last, dtype=np.bool_),
            is_terminal=np.asarray(is_terminal, dtype=np.bool_),
            qpos=np.asarray(qpos, dtype=np.float32),
            qvel=np.asarray(qvel, dtype=np.float32),
            physics_state=np.asarray(physics_state, dtype=np.float64),
            geom_xpos=np.asarray(geom_xpos, dtype=np.float64),
            domain=np.asarray(args.domain),
            task=np.asarray(args.task),
            episode_seed=np.asarray(args.seed + episode_id, dtype=np.int64),
            camera=np.asarray(render_camera, dtype=np.int64),
            requested_camera=np.asarray(args.camera, dtype=np.int64),
            render_size=np.asarray(args.size, dtype=np.int64),
            action_repeat=np.asarray(args.action_repeat, dtype=np.int64),
            environment_steps=np.asarray(environment_steps, dtype=np.int64),
            episode_complete=np.asarray(bool(timestep.last())),
            observation_mode=np.asarray("image_only"),
            protocol=np.asarray(DREAMERV3_DMC_VISION),
            dm_control_version=np.asarray(version("dm-control")),
            mujoco_version=np.asarray(version("mujoco")),
            graph_node_state=np.asarray(graph_node_state, dtype=np.float32),
            graph_node_mask=np.asarray(graph_node_mask, dtype=np.float32),
            graph_relation=np.asarray(graph_relation, dtype=np.int64),
            graph_node_pixel=np.asarray(graph_node_pixel, dtype=np.float32),
            graph_node_visible=np.asarray(graph_node_visible, dtype=np.float32),
            graph_node_name=np.asarray(
                [item[2] for item in graph_schema]
                + ["padding"] * (args.graph_nodes - len(graph_schema))
            ),
            graph_node_kind=np.asarray(
                [item[0] for item in graph_schema]
                + ["padding"] * (args.graph_nodes - len(graph_schema))
            ),
            graph_node_id=np.asarray(
                [item[1] for item in graph_schema]
                + [-1] * (args.graph_nodes - len(graph_schema)),
                dtype=np.int64,
            ),
            graph_schema_version=np.asarray(
                EXPLICIT_GRAPH_SCHEMA_VERSION, dtype=np.int64
            ),
            graph_coordinate_frame=np.asarray(
                EXPLICIT_GRAPH_COORDINATE_FRAME
            ),
            graph_node_geom_id=np.asarray(
                [
                    _representative_geom_id(
                        env.physics, kind, object_id, name
                    )
                    for kind, object_id, name in graph_schema
                ]
                + [-1] * (args.graph_nodes - len(graph_schema)),
                dtype=np.int64,
            ),
            graph_body_id=np.asarray(
                [item[1] if item[0] == "body" else -1 for item in graph_schema]
                + [-1] * (args.graph_nodes - len(graph_schema)),
                dtype=np.int64,
            ),
            action_source=np.asarray(
                "policy" if use_policy
                else "privileged_lqr" if lqr_policy is not None
                else "cartpole_actor" if cartpole_actor is not None
                else "task_exploration" if explore is not None
                else "random"
            ),
            **(
                {
                    "action_controller": np.asarray(controller_ids, dtype=np.int64),
                    "actor_checkpoint": np.asarray(str(args.cartpole_actors[actor_index])),
                    "actor_noise": np.asarray(actor_noise, dtype=np.float32),
                }
                if cartpole_actor is not None
                else {}
            ),
            **(
                {
                    "hold_action_rate": np.asarray(args.hold_action_rate, dtype=np.float32),
                    "hold_action_seconds": np.asarray(args.hold_action_seconds, dtype=np.float32),
                    "hold_action_count": np.asarray(holds, dtype=np.int64),
                }
                if hold_probability > 0
                else {}
            ),
            **(
                {
                    "lqr_reference_cm": np.asarray(args.lqr_reference_cm, dtype=np.float32),
                    "lqr_reference_seconds": np.asarray(
                        args.lqr_reference_seconds, dtype=np.float32
                    ),
                    "lqr_action_noise": np.asarray(episode_noise, dtype=np.float32),
                    "lqr_handoff_step": np.asarray(handoff_step, dtype=np.int64),
                    "lqr_full_fraction": np.asarray(args.lqr_full_fraction, dtype=np.float32),
                    "lqr_handoff_seconds": np.asarray(args.lqr_handoff_seconds, dtype=np.float32),
                    "action_controller": np.asarray(controller_ids, dtype=np.int64),
                    **{
                        f"corrector_{name}": np.asarray(value, dtype=np.float32)
                        for name, value in corrector.parameters().items()
                    },
                    "lqr_action_noise_range": np.asarray(
                        [noise_low, noise_high], dtype=np.float32
                    ),
                }
                if lqr_policy is not None
                else {}
            ),
            policy_checkpoint=np.asarray(
                str(args.policy_checkpoint) if use_policy else ""
            ),
            policy_noise=np.asarray(
                args.policy_noise if use_policy else 0.0, dtype=np.float32
            ),
            **(
                {
                    "task_exploration_speed": np.asarray(
                        env.task.exploration_speed, dtype=np.float32
                    )
                }
                if hasattr(env.task, "exploration_speed")
                else {}
            ),
            **(
                {
                    "task_initial_angle_range_deg": np.asarray(
                        env.task.initial_angle_range_deg, dtype=np.float32
                    )
                }
                if hasattr(env.task, "initial_angle_range_deg")
                else {}
            ),
            **(
                {
                    "cartpole_min_steps": np.asarray(args.cartpole_min_steps, dtype=np.int64),
                    "cartpole_time_limit": np.asarray(
                        env._step_limit * env.control_timestep(), dtype=np.float32
                    ),
                    # The hinge actually simulated, read back from the model.
                    "cartpole_hinge_damping": np.asarray(
                        float(np.ravel(env.physics.named.model.dof_damping["pole"])[0]), dtype=np.float64
                    ),
                    "cartpole_hinge_friction": np.asarray(
                        float(np.ravel(env.physics.named.model.dof_frictionloss["pole"])[0]), dtype=np.float64
                    ),
                    "fall_step": np.asarray(fall_step, dtype=np.int64),
                    "past_horizontal": np.asarray(past_horizontal, dtype=np.bool_),
                }
                if is_cartpole
                else {}
            ),
            **(
                {
                    "task_initial_angular_velocity": np.asarray(
                        env.task.initial_angular_velocity, dtype=np.float32
                    )
                }
                if hasattr(env.task, "initial_angular_velocity")
                else {}
            ),
            **(
                {
                    "task_target_position": np.asarray(
                        env.physics.target_position(), dtype=np.float32
                    ),
                    "task_target_plane": np.asarray("camera_projected_xz"),
                    **(
                        {
                            "task_target_tcp_position": np.asarray(
                                env.physics.target_tcp_position(),
                                dtype=np.float32,
                            )
                        }
                        if hasattr(env.physics, "target_tcp_position")
                        else {}
                    ),
                }
                if args.domain == "rm65b" and hasattr(env.physics, "target_position")
                else {}
            ),
        )
        source = (
            "policy" if use_policy
            else (
                "LQR" if handoff_step < 0
                else f"LQR->correction at step {handoff_step}"
                if handoff_step < len(actions)
                else f"LQR (fell before the step-{handoff_step} hand-over)"
            ) if lqr_policy is not None
            else f"actor {cartpole_actor.labels[actor_index].split(' ')[0]} "
            f"#{actor_index} (noise {actor_noise:.2f})" if cartpole_actor is not None
            else "exploration" if explore is not None
            else "random"
        )
        if hold_probability > 0:
            source += f" + {holds} held ({controller_ids.count(2)} steps)"
        motion = ""
        if is_cartpole:
            # Pole world angle per stored frame; how much it moves inside
            # 32-step prediction windows (33 frames, stride 16).
            angle = np.degrees(np.asarray(qpos)[:, :4].sum(axis=1))
            swings = [
                float(np.ptp(angle[start : start + 33]))
                for start in range(0, len(angle) - 32, 16)
            ]
            rate = float(np.asarray(qvel)[0, :4].sum())
            motion = (
                f", initial lean {angle[0]:+.1f} deg at {rate:+.1f} rad/s, max |lean| "
                f"{np.abs(angle).max():.1f} deg, 32-step windows: "
                + (
                    f"median swing {np.median(swings):.1f} deg, "
                    f"{sum(s >= 5.0 for s in swings)}/{len(swings)} swing >= 5 deg"
                    if swings else "episode shorter than 33 frames"
                )
            )
        if padding:
            motion += f", padded {len(actions) - fall_step - 1} steps below horizontal"
        print(
            f"[{episode_id + 1}/{args.episodes}] {path} ({len(actions)} "
            f"transitions, {source} actions, return {sum(rewards):.1f}{motion})"
        )


if __name__ == "__main__":
    main()
