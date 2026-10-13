"""DeepMind Control Suite protocol shared by collection and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DREAMERV3_DMC_VISION = "dreamerv3_dmc_vision"
DREAMERV3_IMAGE_SIZE = 64
DREAMERV3_ACTION_REPEAT = 2
EXPLICIT_GRAPH_SCHEMA_VERSION = 3
EXPLICIT_GRAPH_COORDINATE_FRAME = "representative_geom_center_world"
DREAMERV3_DEFAULT_CAMERAS = {
    "quadruped": 2,
    "rodent": 4,
}


def resolve_camera(domain: str, requested_camera: int) -> int:
    """Match the camera resolution used by DreamerV3's official DMC wrapper."""
    if requested_camera != -1:
        return requested_camera
    return DREAMERV3_DEFAULT_CAMERAS.get(domain, 0)


def validate_dreamerv3_vision(
    image_size: int,
    action_repeat: int,
    requested_camera: int,
) -> None:
    """Reject deviations from the pinned burchim DMC vision settings."""
    problems = []
    if image_size != DREAMERV3_IMAGE_SIZE:
        problems.append(
            f"image size must be {DREAMERV3_IMAGE_SIZE}, got {image_size}"
        )
    if action_repeat != DREAMERV3_ACTION_REPEAT:
        problems.append(
            f"action repeat must be {DREAMERV3_ACTION_REPEAT}, "
            f"got {action_repeat}"
        )
    if requested_camera != -1:
        problems.append(
            "camera must be -1 so the official domain camera mapping is used"
        )
    if problems:
        raise ValueError(
            "burchim DreamerV3 DMC protocol mismatch: " + "; ".join(problems)
        )


def validate_dmc_dataset(
    root: str | Path,
    *,
    allow_partial: bool = False,
    require_graph_schema_version: int | None = None,
) -> dict[str, object]:
    """Validate every DMC episode before it can be used for training.

    Non-DMC datasets such as converted LIBERO demonstrations do not contain
    ``domain`` and ``task`` and are left untouched.
    """
    files = sorted(Path(root).glob("*.npz"))
    dmc_files = 0
    incomplete = 0
    for path in files:
        with np.load(path, allow_pickle=False) as episode:
            is_dmc = "domain" in episode.files and "task" in episode.files
            if not is_dmc:
                continue
            dmc_files += 1
            required = {
                "protocol",
                "observation_mode",
                "render_size",
                "action_repeat",
                "requested_camera",
                "camera",
                "environment_steps",
                "episode_complete",
                "is_last",
                "is_terminal",
                "episode_seed",
                "physics_state",
                "geom_xpos",
                "dm_control_version",
                "mujoco_version",
            }
            missing = required.difference(episode.files)
            if missing:
                raise ValueError(
                    f"{path} predates the pinned DMC protocol metadata "
                    f"(missing {sorted(missing)}). Recollect this dataset."
                )
            domain = str(episode["domain"].item())
            protocol = str(episode["protocol"].item())
            observation_mode = str(episode["observation_mode"].item())
            size = int(episode["render_size"].item())
            repeat = int(episode["action_repeat"].item())
            requested_camera = int(episode["requested_camera"].item())
            camera = int(episode["camera"].item())
            validate_dreamerv3_vision(size, repeat, requested_camera)
            if protocol != DREAMERV3_DMC_VISION:
                raise ValueError(
                    f"{path} uses protocol {protocol!r}, expected "
                    f"{DREAMERV3_DMC_VISION!r}"
                )
            if observation_mode != "image_only":
                raise ValueError(
                    f"{path} uses {observation_mode!r}, expected image_only"
                )
            if require_graph_schema_version is not None:
                graph_required = {
                    "graph_schema_version",
                    "graph_coordinate_frame",
                    "graph_node_geom_id",
                    "graph_node_state",
                    "graph_node_pixel",
                    "graph_node_visible",
                }
                graph_missing = graph_required.difference(episode.files)
                if graph_missing:
                    raise ValueError(
                        f"{path} lacks Explicit Graph schema metadata "
                        f"{sorted(graph_missing)}. Recollect this dataset."
                    )
                graph_version = int(
                    episode["graph_schema_version"].item()
                )
                coordinate_frame = str(
                    episode["graph_coordinate_frame"].item()
                )
                if graph_version != require_graph_schema_version:
                    raise ValueError(
                        f"{path} uses graph schema v{graph_version}, expected "
                        f"v{require_graph_schema_version}. Recollect it."
                    )
                if coordinate_frame != EXPLICIT_GRAPH_COORDINATE_FRAME:
                    raise ValueError(
                        f"{path} uses graph frame {coordinate_frame!r}, "
                        f"expected {EXPLICIT_GRAPH_COORDINATE_FRAME!r}"
                    )
            if camera != resolve_camera(domain, requested_camera):
                raise ValueError(f"{path} stores an incorrect resolved camera")
            transitions = len(episode["action"])
            if len(episode["image"]) != transitions + 1:
                raise ValueError(f"{path} must contain T+1 images for T actions")
            for key in ("physics_state", "geom_xpos"):
                if len(episode[key]) != transitions + 1:
                    raise ValueError(
                        f"{path}: {key} must contain exactly T+1 values"
                    )
            for key in ("reward", "discount", "is_last", "is_terminal"):
                if len(episode[key]) != transitions:
                    raise ValueError(
                        f"{path}: {key} must contain exactly T values"
                    )
            environment_steps = int(episode["environment_steps"].item())
            minimum_steps = max(0, (transitions - 1) * repeat + 1)
            maximum_steps = transitions * repeat
            if not minimum_steps <= environment_steps <= maximum_steps:
                raise ValueError(
                    f"{path}: environment_steps={environment_steps} is "
                    f"incompatible with T={transitions}, repeat={repeat}"
                )
            complete = bool(episode["episode_complete"].item())
            if complete:
                if transitions == 0 or not bool(episode["is_last"][-1]):
                    raise ValueError(
                        f"{path} claims completion without final is_last"
                    )
            else:
                incomplete += 1
                if not allow_partial:
                    raise ValueError(
                        f"{path} is a truncated smoke episode. Recollect "
                        "without --steps, or pass --allow-partial-dmc only "
                        "for a non-reportable smoke test."
                    )
    return {
        "protocol": DREAMERV3_DMC_VISION if dmc_files else None,
        "dmc_files": dmc_files,
        "incomplete_dmc_files": incomplete,
    }
