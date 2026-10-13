from __future__ import annotations

from pathlib import Path
import csv
import numpy as np


def export_angle_validation(sensor_csv: str | Path, output: str | Path) -> Path:
    """Export angle/time validation only; this is deliberately not a training episode."""
    with Path(sensor_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row.get("valid", "1") == "1" and row.get("angle_deg", "") != ""]
    t = np.asarray([int(row["t_pc_ns"]) for row in selected], dtype=np.int64)
    angle = np.asarray([float(row["angle_deg"]) for row in selected], dtype=np.float32)
    if not len(t):
        raise ValueError("sensor CSV contains no valid angle samples")
    target = Path(output)
    np.savez_compressed(target, t_pc_ns=t, angle_deg=angle, data_kind=np.asarray("real_angle_validation_only"))
    return target


def export_training_episode(*args, action_semantics: str | None = None, graph_labels=None, **kwargs):
    if not action_semantics:
        raise ValueError("refusing training export: measured real action semantics are required; unknown hand support must not be encoded as zero")
    if graph_labels is None:
        raise ValueError("refusing training export: required graph labels are unavailable")
    raise NotImplementedError("a project-specific mapping is required after real actions and graph labels are defined")
