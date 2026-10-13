from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from .angles import circular_difference_deg


def angular_metrics(reference_deg, predicted_deg, valid=None) -> dict[str, float | int]:
    ref = np.asarray(reference_deg, dtype=float)
    pred = np.asarray(predicted_deg, dtype=float)
    mask = np.isfinite(ref) & np.isfinite(pred)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    if not np.any(mask):
        return {"count": 0, "mae_deg": float("nan"), "rmse_deg": float("nan"), "max_abs_deg": float("nan")}
    err = np.asarray(circular_difference_deg(pred[mask], ref[mask]))
    return {"count": int(mask.sum()), "mae_deg": float(np.mean(np.abs(err))), "rmse_deg": float(np.sqrt(np.mean(err**2))), "max_abs_deg": float(np.max(np.abs(err)))}


def load_prediction_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "angle_deg" not in rows[0]:
        raise ValueError("prediction CSV must contain angle_deg")
    has_time = "t_pc_ns" in rows[0]
    has_frame = "frame_id" in rows[0]
    if has_time == has_frame:
        raise ValueError("prediction CSV must contain exactly one key: t_pc_ns or frame_id")
    return rows


def image_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Pixel MSE and a dependency-free global SSIM; LPIPS requires a separate model."""
    a = np.asarray(reference, dtype=np.float64) / 255.0
    b = np.asarray(predicted, dtype=np.float64) / 255.0
    if a.shape != b.shape:
        raise ValueError("images must have identical shapes")
    mse = float(np.mean((a - b) ** 2))
    mu_a, mu_b = float(a.mean()), float(b.mean())
    va, vb = float(a.var()), float(b.var())
    cov = float(np.mean((a - mu_a) * (b - mu_b)))
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
    return {"mse": mse, "ssim_global": float(ssim)}
