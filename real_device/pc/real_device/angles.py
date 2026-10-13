from __future__ import annotations

import numpy as np


def wrap_deg(angle):
    values = (np.asarray(angle, dtype=np.float64) + 180.0) % 360.0 - 180.0
    return float(values) if values.ndim == 0 else values


def circular_difference_deg(a, b):
    return wrap_deg(np.asarray(a) - np.asarray(b))


def unwrap_deg(angle):
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(angle, dtype=np.float64))))


def interpolate_circular(source_ns, source_deg, target_ns, *, max_gap_ms: float = 25.0):
    """Circular interpolation with no extrapolation and no interpolation over large gaps."""
    sx = np.asarray(source_ns, dtype=np.int64)
    sy = np.asarray(source_deg, dtype=np.float64)
    tx = np.asarray(target_ns, dtype=np.int64)
    out = np.full(tx.shape, np.nan, dtype=np.float64)
    valid = np.zeros(tx.shape, dtype=bool)
    finite = np.isfinite(sy)
    sx, sy = sx[finite], sy[finite]
    if len(sx) < 2:
        return out, valid
    order = np.argsort(sx, kind="stable")
    sx, sy = sx[order], sy[order]
    unique = np.r_[True, np.diff(sx) > 0]
    sx, sy = sx[unique], sy[unique]
    unwrapped = unwrap_deg(sy)
    right = np.searchsorted(sx, tx, side="left")
    exact = (right < len(sx)) & (sx[np.minimum(right, len(sx) - 1)] == tx)
    out[exact] = sy[right[exact]]
    valid[exact] = True
    inside = (~exact) & (right > 0) & (right < len(sx))
    indices = np.where(inside)[0]
    for index in indices:
        r = right[index]
        gap = sx[r] - sx[r - 1]
        if gap <= max_gap_ms * 1e6:
            fraction = (tx[index] - sx[r - 1]) / gap
            out[index] = wrap_deg(unwrapped[r - 1] + fraction * (unwrapped[r] - unwrapped[r - 1]))
            valid[index] = True
    return out, valid
