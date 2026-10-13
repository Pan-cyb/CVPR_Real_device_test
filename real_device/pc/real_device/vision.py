from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from .angles import wrap_deg


@dataclass(frozen=True)
class HSVRange:
    low: tuple[int, int, int]
    high: tuple[int, int, int]


@dataclass(frozen=True)
class MarkerDetection:
    valid: bool
    angle_deg: float = float("nan")
    pivot: tuple[float, float] | None = None
    tip: tuple[float, float] | None = None
    reason: str = ""


def _centroid(hsv: np.ndarray, bounds: HSVRange, minimum_area: float):
    mask = cv2.inRange(hsv, np.asarray(bounds.low, np.uint8), np.asarray(bounds.high, np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < minimum_area:
        return None
    moment = cv2.moments(contour)
    if moment["m00"] == 0:
        return None
    return moment["m10"] / moment["m00"], moment["m01"] / moment["m00"]


def detect_markers(frame_bgr: np.ndarray, pivot_hsv: HSVRange, tip_hsv: HSVRange, *, minimum_area: float = 20.0, zero_offset_deg: float = 0.0, direction: float = 1.0) -> MarkerDetection:
    """Measure rod angle from image upward vertical; missing/occluded markers are invalid."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    pivot = _centroid(hsv, pivot_hsv, minimum_area)
    tip = _centroid(hsv, tip_hsv, minimum_area)
    if pivot is None or tip is None:
        missing = "pivot" if pivot is None else "tip"
        return MarkerDetection(False, reason=f"missing_{missing}_marker")
    dx, dy = tip[0] - pivot[0], tip[1] - pivot[1]
    if dx * dx + dy * dy < 25.0:
        return MarkerDetection(False, pivot=pivot, tip=tip, reason="markers_too_close")
    raw = np.degrees(np.arctan2(dx, -dy))
    return MarkerDetection(True, wrap_deg(direction * raw - zero_offset_deg), pivot, tip, "ok")

