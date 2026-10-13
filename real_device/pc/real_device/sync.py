from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SyncExchange:
    t0_pc_ns: int
    t1_device_us: int
    t2_device_us: int
    t3_pc_ns: int

    def __post_init__(self):
        if min(self.t0_pc_ns, self.t1_device_us, self.t2_device_us, self.t3_pc_ns) < 0:
            raise ValueError("sync timestamps must be non-negative")
        if self.t2_device_us < self.t1_device_us or self.t3_pc_ns < self.t0_pc_ns:
            raise ValueError("sync timestamps are not monotonic")
        if (self.t3_pc_ns-self.t0_pc_ns) < (self.t2_device_us-self.t1_device_us)*1000:
            raise ValueError("sync exchange has negative network RTT")

    @property
    def rtt_ns(self) -> int:
        return max(0, (self.t3_pc_ns - self.t0_pc_ns) - (self.t2_device_us - self.t1_device_us) * 1000)

    @property
    def device_mid_ns(self) -> float:
        return (self.t1_device_us + self.t2_device_us) * 500.0

    @property
    def pc_mid_ns(self) -> float:
        return (self.t0_pc_ns + self.t3_pc_ns) / 2.0


@dataclass(frozen=True)
class ClockMap:
    slope: float
    intercept_ns: float
    rms_residual_ns: float
    max_rtt_ns: int
    used: int
    total: int

    def device_us_to_pc_ns(self, value: int | float | np.ndarray) -> np.ndarray:
        return self.slope * (np.asarray(value, dtype=np.float64) * 1000.0) + self.intercept_ns

    @property
    def drift_ppm(self) -> float:
        return (self.slope - 1.0) * 1e6


def fit_clock(exchanges: list[SyncExchange], keep_fraction: float = 0.5, min_points: int = 4) -> ClockMap:
    """Fit PC_ns = slope*device_ns + intercept using the lowest-RTT exchanges."""
    if len(exchanges) < min_points:
        raise ValueError(f"need at least {min_points} sync exchanges")
    keep = max(min_points, int(np.ceil(len(exchanges) * keep_fraction)))
    selected = sorted(exchanges, key=lambda item: item.rtt_ns)[:keep]
    x = np.asarray([item.device_mid_ns for item in selected], dtype=np.float64)
    y = np.asarray([item.pc_mid_ns for item in selected], dtype=np.float64)
    origin = x.mean()
    centred = x - origin
    if np.ptp(x) < 1e6:
        raise ValueError("sync exchanges span too little device time for drift fit")
    slope = float(np.dot(centred, y - y.mean()) / np.dot(centred, centred))
    intercept = float(y.mean() - slope * origin)
    residual = y - (slope * x + intercept)
    return ClockMap(slope, intercept, float(np.sqrt(np.mean(residual**2))), max(i.rtt_ns for i in selected), len(selected), len(exchanges))


def clock_quality(clock: ClockMap, *, max_rtt_ms: float = 20.0, max_residual_ms: float = 2.0, max_drift_ppm: float = 1000.0) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not all(math.isfinite(value) for value in (clock.slope, clock.intercept_ns, clock.rms_residual_ns, clock.drift_ppm)):
        reasons.append("sync_fit_nonfinite")
    if clock.max_rtt_ns > max_rtt_ms * 1e6:
        reasons.append("sync_rtt_too_high")
    if clock.rms_residual_ns > max_residual_ms * 1e6:
        reasons.append("sync_residual_too_high")
    if abs(clock.drift_ppm) > max_drift_ppm:
        reasons.append("sync_drift_implausible")
    return not reasons, reasons
