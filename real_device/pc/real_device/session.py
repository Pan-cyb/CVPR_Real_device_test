from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math


class TrialState(str, Enum):
    WAIT_READY = "wait_ready"
    READY = "ready"
    RELEASED = "released"
    RECORDING = "recording"
    STOPPED = "stopped"
    INVALID = "invalid"


@dataclass
class TrialController:
    duration_ns: int = 6_000_000_000
    angle_limit_deg: float = 120.0
    initial_limit_deg: float = 10.0
    stale_sensor_ns: int = 100_000_000
    stale_camera_ns: int = 500_000_000
    state: TrialState = TrialState.WAIT_READY
    start_ns: int | None = None
    stop_ns: int | None = None
    reason: str | None = None
    last_sensor_ns: int | None = None
    last_camera_ns: int | None = None
    events: list[dict] = field(default_factory=list)
    readiness: dict[str, bool] = field(default_factory=lambda: {
        "sensor_stable": False, "camera_stable": False, "clock_valid": False,
        "geometry_confirmed": False, "zero_calibrated": False,
    })

    def set_ready(self, at_ns: int, **values: bool) -> None:
        before = (self.state, dict(self.readiness))
        for key, value in values.items():
            if key not in self.readiness:
                raise KeyError(key)
            self.readiness[key] = bool(value)
        self.state = TrialState.READY if all(self.readiness.values()) else TrialState.WAIT_READY
        if before != (self.state, self.readiness):
            self.events.append({"t_pc_ns": at_ns, "event": "readiness", **self.readiness})

    def release(self, at_ns: int, initial_angle_deg: float) -> None:
        if self.state != TrialState.READY:
            raise RuntimeError("sensor/camera/sync/geometry/zero are not all ready")
        if not math.isfinite(initial_angle_deg):
            self._stop(at_ns, "initial_angle_nonfinite", invalid=True)
            return
        sensor_age = None if self.last_sensor_ns is None else at_ns-self.last_sensor_ns
        camera_age = None if self.last_camera_ns is None else at_ns-self.last_camera_ns
        if (sensor_age is None or not 0 <= sensor_age <= self.stale_sensor_ns or
                camera_age is None or not 0 <= camera_age <= self.stale_camera_ns):
            raise RuntimeError("sensor or camera readiness is stale")
        if abs(initial_angle_deg) > self.initial_limit_deg:
            self._stop(at_ns, "initial_angle_out_of_range", invalid=True)
            return
        self.state = TrialState.RELEASED
        self.events.append({"t_pc_ns": at_ns, "event": "release_key", "visual_release_verified": False})
        self.start_ns = at_ns
        self.state = TrialState.RECORDING
        self.events.append({"t_pc_ns": at_ns, "event": "recording_start", "initial_angle_deg": initial_angle_deg})

    def sensor(self, measurement_ns: int, angle_deg: float, valid: bool = True) -> None:
        if not valid or not math.isfinite(angle_deg):
            return
        self.last_sensor_ns = max(measurement_ns, self.last_sensor_ns or measurement_ns)
        if self.start_ns is None or measurement_ns < self.start_ns:
            return
        deadline = self.start_ns + self.duration_ns
        if self.state == TrialState.RECORDING:
            if measurement_ns >= deadline:
                self._stop(deadline, "duration")
            elif abs(angle_deg) > self.angle_limit_deg:
                self._stop(measurement_ns, "angle_limit")
        elif (self.state == TrialState.STOPPED and self.reason == "duration" and
              measurement_ns < (self.stop_ns or deadline) and abs(angle_deg) > self.angle_limit_deg):
            self.stop_ns, self.reason = measurement_ns, "angle_limit"
            self.events.append({"t_pc_ns": measurement_ns, "event": "stop_correction", "reason": "late_angle_sample_before_deadline", "valid_trial": True})

    def camera(self, timestamp_ns: int) -> None:
        self.last_camera_ns = timestamp_ns

    def poll(self, now_ns: int) -> None:
        if self.state != TrialState.RECORDING or self.start_ns is None:
            return
        if now_ns >= self.start_ns + self.duration_ns:
            self._stop(self.start_ns + self.duration_ns, "duration")
        elif self.last_sensor_ns is None or now_ns - self.last_sensor_ns > self.stale_sensor_ns:
            self._stop(now_ns, "sensor_stale", invalid=True)
        elif self.last_camera_ns is None or now_ns - self.last_camera_ns > self.stale_camera_ns:
            self._stop(now_ns, "camera_stale", invalid=True)

    def disconnect(self, at_ns: int, source: str) -> None:
        if self.state == TrialState.RECORDING:
            self._stop(at_ns, f"{source}_disconnected", invalid=True)

    def cancel(self, at_ns: int) -> None:
        self._stop(at_ns, "operator_cancelled", invalid=True)

    def invalidate(self, at_ns: int, reason: str) -> None:
        self.stop_ns = min(at_ns, self.stop_ns) if self.stop_ns is not None else at_ns
        self.reason, self.state = reason, TrialState.INVALID
        self.events.append({"t_pc_ns": at_ns, "event": "recording_invalid", "reason": reason, "valid_trial": False})

    def _stop(self, at_ns: int, reason: str, invalid: bool = False) -> None:
        if self.state in (TrialState.STOPPED, TrialState.INVALID):
            return
        self.stop_ns, self.reason = at_ns, reason
        self.state = TrialState.INVALID if invalid else TrialState.STOPPED
        self.events.append({"t_pc_ns": at_ns, "event": "recording_stop", "reason": reason, "valid_trial": not invalid})

