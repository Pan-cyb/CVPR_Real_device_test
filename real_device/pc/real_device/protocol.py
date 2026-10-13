from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class SensorSample:
    version: int
    boot_id: str
    seq: int
    t_device_us: int
    high_ticks: int
    period_ticks: int
    raw_angle_deg: float
    valid: bool
    capture_drops: int = 0
    tx_drops: int = 0
    transport: str = "unknown"
    t_receive_ns: int | None = None


class SequenceTracker:
    """Reject duplicates/reordering and expose gaps and device restarts."""

    def __init__(self) -> None:
        self.boot_id: str | None = None
        self.last_seq: int | None = None
        self.gaps = 0
        self.duplicates = 0
        self.reordered = 0
        self.restarts = 0

    def accept(self, sample: SensorSample) -> tuple[bool, str]:
        if self.boot_id != sample.boot_id:
            if self.boot_id is not None:
                self.restarts += 1
            self.boot_id = sample.boot_id
            self.last_seq = sample.seq
            return True, "new_boot"
        assert self.last_seq is not None
        if sample.seq == self.last_seq:
            self.duplicates += 1
            return False, "duplicate"
        if sample.seq < self.last_seq:
            self.reordered += 1
            return False, "reordered"
        if sample.seq > self.last_seq + 1:
            self.gaps += sample.seq - self.last_seq - 1
        self.last_seq = sample.seq
        return True, "ok"


def parse_packet(data: bytes | str, transport: str = "unknown", receive_ns: int | None = None) -> SensorSample | dict[str, Any] | None:
    """Parse one JSON packet. Boot chatter and malformed lines are ignored."""
    try:
        obj = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or obj.get("v") != PROTOCOL_VERSION:
        return None
    if obj.get("type") != "sample":
        return obj
    required = ("boot_id", "seq", "t_device_us", "high_ticks", "period_ticks", "raw_angle_deg", "valid")
    if any(key not in obj for key in required):
        return None
    try:
        if type(obj["valid"]) is not bool:
            return None
        sample = SensorSample(
            version=int(obj["v"]), boot_id=str(obj["boot_id"]), seq=int(obj["seq"]),
            t_device_us=int(obj["t_device_us"]), high_ticks=int(obj["high_ticks"]),
            period_ticks=int(obj["period_ticks"]), raw_angle_deg=float(obj["raw_angle_deg"]),
            valid=obj["valid"], capture_drops=int(obj.get("capture_drops", 0)),
            tx_drops=int(obj.get("tx_drops", 0)), transport=transport, t_receive_ns=receive_ns,
        )
        if (sample.seq < 0 or sample.t_device_us < 0 or sample.high_ticks < 0 or
                sample.period_ticks < 0 or sample.capture_drops < 0 or sample.tx_drops < 0 or
                not math.isfinite(sample.raw_angle_deg)):
            return None
        return sample
    except (TypeError, ValueError):
        return None


def sync_request(request_id: int, t0_pc_ns: int) -> bytes:
    return (json.dumps({"v": PROTOCOL_VERSION, "type": "sync_req", "id": request_id, "t0_pc_ns": t0_pc_ns}, separators=(",", ":")) + "\n").encode()
