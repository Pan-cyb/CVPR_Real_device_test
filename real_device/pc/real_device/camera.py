from __future__ import annotations

from dataclasses import dataclass
import time
import cv2


@dataclass
class CameraFrame:
    frame_id: int
    image: object
    read_start_ns: int
    read_end_ns: int
    timestamp_ns: int
    timestamp_kind: str


class OpenCVCamera:
    """Generic USB RGB adapter. read_end is a host-side proxy, not exposure time."""

    timestamp_kind = "host_monotonic_read_end_proxy"

    def __init__(self, source=0, width: int | None = None, height: int | None = None, fps: float | None = None, latency_offset_ns: int = 0):
        self.capture = cv2.VideoCapture(source)
        if width: self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height: self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps: self.capture.set(cv2.CAP_PROP_FPS, fps)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open camera {source!r}")
        self.frame_id = 0
        self.latency_offset_ns = latency_offset_ns
        self.actual_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> CameraFrame | None:
        start = time.monotonic_ns()
        ok, image = self.capture.read()
        end = time.monotonic_ns()
        if not ok:
            return None
        item = CameraFrame(self.frame_id, image, start, end, end - self.latency_offset_ns, self.timestamp_kind)
        self.frame_id += 1
        return item

    def close(self) -> None:
        self.capture.release()


class DeviceTimestampCamera:
    """Extension contract for RealSense/depth adapters with device-clock mapping.

    Concrete adapters must return exposure/device timestamps mapped to PC monotonic
    time and name the mapping in timestamp_kind. No such adapter is bundled yet.
    """

    def read(self) -> CameraFrame | None:
        raise NotImplementedError

