from __future__ import annotations

import csv
import json
import math
import queue
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .angles import interpolate_circular, wrap_deg
from .camera import OpenCVCamera
from .metrics import angular_metrics
from .protocol import SensorSample, SequenceTracker, parse_packet, sync_request
from .session import TrialController, TrialState
from .sync import ClockMap, SyncExchange, clock_quality, fit_clock
from .vision import HSVRange, detect_markers


SENSOR_FIELDS = ["boot_id", "seq", "t_device_us", "t_pc_ns", "t_receive_ns", "high_ticks", "period_ticks", "raw_angle_deg", "angle_deg", "valid", "capture_drops", "tx_drops"]
FRAME_FIELDS = ["frame_id", "read_start_ns", "read_end_ns", "t_pc_ns", "timestamp_kind"]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _frequency(timestamps_ns: list[int]) -> dict[str, float | int]:
    if len(timestamps_ns) < 2:
        return {"count": len(timestamps_ns), "mean_hz": 0.0, "median_hz": 0.0}
    dt = np.diff(np.asarray(timestamps_ns, np.int64)) / 1e9
    dt = dt[dt > 0]
    return {"count": len(timestamps_ns), "mean_hz": float((len(timestamps_ns) - 1) / ((timestamps_ns[-1] - timestamps_ns[0]) / 1e9)), "median_hz": float(1.0 / np.median(dt))}


def _pwm_frequency(rows: list[dict]) -> dict[str, float | int]:
    periods = np.asarray([int(row["period_ticks"]) for row in rows if int(row["period_ticks"]) > 0], dtype=float)
    if not len(periods):
        return {"count": 0, "mean_hz": 0.0, "median_hz": 0.0, "assumed_capture_clock_hz": 80_000_000}
    hz = 80_000_000.0 / periods
    return {"count": len(periods), "mean_hz": float(hz.mean()), "median_hz": float(np.median(hz)), "assumed_capture_clock_hz": 80_000_000}


def _bounded_close(resource, timeout_s: float = 1.0) -> bool:
    if resource is None: return True
    thread=threading.Thread(target=resource.close,daemon=True); thread.start(); thread.join(timeout_s)
    return not thread.is_alive()


def _write_failure(root: Path, stage: str, exc: Exception) -> None:
    (root/"metadata.json").write_text(json.dumps({"schema_version":1,"valid_trial":False,"stop_reason":f"{stage}_failed","error":f"{type(exc).__name__}: {exc}"},ensure_ascii=False,indent=2),encoding="utf-8")


def _compress_frame(frame) -> tuple[dict, bytes]:
    """Detach small immutable metadata and JPEG bytes from the raw image array."""
    ok,encoded=cv2.imencode(".jpg",frame.image,[cv2.IMWRITE_JPEG_QUALITY,95])
    if not ok: raise RuntimeError("JPEG encoding failed")
    metadata={"read_start_ns":frame.read_start_ns,"read_end_ns":frame.read_end_ns,"t_pc_ns":frame.timestamp_ns,"timestamp_kind":frame.timestamp_kind}
    return metadata,encoded.tobytes()


def align_session(session_dir: str | Path, max_gap_ms: float = 25.0) -> dict:
    root = Path(session_dir)
    with (root / "sensor.csv").open(newline="", encoding="utf-8") as handle:
        sensors = [row for row in csv.DictReader(handle) if row["valid"] == "1"]
    with (root / "frames.csv").open(newline="", encoding="utf-8") as handle:
        frames = list(csv.DictReader(handle))
    st = np.asarray([int(row["t_pc_ns"]) for row in sensors], np.int64)
    sa = np.asarray([float(row["angle_deg"]) for row in sensors], float)
    ft = np.asarray([int(row["t_pc_ns"]) for row in frames], np.int64)
    angle, valid = interpolate_circular(st, sa, ft, max_gap_ms=max_gap_ms)
    aligned = []
    for row, value, ok in zip(frames, angle, valid):
        aligned.append({"frame_id": row["frame_id"], "t_pc_ns": row["t_pc_ns"], "sensor_angle_deg": "" if not ok else f"{value:.8f}", "aligned_valid": int(ok), "vision_angle_deg": "", "vision_valid": 0})
    _write_csv(root / "aligned.csv", ["frame_id", "t_pc_ns", "sensor_angle_deg", "aligned_valid", "vision_angle_deg", "vision_valid"], aligned)
    return {"frames": len(frames), "aligned_valid": int(valid.sum()), "aligned_invalid": int((~valid).sum()), "max_gap_ms": max_gap_ms}


def analyze_vision(session_dir: str | Path, *, pivot_hsv: HSVRange, tip_hsv: HSVRange, zero_offset_deg: float = 0.0, direction: float = 1.0, minimum_area: float = 20.0) -> dict:
    root = Path(session_dir)
    with (root / "aligned.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    capture = cv2.VideoCapture(str(root / "rgb.mp4"))
    detections = []
    while True:
        ok, frame = capture.read()
        if not ok: break
        detections.append(detect_markers(frame, pivot_hsv, tip_hsv, minimum_area=minimum_area, zero_offset_deg=zero_offset_deg, direction=direction))
    capture.release()
    if len(detections) != len(rows):
        raise ValueError(f"video/frame table mismatch: {len(detections)} decoded frames vs {len(rows)} rows")
    for row, detection in zip(rows, detections):
        row["vision_angle_deg"] = f"{detection.angle_deg:.8f}" if detection.valid else ""
        row["vision_valid"] = int(detection.valid)
    _write_csv(root / "aligned.csv", ["frame_id", "t_pc_ns", "sensor_angle_deg", "aligned_valid", "vision_angle_deg", "vision_valid"], rows)
    metrics = angular_metrics(
        [float(row["sensor_angle_deg"]) if row["aligned_valid"] == "1" else np.nan for row in rows],
        [d.angle_deg for d in detections], [d.valid for d in detections],
    )
    result = {"vision_vs_sensor": metrics, "invalid_vision_frames": sum(not d.valid for d in detections), "note": "angle metrics; RGB prediction metrics require paired predicted images"}
    (root / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def synthetic_demo(output: str | Path, *, category: str = "free_fall", duration_s: float = 6.0, seed: int = 7) -> Path:
    if category not in {"free_fall", "assisted_balance"}:
        raise ValueError("category must be free_fall or assisted_balance")
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    base_ns = 10_000_000_000
    controller = TrialController(duration_ns=int(duration_s * 1e9))
    controller.sensor(base_ns,2.0); controller.camera(base_ns)
    controller.set_ready(base_ns, sensor_stable=True, camera_stable=True, clock_valid=True, geometry_confirmed=True, zero_calibrated=True)
    controller.release(base_ns, 2.0)
    sensor_rows: list[dict] = []
    stop_candidate = base_ns + int(duration_s * 1e9)
    for seq, elapsed in enumerate(np.arange(0.0, duration_s + 0.1, 0.005)):
        if category == "free_fall":
            angle = 2.0 + 15.0 * elapsed * elapsed
        else:
            angle = 4.0 * math.sin(2 * math.pi * 0.7 * elapsed) + 0.5 * math.sin(2 * math.pi * 2.1 * elapsed)
        measured_ns = base_ns + int(elapsed * 1e9)
        noisy = wrap_deg(angle + rng.normal(0.0, 0.08))
        controller.sensor(measured_ns, noisy)
        sensor_rows.append({"boot_id": "synthetic", "seq": seq, "t_device_us": int(elapsed * 1e6), "t_pc_ns": measured_ns, "t_receive_ns": measured_ns + 800_000, "high_ticks": int(((noisy % 360) / 360 * 4098 + 1) * 80), "period_ticks": 4098 * 80, "raw_angle_deg": noisy % 360, "angle_deg": noisy, "valid": 1, "capture_drops": 0, "tx_drops": 0})
        if controller.state == TrialState.STOPPED:
            stop_candidate = controller.stop_ns or measured_ns
            break
    controller.poll(stop_candidate)
    if controller.state == TrialState.RECORDING:
        controller.poll(base_ns + int(duration_s * 1e9))
    stop_ns = controller.stop_ns or stop_candidate
    sensor_rows = [row for row in sensor_rows if row["t_pc_ns"] <= stop_ns]

    width, height, fps = 640, 480, 30.0
    writer = cv2.VideoWriter(str(root / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_rows: list[dict] = []
    vision_angles: list[float] = []
    vision_valid: list[bool] = []
    pivot_hsv = HSVRange((45, 100, 100), (85, 255, 255))
    tip_hsv = HSVRange((0, 150, 120), (10, 255, 255))
    frame_times = np.arange(base_ns, stop_ns + 1, int(1e9 / fps), dtype=np.int64)
    sensor_t = [row["t_pc_ns"] for row in sensor_rows]
    sensor_a = [row["angle_deg"] for row in sensor_rows]
    truth, truth_valid = interpolate_circular(sensor_t, sensor_a, frame_times, max_gap_ms=20)
    for frame_id, (stamp, angle, ok) in enumerate(zip(frame_times, truth, truth_valid)):
        image = np.full((height, width, 3), 30, np.uint8)
        pivot = (320, 380); length = 230
        rad = math.radians(float(angle if ok else 0.0))
        tip = (round(pivot[0] + length * math.sin(rad)), round(pivot[1] - length * math.cos(rad)))
        cv2.line(image, pivot, tip, (0, 190, 240), 12)
        cv2.circle(image, pivot, 18, (0, 255, 0), -1)
        cv2.circle(image, tip, 18, (0, 0, 255), -1)
        writer.write(image)
        read_start, read_end = int(stamp - 2_000_000), int(stamp)
        frame_rows.append({"frame_id": frame_id, "read_start_ns": read_start, "read_end_ns": read_end, "t_pc_ns": int(stamp), "timestamp_kind": "synthetic_exposure_time"})
        detection = detect_markers(image, pivot_hsv, tip_hsv, minimum_area=50)
        vision_angles.append(detection.angle_deg); vision_valid.append(detection.valid)
    writer.release()
    _write_csv(root / "sensor.csv", SENSOR_FIELDS, sensor_rows)
    _write_csv(root / "frames.csv", FRAME_FIELDS, frame_rows)
    align_summary = align_session(root, max_gap_ms=20)
    with (root / "aligned.csv").open(newline="", encoding="utf-8") as handle:
        aligned = list(csv.DictReader(handle))
    for row, value, ok in zip(aligned, vision_angles, vision_valid):
        row["vision_angle_deg"] = "" if not ok else f"{value:.8f}"; row["vision_valid"] = int(ok)
    _write_csv(root / "aligned.csv", ["frame_id", "t_pc_ns", "sensor_angle_deg", "aligned_valid", "vision_angle_deg", "vision_valid"], aligned)
    metric = angular_metrics([float(row["sensor_angle_deg"]) if row["aligned_valid"] == "1" else np.nan for row in aligned], vision_angles, vision_valid)
    (root / "metrics.json").write_text(json.dumps({"vision_vs_sensor": metric, "note": "angular metrics only; image MSE/SSIM/LPIPS require paired predicted RGB images"}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(root / "events.csv", sorted({key for event in controller.events for key in event}), controller.events)
    sync_rows = []
    for i in range(8):
        t0 = base_ns - 2_000_000_000 + i * 200_000_000
        sync_rows.append({"id": i, "t0_pc_ns": t0, "t1_device_us": i * 200_000, "t2_device_us": i * 200_000 + 50, "t3_pc_ns": t0 + 1_000_000, "rtt_ns": 950_000})
    _write_csv(root / "sync.csv", list(sync_rows[0]), sync_rows)
    metadata = {"schema_version": 1, "session_id": str(uuid.uuid4()), "source": "synthetic", "category": category, "category_meaning": "unassisted after release key" if category == "free_fall" else "human may support after release key", "release_semantics": "keyboard event only; synthetic demo has no human-release detector", "valid_trial": controller.state == TrialState.STOPPED, "stop_reason": controller.reason, "start_pc_ns": controller.start_ns, "stop_pc_ns": controller.stop_ns, "duration_s": (stop_ns - base_ns) / 1e9, "sensor_frequency": _frequency([row["t_pc_ns"] for row in sensor_rows]), "camera_frequency": _frequency([row["t_pc_ns"] for row in frame_rows]), "camera": {"adapter": "synthetic", "timestamp_kind": "synthetic_exposure_time", "latency_offset_ns": 0}, "clock": {"kind": "synthetic_exact"}, "geometry": {"sensor_housing_fixed": True, "single_plane": True, "zero_calibrated": True}, "alignment": align_summary}
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


class _ReceiverBase:
    def __init__(self):
        self.samples: queue.Queue[SensorSample] = queue.Queue(maxsize=10000)
        self.responses: queue.Queue[tuple[dict, int]] = queue.Queue()
        self.stop = threading.Event(); self.tracker = SequenceTracker(); self.boot_id: str | None = None
        self.host_queue_drops = 0; self.error: str | None = None

    def _consume(self, data: bytes, transport: str, stamp: int) -> None:
        item = parse_packet(data, transport, stamp)
        if isinstance(item, SensorSample):
            accepted, _ = self.tracker.accept(item)
            if accepted:
                try: self.samples.put_nowait(item)
                except queue.Full: self.host_queue_drops += 1
        elif isinstance(item, dict) and item.get("type") == "sync_resp":
            self.responses.put((item, stamp))

    def _send_sync(self, payload: bytes) -> None:
        raise NotImplementedError

    def synchronize(self, count: int = 16, timeout_s: float = 1.0) -> tuple[ClockMap, list[SyncExchange]]:
        exchanges=[]; boots=set()
        for request_id in range(count):
            while not self.responses.empty(): self.responses.get_nowait()
            t0=time.monotonic_ns(); self._send_sync(sync_request(request_id,t0)); deadline=time.monotonic()+timeout_s
            while time.monotonic()<deadline:
                try: obj,t3=self.responses.get(timeout=.05)
                except queue.Empty: continue
                try:
                    response_id=int(obj.get("id",-1)); echoed_t0=int(obj.get("t0_pc_ns",-1)); boot=str(obj.get("boot_id",""))
                    if response_id!=request_id or echoed_t0!=t0 or not boot: continue
                    exchange=SyncExchange(t0,int(obj["t1_device_us"]),int(obj["t2_device_us"]),t3)
                except (KeyError,TypeError,ValueError):
                    continue
                boots.add(boot); exchanges.append(exchange); break
            time.sleep(.03)
        if len(boots)!=1 or "" in boots: raise RuntimeError("sync responses have missing or changing boot_id")
        self.boot_id=boots.pop()
        return fit_clock(exchanges),exchanges


class SerialReceiver(_ReceiverBase):
    def __init__(self, port: str, baud: int):
        import serial
        super().__init__()
        self.serial = serial.Serial(port, baudrate=baud, timeout=0.05)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self): self.thread.start()
    def close(self): self.stop.set(); self.thread.join(1); self.serial.close()
    def _run(self):
        while not self.stop.is_set():
            try: data = self.serial.readline()
            except Exception as exc:
                self.error=f"{type(exc).__name__}: {exc}"; self.stop.set(); break
            stamp = time.monotonic_ns()
            if not data: continue
            self._consume(data,"serial",stamp)

    def _send_sync(self,payload: bytes) -> None: self.serial.write(payload)


class UDPReceiver(_ReceiverBase):
    def __init__(self, host: str, port: int, local_port: int = 0):
        import socket
        if not 1<=int(port)<=65535: raise ValueError("UDP port must be in 1..65535")
        resolved=socket.gethostbyname(host)
        super().__init__(); self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); self.socket.bind(("0.0.0.0",local_port)); self.socket.settimeout(.05)
        self.device=(resolved,int(port)); self.thread=threading.Thread(target=self._run,daemon=True)

    def start(self): self.thread.start()
    def close(self): self.stop.set(); self.thread.join(1); self.socket.close()
    def _send_sync(self,payload: bytes) -> None: self.socket.sendto(payload,self.device)
    def _run(self):
        while not self.stop.is_set():
            try: data,address=self.socket.recvfrom(2048)
            except TimeoutError: continue
            except OSError as exc:
                if not self.stop.is_set(): self.error=f"{type(exc).__name__}: {exc}"
                break
            if address[0] != self.device[0]: continue
            self._consume(data,"udp",time.monotonic_ns())


def probe_sensor(*, port: str | None = None, baud: int = 921600, udp_device: tuple[str,int] | None = None, duration_s: float = 3.0) -> dict:
    receiver = UDPReceiver(*udp_device) if udp_device else SerialReceiver(str(port),baud)
    receiver.start(); samples=[]
    try:
        if udp_device:
            clock,exchanges=receiver.synchronize(count=6,timeout_s=.5)
            good,reasons=clock_quality(clock)
            if not good: raise RuntimeError("UDP probe sync gate failed: "+", ".join(reasons))
        deadline=time.monotonic()+duration_s
        while time.monotonic()<deadline:
            try: item=receiver.samples.get(timeout=.1)
            except queue.Empty: continue
            if item.valid: samples.append(item)
    finally: receiver.close()
    if not samples: raise RuntimeError("no valid SA100 samples received")
    radians=np.deg2rad([item.raw_angle_deg for item in samples]); mean=float(np.rad2deg(np.arctan2(np.sin(radians).mean(),np.cos(radians).mean()))%360)
    return {"valid_samples":len(samples),"suggested_zero_deg":mean,"raw_min_deg":min(i.raw_angle_deg for i in samples),"raw_max_deg":max(i.raw_angle_deg for i in samples),"reported_hz":_frequency([i.t_device_us*1000 for i in samples]),"boot_id":samples[-1].boot_id}

def capture_real(output: str | Path, *, port: str | None, camera_source: int, category: str, zero_deg: float, direction: float, geometry_confirmed: bool, baud: int = 921600, camera_latency_ms: float = 0.0, duration_s: float = 6.0, udp_device: tuple[str,int] | None = None, fps: float = 30.0, width: int = 640, height: int = 480, readiness_timeout_s: float = 15.0, receiver_override=None, camera_override=None, key_provider=None, show_preview: bool = True) -> Path:
    if not geometry_confirmed: raise ValueError("confirm that sensor-housing orientation is world-fixed and motion is planar")
    if category not in {"free_fall", "assisted_balance"}: raise ValueError("invalid category")
    numeric=(zero_deg,direction,camera_latency_ms,duration_s,fps,readiness_timeout_s)
    if not all(math.isfinite(float(value)) for value in numeric): raise ValueError("numeric capture arguments must be finite")
    if duration_s<=0 or fps<=0 or readiness_timeout_s<=0 or camera_latency_ms<0 or width<=0 or height<=0: raise ValueError("duration/fps/timeout/size must be positive and camera latency non-negative")
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    receiver=camera=None
    try:
        if receiver_override is not None: receiver=receiver_override; transport_kind="test"
        elif udp_device is not None: receiver = UDPReceiver(*udp_device); transport_kind="udp"
        elif port is not None: receiver = SerialReceiver(port,baud); transport_kind="serial"
        else: raise ValueError("serial port or UDP device is required")
        receiver.start()
        camera = camera_override or OpenCVCamera(camera_source,width=width,height=height,fps=fps,latency_offset_ns=int(camera_latency_ms * 1e6))
        clock, exchanges = receiver.synchronize(); clock_ref=[clock]; sync_fit_ns=[time.monotonic_ns()]
        good_clock, reasons = clock_quality(clock)
        if not good_clock: raise RuntimeError("clock sync gate failed: "+", ".join(reasons))
    except Exception as exc:
        _bounded_close(camera); _bounded_close(receiver); _write_failure(root,"initialization",exc); raise
    frame_queue: queue.Queue = queue.Queue(maxsize=8); camera_stop = threading.Event(); camera_done=threading.Event(); camera_drops=[0]
    def camera_worker():
        while not camera_stop.is_set():
            frame = camera.read()
            if frame is None: break
            try: frame_queue.put_nowait(frame)
            except queue.Full:
                camera_drops[0]+=1
                try: frame_queue.get_nowait(); frame_queue.put_nowait(frame)
                except queue.Empty: pass
        camera_done.set()
    camera_thread = threading.Thread(target=camera_worker, daemon=True); camera_thread.start()
    recent_sensor = deque(maxlen=200); recent_frames = deque(maxlen=30)
    sensor_rows=[]; buffered_frames=[]; max_buffer_frames=int(math.ceil((duration_s+1.0)*fps)); runtime_error=None
    controller = TrialController(duration_ns=int(duration_s*1e9)); lock=threading.Lock(); stop_monitor=threading.Event()
    readiness_deadline=time.monotonic_ns()+int(readiness_timeout_s*1e9)
    latest_angle = [None]
    def sensor_worker():
        while not stop_monitor.is_set():
            try: sample=receiver.samples.get(timeout=.02)
            except queue.Empty:
                with lock: controller.poll(time.monotonic_ns())
                continue
            with lock:
                if sample.boot_id != receiver.boot_id:
                    recent_sensor.clear(); controller.invalidate(time.monotonic_ns(),"device_restart_requires_resync"); continue
                active_clock=clock_ref[0]
            stamp=int(active_clock.device_us_to_pc_ns(sample.t_device_us)); angle=wrap_deg(direction*(sample.raw_angle_deg-zero_deg))
            row={**sample.__dict__, "t_pc_ns":stamp, "angle_deg":angle, "valid":int(sample.valid)}; sensor_rows.append(row)
            if sample.valid:
                latest_angle[0]=angle; recent_sensor.append(stamp)
            with lock: controller.sensor(stamp,angle,sample.valid); controller.poll(time.monotonic_ns())
    monitor=threading.Thread(target=sensor_worker,daemon=True); monitor.start()
    try:
        if show_preview: print("预览中：传感器、相机和时钟稳定后按 r，随后同时松手；Esc 取消。")
        while True:
            try: frame=frame_queue.get(timeout=.05)
            except queue.Empty:
                empty_key=(key_provider(controller.state,latest_angle[0]) if key_provider else (cv2.waitKey(1)&0xff if show_preview else -1))
                with lock:
                    now=time.monotonic_ns(); controller.poll(now)
                    if empty_key==27: controller.cancel(now)
                    if receiver.error: controller.invalidate(now,"transport_error")
                    elif controller.state in (TrialState.WAIT_READY,TrialState.READY) and now>=readiness_deadline: controller.cancel(now)
                    if camera_done.is_set():
                        if controller.state==TrialState.RECORDING: controller.disconnect(now,"camera")
                        else: controller.cancel(now)
                if controller.state in (TrialState.STOPPED,TrialState.INVALID): break
                continue
            recent_frames.append(frame.timestamp_ns)
            if controller.state in (TrialState.WAIT_READY,TrialState.READY) and time.monotonic_ns()>=readiness_deadline:
                with lock: controller.cancel(time.monotonic_ns())
            if controller.state in (TrialState.WAIT_READY,TrialState.READY) and time.monotonic_ns()-sync_fit_ns[0]>5_000_000_000:
                old_boot=receiver.boot_id
                try:
                    refreshed,new_exchanges=receiver.synchronize(count=8,timeout_s=.5)
                    refresh_good,_=clock_quality(refreshed)
                    if not refresh_good or receiver.boot_id!=old_boot: raise RuntimeError("clock refresh failed or boot changed")
                    with lock: clock_ref[0]=refreshed; sync_fit_ns[0]=time.monotonic_ns()
                    exchanges.extend(new_exchanges)
                except Exception:
                    with lock:
                        sync_fit_ns[0]=0
                        if receiver.boot_id!=old_boot: controller.invalidate(time.monotonic_ns(),"device_restart_during_sync")
            with lock:
                controller.camera(frame.timestamp_ns)
                if controller.state in (TrialState.WAIT_READY, TrialState.READY):
                    now=time.monotonic_ns(); stable_sensor=len(recent_sensor)>=100 and now-recent_sensor[-1]<controller.stale_sensor_ns
                    stable_camera=len(recent_frames)>=10 and now-recent_frames[-1]<controller.stale_camera_ns
                    sync_fresh=now-sync_fit_ns[0]<10_000_000_000
                    controller.set_ready(now,sensor_stable=stable_sensor,camera_stable=stable_camera,clock_valid=sync_fresh,geometry_confirmed=True,zero_calibrated=True)
                state=controller.state
            if show_preview:
                display=frame.image.copy(); cv2.putText(display,f"state={state.value} angle={latest_angle[0]}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2); cv2.imshow("SA100 capture",display)
            key=(key_provider(state,latest_angle[0]) if key_provider else (cv2.waitKey(1)&0xff if show_preview else -1))
            if key==27:
                with lock: controller.cancel(time.monotonic_ns())
                break
            if key==ord('r') and latest_angle[0] is not None:
                with lock:
                    try: controller.release(time.monotonic_ns(),float(latest_angle[0]))
                    except RuntimeError: pass
            with lock: recording=controller.state==TrialState.RECORDING; finished=controller.state in (TrialState.STOPPED,TrialState.INVALID)
            if recording and controller.start_ns is not None and frame.timestamp_ns>=controller.start_ns:
                try: compressed=_compress_frame(frame)
                except RuntimeError: compressed=None
                if compressed is None or len(buffered_frames)>=max_buffer_frames:
                    with lock: controller.invalidate(time.monotonic_ns(),"frame_buffer_failed")
                else: buffered_frames.append(compressed)
            if finished: break
    except Exception as exc:
        runtime_error=f"{type(exc).__name__}: {exc}"
        with lock: controller.invalidate(time.monotonic_ns(),"capture_runtime_error")
    finally:
        if controller.reason=="duration": time.sleep(.05)
        stop_monitor.set(); camera_stop.set(); monitor.join(1); camera_thread.join(.2)
        camera_closed=_bounded_close(camera); receiver_closed=_bounded_close(receiver)
        camera_thread.join(.8)
        if show_preview: cv2.destroyAllWindows()
        if not camera_closed or camera_thread.is_alive(): controller.invalidate(time.monotonic_ns(),"camera_close_timeout")
        if not receiver_closed: controller.invalidate(time.monotonic_ns(),"transport_close_timeout")
    start_ns,stop_ns=controller.start_ns,controller.stop_ns
    if start_ns is not None and stop_ns is not None:
        sensor_rows[:]=[r for r in sensor_rows if start_ns<=int(r['t_pc_ns'])<=stop_ns]; buffered_frames[:]=[(m,b) for m,b in buffered_frames if start_ns<=m["t_pc_ns"]<=stop_ns]
    frame_rows=[]
    if buffered_frames:
        first=cv2.imdecode(np.frombuffer(buffered_frames[0][1],np.uint8),cv2.IMREAD_COLOR); h,w=first.shape[:2]
        video=cv2.VideoWriter(str(root/'rgb.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not video.isOpened(): controller.invalidate(time.monotonic_ns(),"video_writer_open_failed")
        for new_id,(frame_meta,jpeg) in enumerate(buffered_frames):
            image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
            if image is None or not video.isOpened(): controller.invalidate(time.monotonic_ns(),"video_writer_failed"); break
            video.write(image); frame_rows.append({"frame_id":new_id,**frame_meta})
        video.release()
    _write_csv(root/'sensor.csv',SENSOR_FIELDS,sensor_rows); _write_csv(root/'frames.csv',FRAME_FIELDS,frame_rows)
    sync_rows=[{"t0_pc_ns":e.t0_pc_ns,"t1_device_us":e.t1_device_us,"t2_device_us":e.t2_device_us,"t3_pc_ns":e.t3_pc_ns,"rtt_ns":e.rtt_ns} for e in exchanges]; _write_csv(root/'sync.csv',list(sync_rows[0]),sync_rows)
    alignment=align_session(root)
    if controller.state==TrialState.STOPPED and (not sensor_rows or not frame_rows or alignment["aligned_valid"]==0):
        controller.invalidate(controller.stop_ns or time.monotonic_ns(),"insufficient_recorded_data")
    start_ns,stop_ns=controller.start_ns,controller.stop_ns
    _write_csv(root/'events.csv',sorted({k for e in controller.events for k in e}),controller.events)
    pwm_frequency=_pwm_frequency(sensor_rows)
    metadata={
        "schema_version":1,"category":category,
        "release_semantics":"r key event; visual verification remains required",
        "valid_trial":controller.state==TrialState.STOPPED,"stop_reason":controller.reason,
        "start_pc_ns":start_ns,"stop_pc_ns":stop_ns,
        "sensor_report_frequency":_frequency([r['t_pc_ns'] for r in sensor_rows]),
        "pwm_frequency_from_period_ticks":pwm_frequency,
        "camera_frequency":_frequency([r['t_pc_ns'] for r in frame_rows]),
        "clock":{"slope":clock_ref[0].slope,"intercept_ns":clock_ref[0].intercept_ns,"drift_ppm":clock_ref[0].drift_ppm,"rms_residual_ns":clock_ref[0].rms_residual_ns,"max_rtt_ns":clock_ref[0].max_rtt_ns,"fit_pc_ns":sync_fit_ns[0],"exchange_start_pc_ns":min(e.t0_pc_ns for e in exchanges),"exchange_stop_pc_ns":max(e.t3_pc_ns for e in exchanges)},
        "camera":{"adapter":"opencv_usb_rgb","timestamp_kind":camera.timestamp_kind,"latency_offset_ns":int(camera_latency_ms*1e6),"requested_fps":fps,"actual_fps":getattr(camera,"actual_fps",None),"requested_width":width,"requested_height":height,"actual_width":getattr(camera,"actual_width",None),"actual_height":getattr(camera,"actual_height",None),"queue_drops":camera_drops[0],"buffer_limit_frames":max_buffer_frames},
        "geometry":{"sensor_housing_orientation_world_fixed":True,"translation_allowed":True,"single_plane":True,"zero_deg":zero_deg,"direction":direction},
        "transport":{"kind":transport_kind,"baud":baud if transport_kind=='serial' else None,"sequence_gaps":receiver.tracker.gaps,"duplicates":receiver.tracker.duplicates,"reordered":receiver.tracker.reordered,"restarts":receiver.tracker.restarts,"host_queue_drops":receiver.host_queue_drops,"error":receiver.error,"firmware_capture_drops":max((int(r['capture_drops']) for r in sensor_rows),default=0),"firmware_tx_drops":max((int(r['tx_drops']) for r in sensor_rows),default=0)},
        "alignment":alignment,
        "runtime_error":runtime_error,
    }
    (root/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return root
