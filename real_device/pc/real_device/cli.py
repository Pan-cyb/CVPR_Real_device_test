from __future__ import annotations

import argparse
import csv
import json
import socket
from pathlib import Path
import numpy as np

from .metrics import angular_metrics, load_prediction_csv
from .angles import interpolate_circular
from .pipeline import align_session, analyze_vision, capture_real, probe_sensor, synthetic_demo
from .export import export_angle_validation, export_training_episode
from .vision import HSVRange


def _parse_udp_device(value: str | None):
    if value is None: return None
    try: host,port_text=value.rsplit(":",1); port=int(port_text)
    except (ValueError,TypeError): raise ValueError("--udp-device must be HOST:PORT") from None
    if not host or not 1<=port<=65535: raise ValueError("--udp-device port must be in 1..65535")
    try: host=socket.gethostbyname(host)
    except OSError as exc: raise ValueError(f"cannot resolve UDP host: {exc}") from None
    return host,port


def evaluate_predictions(session: Path, prediction: Path, max_gap_ms: float = 50.0) -> dict:
    rows = load_prediction_csv(prediction)
    with (session / "aligned.csv").open(newline="", encoding="utf-8") as handle:
        aligned = list(csv.DictReader(handle))
    if "frame_id" in rows[0]:
        keys = [int(row["frame_id"]) for row in rows]
        if len(set(keys)) != len(keys): raise ValueError("duplicate frame_id in predictions")
        truth = {int(r["frame_id"]): float(r["sensor_angle_deg"]) for r in aligned if r["aligned_valid"] == "1"}
        reference, predicted = [], []
        for row in rows:
            key = int(row["frame_id"])
            if key in truth: reference.append(truth[key]); predicted.append(float(row["angle_deg"]))
    else:
        prediction_times = np.asarray([int(row["t_pc_ns"]) for row in rows], np.int64)
        if len(set(prediction_times.tolist())) != len(prediction_times): raise ValueError("duplicate t_pc_ns in predictions")
        times = np.asarray([int(r["t_pc_ns"]) for r in aligned if r["aligned_valid"] == "1"], np.int64)
        angles = np.asarray([float(r["sensor_angle_deg"]) for r in aligned if r["aligned_valid"] == "1"], float)
        interpolated, valid = interpolate_circular(times, angles, prediction_times, max_gap_ms=max_gap_ms)
        reference = interpolated[valid].tolist()
        predicted = [float(row["angle_deg"]) for row, ok in zip(rows, valid) if ok]
    return angular_metrics(reference, predicted)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="real-device", description="SA100 + RGB synchronized capture")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="generate a complete no-hardware demo")
    demo.add_argument("--output", type=Path, required=True); demo.add_argument("--category", choices=("free_fall", "assisted_balance"), default="free_fall")
    demo.add_argument("--duration", type=float, default=6.0)
    capture = commands.add_parser("capture", help="interactive serial + USB RGB capture")
    capture.add_argument("--output", type=Path, required=True); capture.add_argument("--port"); capture.add_argument("--udp-device", help="ESP IP:UDP_PORT instead of serial"); capture.add_argument("--baud", type=int, default=921600)
    capture.add_argument("--camera", type=int, default=0); capture.add_argument("--category", choices=("free_fall", "assisted_balance"), required=True)
    capture.add_argument("--fps",type=float,default=30.0); capture.add_argument("--width",type=int,default=640); capture.add_argument("--height",type=int,default=480)
    capture.add_argument("--zero-deg", type=float, required=True, help="raw SA100 angle measured at world upright")
    capture.add_argument("--direction", type=float, choices=(-1.0, 1.0), required=True); capture.add_argument("--duration", type=float, default=6.0)
    capture.add_argument("--camera-latency-ms", type=float, default=0.0, help="independently calibrated host read-end latency")
    capture.add_argument("--confirm-fixed-housing", action="store_true", help="confirm SA100 housing orientation is world-fixed (translation is allowed) and motion is planar")
    probe = commands.add_parser("probe", help="read raw angles for zero/direction calibration")
    probe.add_argument("--port"); probe.add_argument("--udp-device"); probe.add_argument("--baud",type=int,default=921600); probe.add_argument("--duration",type=float,default=3.0)
    align = commands.add_parser("align"); align.add_argument("session", type=Path); align.add_argument("--max-gap-ms", type=float, default=25.0)
    evaluate = commands.add_parser("evaluate"); evaluate.add_argument("session", type=Path); evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--max-gap-ms", type=float, default=50.0)
    vision = commands.add_parser("vision", help="run marker detection over a captured video")
    vision.add_argument("session", type=Path); vision.add_argument("--pivot-hsv", nargs=6, type=int, default=(45,100,100,85,255,255)); vision.add_argument("--tip-hsv", nargs=6, type=int, default=(0,150,120,10,255,255))
    vision.add_argument("--zero-deg", type=float, default=0.0); vision.add_argument("--direction", type=float, choices=(-1.0,1.0), default=1.0); vision.add_argument("--minimum-area", type=float, default=20.0)
    validation = commands.add_parser("export-validation"); validation.add_argument("sensor_csv", type=Path); validation.add_argument("output", type=Path)
    training = commands.add_parser("export-training", help="guarded: unavailable until real actions and graph labels exist")
    training.add_argument("session", type=Path)
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "demo": result = {"output": str(synthetic_demo(args.output, category=args.category, duration_s=args.duration))}
    elif args.command == "capture":
        if bool(args.port)==bool(args.udp_device): raise ValueError("provide exactly one of --port or --udp-device")
        udp=_parse_udp_device(args.udp_device)
        result={"output":str(capture_real(args.output,port=args.port,udp_device=udp,camera_source=args.camera,category=args.category,zero_deg=args.zero_deg,direction=args.direction,geometry_confirmed=args.confirm_fixed_housing,baud=args.baud,camera_latency_ms=args.camera_latency_ms,duration_s=args.duration,fps=args.fps,width=args.width,height=args.height))}
    elif args.command == "probe":
        if bool(args.port)==bool(args.udp_device): raise ValueError("provide exactly one of --port or --udp-device")
        udp=_parse_udp_device(args.udp_device)
        result=probe_sensor(port=args.port,udp_device=udp,baud=args.baud,duration_s=args.duration)
    elif args.command == "align": result = align_session(args.session, args.max_gap_ms)
    elif args.command == "evaluate": result = evaluate_predictions(args.session, args.predictions, args.max_gap_ms)
    elif args.command == "vision":
        p=args.pivot_hsv; t=args.tip_hsv
        result=analyze_vision(args.session,pivot_hsv=HSVRange(tuple(p[:3]),tuple(p[3:])),tip_hsv=HSVRange(tuple(t[:3]),tuple(t[3:])),zero_offset_deg=args.zero_deg,direction=args.direction,minimum_area=args.minimum_area)
    elif args.command == "export-validation": result = {"output": str(export_angle_validation(args.sensor_csv, args.output))}
    else:
        export_training_episode(args.session)
        result = {}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
