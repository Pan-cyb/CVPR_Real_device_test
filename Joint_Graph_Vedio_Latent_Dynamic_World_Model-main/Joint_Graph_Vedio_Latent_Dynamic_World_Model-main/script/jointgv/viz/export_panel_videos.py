"""Export three synchronized, paper-panel-ready videos for one rollout.

The script is intentionally CPU-only: CUDA is hidden before importing torch,
so it cannot allocate memory on a concurrently training NVIDIA GPU.  It emits
three independent MP4 files from the exact same episode slice and actions:

1. ground-truth RGB + the explicit scene graph observed from that RGB;
2. ground truth + backbone-only rollout + amplified absolute error;
3. ground truth + graph-corrected rollout + amplified absolute error.

The accompanying manifest fixes the episode hash, action hash, start, context,
horizon and playback timing.  A later method can pass ``--protocol`` pointing
at that manifest to guarantee a matched panel comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

# This must be set before importing torch.  DINO/RelTR/world-model inference is
# deliberately CPU-only; Intel Quick Sync, when requested, is used only by the
# separate ffmpeg encoding process managed by imageio.
# ``-1`` is used instead of an empty value because ``conda run`` drops empty
# environment variables on Windows before starting Python.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv import JointGraphVideoWorldModel
from jointgv.planning import limit_history


NODE_COLORS = (
    "#38bdf8",
    "#4ade80",
    "#facc15",
    "#fb7185",
    "#c084fc",
    "#fb923c",
    "#2dd4bf",
    "#e879f9",
)
RELATION_COLORS = {
    1: "#38bdf8",  # kinematic
    2: "#facc15",  # proximity
    3: "#fb7185",  # contact
}


def ui_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned or "method"


def as_uint8(images: torch.Tensor) -> np.ndarray:
    return (
        images.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def to_image_tensor(frames: np.ndarray) -> torch.Tensor:
    return (
        torch.from_numpy(frames.copy())
        .permute(0, 3, 1, 2)
        .float()
        .div(255.0)
        .unsqueeze(0)
    )


def image_panel(frame: np.ndarray, title: str, scale: int) -> Image.Image:
    source = Image.fromarray(frame).convert("RGB")
    body = source.resize(
        (source.width * scale, source.height * scale),
        Image.Resampling.NEAREST,
    )
    panel = Image.new("RGB", (body.width, body.height + 34), "#0f172a")
    panel.paste(body, (0, 34))
    draw = ImageDraw.Draw(panel)
    face = ui_font(16, bold=True)
    draw.text(
        ((panel.width - draw.textlength(title, font=face)) / 2, 7),
        title,
        fill="white",
        font=face,
    )
    return panel


def normalized_pixel(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(round((float(point[0]) + 1.0) * 0.5 * (width - 1)))
    y = int(round((float(point[1]) + 1.0) * 0.5 * (height - 1)))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def graph_panel(
    frame: np.ndarray,
    positions: np.ndarray,
    presence: np.ndarray,
    relations: np.ndarray,
    names: list[str],
    source_shape: tuple[int, int],
    scale: int,
) -> Image.Image:
    source_height, source_width = source_shape
    width, height = source_width * scale, source_height * scale
    source = Image.fromarray(frame).convert("RGB").resize(
        (width, height), Image.Resampling.NEAREST
    )
    panel = Image.new("RGB", (width, height + 34), "#0f172a")
    panel.paste(source, (0, 34))
    draw = ImageDraw.Draw(panel)
    title_face = ui_font(16, bold=True)
    label_face = ui_font(11, bold=True)
    title = "Scene graph overlay"
    draw.text(
        ((width - draw.textlength(title, font=title_face)) / 2, 7),
        title,
        fill="white",
        font=title_face,
    )
    points = [
        normalized_pixel(point, width, height)
        for point in positions
    ]
    points = [(x, y + 34) for x, y in points]
    active = [
        value >= 0.5 and (index >= len(names) or names[index] != "padding")
        for index, value in enumerate(presence)
    ]
    for first in range(len(points)):
        if not active[first]:
            continue
        for second in range(first + 1, len(points)):
            if not active[second]:
                continue
            relation = int(relations[first, second])
            if relation <= 0:
                continue
            draw.line(
                (points[first], points[second]),
                fill=RELATION_COLORS.get(relation, "#e2e8f0"),
                width=3 if relation == 3 else 2,
            )
    for index, point in enumerate(points):
        if not active[index]:
            continue
        color = NODE_COLORS[index % len(NODE_COLORS)]
        radius = max(5, scale + 1)
        draw.ellipse(
            (
                point[0] - radius,
                point[1] - radius,
                point[0] + radius,
                point[1] + radius,
            ),
            fill=color,
            outline="#e2e8f0",
            width=1,
        )
        draw.text(
            (point[0] + radius + 2, point[1] - 7),
            str(index),
            fill=color,
            font=label_face,
            stroke_width=1,
            stroke_fill="#07111f",
        )
    return panel


def join_panels(
    panels: list[Image.Image],
    footer: str,
    *,
    footer_detail: str = "",
) -> np.ndarray:
    gap = 6
    footer_height = 48
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels) + footer_height
    canvas = Image.new("RGB", (width, height), "#0f172a")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    draw = ImageDraw.Draw(canvas)
    draw.text((8, height - 42), footer, fill="#e2e8f0", font=ui_font(13))
    if footer_detail:
        draw.text(
            (8, height - 22),
            footer_detail,
            fill="#94a3b8",
            font=ui_font(11),
        )
    # H.264 yuv420p requires even dimensions.  Pad rather than allowing ffmpeg
    # to resize a scientific visualization silently.
    padded_width = width + (width % 2)
    padded_height = height + (height % 2)
    if (padded_width, padded_height) != canvas.size:
        padded = Image.new("RGB", (padded_width, padded_height), "#0f172a")
        padded.paste(canvas, (0, 0))
        canvas = padded
    return np.asarray(canvas)


def render_videos(
    ground_truth: np.ndarray,
    backbone: np.ndarray,
    corrected: np.ndarray,
    backbone_error: np.ndarray,
    corrected_error: np.ndarray,
    graph_positions: np.ndarray,
    graph_presence: np.ndarray,
    graph_relations: np.ndarray,
    role_names: list[str],
    *,
    method_name: str,
    start_frame: int,
    scale: int,
    error_scale: float,
) -> dict[str, list[np.ndarray]]:
    videos: dict[str, list[np.ndarray]] = {
        "graph_construction": [],
        "backbone_error": [],
        "corrected_error": [],
    }
    steps = len(ground_truth)
    for step in range(steps):
        frame_id = start_frame + step + 1
        status = f"{method_name} | step {step + 1}/{steps} | episode frame {frame_id}"
        truth_panel = image_panel(ground_truth[step], "Ground truth", scale)
        scene_panel = graph_panel(
            ground_truth[step],
            graph_positions[step],
            graph_presence[step],
            graph_relations[step],
            role_names,
            ground_truth[step].shape[:2],
            scale,
        )
        active = int(
            sum(
                value >= 0.5 and (i >= len(role_names) or role_names[i] != "padding")
                for i, value in enumerate(graph_presence[step])
            )
        )
        videos["graph_construction"].append(
            join_panels(
                [truth_panel, scene_panel],
                status,
                footer_detail=(
                    f"active observed nodes={active} | edges: blue=kinematic, "
                    "yellow=proximity, pink=contact"
                ),
            )
        )

        backbone_mse = float(
            np.mean(
                (
                    backbone[step].astype(np.float32) / 255.0
                    - ground_truth[step].astype(np.float32) / 255.0
                )
                ** 2
            )
        )
        videos["backbone_error"].append(
            join_panels(
                [
                    truth_panel,
                    image_panel(backbone[step], "Backbone rollout", scale),
                    image_panel(
                        backbone_error[step],
                        f"Absolute error x{error_scale:g}",
                        scale,
                    ),
                ],
                status,
                footer_detail=f"frame MSE={backbone_mse:.6f} | graph conditioning OFF",
            )
        )

        corrected_mse = float(
            np.mean(
                (
                    corrected[step].astype(np.float32) / 255.0
                    - ground_truth[step].astype(np.float32) / 255.0
                )
                ** 2
            )
        )
        videos["corrected_error"].append(
            join_panels(
                [
                    truth_panel,
                    image_panel(corrected[step], "Graph corrected rollout", scale),
                    image_panel(
                        corrected_error[step],
                        f"Absolute error x{error_scale:g}",
                        scale,
                    ),
                ],
                status,
                footer_detail=f"frame MSE={corrected_mse:.6f} | graph conditioning ON",
            )
        )
    return videos


def write_mp4(
    path: Path,
    frames: list[np.ndarray],
    *,
    fps: float,
    codec: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_params: list[str] = []
    if codec == "libx264":
        output_params += ["-crf", "18", "-preset", "medium"]
    elif codec.endswith("_qsv"):
        output_params += ["-global_quality", "20"]
    try:
        with imageio.get_writer(
            path,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec=codec,
            pixelformat="nv12" if codec.endswith("_qsv") else "yuv420p",
            macro_block_size=2,
            output_params=output_params,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "MP4 export needs imageio-ffmpeg and the selected codec. "
            "Install it with: python -m pip install imageio-ffmpeg. "
            "Use --codec libx264 if Intel QSV is unavailable. "
            f"Original error: {error}"
        ) from error


def preview_three(videos: dict[str, list[np.ndarray]], fps: float) -> None:
    """Play the three synchronized products in independent Tk windows."""
    try:
        import tkinter as tk
    except ImportError as error:
        raise RuntimeError("Tk is required for --preview") from error

    labels = {
        "graph_construction": "Ground truth + scene graph",
        "backbone_error": "Ground truth + backbone + error",
        "corrected_error": "Ground truth + corrected + error",
    }
    keys = list(labels)
    root = tk.Tk()
    windows = [root]
    for _ in keys[1:]:
        windows.append(tk.Toplevel(root))
    canvases = []
    for index, (window, key) in enumerate(zip(windows, keys)):
        first = videos[key][0]
        window.title(labels[key])
        window.geometry(f"+{40 + index * 70}+{40 + index * 45}")
        canvas = tk.Canvas(
            window,
            width=first.shape[1],
            height=first.shape[0],
            background="#0f172a",
            highlightthickness=0,
        )
        canvas.pack()
        canvases.append(canvas)

    state: dict[str, Any] = {"frame": 0, "photos": [], "paused": False}
    interval_ms = max(1, round(1000.0 / fps))
    frame_count = len(videos[keys[0]])

    def render() -> None:
        photos = []
        for canvas, key in zip(canvases, keys):
            frame = np.ascontiguousarray(videos[key][state["frame"]])
            height, width = frame.shape[:2]
            ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + frame.tobytes()
            photo = tk.PhotoImage(data=ppm, format="PPM")
            canvas.delete("all")
            canvas.create_image(0, 0, image=photo, anchor="nw")
            photos.append(photo)
        state["photos"] = photos

    def tick() -> None:
        if not state["paused"]:
            state["frame"] = (state["frame"] + 1) % frame_count
            render()
        root.after(interval_ms, tick)

    def toggle(_event=None) -> None:
        state["paused"] = not state["paused"]

    def step(delta: int) -> None:
        state["paused"] = True
        state["frame"] = (state["frame"] + delta) % frame_count
        render()

    for window in windows:
        window.bind("<space>", toggle)
        window.bind("<Left>", lambda _event: step(-1))
        window.bind("<Right>", lambda _event: step(1))
        window.bind("<Escape>", lambda _event: root.destroy())
    render()
    root.after(interval_ms, tick)
    root.mainloop()


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    required = {
        "episode",
        "episode_sha256",
        "action_sha256",
        "start_frame",
        "context_frames",
        "horizon",
        "source_fps",
        "target_seconds_per_frame",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            f"protocol manifest is missing fields: {', '.join(sorted(missing))}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, default=None)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="Reuse and verify episode/timing fields from an earlier panel_manifest.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", default="P")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--context-frames", type=int, default=3)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Rollout initial frame; default equals --context-frames.",
    )
    parser.add_argument("--history-window", type=int, default=0)
    parser.add_argument("--correction-strength", type=float, default=1.0)
    parser.add_argument("--observer-batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--source-fps", type=float, default=25.0)
    parser.add_argument("--seconds-per-frame", type=float, default=0.5)
    parser.add_argument(
        "--codec",
        default="libx264",
        help="libx264 (CPU, reliable) or h264_qsv (Intel iGPU encoding if available).",
    )
    parser.add_argument(
        "--visual-context",
        action="store_true",
        help="Also seed the visual backbone; off matches the current rollout protocol.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="After saving, play all three products in synchronized Tk windows.",
    )
    args = parser.parse_args()

    if args.horizon < 1 or args.context_frames < 0:
        parser.error("--horizon must be positive and --context-frames nonnegative")
    if args.start_frame is not None and args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    if args.history_window < 0 or args.history_window == 1:
        parser.error("--history-window must be 0 (full) or at least 2")
    if not 0.0 <= args.correction_strength <= 1.0:
        parser.error("--correction-strength must be in [0, 1]")
    if min(
        args.observer_batch_size,
        args.cpu_threads,
        args.scale,
        args.error_scale,
        args.source_fps,
        args.seconds_per_frame,
    ) <= 0:
        parser.error("batch, thread, scale, timing, and error-scale values must be positive")

    protocol: dict[str, Any] | None = None
    if args.protocol is not None:
        protocol = load_protocol(args.protocol)
        episode_path = Path(protocol["episode"])
        start_frame = int(protocol["start_frame"])
        context_frames = int(protocol["context_frames"])
        requested_horizon = int(protocol["horizon"])
        source_fps = float(protocol["source_fps"])
        target_seconds = float(protocol["target_seconds_per_frame"])
        # Presentation and model-side rollout settings are part of the fixed
        # comparison protocol too.  A later method must not silently change
        # them merely because its command omitted an option.
        args.visual_context = bool(protocol.get("visual_context", False))
        args.history_window = int(protocol.get("history_window", 0))
        args.correction_strength = float(
            protocol.get("correction_strength", 1.0)
        )
        args.error_scale = float(protocol.get("error_scale", 4.0))
        args.scale = int(protocol.get("render_scale", args.scale))
        if args.episode is not None and args.episode.resolve() != episode_path.resolve():
            parser.error("--episode disagrees with the episode fixed by --protocol")
        print(f"reusing matched protocol: {args.protocol}", flush=True)
    else:
        if args.episode is None:
            parser.error("--episode is required unless --protocol is supplied")
        episode_path = args.episode
        context_frames = args.context_frames
        start_frame = (
            args.context_frames if args.start_frame is None else args.start_frame
        )
        requested_horizon = args.horizon
        source_fps = args.source_fps
        target_seconds = args.seconds_per_frame

    if start_frame < context_frames:
        parser.error("--start-frame must be at least --context-frames")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not episode_path.is_file():
        parser.error(f"episode does not exist: {episode_path}")

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    except RuntimeError:
        pass
    print(
        "inference device=cpu; CUDA is hidden for this process, so RTX VRAM "
        "usage remains zero",
        flush=True,
    )
    print(
        f"loading checkpoint on CPU (threads={args.cpu_threads}): {args.checkpoint}",
        flush=True,
    )
    load_started = time.perf_counter()
    try:
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    model_config = dict(checkpoint["model_config"])
    # Every observer parameter is restored from the world-model checkpoint;
    # avoid redundantly loading the standalone RelTR weights during construction.
    model_config["graph_backbone_pretrained"] = False
    model = JointGraphVideoWorldModel(**model_config)
    model.load_state_dict(checkpoint["model"])
    checkpoint_epoch_raw = checkpoint.get("epoch")
    checkpoint_epoch = (
        None if checkpoint_epoch_raw is None else int(checkpoint_epoch_raw)
    )
    del checkpoint
    model.eval()
    limit_history(model, args.history_window)
    model.set_graph_conditioning_strength(args.correction_strength)
    builder = model.graph_builder
    if not getattr(builder, "requires_images", False):
        raise RuntimeError(
            "Panel scene-graph construction requires an explicit RGB graph observer"
        )
    if hasattr(builder, "reltr_batch_size"):
        builder.reltr_batch_size = args.observer_batch_size
    if hasattr(builder, "visual_backbone") and hasattr(
        builder.visual_backbone, "inference_batch_size"
    ):
        builder.visual_backbone.inference_batch_size = args.observer_batch_size
    role_names = list(getattr(builder, "role_names", ()))
    print(
        f"model ready in {time.perf_counter() - load_started:.1f}s; "
        f"observer CPU chunk={args.observer_batch_size}",
        flush=True,
    )

    episode_hash = file_sha256(episode_path)
    with np.load(episode_path, allow_pickle=False) as episode:
        required = {"image", "action"}
        missing = required.difference(episode.files)
        if missing:
            parser.error("episode is missing: " + ", ".join(sorted(missing)))
        horizon = min(
            requested_horizon,
            len(episode["action"]) - start_frame,
            len(episode["image"]) - 1 - start_frame,
        )
        if horizon < 1:
            parser.error("episode has no transition at the requested start frame")
        if protocol is not None and horizon != requested_horizon:
            parser.error(
                "episode is now shorter than the horizon fixed by --protocol"
            )
        initial_frames = episode["image"][start_frame : start_frame + 1].copy()
        context_start = start_frame - context_frames
        context_frames_np = episode["image"][context_start:start_frame].copy()
        context_actions_np = episode["action"][context_start:start_frame].copy()
        actions_np = episode["action"][start_frame : start_frame + horizon].copy()
        ground_truth = episode[
            "image"
        ][start_frame + 1 : start_frame + horizon + 1].copy()
        rewards = (
            episode["reward"][start_frame : start_frame + horizon].copy()
            if "reward" in episode.files
            else np.zeros(horizon, dtype=np.float32)
        )
    action_hash = array_sha256(actions_np)
    if protocol is not None:
        if episode_hash != protocol["episode_sha256"]:
            parser.error("episode file hash no longer matches --protocol")
        if action_hash != protocol["action_sha256"]:
            parser.error("action slice hash no longer matches --protocol")

    initial = to_image_tensor(initial_frames)[:, 0]
    context = to_image_tensor(context_frames_np) if context_frames else None
    context_actions = (
        torch.from_numpy(context_actions_np).float().unsqueeze(0)
        if context_frames
        else None
    )
    actions = torch.from_numpy(actions_np).float().unsqueeze(0)
    rollout_kwargs = {
        "context_images": context,
        "context_actions": context_actions,
        "visual_context": args.visual_context,
    }

    print(
        f"running backbone-only CPU rollout: start={start_frame}, H={horizon}",
        flush=True,
    )
    phase_started = time.perf_counter()
    with torch.inference_mode():
        base_rollout = model.rollout(
            initial, actions, use_graph=False, **rollout_kwargs
        )
    backbone = as_uint8(base_rollout["images"][0])
    del base_rollout
    print(
        f"backbone rollout complete in {time.perf_counter() - phase_started:.1f}s",
        flush=True,
    )

    print("running graph-corrected CPU rollout", flush=True)
    phase_started = time.perf_counter()
    with torch.inference_mode():
        corrected_rollout = model.rollout(
            initial, actions, use_graph=True, **rollout_kwargs
        )
    corrected = as_uint8(corrected_rollout["images"][0])
    del corrected_rollout
    print(
        f"corrected rollout complete in {time.perf_counter() - phase_started:.1f}s",
        flush=True,
    )

    print(
        f"constructing scene graph from {horizon} ground-truth RGB frames on CPU",
        flush=True,
    )
    phase_started = time.perf_counter()
    truth_tensor = to_image_tensor(ground_truth)
    with torch.inference_mode():
        observed = builder.forward_sequence(truth_tensor, sample=False)
    graph_positions = observed["positions_2d"][0].cpu().numpy()
    graph_presence = observed["presence_logits"][0].sigmoid().cpu().numpy()
    graph_relations = observed["relation_logits"][0].argmax(-1).cpu().numpy()
    del observed, truth_tensor, model
    print(
        f"scene graph complete in {time.perf_counter() - phase_started:.1f}s",
        flush=True,
    )

    backbone_error = np.clip(
        np.abs(backbone.astype(np.float32) - ground_truth.astype(np.float32))
        * args.error_scale,
        0,
        255,
    ).astype(np.uint8)
    corrected_error = np.clip(
        np.abs(corrected.astype(np.float32) - ground_truth.astype(np.float32))
        * args.error_scale,
        0,
        255,
    ).astype(np.uint8)
    method = safe_name(args.method_name)
    slowdown = max(1, int(math.floor(source_fps * target_seconds + 0.5)))
    output_fps = source_fps / slowdown
    actual_seconds = 1.0 / output_fps
    video_frames = render_videos(
        ground_truth,
        backbone,
        corrected,
        backbone_error,
        corrected_error,
        graph_positions,
        graph_presence,
        graph_relations,
        role_names,
        method_name=args.method_name,
        start_frame=start_frame,
        scale=args.scale,
        error_scale=args.error_scale,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(episode_path.stem)
    output_paths = {
        key: args.output_dir / f"{stem}_{method}_{key}_slow{slowdown}x.mp4"
        for key in video_frames
    }
    for key, path in output_paths.items():
        print(
            f"encoding {key} -> {path.name} ({args.codec}, {output_fps:.3f} fps)",
            flush=True,
        )
        write_mp4(path, video_frames[key], fps=output_fps, codec=args.codec)

    backbone_mse = float(
        np.mean(
            (
                backbone.astype(np.float32) / 255.0
                - ground_truth.astype(np.float32) / 255.0
            )
            ** 2
        )
    )
    corrected_mse = float(
        np.mean(
            (
                corrected.astype(np.float32) / 255.0
                - ground_truth.astype(np.float32) / 255.0
            )
            ** 2
        )
    )
    manifest = {
        "schema_version": 1,
        "method_name": args.method_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "episode": str(episode_path.resolve()),
        "episode_sha256": episode_hash,
        "action_sha256": action_hash,
        "start_frame": start_frame,
        "context_frames": context_frames,
        "horizon": horizon,
        "visual_context": bool(args.visual_context),
        "history_window": args.history_window,
        "correction_strength": args.correction_strength,
        "scene_graph_source": "checkpoint_explicit_observer_from_ground_truth_rgb",
        "inference_device": "cpu",
        "cpu_threads": args.cpu_threads,
        "observer_batch_size": args.observer_batch_size,
        "source_fps": source_fps,
        "target_seconds_per_frame": target_seconds,
        "slowdown_factor": slowdown,
        "output_fps": output_fps,
        "actual_seconds_per_frame": actual_seconds,
        "error_scale": args.error_scale,
        "render_scale": args.scale,
        "codec": args.codec,
        "backbone_mse": backbone_mse,
        "corrected_mse": corrected_mse,
        "role_names": role_names,
        "videos": {key: str(path.resolve()) for key, path in output_paths.items()},
    }
    manifest_path = args.output_dir / "panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "panel_data.npz",
        ground_truth=ground_truth,
        backbone_prediction=backbone,
        corrected_prediction=corrected,
        backbone_error=backbone_error,
        corrected_error=corrected_error,
        graph_positions_2d=graph_positions,
        graph_presence=graph_presence,
        graph_relations=graph_relations,
        graph_role_names=np.asarray(role_names),
        actions=actions_np,
        rewards=rewards,
        start_frame=np.asarray(start_frame),
        horizon=np.asarray(horizon),
    )
    print(
        json.dumps(
            {
                "saved": {key: str(path) for key, path in output_paths.items()},
                "manifest": str(manifest_path),
                "slowdown": f"{slowdown}x",
                "frame_hold_seconds": round(actual_seconds, 3),
                "backbone_mse": backbone_mse,
                "corrected_mse": corrected_mse,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.preview:
        print(
            "opening three synchronized windows (Space pause; arrows step; Esc close)",
            flush=True,
        )
        preview_three(video_frames, output_fps)


if __name__ == "__main__":
    main()
