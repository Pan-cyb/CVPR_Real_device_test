"""Frame-synchronous MuJoCo RGB / GT graph / predicted graph debugger."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv.blocks import ExplicitGraphObserver, VisualFeatures
from jointgv.dmc_protocol import (
    EXPLICIT_GRAPH_COORDINATE_FRAME,
    EXPLICIT_GRAPH_SCHEMA_VERSION,
)


COLORS = (
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
    """Load a compact UI font on Windows/Linux, with a Pillow fallback."""
    filenames = (
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for filename in filenames:
        try:
            return ImageFont.truetype(filename, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def normalized_pixel(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(round((float(point[0]) + 1.0) * 0.5 * (width - 1)))
    y = int(round((float(point[1]) + 1.0) * 0.5 * (height - 1)))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def overlay_graph(
    frame: np.ndarray,
    positions: np.ndarray,
    presence: np.ndarray,
    relations: np.ndarray,
    title: str,
    scale: int,
) -> Image.Image:
    source = Image.fromarray(frame).convert("RGB")
    image = source.resize(
        (source.width * scale, source.height * scale),
        Image.Resampling.NEAREST,
    )
    width, height = image.size
    draw = ImageDraw.Draw(image)
    title_font = ui_font(12, bold=True)
    node_font = ui_font(10, bold=True)
    points = [normalized_pixel(point, width, height) for point in positions]
    for first in range(len(points)):
        if presence[first] < 0.5:
            continue
        for second in range(first + 1, len(points)):
            if presence[second] < 0.5:
                continue
            relation = int(relations[first, second])
            if relation <= 0:
                continue
            draw.line(
                (points[first], points[second]),
                fill=RELATION_COLORS.get(relation, "#e2e8f0"),
                width=2 if relation == 3 else 1,
            )
    for index, point in enumerate(points):
        if presence[index] < 0.5:
            continue
        color = COLORS[index % len(COLORS)]
        radius = max(3, int(round(scale * 0.65)))
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=color,
            outline="#0f172a",
            width=1,
        )
        # Keep the overlay compact: full semantic names are shown once in the
        # footer legend rather than repeated over the robot in every panel.
        draw.text(
            (point[0] + radius + 2, point[1] - 6),
            str(index),
            fill=color,
            font=node_font,
            stroke_width=1,
            stroke_fill="#0f172a",
        )
    header_height = 22
    draw.rectangle((0, 0, width, header_height), fill="#0f172a")
    draw.text((7, 3), title, fill="white", font=title_font)
    return image


def plain_panel(frame: np.ndarray, title: str, scale: int) -> Image.Image:
    source = Image.fromarray(frame).convert("RGB")
    image = source.resize(
        (source.width * scale, source.height * scale),
        Image.Resampling.NEAREST,
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 22), fill="#0f172a")
    draw.text((7, 3), title, fill="white", font=ui_font(12, bold=True))
    return image


def compose_frame(
    raw: np.ndarray,
    gt_positions: np.ndarray,
    gt_presence: np.ndarray,
    gt_relations: np.ndarray,
    predicted_positions: np.ndarray,
    predicted_presence: np.ndarray,
    predicted_relations: np.ndarray,
    names: list[str],
    frame_id: int,
    metrics: dict[str, float],
    scale: int,
) -> np.ndarray:
    raw_panel = plain_panel(raw, "MuJoCo RGB", scale)
    width, height = raw_panel.size
    gt_panel = overlay_graph(
        raw,
        gt_positions,
        gt_presence,
        gt_relations,
        "GT physical graph",
        scale,
    )
    prediction_panel = overlay_graph(
        raw,
        predicted_positions,
        predicted_presence,
        predicted_relations,
        "Predicted explicit graph",
        scale,
    )
    status_font = ui_font(13)
    legend_font = ui_font(11)
    status_height = 74
    canvas = Image.new(
        "RGB", (3 * width, height + status_height), color="#0f172a"
    )
    canvas.paste(raw_panel, (0, 0))
    canvas.paste(gt_panel, (width, 0))
    canvas.paste(prediction_panel, (2 * width, 0))
    draw = ImageDraw.Draw(canvas)
    status = (
        f"frame={frame_id:04d}  state_mae={metrics['state_mae']:.4f}  "
        f"pixel_mae={metrics['pixel_mae']:.4f}  "
        f"relation_acc={metrics['relation_accuracy']:.3f}  "
        f"attention_overlap={metrics['attention_overlap']:.3f}"
    )
    draw.text((8, height + 6), status, fill="#e2e8f0", font=status_font)
    legend_x = 8
    legend_y = height + 29
    for index, name in enumerate(names):
        if name == "padding":
            continue
        label = f"{index}={name}"
        draw.text(
            (legend_x, legend_y),
            label,
            fill=COLORS[index % len(COLORS)],
            font=legend_font,
        )
        bbox = draw.textbbox((legend_x, legend_y), label, font=legend_font)
        legend_x = bbox[2] + 14
    relation_legend = "edges:  blue=kinematic   yellow=proximity   pink=contact"
    draw.text(
        (8, height + 50),
        relation_legend,
        fill="#94a3b8",
        font=legend_font,
    )
    return np.asarray(canvas)


@torch.no_grad()
def infer(observer, frames: np.ndarray, device, batch_size: int):
    previous = None
    collected: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "nodes",
            "relation_logits",
            "positions_2d",
            "presence_logits",
            "attention",
            "temporal_gate",
            "identity_features",
            "temporal_correction",
        )
    }
    for start in range(0, len(frames), batch_size):
        images = torch.from_numpy(frames[start : start + batch_size].copy())
        images = images.permute(0, 3, 1, 2).float().to(device) / 255.0
        visual = observer.visual_backbone(images)
        for offset in range(visual.tokens.shape[0]):
            frame_features = VisualFeatures(
                visual.tokens[offset : offset + 1],
                visual.positions[offset : offset + 1],
                visual.dense_tokens[offset : offset + 1],
                visual.dense_positions[offset : offset + 1],
            )
            observation = observer._observe_visual(frame_features, previous)
            previous = observation.node_mu
            for key in collected:
                collected[key].append(getattr(observation, key).cpu())
    result = {key: torch.cat(value).numpy() for key, value in collected.items()}
    result["state"] = observer.decode_state_sequence(
        torch.from_numpy(result["nodes"]).to(device),
        torch.from_numpy(result["positions_2d"]).to(device),
    ).cpu().numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.scale < 1 or args.fps <= 0:
        parser.error("batch size, scale, and fps must be positive")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    print(f"loading observer checkpoint on CPU: {args.checkpoint}", flush=True)
    # Load on CPU first: the training checkpoint also contains Adam states.
    # Mapping that entire payload to CUDA while constructing a second model
    # creates a large, unnecessary peak and can kill an 8 GB process.
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_version = int(checkpoint.get("checkpoint_version", 1))
    config = dict(checkpoint["observer_config"])
    config["pretrained_backbone"] = False
    observer = ExplicitGraphObserver(**config)
    incompatible = observer.load_state_dict(checkpoint["model"], strict=False)
    upgrade_prefixes = (
        "temporal_norm.",
        "temporal_gate.",
        "motion_velocity.",
        "dense_visual_projection.",
        "dense_visual_norm.",
        "position_residual.",
    )
    unsupported_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(upgrade_prefixes)
    ]
    if unsupported_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "observer checkpoint mismatch: missing="
            f"{unsupported_missing}, unexpected={incompatible.unexpected_keys}"
        )
    if incompatible.missing_keys:
        print(
            "upgraded a pre-v3 observer checkpoint with dense/gated adapter "
            "defaults; retraining is required for reportable results",
            flush=True,
        )
    observer.set_temporal_strength(
        float(checkpoint.get("temporal_strength", 1.0))
    )
    del checkpoint
    observer = observer.to(device).eval()
    print(f"observer ready on {device}", flush=True)

    with np.load(args.episode, allow_pickle=False) as episode:
        required = {
            "image", "graph_node_state", "graph_node_mask", "graph_relation"
        }
        missing = required.difference(episode.files)
        if missing:
            parser.error("episode is missing: " + ", ".join(sorted(missing)))
        if checkpoint_version >= 3:
            if "graph_schema_version" not in episode.files:
                parser.error(
                    "v3 observer visualization requires a v3 geom-centric "
                    "episode; recollect it with the current collect_dmc.py"
                )
            schema_version = int(episode["graph_schema_version"].item())
            coordinate_frame = str(episode["graph_coordinate_frame"].item())
            if schema_version != EXPLICIT_GRAPH_SCHEMA_VERSION:
                parser.error(
                    f"episode graph schema is v{schema_version}, expected "
                    f"v{EXPLICIT_GRAPH_SCHEMA_VERSION}"
                )
            if coordinate_frame != EXPLICIT_GRAPH_COORDINATE_FRAME:
                parser.error(
                    f"episode graph frame is {coordinate_frame!r}, expected "
                    f"{EXPLICIT_GRAPH_COORDINATE_FRAME!r}"
                )
        frames = episode["image"]
        if args.max_frames > 0:
            frames = frames[: args.max_frames]
        gt_state = episode["graph_node_state"][: len(frames)]
        gt_mask = episode["graph_node_mask"][: len(frames)]
        gt_relation = episode["graph_relation"][: len(frames)]
        if "graph_node_pixel" not in episode.files:
            parser.error(
                "camera-aligned visualization requires graph_node_pixel. "
                "Recollect this split with the updated collect_dmc.py."
            )
        gt_pixel = episode["graph_node_pixel"][: len(frames)]
        gt_visible = episode["graph_node_visible"][: len(frames)]
        if "graph_node_name" in episode.files:
            names = [str(value) for value in episode["graph_node_name"]]
        else:
            names = list(observer.role_names)

    print(f"running RGB-only graph inference for {len(frames)} frames", flush=True)
    prediction = infer(observer, frames, device, args.batch_size)
    print("graph inference complete; rendering synchronized panels", flush=True)
    predicted_relation = prediction["relation_logits"].argmax(-1)
    predicted_presence = (1.0 / (1.0 + np.exp(-prediction["presence_logits"])))
    node_mask = gt_mask.astype(bool)
    pair_mask = node_mask[..., :, None] & node_mask[..., None, :]
    diagonal = np.eye(node_mask.shape[-1], dtype=bool)
    pair_mask &= ~diagonal
    state_absolute = np.abs(prediction["state"] - gt_state).mean(-1)
    attention = np.clip(prediction["attention"], 1e-8, None)
    attention_entropy = -(attention * np.log(attention)).sum(-1) / math.log(
        attention.shape[-1]
    )
    # Avoid numpy.linalg here. On some Windows Conda installations its lazy
    # BLAS DLL load conflicts with PyTorch's already loaded OpenMP runtime and
    # terminates Python with native exception 0xc06d007f (no traceback).
    attention_norm = np.sqrt(
        np.square(attention).sum(axis=-1, keepdims=True)
    )
    normalized_attention = attention / np.maximum(attention_norm, 1e-8)
    overlap = np.einsum(
        "...ik,...jk->...ij", normalized_attention, normalized_attention
    )
    off_diagonal = ~diagonal

    rows: list[dict[str, float]] = []
    comparison_frames: list[np.ndarray] = []
    role_metric_indices = [
        index
        for index, name in enumerate(names)
        if name != "padding" and gt_visible[:, index].astype(bool).any()
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(exist_ok=True)
    for frame_id, raw in enumerate(frames):
        active = node_mask[frame_id]
        visible = gt_visible[frame_id].astype(bool)
        relation_active = pair_mask[frame_id]
        frame_metrics = {
            "frame": float(frame_id),
            "state_mae": float(state_absolute[frame_id][active].mean()) if active.any() else 0.0,
            "position_mae": float(
                np.abs(prediction["state"][frame_id, :, :3] - gt_state[frame_id, :, :3])
                .mean(-1)[active].mean()
            ) if active.any() else 0.0,
            "velocity_mae": float(
                np.abs(prediction["state"][frame_id, :, 7:] - gt_state[frame_id, :, 7:])
                .mean(-1)[active].mean()
            ) if active.any() else 0.0,
            "pixel_mae": float(
                np.abs(prediction["positions_2d"][frame_id] - gt_pixel[frame_id])
                .mean(-1)[visible].mean()
            ) if visible.any() else 0.0,
            "presence_accuracy": float(
                ((predicted_presence[frame_id] >= 0.5) == active).mean()
            ),
            "relation_accuracy": float(
                (predicted_relation[frame_id][relation_active] == gt_relation[frame_id][relation_active]).mean()
            ) if relation_active.any() else 0.0,
            "attention_entropy": float(attention_entropy[frame_id].mean()),
            "attention_overlap": float(overlap[frame_id][off_diagonal].mean()),
            "node_dispersion": float(
                np.square(
                    prediction["nodes"][frame_id]
                    - prediction["nodes"][frame_id].mean(0, keepdims=True)
                ).sum(-1).mean()
            ),
            "temporal_gate_mean": float(
                prediction["temporal_gate"][frame_id].mean()
            ),
            "temporal_gate_max": float(
                prediction["temporal_gate"][frame_id].max()
            ),
            "temporal_correction_rms": float(
                np.sqrt(
                    np.square(
                        prediction["temporal_correction"][frame_id]
                    ).mean()
                )
            ),
            "temporal_correction_absmax": float(
                np.abs(
                    prediction["temporal_correction"][frame_id]
                ).max()
            ),
        }
        for index in role_metric_indices:
            role_key = "".join(
                character if character.isalnum() else "_"
                for character in names[index].lower()
            ).strip("_")
            frame_metrics[f"pixel_mae_role_{index}_{role_key}"] = (
                float(
                    np.abs(
                        prediction["positions_2d"][frame_id, index]
                        - gt_pixel[frame_id, index]
                    ).mean()
                )
                if visible[index]
                else float("nan")
            )
        rows.append(frame_metrics)
        comparison = compose_frame(
            raw,
            gt_pixel[frame_id],
            gt_visible[frame_id],
            gt_relation[frame_id],
            prediction["positions_2d"][frame_id],
            predicted_presence[frame_id],
            predicted_relation[frame_id],
            names,
            frame_id,
            frame_metrics,
            args.scale,
        )
        comparison_frames.append(comparison)
        Image.fromarray(comparison).save(frame_dir / f"frame_{frame_id:05d}.png")
    print(f"rendered {len(comparison_frames)} panel frames", flush=True)

    with (args.output / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        key: float(np.nanmean([row[key] for row in rows]))
        for key in rows[0]
        if key != "frame"
    }
    summary.update(
        {
            "frames": len(frames),
            "checkpoint": str(args.checkpoint),
            "episode": str(args.episode),
            "role_names": names,
        }
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "graph_debug.npz",
        **prediction,
        gt_state=gt_state,
        gt_mask=gt_mask,
        gt_relation=gt_relation,
        gt_pixel=gt_pixel,
        gt_visible=gt_visible,
        role_names=np.asarray(names),
    )
    stacked = np.stack(comparison_frames)
    np.savez_compressed(args.output / "comparison_episode.npz", image=stacked)
    gif_frames = [Image.fromarray(frame) for frame in comparison_frames]
    gif_frames[0].save(
        args.output / "comparison.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=max(1, round(1000.0 / args.fps)),
        loop=0,
    )
    print(json.dumps(summary, indent=2))
    print(f"saved synchronized debug output to {args.output}")


if __name__ == "__main__":
    main()
