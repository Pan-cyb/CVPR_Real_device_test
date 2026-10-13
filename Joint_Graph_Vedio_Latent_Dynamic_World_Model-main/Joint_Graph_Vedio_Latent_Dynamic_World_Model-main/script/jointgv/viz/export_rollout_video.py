"""Save an evaluate_rollout.py comparison as a labelled GIF (MP4 if possible).

Columns (titles overridable): "Ground truth", "Graph corrected", "Backbone
graph off (same round)" -- the same weights with the graph module off -- and,
when present, the --compare-checkpoint model, "DINO-WM best graph off";
long titles wrap. --with-errors adds the 4x error maps under each
prediction. Reads the .npz only; no torch.

    python script/jointgv/viz/export_rollout_video.py runs/eval/P_best_test00.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLUMNS = (
    ("ground_truth", None),
    ("graph_prediction", "graph_prediction_error"),
    ("base_prediction", "base_prediction_error"),
    ("compare_prediction", "compare_prediction_error"),
)


def font(size: int = 13) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has one fixed-size bitmap font
        return ImageFont.load_default()


def wrap(text: str, width: int, face: ImageFont.ImageFont) -> list[str]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: list[str] = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and draw.textlength(candidate, font=face) <= width - 6:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def label(lines: list[str], width: int, height: int, face: ImageFont.ImageFont) -> np.ndarray:
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    line_height = face.getbbox("Ag")[3] + 3
    top = (height - line_height * len(lines)) // 2
    for index, line in enumerate(lines):
        draw.text(
            ((width - draw.textlength(line, font=face)) / 2, top + index * line_height),
            line,
            fill=(235, 235, 235),
            font=face,
        )
    return np.asarray(canvas)


def upscale(frames: np.ndarray, scale: int) -> np.ndarray:
    return frames.repeat(scale, axis=1).repeat(scale, axis=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("comparison", type=Path, help="evaluate_rollout.py --output .npz")
    parser.add_argument("--output", type=Path, default=None, help="default: <npz>.gif")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--with-errors", action="store_true")
    parser.add_argument("--truth-name", default="Ground truth")
    parser.add_argument("--graph-name", default="Graph corrected")
    parser.add_argument("--base-name", default="Backbone graph off (same round)")
    parser.add_argument("--compare-name", default="DINO-WM best graph off")
    args = parser.parse_args()
    output = args.output or args.comparison.with_suffix(".gif")
    titles = {
        "ground_truth": args.truth_name,
        "graph_prediction": args.graph_name,
        "base_prediction": args.base_name,
        "compare_prediction": args.compare_name,
    }

    with np.load(args.comparison, allow_pickle=False) as data:
        tracks = {key: data[key] for key in data.files}
    steps = len(tracks["ground_truth"])
    face = font()
    column_width = tracks["ground_truth"].shape[2] * args.scale
    present = [(key, error_key) for key, error_key in COLUMNS if key in tracks]
    wrapped = {key: wrap(titles[key], column_width, face) for key, _ in present}
    line_height = face.getbbox("Ag")[3] + 3
    header_height = line_height * max(len(lines) for lines in wrapped.values()) + 8
    columns = []
    for key, error_key in present:
        frames = upscale(tracks[key], args.scale)
        if args.with_errors:
            below = (
                upscale(tracks[error_key], args.scale)
                if error_key is not None and error_key in tracks
                else np.zeros_like(frames)
            )
            frames = np.concatenate((frames, below), axis=1)
        header = label(wrapped[key], frames.shape[2], header_height, face)
        columns.append(
            np.concatenate(
                (np.broadcast_to(header, (steps, *header.shape)), frames), axis=1
            )
        )
    gap = np.full((steps, columns[0].shape[1], 4, 3), 20, dtype=np.uint8)
    body = np.concatenate(
        [part for column in columns for part in (column, gap)][:-1], axis=2
    )
    start = int(tracks.get("start_frame", np.asarray(0)))
    video = []
    for step in range(steps):
        footer = label(
            [f"rollout step {step + 1}/{steps}  (episode frame {start + step + 1})"],
            body.shape[2],
            line_height + 8,
            face,
        )
        video.append(np.concatenate((body[step], footer), axis=0))

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".mp4":
        try:
            imageio.mimsave(output, video, fps=args.fps)
        except (ImportError, ValueError, RuntimeError) as error:
            raise SystemExit(
                f"MP4 needs ffmpeg (python -m pip install imageio-ffmpeg): {error}"
            )
    else:
        images = [Image.fromarray(frame) for frame in video]
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=round(1000.0 / args.fps),
            loop=0,
        )
    print(f"saved {output} ({steps} frames, {video[0].shape[1]}x{video[0].shape[0]})")


if __name__ == "__main__":
    main()
