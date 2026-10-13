"""Grid preview of the stored 64x64 frames, several windows per episode.

Shows exactly what the world model is trained on: rows are episodes, columns
are prediction windows (``--frames`` + 1 frames each) starting at the
beginning, the middle and the end of every episode (evenly spaced), or at
random starts with ``--random``. Every cell is labelled with its start frame
and, for the RM65-B cart-pole, the pole's lean swing inside that window (deg)
and the tip's motion relative to the hinge in the 64x64 frame (px). The same
numbers are printed per window. Reads the .npz files only; no torch, no
MuJoCo.

    python script/jointgv/viz/preview_windows.py runs/debug/rm65b_cartpole_lqr_pilot_move `
        --episodes 4 --windows 5 --output runs/debug/preview_move.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def window_starts(length: int, frames: int, count: int, rng: np.random.Generator | None) -> list[int]:
    """Start indices of ``count`` windows of ``frames + 1`` stored frames."""
    last = length - (frames + 1)
    if last < 0:
        return []
    if rng is not None:
        return sorted(int(s) for s in rng.choice(last + 1, size=min(count, last + 1), replace=False))
    return sorted({int(round(s)) for s in np.linspace(0, last, count)})


def cartpole_signals(episode) -> tuple[np.ndarray, np.ndarray] | None:
    """Pole world lean (deg) and tip-minus-hinge pixel vector, if available."""
    if "qpos" not in episode.files or "graph_node_name" not in episode.files:
        return None
    roles = [str(role) for role in episode["graph_node_name"]]   # role names
    if "end_effector" not in roles or "object_1" not in roles:
        return None
    if str(episode["domain"].item()) != "rm65b" or str(episode["task"].item()) != "cartpole":
        return None
    lean = np.degrees(episode["qpos"][:, :4].sum(axis=1))
    size = episode["image"].shape[1]
    pixels = episode["graph_node_pixel"] * (size - 1) / 2.0     # [-1, 1] -> px
    pole = pixels[:, roles.index("object_1")] - pixels[:, roles.index("end_effector")]
    return lean, pole


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, help="Folder of collected .npz episodes.")
    parser.add_argument("--episodes", type=int, default=4, help="First N episodes (sorted).")
    parser.add_argument("--files", nargs="+", default=None, help="Explicit episode file names instead.")
    parser.add_argument("--windows", type=int, default=5, help="Windows per episode.")
    parser.add_argument("--frames", type=int, default=32, help="Predicted steps per window.")
    parser.add_argument("--random", action="store_true", help="Random window starts instead of evenly spaced.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=None, help="Default: <data>/preview_windows.gif")
    args = parser.parse_args()
    if min(args.episodes, args.windows, args.frames, args.scale) < 1 or args.fps <= 0:
        parser.error("--episodes, --windows, --frames, --scale and --fps must be positive")

    if args.files:
        paths = [args.data / name for name in args.files]
    else:
        paths = sorted(args.data.glob("*.npz"))[: args.episodes]
    if not paths:
        raise SystemExit(f"no .npz episodes in {args.data}")
    rng = np.random.default_rng(args.seed) if args.random else None

    rows = []          # per episode: list of (label lines, frames [F+1, H, W, 3])
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            images = episode["image"]
            signals = cartpole_signals(episode)
            starts = window_starts(len(images), args.frames, args.windows, rng)
            if not starts:
                print(f"{path.name}: only {len(images)} frames, shorter than one window; skipped")
                continue
            cells = []
            summary = []
            for start in starts:
                stop = start + args.frames + 1
                lines = [f"{path.stem[-5:]}  frames {start}-{stop - 1}"]
                if signals is not None:
                    lean, pole = signals
                    swing = float(np.ptp(lean[start:stop]))
                    tip = float(np.ptp(pole[start:stop, 0]))
                    lines.append(f"lean {lean[start]:+.1f}deg swing {swing:.1f}deg tip {tip:.1f}px")
                    summary.append(f"[{start}:{stop}] swing {swing:4.1f} deg, tip {tip:3.1f} px")
                else:
                    summary.append(f"[{start}:{stop}]")
                cells.append((lines, images[start:stop]))
            print(f"{path.name} ({len(images)} frames): " + " | ".join(summary))
            rows.append(cells)
    if not rows:
        raise SystemExit("no episode is long enough for the requested window")

    side = rows[0][0][1].shape[1] * args.scale
    header = 30
    columns = max(len(cells) for cells in rows)
    width = columns * (side + 4)
    height = len(rows) * (side + header + 4) + 18
    canvases = []
    for step in range(args.frames + 1):
        canvas = Image.new("RGB", (width, height), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for row, cells in enumerate(rows):
            top = row * (side + header + 4)
            for column, (lines, frames) in enumerate(cells):
                left = column * (side + 4)
                frame = frames[step].repeat(args.scale, axis=0).repeat(args.scale, axis=1)
                canvas.paste(Image.fromarray(frame), (left, top + header))
                for index, line in enumerate(lines):
                    draw.text((left + 3, top + 2 + 13 * index), line, fill=(235, 235, 235))
        draw.text(
            (4, height - 15),
            f"step {step}/{args.frames} of each window (64x64 stored frames, x{args.scale})",
            fill=(200, 200, 200),
        )
        canvases.append(canvas)
    output = args.output or args.data / "preview_windows.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvases[0].save(
        output, save_all=True, append_images=canvases[1:],
        duration=max(1, round(1000.0 / args.fps)), loop=0,
    )
    print(f"saved {output} ({len(rows)} episodes x up to {columns} windows, {args.frames + 1} frames)")


if __name__ == "__main__":
    main()
