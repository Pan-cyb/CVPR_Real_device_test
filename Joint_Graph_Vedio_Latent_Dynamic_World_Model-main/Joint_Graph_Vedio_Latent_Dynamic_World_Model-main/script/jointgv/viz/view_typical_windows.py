"""Play typical cart-pole training clips one at a time in small windows.

Picks a few representative clips from a collected RM65-B cart-pole folder
(``utils/collect_dmc.py --cartpole-policy lqr``) and plays each in its own
window of stored 64x64 frames (what the world model sees), enlarged. The
title says what the clip is; the line underneath shows, per frame, who is
controlling the arm (LQR / casual correction / past horizontal) and the pole
lean. Close a window to open the next. Space pauses, left/right step.

Clip types (each skipped if the folder has none):

1. LQR balancing: an episode the LQR drives to the end, mid-episode.
2. LQR hands over, casual correction fails: from shortly before the
   hand-over to the fall.
3. Casual correction holds a while before failing: the longest
   corrector-only stretch, ending in the fall.
4. Wide start, fast fall, padded below horizontal: a short failure.
5. LQR catches a wide start: an episode starting at >15 deg or >1.5 rad/s
   that is still up 2 s later.

    python script/jointgv/viz/view_typical_windows.py runs/debug/rm65b_cartpole_v3_pilot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as episode:
        data = {key: episode[key] for key in episode.files}
    steps = len(data["action"])
    lean = np.degrees(data["qpos"][:, :4].sum(axis=1))
    data["lean"] = (lean + 180.0) % 360.0 - 180.0
    data["rate"] = data["qvel"][:, :4].sum(axis=1)
    data["controller"] = data.get("action_controller", np.zeros(steps, dtype=np.int64))
    data["past"] = data.get("past_horizontal", np.zeros(steps, dtype=bool))
    data["handoff"] = int(data.get("lqr_handoff_step", -1))
    data["fall"] = int(data.get("fall_step", -1))
    data["steps"] = steps
    data["path"] = path
    return data


def clip(episode: dict, start: int, length: int) -> tuple[int, int]:
    start = max(0, min(start, episode["steps"] - length))
    return start, min(episode["steps"], start + length)


def pick_clips(episodes: list[dict], length: int) -> list[tuple[str, dict, int, int]]:
    clips = []
    def upright_steps(e: dict) -> int:
        return e["fall"] if e["fall"] >= 0 else e["steps"]

    full = [e for e in episodes if e["handoff"] < 0 and upright_steps(e) >= 2 * length]
    if full:
        e = max(full, key=upright_steps)
        clips.append(("1. LQR balancing (episode under LQR), middle of the balanced part", e,
                      *clip(e, upright_steps(e) // 2 - length // 2, length)))
    handed = [e for e in episodes if 0 <= e["handoff"] < e["steps"] and (e["controller"] == 1).any()]
    # Hand-over AND fall must both be in the clip: prefer a fall within the
    # clip length after the hand-over; otherwise stretch the clip to the fall.
    quick = [e for e in handed if 0 <= e["fall"] - e["handoff"] <= length - 8]
    if quick:
        e = max(quick, key=lambda e: e["fall"] - e["handoff"])
        clips.append(("2. LQR hands over -> casual correction fails", e,
                      *clip(e, e["handoff"] - 4, length)))
    elif any(e["fall"] >= 0 for e in handed):
        e = min((e for e in handed if e["fall"] >= 0), key=lambda e: e["fall"] - e["handoff"])
        clips.append(("2. LQR hands over -> casual correction fails (long clip)", e,
                      max(0, e["handoff"] - 4), min(e["steps"], e["fall"] + 6)))
    corrected = [e for e in handed if (e["controller"] == 1).sum() >= length]
    if corrected:
        e = max(corrected, key=lambda e: int((e["controller"] == 1).sum()))
        end = e["fall"] if e["fall"] >= 0 else e["steps"]
        clips.append(("3. Casual correction holds a while, then fails", e,
                      *clip(e, end - length + 6, length)))
    padded = [e for e in episodes if e["fall"] >= 0 and e["past"].any() and e["steps"] < length]
    if padded:
        e = max(padded, key=lambda e: abs(e["lean"][0]))
        clips.append(("4. Wide start, fast fall, padded below horizontal (whole episode)", e,
                      0, e["steps"]))
    caught = [
        e for e in episodes
        if (abs(e["lean"][0]) > 15.0 or abs(e["rate"][0]) > 1.5)
        and e["steps"] >= 50 and (e["fall"] < 0 or e["fall"] >= 50)
        and (e["controller"][:50] == 0).all()
    ]
    if caught:
        e = max(caught, key=lambda e: abs(e["lean"][0]) + 10.0 * abs(e["rate"][0]))
        clips.append(("5. LQR catches a wide start", e, *clip(e, 0, length)))
    return clips


def play(title: str, episode: dict, start: int, stop: int, scale: int, fps: float) -> None:
    import tkinter as tk

    frames = episode["image"][start : stop + 1]
    height, width = frames.shape[1:3]
    root = tk.Tk()
    root.title(title)
    events = []
    if episode["handoff"] >= 0:
        events.append(f"hand-over at frame {episode['handoff']}")
    if episode["fall"] >= 0:
        events.append(f"passes horizontal at frame {episode['fall'] + 1}")
    header = (
        f"{episode['path'].name}   frames {start}-{stop}   start: lean "
        f"{episode['lean'][0]:+.1f} deg at {episode['rate'][0]:+.1f} rad/s"
        + ("   |   " + ", ".join(events) if events else "")
    )
    tk.Label(root, text=title, font=("Segoe UI", 12, "bold")).pack(fill="x")
    tk.Label(root, text=header, font=("Consolas", 9)).pack(fill="x")
    canvas = tk.Canvas(root, width=width * scale, height=height * scale,
                       background="#15171a", highlightthickness=0)
    canvas.pack()
    status = tk.StringVar()
    tk.Label(root, textvariable=status, anchor="w", font=("Consolas", 10)).pack(fill="x")
    state = {"frame": 0, "paused": False, "photo": None}

    def who(step: int) -> str:
        if step >= episode["steps"]:
            return "end"
        if episode["past"][step]:
            return "PAST HORIZONTAL (padded)"
        return "LQR" if episode["controller"][step] == 0 else "casual correction"

    def render() -> None:
        index = state["frame"]
        frame = np.ascontiguousarray(frames[index])
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + frame.tobytes()
        state["photo"] = tk.PhotoImage(data=ppm, format="PPM").zoom(scale, scale)
        canvas.delete("all")
        canvas.create_image(0, 0, image=state["photo"], anchor="nw")
        step = start + index
        status.set(
            f"frame {step:4d} ({index}/{len(frames) - 1})  lean {episode['lean'][step]:+7.1f} deg  "
            f"next action: {who(step)}" + ("   [paused]" if state["paused"] else "")
        )

    def tick() -> None:
        if not state["paused"]:
            state["frame"] = (state["frame"] + 1) % len(frames)
            render()
        root.after(max(1, round(1000.0 / fps)), tick)

    def toggle(_event=None) -> None:
        state["paused"] = not state["paused"]
        render()

    def step_by(delta: int) -> None:
        state["paused"] = True
        state["frame"] = (state["frame"] + delta) % len(frames)
        render()

    root.bind("<space>", toggle)
    root.bind("<Left>", lambda _e: step_by(-1))
    root.bind("<Right>", lambda _e: step_by(1))
    render()
    root.after(max(1, round(1000.0 / fps)), tick)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--steps", type=int, default=32, help="Clip length in steps.")
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--list", action="store_true", help="Print the picked clips without opening windows.")
    args = parser.parse_args()
    paths = sorted(args.data.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no .npz episodes in {args.data}")
    episodes = [load(path) for path in paths]
    clips = pick_clips(episodes, args.steps)
    if not clips:
        raise SystemExit("no typical clips found (collect with --cartpole-policy lqr)")
    for index, (title, episode, start, stop) in enumerate(clips, start=1):
        print(f"[{index}/{len(clips)}] {title}: {episode['path'].name} frames {start}-{stop}")
        if not args.list:
            play(f"[{index}/{len(clips)}] {title}", episode, start, stop, args.scale, args.fps)


if __name__ == "__main__":
    main()
