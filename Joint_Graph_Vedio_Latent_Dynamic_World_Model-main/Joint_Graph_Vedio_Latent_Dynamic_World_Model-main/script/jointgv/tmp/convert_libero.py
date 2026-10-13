from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


IMAGE_CANDIDATES = (
    "agentview_rgb",
    "agentview_image",
    "rgb",
)


def natural_key(value: str) -> list[str | int]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def print_tree(group: h5py.Group, prefix: str = "") -> None:
    for key in group:
        value = group[key]
        path = f"{prefix}/{key}"
        if isinstance(value, h5py.Dataset):
            print(f"{path}: shape={value.shape} dtype={value.dtype}")
        else:
            print(f"{path}/")
            print_tree(value, path)


def find_image_key(obs: h5py.Group, requested: str | None) -> str:
    if requested is not None:
        if requested not in obs:
            raise KeyError(
                f"Requested image key {requested!r} is absent; "
                f"available keys: {sorted(obs.keys())}"
            )
        return requested
    for candidate in IMAGE_CANDIDATES:
        if candidate in obs:
            return candidate
    rgb_keys = [key for key in obs if "rgb" in key.lower() or "image" in key.lower()]
    if len(rgb_keys) == 1:
        return rgb_keys[0]
    raise KeyError(
        "Could not choose an RGB observation automatically; "
        f"available keys: {sorted(obs.keys())}. Pass --image-key."
    )


def resize_images(images: np.ndarray, size: int) -> np.ndarray:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected RGB images [T,H,W,3], got {images.shape}")
    if images.shape[1:3] == (size, size):
        return images.astype(np.uint8, copy=False)
    tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return tensor.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def metadata_from_file(handle: h5py.File) -> dict[str, str]:
    metadata = {}
    for key, value in handle.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, (str, int, float, bool, np.generic)):
            metadata[key] = str(value)
    if "data" in handle:
        for key, value in handle["data"].attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if isinstance(value, (str, int, float, bool, np.generic)):
                metadata[f"data.{key}"] = str(value)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official LIBERO HDF5 demonstrations to JointGV episodes."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/libero_10_tiny"))
    parser.add_argument("--max-demos", type=int, default=2)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--image-key")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the HDF5 structure and exit without converting.",
    )
    args = parser.parse_args()
    if args.max_demos < 1:
        parser.error("--max-demos must be positive")
    if args.size < 1:
        parser.error("--size must be positive")
    if args.stride < 1:
        parser.error("--stride must be positive")

    with h5py.File(args.input, "r") as handle:
        if args.inspect:
            print_tree(handle)
            return
        if "data" not in handle or not isinstance(handle["data"], h5py.Group):
            raise KeyError("Expected a robomimic-style /data group in the HDF5 file")
        demos = sorted(handle["data"].keys(), key=natural_key)[: args.max_demos]
        if not demos:
            raise ValueError("The HDF5 file does not contain any demonstrations")
        metadata = metadata_from_file(handle)
        args.output.mkdir(parents=True, exist_ok=True)
        task_stem = args.input.stem.removesuffix("_demo")

        for output_id, demo_name in enumerate(demos):
            demo = handle["data"][demo_name]
            if "obs" not in demo or "actions" not in demo:
                raise KeyError(f"{demo_name} must contain obs and actions")
            obs = demo["obs"]
            image_key = find_image_key(obs, args.image_key)
            images = np.asarray(obs[image_key])[:: args.stride]
            actions = np.asarray(demo["actions"], dtype=np.float32)[:: args.stride]
            rewards = (
                np.asarray(demo["rewards"], dtype=np.float32)[:: args.stride]
                if "rewards" in demo
                else np.zeros(len(actions), dtype=np.float32)
            )
            if args.flip_vertical:
                images = images[:, ::-1]
            if args.flip_horizontal:
                images = images[:, :, ::-1]
            images = resize_images(np.ascontiguousarray(images), args.size)

            transitions = min(len(actions), len(images) - 1)
            if transitions < 1:
                raise ValueError(
                    f"{demo_name} has insufficient aligned images/actions: "
                    f"{len(images)} images, {len(actions)} actions"
                )
            payload: dict[str, np.ndarray] = {
                "image": images[: transitions + 1],
                "action": actions[:transitions],
                "reward": rewards[:transitions],
                "source_hdf5": np.asarray(str(args.input)),
                "source_demo": np.asarray(demo_name),
                "source_image_key": np.asarray(image_key),
                "source_metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            }
            if "states" in demo:
                payload["sim_state"] = np.asarray(demo["states"])[
                    :: args.stride
                ][: transitions + 1]
            for source_key, target_key in (
                ("joint_states", "robot_joint_state"),
                ("gripper_states", "robot_gripper_state"),
                ("object", "object_state"),
            ):
                if source_key in obs:
                    payload[target_key] = np.asarray(obs[source_key])[
                        :: args.stride
                    ][: transitions + 1]
            output = args.output / f"{task_stem}_{output_id:05d}.npz"
            np.savez_compressed(output, **payload)
            print(
                f"[{output_id + 1}/{len(demos)}] {output} "
                f"({transitions} transitions, image_key={image_key})"
            )


if __name__ == "__main__":
    main()
