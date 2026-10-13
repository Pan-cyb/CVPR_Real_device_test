"""Render frozen-DINO encode/decode reconstructions next to ground truth.

Loads a checkpoint produced by ``pretrain_dino_decoder.py`` and saves an
episode comparison viewable with the existing ``view_episode.py`` -- no
viewer changes needed, since this reuses the ``frames``/``actor_rgb`` track
pair it already recognizes as "Ground truth" / "Tokenizer reconstruction".
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv.blocks import DinoPatchVisualCodec


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ground-truth DMC frames against the frozen-DINO "
            "encode/decode reconstruction from a pretrain_dino_decoder.py "
            "checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Cap the number of frames rendered from the episode.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/jointgv/dino_decoder_comparison.npz"),
    )
    args = parser.parse_args()
    if args.max_frames < 1 or args.batch_size < 1:
        parser.error("--max-frames and --batch-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload.get("visual_kind") != "dino_patch":
        raise ValueError(
            f"{args.checkpoint} is not a pretrain_dino_decoder.py checkpoint "
            f"(visual_kind={payload.get('visual_kind')!r})"
        )
    model = DinoPatchVisualCodec(
        token_dim=int(payload["token_dim"]),
        dino_model=str(payload["dino_model"]),
        dino_image_size=int(payload["dino_image_size"]),
        dino_freeze_encoder=True,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    with np.load(args.episode, allow_pickle=False) as episode:
        frames = episode["image"][: args.max_frames].copy()
    images = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)

    reconstructions = []
    with torch.inference_mode():
        for start in range(0, images.shape[0], args.batch_size):
            chunk = images[start : start + args.batch_size].to(device)
            tokens, _, _ = model.encode(chunk, sample=False)
            reconstructions.append(as_uint8(model.decode(tokens)))
    reconstruction = np.concatenate(reconstructions, axis=0)
    reconstruction_error = float(
        np.mean(
            (reconstruction.astype(np.float32) - frames.astype(np.float32)) ** 2
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frames=frames,
        actor_rgb=reconstruction,
        reconstruction_mse=np.asarray(reconstruction_error, dtype=np.float32),
        source_episode=np.asarray(str(args.episode)),
        checkpoint=np.asarray(str(args.checkpoint)),
    )
    print(f"saved: {args.output}")
    print(
        f"frames={frames.shape[0]} pixel_mse={reconstruction_error:.6f} "
        f"(pixel values in [0,1]^2)"
    )


if __name__ == "__main__":
    main()
