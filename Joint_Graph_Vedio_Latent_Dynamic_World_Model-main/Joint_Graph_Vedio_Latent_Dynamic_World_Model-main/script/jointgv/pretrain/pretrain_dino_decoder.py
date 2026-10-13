"""Pretrain the DINO-WM visual reconstruction module (projection + decoder).

The DINOv2 encoder itself is always frozen (a fixed pretrained visual
representation is DINO-WM's defining choice, not something this repo
retrains). This script only fits the linear projection into ``token_dim``
and the diagnostic conv decoder used to visualize DINO patch tokens as RGB,
mirroring the existing ``pretrain_visual_vae.py`` -> ``--visual-checkpoint``
-> ``--freeze-visual`` workflow used by the David Ha and DreamerV3-VAE
backbones. There is no KL term here: DINO patch features are deterministic,
unlike the Gaussian ``VisualVAE``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv.blocks import DinoPatchVisualCodec
from jointgv.data import EpisodeSequences
from jointgv.dmc_protocol import validate_dmc_dataset


def reconstruction_objective(
    model: DinoPatchVisualCodec, images: Tensor
) -> tuple[Tensor, dict[str, float]]:
    flat = images.flatten(0, 1)
    tokens, mu, _ = model.encode(flat, sample=False)
    reconstruction = model.decode(tokens)
    loss = torch.nn.functional.mse_loss(reconstruction, flat)
    metrics = {
        "total": float(loss.detach().item()),
        "reconstruction": float(loss.detach().item()),
        "latent_rms": float(mu.detach().square().mean().sqrt().item()),
        "latent_absmax": float(mu.detach().abs().amax().item()),
    }
    return loss, metrics


class Averages:
    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.weight = 0

    def update(self, metrics: dict[str, float], weight: int) -> None:
        self.weight += weight
        for key, value in metrics.items():
            self.total[key] = self.total.get(key, 0.0) + value * weight

    def compute(self) -> dict[str, float]:
        return {key: value / self.weight for key, value in self.total.items()}


def unpack_images(batch: object, device: torch.device) -> Tensor:
    if not isinstance(batch, (tuple, list)) or not batch:
        raise TypeError("Unexpected DMC batch structure")
    images = batch[0]
    if not isinstance(images, Tensor):
        raise TypeError("DMC batch does not contain image tensors")
    return images.to(device)


@torch.no_grad()
def validate(
    model: DinoPatchVisualCodec, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    averages = Averages()
    for batch in loader:
        images = unpack_images(batch, device)
        _, metrics = reconstruction_objective(model, images)
        averages.update(metrics, images.shape[0])
    return averages.compute()


def atomic_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return prefix + " " + " ".join(
        f"{key}={value:.6f}" for key, value in metrics.items()
    )


def check_resume_architecture_matches(
    payload: dict[str, object], args: argparse.Namespace
) -> None:
    """Refuse a resume that would silently reload weights into a mismatched
    architecture (e.g. --token-dim or --dino-image-size changed)."""
    expected = {
        "token_dim": args.token_dim,
        "dino_model": args.dino_model,
        "dino_image_size": args.dino_image_size,
    }
    for key, value in expected.items():
        checkpoint_value = payload.get(key)
        if checkpoint_value != value:
            raise ValueError(
                f"--resume checkpoint {key}={checkpoint_value!r} does not "
                f"match --{key.replace('_', '-')}={value!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the DINO-WM visual reconstruction module (frozen DINOv2 -> "
            "optional width projection -> conv decoder) on DMC RGB frames. "
            "The projection is an identity at the encoder's native width."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument(
        "--dino-model",
        type=str,
        default="dinov2_vits14",
        help=(
            "torch.hub facebookresearch/dinov2 variant name (downloaded and "
            "cached on first use), or 'debug_conv' for a tiny untrained "
            "stand-in used only by smoke tests."
        ),
    )
    parser.add_argument(
        "--dino-image-size",
        type=int,
        default=112,
        help="Must match the value later passed to train.py --dino-image-size.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--allow-partial-dmc", action="store_true")
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "Resume model, optimizer, epoch, and best metric from a "
            "checkpoint (e.g. .../checkpoints/epoch_0020.pt or latest.pt). "
            "--token-dim/--dino-model/--dino-image-size must match the run "
            "being resumed."
        ),
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.sequence_length < 1 or args.stride < 1:
        parser.error("--sequence-length and --stride must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"--resume checkpoint does not exist: {args.resume}")

    protocol = validate_dmc_dataset(
        args.data, allow_partial=args.allow_partial_dmc
    )
    files = sorted(args.data.glob("*.npz"))
    if not files:
        parser.error(f"No .npz episodes found in {args.data}")
    if args.validation_data is not None:
        if args.validation_data.resolve() == args.data.resolve():
            parser.error("--validation-data must differ from --data")
        validate_dmc_dataset(
            args.validation_data, allow_partial=args.allow_partial_dmc
        )
        training_files = files
        validation_files = sorted(args.validation_data.glob("*.npz"))
        if not validation_files:
            parser.error(
                "--validation-data contains no top-level .npz episodes: "
                f"{args.validation_data}. Collect the validation split first "
                "or correct the path."
            )
        validation_root = args.validation_data
    elif args.validation_fraction > 0 and len(files) >= 2:
        count = max(1, round(len(files) * args.validation_fraction))
        count = min(count, len(files) - 1)
        training_files, validation_files = files[:-count], files[-count:]
        validation_root = args.data
    else:
        training_files, validation_files = files, []
        validation_root = args.data
    print(
        "verified DMC protocol:",
        protocol["protocol"],
        f"train_episodes={len(training_files)}",
        f"validation_episodes={len(validation_files)}",
    )

    training = EpisodeSequences(
        args.data,
        args.sequence_length,
        args.stride,
        include_graph_targets=False,
        episode_files=training_files,
    )
    validation_data = (
        EpisodeSequences(
            validation_root,
            args.sequence_length,
            args.stride,
            include_graph_targets=False,
            episode_files=validation_files,
        )
        if validation_files
        else None
    )
    loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    validation_loader = (
        DataLoader(
            validation_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        if validation_data is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoPatchVisualCodec(
        token_dim=args.token_dim,
        dino_model=args.dino_model,
        dino_image_size=args.dino_image_size,
        dino_freeze_encoder=True,
    ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    best_value = float("inf")
    best_epoch = 0
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        try:
            check_resume_architecture_matches(checkpoint, args)
        except ValueError as error:
            parser.error(str(error))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1
        best_value = float(checkpoint.get("best_metric_value", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", completed_epoch))
        if start_epoch > args.epochs:
            parser.error(
                f"--resume checkpoint already completed epoch {completed_epoch}, "
                f"but --epochs is only {args.epochs}"
            )
        print(
            f"resumed {args.resume} after epoch={completed_epoch} "
            f"best_value={best_value:.6f} best_epoch={best_epoch}"
        )
    elif metrics_path.exists():
        metrics_path.write_text("", encoding="utf-8")
    else:
        metrics_path.touch()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        averages = Averages()
        progress = tqdm(loader, desc=f"DINO decoder epoch {epoch}")
        for batch in progress:
            images = unpack_images(batch, device)
            total, metrics = reconstruction_objective(model, images)
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError(
                    f"Non-finite DINO decoder loss at epoch={epoch}"
                )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            metrics["grad_norm"] = float(grad_norm.detach().item())
            averages.update(metrics, images.shape[0])
            progress.set_postfix(
                loss=f"{metrics['total']:.4f}",
                rms=f"{metrics['latent_rms']:.4f}",
            )
        train_metrics = averages.compute()
        validation_metrics = (
            validate(model, validation_loader, device)
            if validation_loader is not None
            else None
        )
        selection = (
            validation_metrics["total"]
            if validation_metrics is not None
            else train_metrics["total"]
        )
        is_best = selection < best_value
        if is_best:
            best_value = selection
            best_epoch = epoch
        payload: dict[str, object] = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "visual_kind": "dino_patch",
            "token_dim": args.token_dim,
            "dino_model": args.dino_model,
            "dino_image_size": args.dino_image_size,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "best_metric_value": best_value,
            "best_epoch": best_epoch,
            "data_protocol": protocol["protocol"],
        }
        epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
        atomic_save(payload, epoch_path)
        atomic_copy(epoch_path, args.output / "latest.pt")
        if is_best:
            atomic_copy(epoch_path, args.output / "best.pt")
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "best_metric_value": best_value,
            "best_epoch": best_epoch,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(format_metrics("train", train_metrics))
        if validation_metrics is not None:
            print(format_metrics("validation", validation_metrics))
        suffix = " |--> saved best.pt" if is_best else ""
        print(
            f"epoch={epoch} best_value={best_value:.6f} "
            f"best_epoch={best_epoch}{suffix}"
        )


if __name__ == "__main__":
    main()
