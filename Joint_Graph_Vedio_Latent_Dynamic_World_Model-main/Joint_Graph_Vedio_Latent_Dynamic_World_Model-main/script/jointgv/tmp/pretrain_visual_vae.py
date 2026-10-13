"""Pretrain the Gaussian VAE used by David Ha and burchim DreamerV3-VAE."""

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
from jointgv.blocks import VisualVAE
from jointgv.data import EpisodeSequences
from jointgv.dmc_protocol import validate_dmc_dataset


def vae_objective(
    model: VisualVAE,
    images: Tensor,
    beta: float,
    *,
    sample: bool,
) -> tuple[Tensor, dict[str, float]]:
    flat = images.flatten(0, 1)
    tokens, mu, logvar = model.encode(flat, sample=sample)
    reconstruction = model.decode(tokens)
    reconstruction_loss = torch.nn.functional.mse_loss(
        reconstruction, flat
    )
    kl = -0.5 * (
        1 + logvar - mu.square() - logvar.exp()
    ).mean()
    total = reconstruction_loss + beta * kl
    metrics = {
        "total": float(total.detach().item()),
        "reconstruction": float(reconstruction_loss.detach().item()),
        "vae_kl": float(kl.detach().item()),
        "latent_rms": float(mu.detach().square().mean().sqrt().item()),
        "latent_absmax": float(mu.detach().abs().amax().item()),
    }
    return total, metrics


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
    model: VisualVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    model.eval()
    averages = Averages()
    for batch in loader:
        images = unpack_images(batch, device)
        _, metrics = vae_objective(
            model, images, beta, sample=False
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain a stable Gaussian visual VAE on DMC RGB frames."
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--allow-partial-dmc", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.sequence_length < 1 or args.stride < 1:
        parser.error("--sequence-length and --stride must be positive")
    if args.lr <= 0 or args.beta_kl < 0:
        parser.error("--lr must be positive and --beta-kl nonnegative")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")

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
    model = VisualVAE(args.token_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_value = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        averages = Averages()
        progress = tqdm(loader, desc=f"visual VAE epoch {epoch}")
        for batch in progress:
            images = unpack_images(batch, device)
            total, metrics = vae_objective(
                model, images, args.beta_kl, sample=True
            )
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError(
                    f"Non-finite VAE loss at epoch={epoch}"
                )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            metrics["grad_norm"] = float(grad_norm.detach().item())
            averages.update(metrics, images.shape[0])
            progress.set_postfix(
                loss=f"{metrics['total']:.4f}",
                kl=f"{metrics['vae_kl']:.4f}",
                rms=f"{metrics['latent_rms']:.4f}",
            )
        train_metrics = averages.compute()
        validation_metrics = (
            validate(model, validation_loader, device, args.beta_kl)
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
            "visual_kind": "vae",
            "token_dim": args.token_dim,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "best_metric_value": best_value,
            "best_epoch": best_epoch,
            "beta_kl": args.beta_kl,
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
