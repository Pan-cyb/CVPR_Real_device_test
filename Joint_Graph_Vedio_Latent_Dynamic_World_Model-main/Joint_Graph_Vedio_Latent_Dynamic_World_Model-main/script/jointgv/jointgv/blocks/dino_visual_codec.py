"""Frozen DINOv2 patch-token visual codec for the DINO-WM backbone.

``DinoV2Encoder`` is imported unmodified from the official ``dino_wm``
submodule at ``benchmark/dino_wm`` (mirroring how RelTR is vendored for the
Explicit Graph Observer). Only the file that defines the encoder is loaded,
by path, under a private module name -- not by inserting the submodule root
onto ``sys.path`` -- so this never collides with RelTR's own top-level
``models`` package when both are active in the same process.

The encoder is always frozen: a fixed pretrained visual representation is
DINO-WM's defining choice, not an incidental default. When ``token_dim`` is
the encoder's native feature width (384 for ``dinov2_vits14``), the patch
tokens are passed through unchanged. A trainable projection is used only for
explicitly requested non-native widths. The diagnostic decoder remains
trainable, matching the existing two-stage workflow already used for
``VisualVAE`` (``pretrain_visual_vae.py`` -> ``--visual-checkpoint`` ->
``--freeze-visual``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .visual_codec import VisualCodec


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _repository_root() -> Path:
    # .../script/jointgv/jointgv/blocks/dino_visual_codec.py -> repository root
    return Path(__file__).resolve().parents[4]


def _load_dino_v2_encoder_class():
    """Import the vendored official DINO-WM encoder without copying its code."""
    module_path = (
        _repository_root() / "benchmark" / "dino_wm" / "models" / "dino.py"
    ).resolve()
    if not module_path.is_file():
        raise FileNotFoundError(
            "Missing official DINO-WM submodule. Run: "
            "git submodule update --init --recursive"
        )
    module_name = "jointgv_vendored_dino_wm_encoder"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Cannot load the DINO-WM encoder module from {module_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.DinoV2Encoder


class _DebugConvEncoder(nn.Module):
    """Tiny random-init stand-in for DINOv2; smoke tests only, never for
    reported runs. Avoids a network fetch while still exercising the exact
    patch-token shapes and freeze/train-mode plumbing that the real encoder
    requires."""

    def __init__(self, patch_size: int = 8, emb_dim: int = 32):
        super().__init__()
        self.name = "debug_conv"
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.net = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        features = self.net(x)
        return features.flatten(2).transpose(1, 2)


class DinoPatchDecoder(nn.Module):
    """Trainable decoder from DINO patch tokens back to RGB, for viewer checks.

    The token grid is resized to ``image_size`` with bilinear interpolation
    before a small conv stack refines it to RGB, so the same decoder shape
    works regardless of ``--dino-image-size`` / the resulting patch grid.
    """

    def __init__(self, token_dim: int, image_size: int = 64, hidden: int = 128):
        super().__init__()
        self.image_size = image_size
        self.refine = nn.Sequential(
            nn.Conv2d(token_dim, hidden, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden // 2, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        batch, count, dim = tokens.shape
        side = int(round(count**0.5))
        if side * side != count:
            raise ValueError("DINO patch token count must form a square grid")
        grid = tokens.transpose(1, 2).reshape(batch, dim, side, side)
        grid = F.interpolate(
            grid,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(grid)


class DinoPatchVisualCodec(VisualCodec):
    """Frozen DINOv2 patch tokens with an optional width projection."""

    is_variational = False

    def __init__(
        self,
        token_dim: int = 64,
        dino_model: str = "dinov2_vits14",
        dino_image_size: int = 112,
        dino_freeze_encoder: bool = True,
    ):
        super().__init__()
        if dino_image_size < 14:
            raise ValueError("dino_image_size must be at least 14")
        self.dino_model_name = dino_model
        self.dino_image_size = int(dino_image_size)
        self.freeze_encoder = bool(dino_freeze_encoder)
        if dino_model == "debug_conv":
            self.encoder = _DebugConvEncoder()
        else:
            dino_v2_encoder_cls = _load_dino_v2_encoder_class()
            self.encoder = dino_v2_encoder_cls(dino_model, "x_norm_patchtokens")
        patch_size = int(self.encoder.patch_size)
        if self.dino_image_size % patch_size != 0:
            raise ValueError(
                f"--dino-image-size {self.dino_image_size} must be a multiple "
                f"of the {dino_model!r} patch size {patch_size}"
            )
        self.patches_per_side = self.dino_image_size // patch_size
        self.token_count = self.patches_per_side**2
        self.token_dim = token_dim
        native_dim = int(self.encoder.emb_dim)
        self.native_token_dim = native_dim
        self.uses_native_tokens = token_dim == native_dim
        # Preserve the official DINO-WM representation when its native width
        # is requested. In particular, do not learn a reconstruction-oriented
        # 384 -> 384 transform and then freeze it before dynamics training.
        self.project_in = (
            nn.Identity()
            if self.uses_native_tokens
            else nn.Linear(native_dim, token_dim)
        )
        print(
            "DINO patch tokens: "
            f"native_dim={native_dim} configured_dim={token_dim} "
            f"projection={'identity' if self.uses_native_tokens else 'linear'}"
        )
        self.decoder = DinoPatchDecoder(token_dim, image_size=64)
        self.register_buffer(
            "_imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True) -> "DinoPatchVisualCodec":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _prepare(self, image: Tensor) -> Tensor:
        resized = F.interpolate(
            image,
            size=(self.dino_image_size, self.dino_image_size),
            mode="bilinear",
            align_corners=False,
        )
        return (resized - self._imagenet_mean) / self._imagenet_std

    def encode(
        self, image: Tensor, sample: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        del sample  # deterministic: conditions on fixed frozen-DINO features
        normalized = self._prepare(image)
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(normalized)
            features = features.detach()
        else:
            features = self.encoder(normalized)
        tokens = self.project_in(features)
        return tokens, tokens, torch.zeros_like(tokens)

    def decode(self, tokens: Tensor) -> Tensor:
        return self.decoder(tokens)
