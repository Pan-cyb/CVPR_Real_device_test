from __future__ import annotations

from .backbones import (
    DavidHaMDNBackbone,
    DinoWMBackbone,
    LatentBackbone,
)
from .blocks import DinoPatchVisualCodec, VisualCodec, VisualVAE


def build_visual_codec(kind: str, token_dim: int, **config: object) -> VisualCodec:
    normalized = kind.lower().replace("-", "_")
    if normalized == "vae":
        return VisualVAE(token_dim)
    if normalized in {"dino_patch", "dino_wm_patch"}:
        return DinoPatchVisualCodec(token_dim=token_dim, **config)
    raise ValueError(f"Unknown visual codec {kind!r}")


def build_backbone(
    name: str,
    *,
    token_count: int,
    token_dim: int,
    action_dim: int,
    **config: object,
) -> LatentBackbone:
    normalized = name.lower().replace("-", "_")
    common = {
        "token_count": token_count,
        "token_dim": token_dim,
        "action_dim": action_dim,
        **config,
    }
    if normalized in {"david_ha", "davidha", "worldmodels2018", "mdn_lstm"}:
        return DavidHaMDNBackbone(**common)
    if normalized in {"dino_wm", "dinowm"}:
        return DinoWMBackbone(**common)
    raise ValueError(
        f"Unknown transition {name!r}; choose 'david_ha' or 'dino_wm'"
    )
