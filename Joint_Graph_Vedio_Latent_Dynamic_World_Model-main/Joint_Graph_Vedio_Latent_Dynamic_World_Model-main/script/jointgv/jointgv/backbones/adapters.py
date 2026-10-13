"""Backbone adapter specifications for native visual representations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneAdapterSpec:
    """Bind a transition family to the visual representation it expects."""

    name: str
    transition: str
    visual: str


def resolve_backbone_adapter(name: str) -> BackboneAdapterSpec:
    normalized = name.lower().replace("-", "_")
    if normalized in {
        "david_ha",
        "davidha",
        "worldmodels2018",
        "mdn_lstm",
    }:
        return BackboneAdapterSpec("david_ha", "david_ha", "vae")
    if normalized in {
        "dreamerv3",
        "dreamer_v3",
        "burchim_dreamerv3",
        "burchim_dreamer_v3",
        "rssm",
    }:
        return BackboneAdapterSpec(
            "dreamerv3", "dreamerv3", "dreamerv3_native"
        )
    if normalized in {
        "dreamerv3_vae",
        "dreamer_v3_vae",
        "burchim_dreamerv3_vae",
        "burchim_dreamer_v3_vae",
    }:
        return BackboneAdapterSpec(
            "dreamerv3_vae", "dreamerv3", "vae"
        )
    if normalized in {"dino_wm", "dinowm", "dino_world_model"}:
        return BackboneAdapterSpec("dino_wm", "dino_wm", "dino_patch")
    raise ValueError(
        f"Unknown backbone {name!r}; choose 'david_ha', "
        "'dreamerv3', 'dreamerv3_vae', or 'dino_wm'"
    )
