"""Selectable latent transition backbones."""

from .adapters import BackboneAdapterSpec, resolve_backbone_adapter
from .base import BackboneOutput, LatentBackbone
from .dino_wm import DinoWMBackbone, DinoWMPredictor
from .dreamerv3 import DreamerV3WorldModel
from .david_ha_2018 import (
    DavidHaMDNBackbone,
    MDNOutput,
    MDNLSTMTransition,
    mdn_nll,
)

__all__ = [
    "BackboneOutput",
    "BackboneAdapterSpec",
    "DinoWMBackbone",
    "DinoWMPredictor",
    "DreamerV3WorldModel",
    "DavidHaMDNBackbone",
    "LatentBackbone",
    "MDNLSTMTransition",
    "MDNOutput",
    "mdn_nll",
    "resolve_backbone_adapter",
]
