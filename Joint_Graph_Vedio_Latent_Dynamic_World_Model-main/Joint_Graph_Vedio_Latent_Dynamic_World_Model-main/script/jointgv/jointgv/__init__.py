"""Joint graph-video latent dynamics with pluggable backbones."""

from .backbones import (
    BackboneAdapterSpec,
    DavidHaMDNBackbone,
    DreamerV3WorldModel,
    MDNLSTMTransition,
    MDNOutput,
    mdn_nll,
    resolve_backbone_adapter,
)
from .losses import JointLoss, compute_loss, compute_sequence_loss
from .loss_balance import GradientRatioOTController
from .world_model import JointGraphVideoWorldModel

__all__ = [
    "DavidHaMDNBackbone",
    "BackboneAdapterSpec",
    "DreamerV3WorldModel",
    "JointGraphVideoWorldModel",
    "JointLoss",
    "GradientRatioOTController",
    "MDNLSTMTransition",
    "MDNOutput",
    "compute_loss",
    "compute_sequence_loss",
    "mdn_nll",
    "resolve_backbone_adapter",
]
