"""Compatibility facade for the refactored JointGV package."""

from .blocks import (
    GraphBlock,
    GraphDynamics,
    GraphObservation,
    LatentGraphBuilder,
    VisualVAE,
)
from .losses import JointLoss, compute_loss, compute_sequence_loss
from .world_model import JointGraphVideoWorldModel

__all__ = [
    "GraphBlock",
    "GraphDynamics",
    "GraphObservation",
    "JointGraphVideoWorldModel",
    "JointLoss",
    "LatentGraphBuilder",
    "VisualVAE",
    "compute_loss",
    "compute_sequence_loss",
]
