"""Visual latent codecs selected by backbone adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class VisualCodec(nn.Module, ABC):
    """Image-to-spatial-token interface shared by JointGV backbones."""

    token_count: int
    token_dim: int
    is_variational: bool

    @abstractmethod
    def encode(
        self, image: Tensor, sample: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return tokens, distribution mean, and distribution log variance."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError
