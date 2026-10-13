from __future__ import annotations

import torch
from torch import Tensor, nn

from .visual_codec import VisualCodec


class VisualVAE(VisualCodec):
    """64x64 RGB VAE whose 4x4 feature grid is exposed as latent tokens."""

    is_variational = True

    def __init__(self, token_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(128, 2 * token_dim, 4, 2, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(token_dim, 128, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )
        self.token_dim = token_dim
        self.token_count = 16

    def encode(
        self, image: Tensor, sample: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        stats = self.encoder(image)
        mu, logvar = stats.chunk(2, dim=1)
        logvar = logvar.clamp(-8, 5)
        z = (
            mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            if sample
            else mu
        )

        def to_tokens(value: Tensor) -> Tensor:
            return value.flatten(2).transpose(1, 2)

        return to_tokens(z), to_tokens(mu), to_tokens(logvar)

    def decode(self, tokens: Tensor) -> Tensor:
        batch, count, dim = tokens.shape
        side = int(count**0.5)
        if side * side != count:
            raise ValueError("Visual token count must form a square grid")
        return self.decoder(
            tokens.transpose(1, 2).reshape(batch, dim, side, side)
        )
