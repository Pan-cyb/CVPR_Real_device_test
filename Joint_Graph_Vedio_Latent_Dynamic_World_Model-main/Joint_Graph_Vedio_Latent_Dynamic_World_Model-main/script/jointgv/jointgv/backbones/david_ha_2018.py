"""David Ha et al. (2018) factorized MDN-LSTM latent transition."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .base import BackboneOutput, CorrectionFunction, LatentBackbone


LSTMState = tuple[Tensor, Tensor]


@dataclass
class MDNOutput:
    """Mixture parameters with shape [B,T,N,D,K]."""

    logmix: Tensor
    mean: Tensor
    logstd: Tensor
    state: LSTMState

    def expectation(self, mean: Tensor | None = None) -> Tensor:
        component_mean = self.mean if mean is None else mean
        return (self.logmix.exp() * component_mean).sum(dim=-1)

    def sample(
        self, mean: Tensor | None = None, temperature: float = 1.0
    ) -> Tensor:
        if temperature <= 0:
            return self.expectation(mean)
        component_mean = self.mean if mean is None else mean
        indices = torch.distributions.Categorical(
            logits=self.logmix / temperature
        ).sample()
        selected_mean = component_mean.gather(
            -1, indices.unsqueeze(-1)
        ).squeeze(-1)
        selected_logstd = self.logstd.gather(
            -1, indices.unsqueeze(-1)
        ).squeeze(-1)
        noise = torch.randn_like(selected_mean) * math.sqrt(temperature)
        return selected_mean + selected_logstd.exp() * noise


def mdn_nll(
    output: MDNOutput, target: Tensor, mean: Tensor | None = None
) -> Tensor:
    component_mean = output.mean if mean is None else mean
    target = target.unsqueeze(-1)
    log_normal = (
        -0.5 * ((target - component_mean) / output.logstd.exp()).square()
        - output.logstd
        - 0.5 * math.log(2.0 * math.pi)
    )
    return -torch.logsumexp(
        output.logmix + log_normal, dim=-1
    ).mean()


class MDNLSTMTransition(nn.Module):
    def __init__(
        self,
        token_count: int,
        token_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        mixtures: int = 5,
        logstd_min: float = -7.0,
        logstd_max: float = 5.0,
    ):
        super().__init__()
        self.token_count = token_count
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mixtures = mixtures
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max
        latent_dim = token_count * token_dim
        self.lstm = nn.LSTM(
            latent_dim + action_dim, hidden_dim, batch_first=True
        )
        self.head = nn.Linear(hidden_dim, latent_dim * mixtures * 3)

    def initial_state(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> LSTMState:
        shape = (1, batch, self.hidden_dim)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def forward(
        self,
        tokens: Tensor,
        action: Tensor,
        state: LSTMState | None = None,
    ) -> MDNOutput:
        batch, time, nodes, dim = tokens.shape
        if (nodes, dim) != (self.token_count, self.token_dim):
            raise ValueError(
                f"Expected (*,{self.token_count},{self.token_dim}), "
                f"got {tuple(tokens.shape)}"
            )
        inputs = torch.cat((tokens.flatten(2), action), dim=-1)
        hidden, next_state = self.lstm(inputs, state)
        raw = self.head(hidden).reshape(
            batch, time, nodes, dim, 3, self.mixtures
        )
        logmix, mean, logstd = raw.unbind(dim=-2)
        return MDNOutput(
            logmix.log_softmax(dim=-1),
            mean,
            logstd.clamp(self.logstd_min, self.logstd_max),
            next_state,
        )


class DavidHaMDNBackbone(LatentBackbone):
    """Adapter that exposes the 2018 MDN-LSTM through the common API."""

    name = "david_ha"

    def __init__(
        self,
        token_count: int,
        token_dim: int,
        action_dim: int,
        rnn_dim: int = 256,
        mixtures: int = 5,
        **_: object,
    ):
        super().__init__()
        self.transition = MDNLSTMTransition(
            token_count, token_dim, action_dim, rnn_dim, mixtures
        )

    def forward_sequence(
        self,
        tokens: Tensor,
        actions: Tensor,
        state: LSTMState | None = None,
        target_tokens: Tensor | None = None,
        deterministic: bool = False,
    ) -> BackboneOutput:
        del target_tokens, deterministic
        mdn = self.transition(tokens, actions, state)
        base = mdn.expectation()
        return BackboneOutput(
            predicted_tokens=base,
            base_predicted_tokens=base,
            base_delta=base - tokens,
            state=mdn.state,
            aux={"mdn": mdn, "conditioned_mean": mdn.mean},
        )

    def apply_graph_condition(
        self,
        output: BackboneOutput,
        source_tokens: Tensor,
        graph_context: Tensor,
        correction: CorrectionFunction,
        enabled: bool,
    ) -> BackboneOutput:
        mdn = output.aux["mdn"]
        assert isinstance(mdn, MDNOutput)
        # [B,T,N,D,K] -> [B,T,N,K,D] so correction operates on D.
        component_delta = (
            mdn.mean - source_tokens.unsqueeze(-1)
        ).permute(0, 1, 2, 4, 3)
        component_correction = correction(
            component_delta, graph_context
        ).permute(0, 1, 2, 4, 3)
        conditioned_mean = (
            mdn.mean + component_correction if enabled else mdn.mean
        )
        output.predicted_tokens = mdn.expectation(conditioned_mean)
        output.aux["conditioned_mean"] = conditioned_mean
        output.aux["correction"] = (
            component_correction
            if enabled
            else torch.zeros_like(component_correction)
        )
        return output
