from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from torch import Tensor, nn


@dataclass
class BackboneOutput:
    """Common transition output consumed by the Graph Dynamics plugin."""

    predicted_tokens: Tensor
    base_predicted_tokens: Tensor
    base_delta: Tensor
    state: Any
    aux: dict[str, Any] = field(default_factory=dict)


CorrectionFunction = Callable[[Tensor, Tensor], Tensor]


class LatentBackbone(nn.Module, ABC):
    """Minimal interface required from a pluggable latent transition model."""

    name: str

    @abstractmethod
    def forward_sequence(
        self,
        tokens: Tensor,
        actions: Tensor,
        state: Any = None,
        target_tokens: Tensor | None = None,
        deterministic: bool = False,
    ) -> BackboneOutput:
        raise NotImplementedError

    @abstractmethod
    def apply_graph_condition(
        self,
        output: BackboneOutput,
        source_tokens: Tensor,
        graph_context: Tensor,
        correction: CorrectionFunction,
        enabled: bool,
    ) -> BackboneOutput:
        raise NotImplementedError

    def feedback(self, state: Any, predicted_tokens: Tensor) -> Any:
        """Update recurrent state from a corrected prediction when necessary."""
        return state
