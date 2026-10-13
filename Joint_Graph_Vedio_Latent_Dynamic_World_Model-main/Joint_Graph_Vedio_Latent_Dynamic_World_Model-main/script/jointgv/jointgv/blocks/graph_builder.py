from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class GraphObservation:
    """A probabilistic scene graph inferred from visual tokens."""

    nodes: Tensor
    node_mu: Tensor
    node_logvar: Tensor
    relation_logits: Tensor
    positions_2d: Tensor
    presence_logits: Tensor
    attention: Tensor
    temporal_gate: Tensor | None = None
    identity_features: Tensor | None = None
    temporal_correction: Tensor | None = None


class LatentGraphBuilder(nn.Module):
    """RGB-only proxy graph: persistent slots plus a VTransE relation head.

    This is intentionally isolated behind a small interface. A future detector,
    object-centric encoder, or user-provided scene graph only needs to produce a
    :class:`GraphObservation` with the same fields.
    """

    requires_images = False

    def __init__(
        self,
        token_dim: int,
        node_dim: int,
        nodes: int,
        token_count: int = 16,
        relation_count: int = 4,
        teacher_state_dim: int = 13,
    ):
        super().__init__()
        self.nodes = nodes
        self.node_dim = node_dim
        self.relation_count = relation_count
        self.teacher_state_dim = teacher_state_dim
        self.initial_slots = nn.Parameter(torch.randn(nodes, node_dim) * 0.02)
        self.token_position = nn.Parameter(
            torch.randn(token_count, node_dim) * 0.01
        )
        self.query = nn.Linear(node_dim, node_dim)
        self.key = nn.Linear(token_dim, node_dim)
        self.value = nn.Linear(token_dim, node_dim)
        self.slot_update = nn.GRUCell(node_dim, node_dim)
        self.slot_norm = nn.LayerNorm(node_dim)
        self.node_mu = nn.Linear(node_dim, node_dim)
        self.node_logvar = nn.Linear(node_dim, node_dim)
        self.presence = nn.Linear(node_dim, 1)

        self.subject = nn.Linear(node_dim, node_dim, bias=False)
        self.object = nn.Linear(node_dim, node_dim, bias=False)
        self.relation_translation = nn.Parameter(
            torch.randn(relation_count, node_dim) * 0.02
        )
        self.relation_residual = nn.Sequential(
            nn.Linear(2 * node_dim + 2, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, relation_count),
        )
        self.teacher_state = nn.Linear(node_dim, teacher_state_dim)

        side = int(token_count**0.5)
        if side * side != token_count:
            raise ValueError("Graph slots require a square visual-token grid")
        axis = torch.linspace(-1.0, 1.0, side)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "token_grid",
            torch.stack((grid_x, grid_y), dim=-1).reshape(token_count, 2),
            persistent=False,
        )

    def _initial_state(self, batch: int) -> Tensor:
        return self.initial_slots.unsqueeze(0).expand(batch, -1, -1)

    def relation_head(
        self, nodes: Tensor, positions_2d: Tensor | None = None
    ) -> Tensor:
        batch, count, _ = nodes.shape
        if positions_2d is None:
            positions_2d = nodes.new_zeros(batch, count, 2)
        subject = self.subject(nodes)[:, :, None, None, :]
        object_ = self.object(nodes)[:, None, :, None, :]
        translation = self.relation_translation[None, None, None, :, :]
        translation_score = -(subject + translation - object_).square().mean(-1)

        sender = nodes[:, :, None, :].expand(-1, -1, count, -1)
        receiver = nodes[:, None, :, :].expand(-1, count, -1, -1)
        relative_position = (
            positions_2d[:, None, :, :] - positions_2d[:, :, None, :]
        )
        residual = self.relation_residual(
            torch.cat((sender, receiver, relative_position), dim=-1)
        )
        logits = translation_score + residual
        diagonal = torch.eye(count, device=nodes.device, dtype=torch.bool)
        logits = logits.clone()
        logits[:, diagonal, 0] += 20.0
        if self.relation_count > 1:
            logits[:, diagonal, 1:] -= 20.0
        return logits

    def forward(
        self,
        tokens: Tensor,
        previous_slots: Tensor | None = None,
        sample: bool = False,
        reset_mask: Tensor | None = None,
    ) -> GraphObservation:
        batch, token_count, _ = tokens.shape
        if token_count != self.token_position.shape[0]:
            raise ValueError(
                f"Expected {self.token_position.shape[0]} tokens, got {token_count}"
            )
        previous = (
            self._initial_state(batch)
            if previous_slots is None
            else previous_slots
        )
        if previous_slots is not None and reset_mask is not None:
            reset = reset_mask.to(tokens.dtype)[:, None, None]
            initial = self._initial_state(batch)
            previous = previous * (1.0 - reset) + initial * reset
        keys = self.key(tokens) + self.token_position[None]
        values = self.value(tokens) + self.token_position[None]
        scores = torch.einsum(
            "bnd,btd->bnt", self.query(previous), keys
        ) / self.node_dim**0.5
        attention = scores.softmax(dim=1) + 1e-8
        attention = attention / attention.sum(dim=-1, keepdim=True)
        observed = torch.bmm(attention, values)
        updated = self.slot_update(
            observed.reshape(batch * self.nodes, self.node_dim),
            previous.reshape(batch * self.nodes, self.node_dim),
        ).reshape(batch, self.nodes, self.node_dim)
        updated = self.slot_norm(updated)
        mu = self.node_mu(updated)
        logvar = self.node_logvar(updated).clamp(-8, 5)
        nodes = (
            mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            if sample
            else mu
        )
        positions = torch.einsum("bnt,td->bnd", attention, self.token_grid)
        return GraphObservation(
            nodes=nodes,
            node_mu=mu,
            node_logvar=logvar,
            relation_logits=self.relation_head(nodes, positions),
            positions_2d=positions,
            presence_logits=self.presence(nodes).squeeze(-1),
            attention=attention,
        )

    def forward_sequence(
        self,
        tokens: Tensor,
        sample: bool = False,
        firsts: Tensor | None = None,
    ) -> dict[str, Tensor]:
        previous: Tensor | None = None
        observations: list[GraphObservation] = []
        for step in range(tokens.shape[1]):
            if previous is not None and firsts is not None:
                reset = firsts[:, step].to(tokens.dtype)[:, None, None]
                initial = self._initial_state(tokens.shape[0])
                previous = previous * (1.0 - reset) + initial * reset
            observation = self(tokens[:, step], previous, sample=sample)
            observations.append(observation)
            previous = observation.node_mu
        return {
            field: torch.stack(
                [getattr(item, field) for item in observations], dim=1
            )
            for field in (
                "nodes",
                "node_mu",
                "node_logvar",
                "relation_logits",
                "positions_2d",
                "presence_logits",
                "attention",
            )
        }
