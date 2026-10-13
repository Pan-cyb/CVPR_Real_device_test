from __future__ import annotations

import torch
from torch import Tensor, nn


class GraphBlock(nn.Module):
    """Relation-weighted Graph Network edge/aggregate/node update."""

    def __init__(self, dim: int, action_dim: int, relation_count: int):
        super().__init__()
        self.relation_embedding = nn.Linear(relation_count, dim, bias=False)
        self.edge = nn.Sequential(
            nn.Linear(3 * dim + action_dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim),
        )
        self.node = nn.Sequential(
            nn.Linear(2 * dim + action_dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, nodes: Tensor, action: Tensor, relation_logits: Tensor
    ) -> Tensor:
        _, count, _ = nodes.shape
        sender = nodes[:, :, None, :].expand(-1, -1, count, -1)
        receiver = nodes[:, None, :, :].expand(-1, count, -1, -1)
        act_edge = action[:, None, None, :].expand(-1, count, count, -1)
        relation_probability = relation_logits.softmax(-1)
        relation_features = self.relation_embedding(relation_probability)
        messages = self.edge(
            torch.cat(
                (sender, receiver, relation_features, act_edge), dim=-1
            )
        )
        edge_weight = 1.0 - relation_probability[..., 0]
        diagonal_mask = 1.0 - torch.eye(
            count, device=nodes.device, dtype=nodes.dtype
        )
        edge_weight = edge_weight * diagonal_mask[None]
        aggregate = (messages * edge_weight[..., None]).sum(
            dim=1
        ) / edge_weight.sum(dim=1).clamp_min(1e-6)[..., None]
        act_node = action[:, None, :].expand(-1, count, -1)
        update = self.node(torch.cat((nodes, aggregate, act_node), dim=-1))
        return self.norm(nodes + update)


class GraphDynamics(nn.Module):
    """Action-conditioned recursive node and relation increment predictor."""

    def __init__(
        self,
        node_dim: int,
        action_dim: int,
        relation_count: int,
        steps: int = 3,
        max_delta: float = 0.25,
        max_relation_delta: float = 0.5,
    ):
        super().__init__()
        if max_delta <= 0 or max_relation_delta <= 0:
            raise ValueError("Graph delta bounds must be positive")
        self.max_delta = float(max_delta)
        self.max_relation_delta = float(max_relation_delta)
        self.blocks = nn.ModuleList(
            GraphBlock(node_dim, action_dim, relation_count)
            for _ in range(steps)
        )
        self.delta = nn.Linear(node_dim, node_dim)
        self.relation_delta = nn.Sequential(
            nn.Linear(2 * node_dim + action_dim, 2 * node_dim),
            nn.SiLU(),
            nn.Linear(2 * node_dim, relation_count),
        )
        # Start as identity dynamics. Supervised and consistency losses then
        # grow bounded increments without an unstable random initial jump.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.relation_delta[-1].weight)
        nn.init.zeros_(self.relation_delta[-1].bias)

    def forward(
        self, nodes: Tensor, action: Tensor, relation_logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = nodes
        for block in self.blocks:
            hidden = block(hidden, action, relation_logits)
        delta = self.max_delta * torch.tanh(self.delta(hidden))
        next_nodes = nodes + delta
        count = nodes.shape[1]
        sender = hidden[:, :, None, :].expand(-1, -1, count, -1)
        receiver = hidden[:, None, :, :].expand(-1, count, -1, -1)
        edge_action = action[:, None, None, :].expand(
            -1, count, count, -1
        )
        relation_delta = self.max_relation_delta * torch.tanh(
            self.relation_delta(
                torch.cat((sender, receiver, edge_action), dim=-1)
            )
        )
        next_relation_logits = relation_logits + relation_delta
        return next_nodes, delta, next_relation_logits


class HistoryAwareGraphDynamics(nn.Module):
    """GraphDynamics variant conditioned on a short window of finite-
    difference velocities instead of only the current instantaneous state.

    A single observed frame cannot represent motion at all. Standalone
    pretraining (script/jointgv/tmp/pretrain_graph_dynamics.py) found this, with
    K-step self-fed training and a direct embedding-space loss, to be the
    difference between runaway and stable long-horizon drift -- consistent
    with graph-network physics simulators, which typically condition on
    several past velocities (Sanchez-Gonzalez et al., ICML 2020).

    A separate class, not a change to GraphDynamics: existing callers (the
    historical DreamerV3 path, direct GraphDynamics users) are unaffected
    unless they opt in.
    """

    def __init__(
        self,
        node_dim: int,
        action_dim: int,
        relation_count: int,
        history: int,
        steps: int = 3,
        max_delta: float = 0.25,
        max_relation_delta: float = 0.5,
    ):
        super().__init__()
        if history < 1:
            raise ValueError("history must be at least 1 (current frame only)")
        if max_delta <= 0 or max_relation_delta <= 0:
            raise ValueError("Graph delta bounds must be positive")
        self.node_dim = node_dim
        self.history = history
        self.history_encoder = nn.Sequential(
            nn.Linear(node_dim * history, node_dim),
            nn.SiLU(),
        )
        self.max_delta = float(max_delta)
        self.max_relation_delta = float(max_relation_delta)
        self.blocks = nn.ModuleList(
            GraphBlock(node_dim, action_dim, relation_count)
            for _ in range(steps)
        )
        self.delta = nn.Linear(node_dim, node_dim)
        self.relation_delta = nn.Sequential(
            nn.Linear(2 * node_dim + action_dim, 2 * node_dim),
            nn.SiLU(),
            nn.Linear(2 * node_dim, relation_count),
        )
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.relation_delta[-1].weight)
        nn.init.zeros_(self.relation_delta[-1].bias)

    def encode_window(self, window: Tensor) -> Tensor:
        """window: [B, history, N, D] (oldest..newest) -> enriched [B, N, D]."""
        batch, history, count, _ = window.shape
        current = window[:, -1]
        if history == 1:
            return self.history_encoder(current)
        velocities = window[:, 1:] - window[:, :-1]
        velocities = velocities.permute(0, 2, 1, 3).reshape(batch, count, -1)
        return self.history_encoder(torch.cat((current, velocities), dim=-1))

    def forward(
        self, window: Tensor, action: Tensor, relation_logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """window [B, history, N, D] -> next_nodes, delta, next_relation_logits."""
        nodes = window[:, -1]
        hidden = self.encode_window(window)
        for block in self.blocks:
            hidden = block(hidden, action, relation_logits)
        delta = self.max_delta * torch.tanh(self.delta(hidden))
        next_nodes = nodes + delta
        count = nodes.shape[1]
        sender = hidden[:, :, None, :].expand(-1, -1, count, -1)
        receiver = hidden[:, None, :, :].expand(-1, count, -1, -1)
        edge_action = action[:, None, None, :].expand(-1, count, count, -1)
        relation_delta = self.max_relation_delta * torch.tanh(
            self.relation_delta(
                torch.cat((sender, receiver, edge_action), dim=-1)
            )
        )
        next_relation_logits = relation_logits + relation_delta
        return next_nodes, delta, next_relation_logits
