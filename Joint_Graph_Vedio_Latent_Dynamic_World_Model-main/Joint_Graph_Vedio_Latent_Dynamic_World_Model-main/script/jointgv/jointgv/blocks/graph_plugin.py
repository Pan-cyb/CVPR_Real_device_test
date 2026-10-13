from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn

from ..ot import sinkhorn_divergence_components, transport_values
from .explicit_graph_observer import ExplicitGraphObserver
from .graph_builder import GraphObservation, LatentGraphBuilder
from .graph_dynamics import GraphDynamics, HistoryAwareGraphDynamics


@dataclass
class GraphPluginOutput:
    graph: Tensor
    graph_next: Tensor
    delta_g: Tensor
    relation_logits: Tensor
    next_relation_logits: Tensor
    context: Tensor
    ot_loss: Tensor
    ot_variance_loss: Tensor
    ot_cross: Tensor
    ot_self_z: Tensor
    ot_self_g: Tensor
    ot_z_proj_rms: Tensor
    ot_g_proj_rms: Tensor
    ot_z_dispersion: Tensor
    ot_g_dispersion: Tensor
    ot_z_pair_cos: Tensor
    ot_g_pair_cos: Tensor
    delta_z_rms: Tensor
    delta_g_rms: Tensor
    delta_z_dispersion: Tensor
    delta_g_dispersion: Tensor
    delta_z_pair_cos: Tensor
    delta_g_pair_cos: Tensor
    graph_context_rms: Tensor
    alignment_mode: str = "sinkhorn"
    alignment_attention_entropy: Tensor | None = None


class GraphDynamicsPlugin(nn.Module):
    """Backbone-agnostic graph dynamics and Δz↔ΔG alignment block.

    The plugin sees visual token increments and actions, recursively predicts
    the graph increment, aligns the two increments with Sinkhorn transport, and
    exposes a correction function. It never reads an LSTM/RSSM hidden state.
    """

    # Per-node pose payload width of the global-latent correction.
    GLOBAL_POSE_DIM = 32

    def __init__(
        self,
        token_dim: int,
        token_count: int,
        action_dim: int,
        node_dim: int = 64,
        graph_nodes: int = 8,
        message_steps: int = 3,
        relation_count: int = 4,
        teacher_state_dim: int = 13,
        ot_dim: int = 32,
        epsilon: float = 0.1,
        sinkhorn_iters: int = 20,
        ot_centering: bool = True,
        max_graph_delta: float = 0.25,
        max_correction: float = 0.25,
        correction_gate_init: float = 0.11920292202211755,
        graph_observer: str = "proxy",
        graph_backbone: str = "reltr_resnet50",
        graph_backbone_pretrained: bool = True,
        freeze_graph_backbone: bool = True,
        reltr_checkpoint: str = "Models/reltr/checkpoint0149.pth",
        reltr_image_size: int = 224,
        reltr_batch_size: int = 8,
        graph_role_names: tuple[str, ...] | list[str] | None = None,
        alignment_dim: int | None = None,
        alignment_mode: str = "sinkhorn",
        graph_history: int = 1,
        pose_correction: bool = False,
        pose_sigma: float = 0.15,
        correction_dim: int | None = None,
    ):
        """``correction_dim`` selects the latent the correction acts on.

        None: a set of ``token_count`` visual tokens (DINO-WM, ``correct``);
        with pose correction the pose is delivered spatially to patches and
        the gate is per token. An int: one global backbone vector (DreamerV3
        RSSM, ``alignment_mode='global'``, ``correct_global``); the
        correction is a ``correction_dim`` vector (the RSSM deter), the
        per-node poses are concatenated in role order and the gate is per
        dimension. Observer, graph dynamics, history, position head and the
        K-step unroll are shared by both.
        """
        super().__init__()
        if graph_history < 1:
            raise ValueError("graph_history must be at least 1")
        if correction_dim is not None and alignment_mode != "global":
            raise ValueError("correction_dim is for alignment_mode='global'")
        if pose_sigma <= 0:
            raise ValueError("pose_sigma must be positive")
        if max_correction <= 0:
            raise ValueError("max_correction must be positive")
        if not 0.0 < correction_gate_init < 1.0:
            raise ValueError("correction_gate_init must be in (0, 1)")
        self.token_dim = token_dim
        self.token_count = token_count
        self.action_dim = action_dim
        self.node_dim = node_dim
        self.graph_nodes = graph_nodes
        self.relation_count = relation_count
        self.teacher_state_dim = teacher_state_dim
        if alignment_mode not in {"sinkhorn", "global"}:
            raise ValueError("alignment_mode must be 'sinkhorn' or 'global'")
        self.alignment_mode = alignment_mode
        self.alignment_dim = int(alignment_dim or token_dim)
        self.graph_observer = graph_observer
        if graph_observer == "proxy":
            self.builder = LatentGraphBuilder(
                token_dim,
                node_dim,
                graph_nodes,
                token_count,
                relation_count,
                teacher_state_dim,
            )
        elif graph_observer == "explicit":
            self.builder = ExplicitGraphObserver(
                node_dim=node_dim,
                nodes=graph_nodes,
                relation_count=relation_count,
                teacher_state_dim=teacher_state_dim,
                backbone=graph_backbone,
                pretrained_backbone=graph_backbone_pretrained,
                freeze_backbone=freeze_graph_backbone,
                reltr_checkpoint=reltr_checkpoint,
                reltr_image_size=reltr_image_size,
                reltr_batch_size=reltr_batch_size,
                role_names=graph_role_names,
            )
        else:
            raise ValueError(f"Unknown graph observer: {graph_observer}")
        self.graph_history = int(graph_history)
        if self.graph_history == 1:
            # Exactly the original single-frame module: unchanged behavior
            # and unchanged state_dict keys, so every existing checkpoint
            # (and the historical DreamerV3 path, which never passes
            # graph_history) loads and runs identically.
            self.dynamics = GraphDynamics(
                node_dim,
                action_dim,
                relation_count,
                message_steps,
                max_delta=max_graph_delta,
            )
        else:
            self.dynamics = HistoryAwareGraphDynamics(
                node_dim,
                action_dim,
                relation_count,
                self.graph_history,
                steps=message_steps,
                max_delta=max_graph_delta,
            )
        # Bias-free projections prevent a shared affine offset from becoming
        # the direction of every token/node after unit normalization.
        self.z_ot = nn.Linear(self.alignment_dim, ot_dim, bias=False)
        self.g_ot = nn.Linear(node_dim, ot_dim, bias=False)
        self.g_value = nn.Linear(node_dim, token_dim)
        self.correction = nn.Sequential(
            nn.Linear(2 * token_dim, token_dim),
            nn.Tanh(),
            nn.Linear(token_dim, token_dim),
        )
        # Graph conditioning starts as an exact no-op for stable warm-up.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        correction_gate_logit = math.log(
            correction_gate_init / (1.0 - correction_gate_init)
        )
        self.correction_gate_logit = nn.Parameter(
            torch.tensor(correction_gate_logit)
        )
        self.max_correction = float(max_correction)
        self.conditioning_strength = 1.0
        self.pose_correction = bool(pose_correction)
        self.pose_sigma = float(pose_sigma)
        self.correction_dim = None if correction_dim is None else int(correction_dim)
        # Mean gate of the latest pose-mode or global correction (diagnostic).
        self.last_correction_gate: Tensor | None = None
        if self.pose_correction and self.correction_dim is None:
            side = int(round(math.sqrt(token_count)))
            if side * side != token_count:
                raise ValueError(
                    "pose correction needs a square patch-token grid"
                )
            # Node 2D image position in [-1, 1] (x = column, y = row), the
            # graph_node_pixel convention; supervised by the pixel labels.
            self.position_head = nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.SiLU(),
                nn.Linear(node_dim, 2),
            )
            # Per-node pose payload delivered to nearby patches.
            self.pose_value = nn.Linear(node_dim, token_dim)
            # Input: [delta_z, OT motion context, pose context, base tokens].
            self.pose_correction_mlp = nn.Sequential(
                nn.Linear(4 * token_dim, 2 * token_dim),
                nn.Tanh(),
                nn.Linear(2 * token_dim, token_dim),
            )
            nn.init.zeros_(self.pose_correction_mlp[-1].weight)
            nn.init.zeros_(self.pose_correction_mlp[-1].bias)
            # Per-token gate; starts at correction_gate_init on every token.
            self.pose_gate = nn.Sequential(
                nn.Linear(4 * token_dim, token_dim),
                nn.Tanh(),
                nn.Linear(token_dim, 1),
            )
            nn.init.zeros_(self.pose_gate[-1].weight)
            nn.init.constant_(self.pose_gate[-1].bias, correction_gate_logit)
            centers = (2.0 * torch.arange(side, dtype=torch.float32) + 1.0) / side - 1.0
            rows, columns = torch.meshgrid(centers, centers, indexing="ij")
            # Row-major like the DINO patch tokens: token i = (i // side, i % side).
            self.register_buffer(
                "patch_centers",
                torch.stack((columns.reshape(-1), rows.reshape(-1)), dim=-1),
                persistent=False,
            )
        if self.correction_dim is not None:
            if self.pose_correction:
                # Same position head as the token path; the pose payload is
                # read per node and concatenated in role order.
                self.position_head = nn.Sequential(
                    nn.Linear(node_dim, node_dim),
                    nn.SiLU(),
                    nn.Linear(node_dim, 2),
                )
                self.pose_value = nn.Linear(node_dim, self.GLOBAL_POSE_DIM)
            # Input: [Δs, pooled ΔG context, node poses, base prediction].
            features = 2 * self.alignment_dim + node_dim
            if self.pose_correction:
                features += graph_nodes * (self.GLOBAL_POSE_DIM + 2)
            self.global_correction_norm = nn.LayerNorm(features)
            self.global_correction_mlp = nn.Sequential(
                nn.Linear(features, self.correction_dim),
                nn.Tanh(),
                nn.Linear(self.correction_dim, self.correction_dim),
            )
            nn.init.zeros_(self.global_correction_mlp[-1].weight)
            nn.init.zeros_(self.global_correction_mlp[-1].bias)
            # Per-dimension gate; starts at correction_gate_init everywhere.
            self.global_correction_gate = nn.Sequential(
                nn.Linear(features, self.correction_dim),
                nn.Tanh(),
                nn.Linear(self.correction_dim, self.correction_dim),
            )
            nn.init.zeros_(self.global_correction_gate[-1].weight)
            nn.init.constant_(
                self.global_correction_gate[-1].bias, correction_gate_logit
            )
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.ot_centering = bool(ot_centering)
        self.ot_variance_target = 1.0 / math.sqrt(float(ot_dim))

    def set_conditioning_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("conditioning strength must be in [0, 1]")
        self.conditioning_strength = float(strength)

    def _active_graph_indices(self, device: torch.device) -> Tensor:
        """Return semantic nodes while excluding fixed padding slots.

        Explicit observers keep a fixed tensor width for batching, but roles
        named ``padding`` are storage only: they must not receive actions,
        exchange messages, or enter the empirical OT distribution. Proxy
        graphs have no semantic role list and therefore keep every node.
        """
        role_names = getattr(self.builder, "role_names", None)
        if role_names is None:
            active = tuple(range(self.graph_nodes))
        else:
            active = tuple(
                index
                for index, name in enumerate(role_names[: self.graph_nodes])
                if name != "padding"
            )
        if not active:
            raise ValueError("Graph observer has no non-padding nodes")
        return torch.tensor(active, device=device, dtype=torch.long)

    @property
    def active_graph_node_count(self) -> int:
        role_names = getattr(self.builder, "role_names", None)
        if role_names is None:
            return self.graph_nodes
        return sum(
            name != "padding" for name in role_names[: self.graph_nodes]
        )

    def infer_sequence(
        self,
        tokens: Tensor,
        images: Tensor | None = None,
        firsts: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if self.builder.requires_images:
            if images is None:
                raise ValueError("Explicit Graph Observer requires RGB images")
            return self.builder.forward_sequence(
                images, sample=False, firsts=firsts
            )
        return self.builder.forward_sequence(
            tokens, sample=False, firsts=firsts
        )

    def observe(
        self,
        tokens: Tensor,
        images: Tensor | None = None,
        previous_slots: Tensor | None = None,
        sample: bool = False,
        reset_mask: Tensor | None = None,
    ) -> GraphObservation:
        """Infer one graph while hiding observer-specific input routing."""
        if self.builder.requires_images:
            if images is None:
                raise ValueError("Explicit Graph Observer requires an RGB frame")
            return self.builder(
                images,
                previous_slots,
                sample=sample,
                reset_mask=reset_mask,
            )
        return self.builder(
            tokens,
            previous_slots,
            sample=sample,
            reset_mask=reset_mask,
        )

    @staticmethod
    def _safe_unit_normalize(value: Tensor) -> Tensor:
        """Normalize OT coordinates without amplifying near-zero gradients."""
        denominator = torch.linalg.vector_norm(
            value, ord=2, dim=-1, keepdim=True
        ).clamp_min(1e-4)
        return value / denominator

    @staticmethod
    def _rms(value: Tensor) -> Tensor:
        return value.square().mean().sqrt()

    @staticmethod
    def _center_points(value: Tensor) -> Tensor:
        """Remove each empirical set's shared projection direction."""
        return value - value.mean(dim=-2, keepdim=True)

    @staticmethod
    def _set_dispersion(unit_value: Tensor) -> Tensor:
        """Mean squared distance from the point-cloud centroid."""
        centroid = unit_value.mean(dim=-2, keepdim=True)
        return (unit_value - centroid).square().sum(dim=-1).mean()

    @staticmethod
    def _mean_pairwise_cosine(unit_value: Tensor) -> Tensor:
        """Mean off-diagonal cosine; values near one indicate collapse."""
        count = unit_value.shape[-2]
        if count < 2:
            return unit_value.new_ones(())
        gram = torch.bmm(unit_value, unit_value.transpose(1, 2))
        diagonal = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        off_diagonal = gram.sum(dim=(-2, -1)) - diagonal
        return (off_diagonal / float(count * (count - 1))).mean()

    def _variance_floor_loss(self, unit_value: Tensor) -> Tensor:
        """VICReg-style per-coordinate standard-deviation floor."""
        flattened = unit_value.reshape(-1, unit_value.shape[-1])
        standard_deviation = torch.sqrt(
            flattened.var(dim=0, unbiased=False) + 1e-4
        )
        return torch.relu(
            standard_deviation.new_tensor(self.ot_variance_target)
            - standard_deviation
        ).mean()

    def history_windows(
        self, graph: Tensor, graph_window: Tensor | None = None
    ) -> Tensor:
        """Per-step history windows [B*T, history, N, D], oldest..newest.

        ``graph_window`` ([B, T, history, N, D]) is supplied by callers that
        maintain their own rolling context (rollouts). Otherwise windows are
        built from the observed sequence ``graph`` ([B, T, N, D]): step t
        uses frames t-history+1..t, with indices before the start clamped to
        frame 0 (zero velocity) since no earlier frame exists in the batch.
        """
        batch, time = graph.shape[:2]
        history = self.graph_history
        if graph_window is not None:
            return graph_window.reshape(
                batch * time, history, self.graph_nodes, self.node_dim
            )
        offsets = torch.arange(-(history - 1), 1, device=graph.device)
        index = (
            torch.arange(time, device=graph.device)[:, None] + offsets[None]
        ).clamp_min(0)
        return graph[:, index].reshape(
            batch * time, history, self.graph_nodes, self.node_dim
        )

    def _predict_graph_step(
        self,
        graph: Tensor,
        flat_graph: Tensor,
        flat_action: Tensor,
        active_index: Tensor,
        active_relation: Tensor,
        graph_window: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One dynamics call for every (batch, time) step in parallel.

        Returns (active_graph, next, delta, next_relation) restricted to the
        active (non-padding) nodes; the first element is the current-frame
        active graph the increment is added to.
        """
        active_graph = flat_graph.index_select(1, active_index)
        if self.graph_history == 1:
            dynamics_input = active_graph
        else:
            dynamics_input = self.history_windows(graph, graph_window).index_select(
                2, active_index
            )
        active_next, active_delta, active_next_relation = self.dynamics(
            dynamics_input, flat_action, active_relation
        )
        return active_graph, active_next, active_delta, active_next_relation

    def predict_next_graph(
        self, window: Tensor, action: Tensor, relation_logits: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Graph-only dynamics step, no Sinkhorn/OT/correction.

        window [B, history, N, D] (oldest..newest), action [B, A],
        relation_logits [B, N, N, R] -> (graph_next [B, N, D],
        next_relation_logits [B, N, N, R]). Padding nodes keep their state
        and relations, exactly as in forward(). Used for K-step self-fed
        training, where only the graph trajectory is needed.
        """
        if window.shape[1] != self.graph_history:
            raise ValueError(
                f"Expected a {self.graph_history}-frame window, got "
                f"{window.shape[1]}"
            )
        current = window[:, -1]
        active_index = self._active_graph_indices(current.device)
        active_relation = relation_logits.index_select(
            1, active_index
        ).index_select(2, active_index)
        _, _, active_delta, active_next_relation = self._predict_graph_step(
            current[:, None],
            current,
            action,
            active_index,
            active_relation,
            window[:, None],
        )
        selection = torch.nn.functional.one_hot(
            active_index, num_classes=self.graph_nodes
        ).transpose(0, 1).to(dtype=current.dtype)
        delta = torch.einsum("na,bad->bnd", selection, active_delta)
        relation_delta = torch.einsum(
            "na,bacr,mc->bnmr",
            selection,
            active_next_relation - active_relation,
            selection,
        )
        return current + delta, relation_logits + relation_delta

    def graph_multistep(
        self,
        nodes: Tensor,
        relation_logits: Tensor,
        actions: Tensor,
        horizon: int,
    ) -> dict[str, Any]:
        """Unroll graph dynamics K steps on its own predictions.

        nodes [B, T+1, N, D] and relation_logits [B, T+1, N, N, R] are the
        observed sequence, actions [B, T, A]. Starts from the first step with
        a full real history window, then feeds each predicted graph (and
        relation logits) back in, exactly as a rollout does. Targets are the
        observed future embeddings; the state head decodes predictions for
        supervision against MuJoCo state. Shared by every backbone adapter.
        """
        history = self.graph_history
        time = actions.shape[1]
        start = min(history - 1, time - 1)
        horizon = min(horizon, time - start)
        offsets = torch.arange(
            start - history + 1, start + 1, device=nodes.device
        ).clamp_min(0)
        window = nodes[:, offsets]
        relation = relation_logits[:, start]
        predictions = []
        for step in range(horizon):
            next_graph, relation = self.predict_next_graph(
                window, actions[:, start + step], relation
            )
            predictions.append(next_graph)
            window = torch.cat((window[:, 1:], next_graph[:, None]), dim=1)
        predicted = torch.stack(predictions, dim=1)
        result: dict[str, Any] = {
            "graph_multistep_prediction": predicted,
            "graph_multistep_state_prediction": self.builder.teacher_state(
                predicted
            ),
            "graph_multistep_target": nodes[
                :, start + 1 : start + 1 + horizon
            ].detach(),
            "graph_multistep_start": start,
        }
        if self.pose_correction:
            result["graph_multistep_position_prediction"] = self.node_position(
                predicted
            )
        return result

    def _zero_output(
        self, base_delta: Tensor, graph: Tensor, relation_logits: Tensor
    ) -> GraphPluginOutput:
        """Cheap placeholder for a caller that will zero-weight every term.

        Only reached with ``compute_dynamics=False``, i.e. graph conditioning
        disabled: every graph/OT loss term is weighted by zero regardless, so
        skipping the GraphDynamics recurrence and Sinkhorn alignment here
        changes no gradient and no trained weight -- only the otherwise
        wasted forward compute for a graph-off run. ``graph_next``/
        ``next_relation_logits`` echo the observed graph/relations unchanged
        (a zero predicted delta) rather than all-zero tensors, so a
        placeholder still decodes to a valid graph state if inspected.
        """
        zero = base_delta.new_zeros(())
        return GraphPluginOutput(
            graph=graph,
            graph_next=graph,
            delta_g=torch.zeros_like(graph),
            relation_logits=relation_logits,
            next_relation_logits=relation_logits,
            context=torch.zeros_like(base_delta),
            ot_loss=zero,
            ot_variance_loss=zero,
            ot_cross=zero,
            ot_self_z=zero,
            ot_self_g=zero,
            ot_z_proj_rms=zero,
            ot_g_proj_rms=zero,
            ot_z_dispersion=zero,
            ot_g_dispersion=zero,
            ot_z_pair_cos=zero,
            ot_g_pair_cos=zero,
            delta_z_rms=zero,
            delta_g_rms=zero,
            delta_z_dispersion=zero,
            delta_g_dispersion=zero,
            delta_z_pair_cos=zero,
            delta_g_pair_cos=zero,
            graph_context_rms=zero,
            alignment_mode="sinkhorn",
            alignment_attention_entropy=None,
        )

    def forward(
        self,
        base_delta: Tensor,
        actions: Tensor,
        graph: Tensor,
        relation_logits: Tensor,
        compute_dynamics: bool = True,
        graph_window: Tensor | None = None,
        context_gradient_scale: float = 1.0,
    ) -> GraphPluginOutput:
        """Process sequences [B,T,...] without depending on a backbone type.

        ``context_gradient_scale`` scales only the gradient that reaches
        ``base_delta`` through the transported context (the correction's
        input), not through the OT loss; 1.0 (default) leaves both intact.

        ``graph_window`` ([B,T,graph_history,N,D]) supplies explicit rolling
        history for rollouts when ``graph_history > 1``; when omitted, windows
        are built from ``graph`` itself (the teacher-forced training path).

        ``compute_dynamics=False`` skips the GraphDynamics recurrence and
        Sinkhorn alignment entirely and returns ``_zero_output`` instead; it
        defaults to ``True`` so every existing caller (any direct use of
        this method) is unaffected unless it opts in.
        """
        if self.alignment_mode != "sinkhorn":
            raise RuntimeError(
                "Token-set forward requires alignment_mode='sinkhorn'; "
                "use forward_global for a genuine global latent state"
            )
        if not compute_dynamics:
            return self._zero_output(base_delta, graph, relation_logits)
        batch, time, token_count, token_dim = base_delta.shape
        flat_graph = graph.reshape(batch * time, self.graph_nodes, self.node_dim)
        flat_relation = relation_logits.reshape(
            batch * time,
            self.graph_nodes,
            self.graph_nodes,
            self.relation_count,
        )
        flat_action = actions.reshape(batch * time, self.action_dim)
        active_index = self._active_graph_indices(flat_graph.device)
        active_relation = flat_relation.index_select(
            1, active_index
        ).index_select(2, active_index)
        (
            active_graph,
            active_graph_next,
            active_delta_g,
            active_next_relation,
        ) = self._predict_graph_step(
            graph,
            flat_graph,
            flat_action,
            active_index,
            active_relation,
            graph_window,
        )
        # Scatter active increments back into the fixed-width graph. Padding
        # nodes retain their observed state, have exactly zero delta, and all
        # relations touching padding remain unchanged.
        selection = torch.nn.functional.one_hot(
            active_index, num_classes=self.graph_nodes
        ).transpose(0, 1).to(dtype=flat_graph.dtype)
        delta_g = torch.einsum(
            "na,bad->bnd", selection, active_delta_g
        )
        active_relation_delta = active_next_relation - active_relation
        relation_delta = torch.einsum(
            "na,bacr,mc->bnmr",
            selection,
            active_relation_delta,
            selection,
        )
        graph_next = flat_graph + delta_g
        next_relation = flat_relation + relation_delta
        # Unit-normalized OT coordinates make squared transport costs
        # invariant to projection scale and prevent runaway Sinkhorn costs.
        flat_base_delta = base_delta.reshape(
            batch * time, token_count, token_dim
        )
        z_projected = self.z_ot(flat_base_delta)
        # Only real semantic nodes define the graph-change distribution.
        # Including fixed padding here previously produced misleadingly
        # common action-conditioned deltas and contaminated Sinkhorn metrics.
        g_projected = self.g_ot(active_delta_g)
        # Centering is optional so experiments can retain global/common-mode
        # motion. Unit normalization remains mandatory to keep Sinkhorn costs
        # independent of arbitrary projector scale.
        if self.ot_centering:
            z_ot_input = self._center_points(z_projected)
            g_ot_input = self._center_points(g_projected)
        else:
            z_ot_input = z_projected
            g_ot_input = g_projected
        z_change = self._safe_unit_normalize(z_ot_input)
        g_change = self._safe_unit_normalize(g_ot_input)
        z_raw_unit = self._safe_unit_normalize(flat_base_delta)
        g_raw_unit = self._safe_unit_normalize(active_delta_g)
        graph_values = torch.tanh(self.g_value(active_delta_g))
        context_z = z_change
        if context_gradient_scale != 1.0:
            context_z = z_change.detach() + context_gradient_scale * (
                z_change - z_change.detach()
            )
        context = transport_values(
            context_z,
            g_change,
            graph_values,
            self.epsilon,
            self.sinkhorn_iters,
        ).reshape(batch, time, token_count, token_dim)
        ot_loss, ot_cross, ot_self_z, ot_self_g = (
            sinkhorn_divergence_components(
                z_change, g_change, self.epsilon, self.sinkhorn_iters
            )
        )
        ot_variance_loss = 0.5 * (
            self._variance_floor_loss(z_change)
            + self._variance_floor_loss(g_change)
        )
        return GraphPluginOutput(
            graph=graph,
            graph_next=graph_next.reshape(
                batch, time, self.graph_nodes, self.node_dim
            ),
            delta_g=delta_g.reshape(
                batch, time, self.graph_nodes, self.node_dim
            ),
            relation_logits=relation_logits,
            next_relation_logits=next_relation.reshape(
                batch,
                time,
                self.graph_nodes,
                self.graph_nodes,
                self.relation_count,
            ),
            context=context,
            ot_loss=ot_loss,
            ot_variance_loss=ot_variance_loss,
            ot_cross=ot_cross.detach(),
            ot_self_z=ot_self_z.detach(),
            ot_self_g=ot_self_g.detach(),
            ot_z_proj_rms=self._rms(z_projected).detach(),
            ot_g_proj_rms=self._rms(g_projected).detach(),
            ot_z_dispersion=self._set_dispersion(z_change).detach(),
            ot_g_dispersion=self._set_dispersion(g_change).detach(),
            ot_z_pair_cos=self._mean_pairwise_cosine(z_change).detach(),
            ot_g_pair_cos=self._mean_pairwise_cosine(g_change).detach(),
            delta_z_rms=self._rms(flat_base_delta).detach(),
            delta_g_rms=self._rms(active_delta_g).detach(),
            delta_z_dispersion=self._set_dispersion(z_raw_unit).detach(),
            delta_g_dispersion=self._set_dispersion(g_raw_unit).detach(),
            delta_z_pair_cos=self._mean_pairwise_cosine(z_raw_unit).detach(),
            delta_g_pair_cos=self._mean_pairwise_cosine(g_raw_unit).detach(),
            graph_context_rms=self._rms(context).detach(),
            alignment_mode="sinkhorn",
            alignment_attention_entropy=None,
        )

    def forward_global(
        self,
        state_delta: Tensor,
        actions: Tensor,
        graph: Tensor,
        relation_logits: Tensor,
        graph_window: Tensor | None = None,
    ) -> GraphPluginOutput:
        """Align one global state change with an attention-pooled graph change.

        ``state_delta`` is a genuine global latent change ``[B,T,D]``.  A
        one-point empirical distribution has no meaningful correspondence
        structure, so this path deliberately does *not* run Sinkhorn.  The
        projected global change queries the active graph-node changes; the
        normalized query and pooled key use half squared L2 distance, which
        equals cosine distance after unit normalization.
        """
        if self.alignment_mode != "global":
            raise RuntimeError(
                "Global forward requires alignment_mode='global'"
            )
        batch, time, state_dim = state_delta.shape
        if state_dim != self.alignment_dim:
            raise ValueError(
                f"Expected global delta dim {self.alignment_dim}, got "
                f"{state_dim}"
            )
        flat_graph = graph.reshape(
            batch * time, self.graph_nodes, self.node_dim
        )
        flat_relation = relation_logits.reshape(
            batch * time,
            self.graph_nodes,
            self.graph_nodes,
            self.relation_count,
        )
        flat_action = actions.reshape(batch * time, self.action_dim)
        active_index = self._active_graph_indices(flat_graph.device)
        active_relation = flat_relation.index_select(
            1, active_index
        ).index_select(2, active_index)
        (
            active_graph,
            active_graph_next,
            active_delta_g,
            active_next_relation,
        ) = self._predict_graph_step(
            graph,
            flat_graph,
            flat_action,
            active_index,
            active_relation,
            graph_window,
        )

        selection = torch.nn.functional.one_hot(
            active_index, num_classes=self.graph_nodes
        ).transpose(0, 1).to(dtype=flat_graph.dtype)
        delta_g = torch.einsum("na,bad->bnd", selection, active_delta_g)
        active_relation_delta = active_next_relation - active_relation
        relation_delta = torch.einsum(
            "na,bacr,mc->bnmr",
            selection,
            active_relation_delta,
            selection,
        )
        graph_next = flat_graph + delta_g
        next_relation = flat_relation + relation_delta

        flat_state_delta = state_delta.reshape(batch * time, state_dim)
        z_projected = self.z_ot(flat_state_delta)
        g_projected = self.g_ot(active_delta_g)
        z_change = self._safe_unit_normalize(z_projected)
        g_change = self._safe_unit_normalize(g_projected)
        # Query/key vectors are already unit normalized, so the usual
        # 1/sqrt(d) scaling would make every score nearly zero and collapse
        # pooling to a mean.  Reuse epsilon as an explicit cosine-attention
        # temperature (0.1 by default).
        logits = torch.einsum("bd,bnd->bn", z_change, g_change) / max(
            float(self.epsilon), 1e-4
        )
        weights = torch.softmax(logits, dim=-1)
        pooled_projected = torch.einsum("bn,bnd->bd", weights, g_projected)
        pooled_change = self._safe_unit_normalize(pooled_projected)
        # 0.5 ||u-v||^2 = 1-cos(u,v) for unit vectors.  Keeping the
        # half-squared-L2 form makes the degenerate alignment explicit.
        alignment_loss = 0.5 * (
            z_change - pooled_change
        ).square().sum(dim=-1).mean()
        context = torch.einsum(
            "bn,bnd->bd", weights, active_delta_g
        ).reshape(batch, time, self.node_dim)
        if weights.shape[-1] > 1:
            attention_entropy = -(
                weights * weights.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(float(weights.shape[-1]))
        else:
            attention_entropy = weights.new_zeros(weights.shape[0])
        g_raw_unit = self._safe_unit_normalize(active_delta_g)
        zero = alignment_loss.new_zeros(())
        return GraphPluginOutput(
            graph=graph,
            graph_next=graph_next.reshape(
                batch, time, self.graph_nodes, self.node_dim
            ),
            delta_g=delta_g.reshape(
                batch, time, self.graph_nodes, self.node_dim
            ),
            relation_logits=relation_logits,
            next_relation_logits=next_relation.reshape(
                batch,
                time,
                self.graph_nodes,
                self.graph_nodes,
                self.relation_count,
            ),
            context=context,
            # Legacy key retained for the existing loss-weight controller.
            # ``alignment_mode`` prevents this value being misreported as OT.
            ot_loss=alignment_loss,
            ot_variance_loss=self._variance_floor_loss(g_change),
            ot_cross=alignment_loss.detach(),
            ot_self_z=zero.detach(),
            ot_self_g=zero.detach(),
            ot_z_proj_rms=self._rms(z_projected).detach(),
            ot_g_proj_rms=self._rms(g_projected).detach(),
            ot_z_dispersion=zero.detach(),
            ot_g_dispersion=self._set_dispersion(g_change).detach(),
            ot_z_pair_cos=z_change.new_ones(()).detach(),
            ot_g_pair_cos=self._mean_pairwise_cosine(g_change).detach(),
            delta_z_rms=self._rms(flat_state_delta).detach(),
            delta_g_rms=self._rms(active_delta_g).detach(),
            delta_z_dispersion=zero.detach(),
            delta_g_dispersion=self._set_dispersion(g_raw_unit).detach(),
            delta_z_pair_cos=z_change.new_ones(()).detach(),
            delta_g_pair_cos=self._mean_pairwise_cosine(g_raw_unit).detach(),
            graph_context_rms=self._rms(context).detach(),
            alignment_mode="global",
            alignment_attention_entropy=attention_entropy.mean().detach(),
        )

    def correction_parameters(self) -> list[nn.Parameter]:
        """Parameters of the correction itself (not graph, observer, or OT)."""
        modules: list[nn.Module] = [self.correction]
        if self.correction_dim is not None:
            if self.pose_correction:
                modules.append(self.pose_value)
            modules += [
                self.global_correction_norm,
                self.global_correction_mlp,
                self.global_correction_gate,
            ]
        elif self.pose_correction:
            modules += [
                self.pose_value,
                self.pose_correction_mlp,
                self.pose_gate,
            ]
        parameters = [self.correction_gate_logit]
        for module in modules:
            parameters.extend(module.parameters())
        return parameters

    def node_position(self, nodes: Tensor) -> Tensor:
        """Decode node embeddings [..., N, D] to 2D image positions in [-1, 1]."""
        return torch.tanh(self.position_head(nodes))

    def pose_context(self, graph_next: Tensor) -> Tensor:
        """Deliver each node's pose to the patches near where the graph puts it.

        graph_next [..., N, D] -> [..., P, token_dim]. Patch i receives
        sum_j exp(-|x_i - pos_j|^2 / 2 sigma^2) * tanh(W_p g_j) over active
        nodes j, independent of what the visual prediction currently shows
        (unlike the OT context, which routes by visual motion). Unnormalized,
        so patches far from every node receive ~0. The graph is read only:
        no gradient flows from the correction back into graph or observer.
        """
        active_index = self._active_graph_indices(graph_next.device)
        nodes = graph_next.detach().index_select(-2, active_index)
        position = self.node_position(nodes).detach()
        payload = torch.tanh(self.pose_value(nodes))
        distance = (
            self.patch_centers[:, None, :] - position[..., None, :, :]
        ).square().sum(dim=-1)
        weight = torch.exp(-distance / (2.0 * self.pose_sigma**2))
        return weight @ payload

    def pose_vector(self, graph_next: Tensor) -> Tensor:
        """Every node's pose as one vector, for a global backbone latent.

        graph_next [..., N, D] -> [..., N * (GLOBAL_POSE_DIM + 2)]: per node
        tanh(W_p g_j) and its decoded 2D position, in role order; padding
        slots are zero. A global latent has no patches to route poses to,
        so the correction reads all of them. The graph is read only, as in
        ``pose_context``.
        """
        nodes = graph_next.detach()
        position = self.node_position(nodes).detach()
        payload = torch.tanh(self.pose_value(nodes))
        active = torch.zeros(
            self.graph_nodes, device=nodes.device, dtype=nodes.dtype
        )
        active[self._active_graph_indices(nodes.device)] = 1.0
        pose = torch.cat((payload, position), dim=-1) * active[:, None]
        return pose.flatten(-2)

    def correct_global(
        self,
        delta_s: Tensor,
        context: Tensor,
        base: Tensor,
        pose: Tensor | None = None,
        strength: float | None = None,
    ) -> Tensor:
        """Correction of one global backbone latent (``correction_dim`` wide).

        ``delta_s`` is the backbone's predicted global change, ``context``
        the pooled ΔG context of ``forward_global``, ``base`` the backbone's
        own prediction and ``pose`` the ``pose_vector`` (pose correction).
        Same bound as the token path; the gate is per output dimension.
        ``strength`` overrides the conditioning strength.
        """
        if self.correction_dim is None:
            raise RuntimeError("correct_global needs correction_dim")
        if strength is None:
            strength = self.conditioning_strength
        parts = [delta_s, context]
        if self.pose_correction:
            if pose is None:
                raise ValueError("pose correction requires pose")
            parts.append(pose)
        parts.append(base)
        features = self.global_correction_norm(torch.cat(parts, dim=-1))
        raw = self.global_correction_mlp(features)
        bounded = self.max_correction * torch.tanh(raw / self.max_correction)
        gate = torch.sigmoid(self.global_correction_gate(features))
        self.last_correction_gate = gate.detach().mean()
        return strength * gate * bounded

    def correct(
        self,
        delta_z: Tensor,
        context: Tensor,
        pose: Tensor | None = None,
        base_tokens: Tensor | None = None,
        strength: float | None = None,
    ) -> Tensor:
        """Return f_correction(Δz, transported ΔG context).

        Extra mixture axes are supported, which lets the same module correct
        every MDN component as well as a single RSSM token prediction.

        With pose correction, ``pose`` (spatially routed graph pose) and
        ``base_tokens`` (the backbone's own prediction) are also inputs and
        the gate is per token. ``strength`` overrides the conditioning
        strength (the correction-unroll objective trains at full strength).
        """
        if strength is None:
            strength = self.conditioning_strength
        if self.pose_correction:
            if pose is None or base_tokens is None:
                raise ValueError(
                    "pose correction requires pose and base_tokens"
                )
            features = torch.cat((delta_z, context, pose, base_tokens), dim=-1)
            raw = self.pose_correction_mlp(features)
            bounded = self.max_correction * torch.tanh(
                raw / self.max_correction
            )
            gate = torch.sigmoid(self.pose_gate(features))
            self.last_correction_gate = gate.detach().mean()
            return strength * gate * bounded
        while context.ndim < delta_z.ndim:
            context = context.unsqueeze(-2)
        context = context.expand(*delta_z.shape[:-1], context.shape[-1])
        raw = self.correction(torch.cat((delta_z, context), dim=-1))
        bounded = self.max_correction * torch.tanh(
            raw / self.max_correction
        )
        learned_gate = torch.sigmoid(self.correction_gate_logit)
        return strength * learned_gate * bounded
