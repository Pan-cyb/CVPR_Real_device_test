from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .backbones import (
    BackboneOutput,
    LatentBackbone,
    MDNOutput,
    DreamerV3WorldModel,
    resolve_backbone_adapter,
)
from .blocks import GraphDynamicsPlugin, VisualCodec
from .factory import build_backbone, build_visual_codec

# Options of the removed IRIS backbone. Every saved model_config carries
# them (train.py always wrote them), so they are accepted and ignored.
REMOVED_CONFIG_KEYS = frozenset({
    "iris_vocab_size",
    "iris_token_dim",
    "iris_model_dim",
    "iris_layers",
    "iris_heads",
    "iris_max_blocks",
    "iris_token_grid",
    "iris_action_bins",
    "iris_action_conditioning",
    "iris_geometry_bins",
    "iris_static_target_tokens",
})


class JointGraphVideoWorldModel(nn.Module):
    """Visual world model with a selectable transition and graph plugin."""

    checkpoint_version = 14

    def __init__(
        self,
        action_dim: int,
        backbone: str = "david_ha",
        token_dim: int = 64,
        node_dim: int = 64,
        graph_nodes: int = 8,
        message_steps: int = 3,
        relation_count: int = 4,
        teacher_state_dim: int = 13,
        ot_dim: int = 32,
        rnn_dim: int = 256,
        mixtures: int = 5,
        epsilon: float = 0.1,
        sinkhorn_iters: int = 20,
        ot_centering: bool = True,
        max_graph_delta: float = 0.25,
        max_graph_correction: float = 0.25,
        graph_correction_gate_init: float = 0.11920292202211755,
        freeze_visual: bool = False,
        graph_observer: str = "proxy",
        graph_backbone: str = "reltr_resnet50",
        graph_backbone_pretrained: bool = True,
        freeze_graph_backbone: bool = True,
        reltr_checkpoint: str = "Models/reltr/checkpoint0149.pth",
        reltr_image_size: int = 224,
        reltr_batch_size: int = 8,
        graph_role_names: tuple[str, ...] | list[str] | None = None,
        dino_model: str = "dinov2_vits14",
        dino_image_size: int = 112,
        dino_freeze_encoder: bool = True,
        dino_predictor_depth: int = 6,
        dino_predictor_heads: int = 8,
        dino_mlp_ratio: float = 4.0,
        dino_dropout: float = 0.1,
        dino_max_frames: int = 40,
        graph_history: int = 1,
        graph_pose_correction: bool = False,
        graph_pose_sigma: float = 0.15,
        graph_global_correction: bool = False,
        **removed_options: Any,
    ):
        """``graph_global_correction`` (DreamerV3 only) builds the plug-in
        path: the shared plugin's global correction head corrects the RSSM
        prior deter of an otherwise graph-off world model (see
        ``DreamerV3WorldModel``). False keeps the historical joint path.

        ``removed_options`` absorbs the options of the removed IRIS backbone
        (``REMOVED_CONFIG_KEYS``) stored in older checkpoints' configs."""
        super().__init__()
        unknown = set(removed_options) - REMOVED_CONFIG_KEYS
        if unknown:
            raise TypeError(f"Unexpected model options: {sorted(unknown)}")
        adapter = resolve_backbone_adapter(backbone)
        if graph_pose_correction and adapter.transition not in {
            "dino_wm",
            "dreamerv3",
        }:
            raise ValueError(
                "graph_pose_correction requires the dino_wm or dreamerv3 backbone"
            )
        if graph_global_correction and adapter.transition != "dreamerv3":
            raise ValueError("graph_global_correction is for DreamerV3")
        # Training-only switches (not architecture, not in model_config);
        # neither changes what rollout() computes. plugin_correction: the
        # backbone never receives gradient from OT, graph context or the
        # correction, and its own loss is the graph-off loss. Otherwise
        # correction_backbone_weight scales only the gradient the corrected
        # prediction's loss sends into the backbone (1.0 = historical joint).
        self.plugin_correction = False
        self.correction_backbone_weight = 1.0
        self.graph_history = int(graph_history)
        self.action_dim = action_dim
        self.token_dim = token_dim
        self.backbone_name = adapter.name
        self.visual_kind = adapter.visual
        self.dreamerv3_core: DreamerV3WorldModel | None = None
        if adapter.transition == "dreamerv3":
            self.dreamerv3_core = DreamerV3WorldModel(
                action_dim=action_dim,
                visual_kind=adapter.visual,
                token_dim=token_dim,
                node_dim=node_dim,
                graph_nodes=graph_nodes,
                message_steps=message_steps,
                relation_count=relation_count,
                teacher_state_dim=teacher_state_dim,
                ot_dim=ot_dim,
                epsilon=epsilon,
                sinkhorn_iters=sinkhorn_iters,
                ot_centering=ot_centering,
                max_graph_delta=max_graph_delta,
                max_graph_correction=max_graph_correction,
                graph_correction_gate_init=graph_correction_gate_init,
                freeze_visual=freeze_visual,
                graph_observer=graph_observer,
                graph_backbone=graph_backbone,
                graph_backbone_pretrained=graph_backbone_pretrained,
                freeze_graph_backbone=freeze_graph_backbone,
                reltr_checkpoint=reltr_checkpoint,
                reltr_image_size=reltr_image_size,
                reltr_batch_size=reltr_batch_size,
                graph_role_names=graph_role_names,
                graph_history=graph_history,
                graph_pose_correction=graph_pose_correction,
                graph_pose_sigma=graph_pose_sigma,
                global_correction=graph_global_correction,
            )
            self.token_dim = self.dreamerv3_core.token_dim
            self.visual_frozen = self.dreamerv3_core.visual_frozen
            return
        # Keep the historical ``vae`` attribute so David Ha checkpoints retain
        # their original ``vae.*`` state-dict keys.
        self.vae: VisualCodec = build_visual_codec(
            adapter.visual,
            token_dim,
            dino_model=dino_model,
            dino_image_size=dino_image_size,
            dino_freeze_encoder=dino_freeze_encoder,
        )
        self.backbone: LatentBackbone = build_backbone(
            adapter.transition,
            token_count=self.vae.token_count,
            token_dim=token_dim,
            action_dim=action_dim,
            rnn_dim=rnn_dim,
            mixtures=mixtures,
            dino_predictor_depth=dino_predictor_depth,
            dino_predictor_heads=dino_predictor_heads,
            dino_mlp_ratio=dino_mlp_ratio,
            dino_dropout=dino_dropout,
            dino_max_frames=dino_max_frames,
        )
        self.graph_plugin = GraphDynamicsPlugin(
            token_dim=token_dim,
            token_count=self.vae.token_count,
            action_dim=action_dim,
            node_dim=node_dim,
            graph_nodes=graph_nodes,
            message_steps=message_steps,
            relation_count=relation_count,
            teacher_state_dim=teacher_state_dim,
            ot_dim=ot_dim,
            epsilon=epsilon,
            sinkhorn_iters=sinkhorn_iters,
            ot_centering=ot_centering,
            max_graph_delta=max_graph_delta,
            max_correction=max_graph_correction,
            correction_gate_init=graph_correction_gate_init,
            graph_observer=graph_observer,
            graph_backbone=graph_backbone,
            graph_backbone_pretrained=graph_backbone_pretrained,
            freeze_graph_backbone=freeze_graph_backbone,
            reltr_checkpoint=reltr_checkpoint,
            reltr_image_size=reltr_image_size,
            reltr_batch_size=reltr_batch_size,
            graph_role_names=graph_role_names,
            graph_history=graph_history,
            pose_correction=graph_pose_correction,
            pose_sigma=graph_pose_sigma,
        )
        self.visual_frozen = bool(freeze_visual)
        if self.visual_frozen:
            self.vae.requires_grad_(False)
        head_input = token_dim
        self.reward_head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, head_input),
            nn.SiLU(),
            nn.Linear(head_input, 1),
        )
        self.continuation_head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, head_input),
            nn.SiLU(),
            nn.Linear(head_input, 1),
        )

    @property
    def graph_builder(self):
        if self.dreamerv3_core is not None:
            return self.dreamerv3_core.graph_builder
        return self.graph_plugin.builder

    @property
    def graph_dynamics(self):
        if self.dreamerv3_core is not None:
            return self.dreamerv3_core.graph_dynamics
        return self.graph_plugin.dynamics

    @property
    def graph_dynamics_plugin(self) -> GraphDynamicsPlugin:
        """Return the plugin through one path for every backbone adapter."""
        if self.dreamerv3_core is not None:
            return self.dreamerv3_core.graph_plugin
        return self.graph_plugin

    @property
    def visual_codec(self) -> VisualCodec:
        if self.dreamerv3_core is not None:
            return self.dreamerv3_core.visual
        return self.vae

    @property
    def visual_is_variational(self) -> bool:
        if self.dreamerv3_core is not None:
            return self.dreamerv3_core.visual_is_variational
        return self.vae.is_variational

    @property
    def token_count(self) -> int:
        if self.dreamerv3_core is not None:
            return self.dreamerv3_core.token_count
        return self.vae.token_count

    @property
    def rollout_latent_matches_visual_tokens(self) -> bool:
        """Whether rollout['tokens'] can be compared to per-frame encodings."""
        # Dreamer returns its genuine global RSSM feature during imagination;
        # it is intentionally no longer reshaped into fake visual tokens.
        return self.dreamerv3_core is None

    def encode_visual_tokens(self, images: Tensor) -> Tensor:
        if self.dreamerv3_core is not None:
            return self.dreamerv3_core.encode_visual_tokens(images)
        return self.vae.encode(images, sample=False)[0]

    def set_graph_conditioning_strength(self, strength: float) -> None:
        if self.dreamerv3_core is not None:
            self.dreamerv3_core.set_graph_conditioning_strength(strength)
            return
        self.graph_plugin.set_conditioning_strength(strength)

    @property
    def mdn_lstm(self):
        """Compatibility accessor for pre-refactor experiments."""
        return getattr(getattr(self, "backbone", None), "transition", None)

    def _condition(
        self,
        transition: BackboneOutput,
        source_tokens: Tensor,
        actions: Tensor,
        graph: Tensor,
        relation_logits: Tensor,
        use_graph: bool,
        graph_window: Tensor | None = None,
    ) -> tuple[BackboneOutput, Any]:
        plugin = self.plugin_correction
        backbone_weight = self.correction_backbone_weight
        graph_output = self.graph_plugin(
            transition.base_delta.detach() if plugin else transition.base_delta,
            actions,
            graph,
            relation_logits,
            compute_dynamics=use_graph,
            graph_window=graph_window,
            # J: the correction's context input also carries only w of its
            # gradient into the backbone; the OT loss keeps its full gradient.
            context_gradient_scale=1.0 if plugin else backbone_weight,
        )
        if plugin or backbone_weight != 1.0:
            return self._condition_decoupled(
                transition, graph_output, use_graph, plugin, backbone_weight
            ), graph_output
        correct = self.graph_plugin.correct
        if self.graph_plugin.pose_correction and use_graph:
            pose = self.graph_plugin.pose_context(graph_output.graph_next)
            base_tokens = transition.base_predicted_tokens

            def correct(delta_z: Tensor, context: Tensor) -> Tensor:
                return self.graph_plugin.correct(
                    delta_z, context, pose=pose, base_tokens=base_tokens
                )

        transition = self.backbone.apply_graph_condition(
            transition,
            source_tokens,
            graph_output.context,
            correct,
            use_graph,
        )
        return transition, graph_output

    def _condition_decoupled(
        self,
        transition: BackboneOutput,
        graph_output: Any,
        use_graph: bool,
        plugin: bool,
        backbone_weight: float,
    ) -> BackboneOutput:
        """Correction with a reduced (or no) gradient path into the backbone.

        Forward values equal the joint path (prediction + correction). The
        correction always learns from its full loss; only what flows back
        into the backbone changes:

        * plugin: the correction reads a detached prediction and the
          backbone's ``predicted_tokens`` stay its own prediction, so the
          training loss on them is exactly the graph-off loss; the corrected
          tokens are exposed as aux["plugin_corrected_tokens"] (trained by a
          separate loss term, and what rollout() feeds back).
        * backbone_weight w < 1: the backbone receives w times the corrected
          prediction's gradient (forward unchanged, x.detach() + w*(x - x.detach())).
        """
        base_tokens = transition.base_predicted_tokens
        base_delta = transition.base_delta
        if plugin:
            base_tokens = base_tokens.detach()
            base_delta = base_delta.detach()
        else:
            base_tokens = base_tokens.detach() + backbone_weight * (
                base_tokens - base_tokens.detach()
            )
            base_delta = base_delta.detach() + backbone_weight * (
                base_delta - base_delta.detach()
            )
        if use_graph:
            pose = (
                self.graph_plugin.pose_context(graph_output.graph_next)
                if self.graph_plugin.pose_correction
                else None
            )
            correction = self.graph_plugin.correct(
                base_delta, graph_output.context, pose=pose, base_tokens=base_tokens
            )
        else:
            correction = torch.zeros_like(transition.base_delta)
        transition.aux["correction"] = correction
        if plugin:
            transition.predicted_tokens = transition.base_predicted_tokens
            transition.aux["plugin_corrected_tokens"] = base_tokens + correction
        else:
            transition.predicted_tokens = base_tokens + correction
        return transition

    def _correction_gate_diagnostic(self, use_graph: bool) -> Tensor:
        plugin = self.graph_plugin
        if not plugin.pose_correction:
            return torch.sigmoid(plugin.correction_gate_logit).detach()
        if use_graph and plugin.last_correction_gate is not None:
            return plugin.last_correction_gate
        return plugin.correction_gate_logit.new_zeros(())

    def forward_sequence(
        self,
        images: Tensor,
        actions: Tensor,
        state: Any = None,
        use_graph: bool = True,
        sample_vae: bool = True,
        graph_multistep_horizon: int = 0,
        visual_multistep_horizon: int = 0,
        visual_multistep_context: int = 3,
        correction_unroll_horizon: int = 0,
    ) -> dict[str, Any]:
        """Teacher-forced sequence, images [B,T+1,3,64,64].

        ``graph_multistep_horizon > 0`` additionally unrolls the graph
        dynamics K steps on its own predictions (generic path only), for the
        K-step self-fed graph loss. ``visual_multistep_horizon > 0`` does the
        same for the visual backbone's own (uncorrected) token prediction,
        after ``visual_multistep_context`` steps of real context.
        ``correction_unroll_horizon > 0`` runs a rollout-style K-step unroll
        in which only the correction receives gradients (see
        ``_correction_unroll``). 0 (default) leaves outputs unchanged.
        """
        if self.dreamerv3_core is not None:
            core = self.dreamerv3_core
            if use_graph and core.global_correction:
                # The plug-in head exists only for plug-in (P) training.
                output = core.forward_sequence_plugin(
                    images,
                    actions,
                    sample_visual=sample_vae,
                    graph_multistep_horizon=graph_multistep_horizon,
                    correction_unroll_horizon=correction_unroll_horizon,
                )
            else:
                if graph_multistep_horizon or correction_unroll_horizon:
                    raise ValueError(
                        "DreamerV3 multistep graph / correction-unroll "
                        "losses need the plug-in path (graph on, "
                        "graph_global_correction)"
                    )
                output = core.forward_sequence(
                    images,
                    actions,
                    use_graph=use_graph,
                    sample_visual=sample_vae,
                )
            output["backbone"] = self.backbone_name
            return output
        batch, frames = images.shape[:2]
        time = frames - 1
        if actions.shape[:2] != (batch, time):
            raise ValueError("Expected one action between adjacent frames")
        flat_images = images.flatten(0, 1)
        all_z, all_mu, all_logvar = self.vae.encode(
            flat_images, sample=sample_vae
        )
        latent_shape = (
            batch,
            frames,
            self.vae.token_count,
            self.token_dim,
        )
        all_z = all_z.reshape(latent_shape)
        all_mu = all_mu.reshape(latent_shape)
        all_logvar = all_logvar.reshape(latent_shape)
        source, target = all_z[:, :-1], all_z[:, 1:]
        observed_graph = self.graph_plugin.infer_sequence(all_z, images)
        transition = self.backbone.forward_sequence(
            source,
            actions,
            state=state,
            target_tokens=target,
            deterministic=not sample_vae,
        )
        transition, graph_output = self._condition(
            transition,
            source,
            actions,
            observed_graph["nodes"][:, :-1],
            observed_graph["relation_logits"][:, :-1],
            use_graph,
        )
        predicted = transition.predicted_tokens
        pooled = predicted.mean(dim=-2)
        correction_rms = (
            transition.aux["correction"].detach().square().mean().sqrt()
        )
        base_transition_rms = (
            transition.base_delta.detach().square().mean().sqrt()
        )
        output: dict[str, Any] = {
            "backbone": self.backbone_name,
            "backbone_output": transition,
            "state": transition.state,
            "z": source,
            "z_mu": all_mu,
            "z_logvar": all_logvar,
            "visual_is_variational": self.visual_is_variational,
            "target_tokens": target,
            "predicted_tokens": predicted,
            "base_predicted_tokens": transition.base_predicted_tokens,
            "base_delta": transition.base_delta,
            "graph": graph_output.graph,
            "graph_next": graph_output.graph_next,
            "delta_g": graph_output.delta_g,
            "relation_logits": graph_output.relation_logits,
            "next_relation_logits": graph_output.next_relation_logits,
            "graph_context": graph_output.context,
            "ot_loss": graph_output.ot_loss,
            "alignment_mode": graph_output.alignment_mode,
            "ot_variance_loss": graph_output.ot_variance_loss,
            "ot_cross": graph_output.ot_cross,
            "ot_self_z": graph_output.ot_self_z,
            "ot_self_g": graph_output.ot_self_g,
            "ot_z_proj_rms": graph_output.ot_z_proj_rms,
            "ot_g_proj_rms": graph_output.ot_g_proj_rms,
            "ot_z_dispersion": graph_output.ot_z_dispersion,
            "ot_g_dispersion": graph_output.ot_g_dispersion,
            "ot_z_pair_cos": graph_output.ot_z_pair_cos,
            "ot_g_pair_cos": graph_output.ot_g_pair_cos,
            "delta_z_rms": graph_output.delta_z_rms,
            "delta_g_rms": graph_output.delta_g_rms,
            "delta_z_dispersion": graph_output.delta_z_dispersion,
            "delta_g_dispersion": graph_output.delta_g_dispersion,
            "delta_z_pair_cos": graph_output.delta_z_pair_cos,
            "delta_g_pair_cos": graph_output.delta_g_pair_cos,
            "graph_context_rms": graph_output.graph_context_rms,
            "graph_correction": transition.aux["correction"],
            "base_transition_rms": base_transition_rms,
            "graph_correction_ratio": (
                correction_rms / base_transition_rms.clamp_min(1e-8)
            ),
            "graph_correction_gate": self._correction_gate_diagnostic(use_graph),
            "target_graph": observed_graph["nodes"][:, 1:].detach(),
            "observed_graph_state": observed_graph.get(
                "state",
                self.graph_builder.teacher_state(observed_graph["nodes"]),
            ),
            "graph_state_prediction": self.graph_builder.teacher_state(
                graph_output.graph_next
            ),
            "observed_graph_presence": observed_graph["presence_logits"],
            "graph_presence_prediction": self.graph_builder.presence(
                graph_output.graph_next
            ).squeeze(-1),
            "observed_relation_logits": observed_graph["relation_logits"],
            "graph_attention": observed_graph["attention"],
            "graph_positions_2d": observed_graph["positions_2d"],
            "predicted_image": self.vae.decode(
                predicted.reshape(
                    batch * time, self.vae.token_count, self.token_dim
                )
            ).reshape(batch, time, 3, 64, 64),
            "vae_reconstruction": self.vae.decode(
                all_z.reshape(
                    batch * frames, self.vae.token_count, self.token_dim
                )
            ).reshape(batch, frames, 3, 64, 64),
            "reward_prediction": self.reward_head(pooled).squeeze(-1),
            "continuation_logit": self.continuation_head(pooled).squeeze(-1),
            # First step whose graph history window is entirely real frames of
            # this clip; earlier steps are padded with frame 0 (fabricated
            # zero velocity), so graph-dynamics supervision skips them.
            "graph_prediction_start": (
                min(self.graph_history - 1, time - 1)
                if self.graph_history > 1
                else 0
            ),
            **transition.aux,
        }
        if "plugin_corrected_tokens" in transition.aux:
            output["plugin_corrected_image"] = self.vae.decode(
                transition.aux["plugin_corrected_tokens"].reshape(
                    batch * time, self.vae.token_count, self.token_dim
                )
            ).reshape(batch, time, 3, 64, 64)
        if use_graph and self.graph_plugin.pose_correction:
            output["graph_position_observed"] = self.graph_plugin.node_position(
                observed_graph["nodes"]
            )
            output["graph_position_prediction"] = (
                self.graph_plugin.node_position(graph_output.graph_next)
            )
        if use_graph and graph_multistep_horizon > 0:
            output.update(
                self._graph_multistep(
                    observed_graph, actions, graph_multistep_horizon
                )
            )
        if use_graph and correction_unroll_horizon > 0:
            output.update(
                self._correction_unroll(
                    all_z, actions, observed_graph, correction_unroll_horizon
                )
            )
        if visual_multistep_horizon > 0:
            output.update(
                self._visual_multistep(
                    all_z,
                    actions,
                    visual_multistep_horizon,
                    visual_multistep_context,
                    deterministic=not sample_vae,
                )
            )
        return output

    def _visual_multistep(
        self,
        all_z: Tensor,
        actions: Tensor,
        horizon: int,
        context: int,
        deterministic: bool,
    ) -> dict[str, Any]:
        """Unroll the visual backbone K steps on its own token predictions.

        Real tokens 0..context condition the first prediction; every later
        step feeds the previous *base* prediction back in (no graph
        correction), through the same forward_sequence/feedback calls
        rollout() uses. Gradients flow through the whole unroll, matching
        the graph K-step loss.
        """
        time = actions.shape[1]
        start = min(context, time - 1)
        horizon = min(horizon, time - start)
        first = self.backbone.forward_sequence(
            all_z[:, : start + 1],
            actions[:, : start + 1],
            state=None,
            deterministic=deterministic,
        )
        prediction = first.base_predicted_tokens[:, -1]
        state = first.state
        predictions = [prediction]
        for step in range(1, horizon):
            state = self.backbone.feedback(state, prediction)
            transition = self.backbone.forward_sequence(
                prediction[:, None],
                actions[:, start + step : start + step + 1],
                state=state,
                deterministic=deterministic,
            )
            prediction = transition.base_predicted_tokens[:, 0]
            state = transition.state
            predictions.append(prediction)
        return {
            "visual_multistep_prediction": torch.stack(predictions, dim=1),
            "visual_multistep_target": all_z[
                :, start + 1 : start + 1 + horizon
            ].detach(),
            "visual_multistep_source": all_z[:, start : start + horizon].detach(),
        }

    def _graph_multistep(
        self,
        observed_graph: dict[str, Tensor],
        actions: Tensor,
        horizon: int,
    ) -> dict[str, Any]:
        """Unroll graph dynamics K steps on its own predictions.

        See ``GraphDynamicsPlugin.graph_multistep`` (shared with DreamerV3).
        """
        return self.graph_plugin.graph_multistep(
            observed_graph["nodes"],
            observed_graph["relation_logits"],
            actions,
            horizon,
        )

    def _correction_unroll(
        self,
        all_z: Tensor,
        actions: Tensor,
        observed_graph: dict[str, Tensor],
        horizon: int,
    ) -> dict[str, Any]:
        """Train the correction on the model's own drifted rollout.

        Mirrors rollout(): the visual backbone starts from one real frame,
        the graph from its real history window at the same frame, and both
        then run on their own outputs for K steps, with the corrected
        prediction fed back. Backbone and graph run under no_grad (backbone
        in eval mode, as in rollout), so this objective updates only the
        correction; the backbone learns from the single-step loss alone.
        The correction runs at full strength regardless of the warm-up
        delay, since nothing here reaches the backbone.
        """
        time = actions.shape[1]
        start = min(self.graph_history - 1, time - 1)
        horizon = min(horizon, time - start)
        with torch.no_grad():
            offsets = torch.arange(
                start - self.graph_history + 1,
                start + 1,
                device=all_z.device,
            ).clamp_min(0)
            window = observed_graph["nodes"][:, offsets].detach()
            graph = window[:, -1]
            relations = observed_graph["relation_logits"][:, start].detach()
        tokens = all_z[:, start].detach()
        state: Any = None
        corrected_steps: list[Tensor] = []
        base_steps: list[Tensor] = []
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            for step in range(horizon):
                index = start + step
                step_action = actions[:, index : index + 1]
                with torch.no_grad():
                    transition = self.backbone.forward_sequence(
                        tokens[:, None], step_action, state=state, deterministic=True
                    )
                    graph_output = self.graph_plugin(
                        transition.base_delta,
                        step_action,
                        graph[:, None],
                        relations[:, None],
                        graph_window=(
                            window[:, None] if self.graph_history > 1 else None
                        ),
                    )
                # Outside no_grad: the pose payload projection belongs to the
                # correction; pose_context itself detaches graph and position.
                pose = (
                    self.graph_plugin.pose_context(graph_output.graph_next)
                    if self.graph_plugin.pose_correction
                    else None
                )
                correction = self.graph_plugin.correct(
                    transition.base_delta,
                    graph_output.context,
                    pose=pose,
                    base_tokens=transition.base_predicted_tokens,
                    strength=1.0,
                )
                base = transition.base_predicted_tokens[:, 0]
                corrected = base + correction[:, 0]
                corrected_steps.append(corrected)
                base_steps.append(base)
                tokens = corrected.detach()
                state = self.backbone.feedback(transition.state, tokens)
                graph = graph_output.graph_next[:, 0]
                relations = graph_output.next_relation_logits[:, 0]
                if self.graph_history > 1:
                    window = torch.cat((window[:, 1:], graph[:, None]), dim=1)
        finally:
            self.backbone.train(was_training)
        return {
            "correction_unroll_prediction": torch.stack(corrected_steps, dim=1),
            "correction_unroll_base": torch.stack(base_steps, dim=1),
            "correction_unroll_target": all_z[
                :, start + 1 : start + 1 + horizon
            ].detach(),
            "correction_unroll_source": all_z[
                :, start : start + horizon
            ].detach(),
        }

    def forward(
        self,
        image: Tensor,
        action: Tensor,
        state: Any = None,
        use_graph: bool = True,
        sample_vae: bool = True,
    ) -> dict[str, Any]:
        """Single prior transition used for inference and compatibility."""
        if self.dreamerv3_core is not None:
            raise NotImplementedError(
                "Use forward_sequence or rollout for burchim DreamerV3"
            )
        tokens, mu, logvar = self.vae.encode(image, sample=sample_vae)
        observation = self.graph_plugin.observe(
            tokens, image, sample=False
        )
        transition = self.backbone.forward_sequence(
            tokens[:, None],
            action[:, None],
            state=state,
            deterministic=not sample_vae,
        )
        transition, graph_output = self._condition(
            transition,
            tokens[:, None],
            action[:, None],
            observation.nodes[:, None],
            observation.relation_logits[:, None],
            use_graph,
        )
        predicted = transition.predicted_tokens[:, 0]
        correction_rms = (
            transition.aux["correction"].detach().square().mean().sqrt()
        )
        base_transition_rms = (
            transition.base_delta.detach().square().mean().sqrt()
        )
        result: dict[str, Any] = {
            "backbone": self.backbone_name,
            "backbone_output": transition,
            "state": transition.state,
            "z": tokens,
            "z_mu": mu,
            "z_logvar": logvar,
            "visual_is_variational": self.visual_is_variational,
            "predicted_tokens": predicted,
            "base_predicted_tokens": transition.base_predicted_tokens[:, 0],
            "base_delta": transition.base_delta[:, 0],
            "graph": graph_output.graph[:, 0],
            "graph_next": graph_output.graph_next[:, 0],
            "delta_g": graph_output.delta_g[:, 0],
            "relation_logits": graph_output.relation_logits[:, 0],
            "next_relation_logits": graph_output.next_relation_logits[:, 0],
            "graph_context": graph_output.context[:, 0],
            "ot_loss": graph_output.ot_loss,
            "alignment_mode": graph_output.alignment_mode,
            "ot_variance_loss": graph_output.ot_variance_loss,
            "ot_cross": graph_output.ot_cross,
            "ot_self_z": graph_output.ot_self_z,
            "ot_self_g": graph_output.ot_self_g,
            "ot_z_proj_rms": graph_output.ot_z_proj_rms,
            "ot_g_proj_rms": graph_output.ot_g_proj_rms,
            "ot_z_dispersion": graph_output.ot_z_dispersion,
            "ot_g_dispersion": graph_output.ot_g_dispersion,
            "ot_z_pair_cos": graph_output.ot_z_pair_cos,
            "ot_g_pair_cos": graph_output.ot_g_pair_cos,
            "delta_z_rms": graph_output.delta_z_rms,
            "delta_g_rms": graph_output.delta_g_rms,
            "delta_z_dispersion": graph_output.delta_z_dispersion,
            "delta_g_dispersion": graph_output.delta_g_dispersion,
            "delta_z_pair_cos": graph_output.delta_z_pair_cos,
            "delta_g_pair_cos": graph_output.delta_g_pair_cos,
            "graph_context_rms": graph_output.graph_context_rms,
            "graph_correction": transition.aux["correction"][:, 0],
            "base_transition_rms": base_transition_rms,
            "graph_correction_ratio": (
                correction_rms / base_transition_rms.clamp_min(1e-8)
            ),
            "graph_correction_gate": self._correction_gate_diagnostic(use_graph),
            "predicted_image": self.vae.decode(predicted),
            "vae_reconstruction": self.vae.decode(tokens),
            "reward_prediction": self.reward_head(
                predicted.mean(-2)
            ).squeeze(-1),
            "continuation_logit": self.continuation_head(
                predicted.mean(-2)
            ).squeeze(-1),
        }
        for key, value in transition.aux.items():
            if isinstance(value, Tensor) and value.ndim > 1 and value.shape[1] == 1:
                result[key] = value[:, 0]
            else:
                result[key] = value
        return result

    @torch.no_grad()
    def rollout(
        self,
        initial_image: Tensor,
        actions: Tensor,
        use_graph: bool = True,
        deterministic: bool = True,
        temperature: float = 1.0,
        decode: bool = True,
        context_images: Tensor | None = None,
        context_actions: Tensor | None = None,
        visual_context: bool = False,
    ) -> dict[str, Tensor]:
        """Open-loop recursive prior imagination for either backbone.

        ``context_images`` ([B, C, 3, H, W], oldest first) are real frames
        immediately *before* ``initial_image``. They always seed the graph
        history window when ``graph_history > 1``. By default the visual
        backbone still starts from ``initial_image`` alone (historical
        protocol). With ``visual_context=True`` the backbone also receives
        the same real frames and the actions between them
        (``context_actions`` [B, C, A]) as history before the first
        prediction -- exactly the input it sees in teacher-forced training
        -- so graph and backbone start from the same real information.
        Without context, the graph window is padded by repeating the first
        observed frame (zero velocity), the same convention training uses at
        sequence starts.
        """
        if self.dreamerv3_core is not None:
            return self.dreamerv3_core.rollout(
                initial_image,
                actions,
                use_graph=use_graph,
                decode=decode,
                context_images=context_images,
                context_actions=context_actions,
                visual_context=visual_context,
                deterministic=deterministic,
            )
        tokens, _, _ = self.vae.encode(initial_image, sample=False)
        graph_window: Tensor | None = None
        if self.graph_history == 1:
            # Unchanged single-frame path.
            observation = self.graph_plugin.observe(
                tokens, initial_image, sample=False
            )
            graph = observation.nodes
            relations = observation.relation_logits
        else:
            seed_images = initial_image[:, None]
            if context_images is not None:
                seed_images = torch.cat(
                    (context_images[:, -(self.graph_history - 1):], seed_images),
                    dim=1,
                )
            batch, seed_frames = seed_images.shape[:2]
            seed_tokens, _, _ = self.vae.encode(
                seed_images.flatten(0, 1), sample=False
            )
            seed_tokens = seed_tokens.reshape(
                batch, seed_frames, *seed_tokens.shape[1:]
            )
            observed = self.graph_plugin.infer_sequence(seed_tokens, seed_images)
            offsets = torch.arange(
                seed_frames - self.graph_history,
                seed_frames,
                device=initial_image.device,
            ).clamp_min(0)
            graph_window = observed["nodes"][:, offsets]
            graph = graph_window[:, -1]
            relations = observed["relation_logits"][:, -1]
        state: Any = None
        if visual_context and context_images is not None and context_images.shape[1]:
            if (
                context_actions is None
                or context_actions.shape[1] != context_images.shape[1]
            ):
                raise ValueError(
                    "visual_context needs one action per context frame"
                )
            batch, count = context_images.shape[:2]
            context_tokens, _, _ = self.vae.encode(
                context_images.flatten(0, 1), sample=False
            )
            context_tokens = context_tokens.reshape(
                batch, count, *context_tokens.shape[1:]
            )
            # Only the resulting history is kept; these predictions of the
            # context frames themselves are discarded.
            state = self.backbone.forward_sequence(
                context_tokens,
                context_actions,
                state=None,
                deterministic=deterministic,
            ).state
        token_rollout: list[Tensor] = []
        graph_rollout: list[Tensor] = []
        relation_rollout: list[Tensor] = []
        image_rollout: list[Tensor] = []
        reward_rollout: list[Tensor] = []
        continuation_rollout: list[Tensor] = []
        for step in range(actions.shape[1]):
            transition = self.backbone.forward_sequence(
                tokens[:, None],
                actions[:, step : step + 1],
                state=state,
                deterministic=deterministic,
            )
            transition, graph_output = self._condition(
                transition,
                tokens[:, None],
                actions[:, step : step + 1],
                graph[:, None],
                relations[:, None],
                use_graph,
                graph_window=(
                    graph_window[:, None] if graph_window is not None else None
                ),
            )
            if (
                not deterministic
                and isinstance(transition.aux.get("mdn"), MDNOutput)
            ):
                mdn = transition.aux["mdn"]
                tokens = mdn.sample(
                    transition.aux["conditioned_mean"], temperature
                )[:, 0]
            else:
                # Plug-in mode keeps the backbone's own prediction in
                # predicted_tokens; the rollout feeds back the corrected one.
                tokens = transition.aux.get(
                    "plugin_corrected_tokens", transition.predicted_tokens
                )[:, 0]
            state = self.backbone.feedback(transition.state, tokens)
            graph = graph_output.graph_next[:, 0]
            relations = graph_output.next_relation_logits[:, 0]
            if graph_window is not None:
                graph_window = torch.cat(
                    (graph_window[:, 1:], graph[:, None]), dim=1
                )
            pooled = tokens.mean(-2)
            token_rollout.append(tokens)
            graph_rollout.append(graph)
            relation_rollout.append(relations)
            reward_rollout.append(self.reward_head(pooled).squeeze(-1))
            continuation_rollout.append(
                self.continuation_head(pooled).squeeze(-1).sigmoid()
            )
            if decode:
                image_rollout.append(self.vae.decode(tokens))
        result = {
            "tokens": torch.stack(token_rollout, dim=1),
            "graphs": torch.stack(graph_rollout, dim=1),
            "relations": torch.stack(relation_rollout, dim=1),
            "rewards": torch.stack(reward_rollout, dim=1),
            "continuations": torch.stack(continuation_rollout, dim=1),
        }
        if decode:
            result["images"] = torch.stack(image_rollout, dim=1)
        return result
