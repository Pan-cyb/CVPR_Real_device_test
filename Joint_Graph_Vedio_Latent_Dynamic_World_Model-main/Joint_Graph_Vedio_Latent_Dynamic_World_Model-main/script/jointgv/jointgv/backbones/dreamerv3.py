"""Adapter around the pinned burchim DreamerV3-PyTorch world model.

The neural-network components are imported directly from the pinned submodule
without executing its heavyweight top-level ``nnet`` package initializer.  The
native path preserves the DMC-S encoder, RSSM, decoder, reward head, and
discount head.  The VAE path changes only the visual encoder/decoder boundary;
both paths share the same burchim RSSM and prediction heads.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import types
from typing import Any

import torch
from torch import Tensor, nn

from ..blocks import GraphDynamicsPlugin, VisualVAE


def _load_burchim_components():
    root = (
        Path(__file__).resolve().parents[4]
        / "benchmark"
        / "burchim-dreamerv3-pytorch"
    )
    nnet_root = root / "nnet"
    if not nnet_root.is_dir():
        raise FileNotFoundError(
            "Missing burchim DreamerV3 submodule. Run: "
            "git submodule update --init --recursive"
        )
    # Importing burchim's nnet/__init__.py eagerly loads Atari, WandB, AV, and
    # plotting dependencies that its world-model components do not need.  A
    # namespace package keeps the imported source files unchanged while
    # loading only nnet.modules, nnet.distributions, and nnet.structs.
    package = sys.modules.get("nnet")
    if package is None:
        package = types.ModuleType("nnet")
        package.__path__ = [str(nnet_root)]
        package.__package__ = "nnet"
        sys.modules["nnet"] = package
    elif str(nnet_root) not in getattr(package, "__path__", []):
        raise RuntimeError("A different top-level nnet package is already loaded")
    from nnet.modules.dreamer.v3 import (  # type: ignore[import-not-found]
        DiscountNetwork,
        ObsNetwork,
        PolicyNetwork,
        ReprNetwork,
        RewardNetwork,
        RSSM,
        ValueNetwork,
    )

    return (
        ReprNetwork,
        RSSM,
        ObsNetwork,
        RewardNetwork,
        DiscountNetwork,
        PolicyNetwork,
        ValueNetwork,
    )


(
    BurchimReprNetwork,
    BurchimRSSM,
    BurchimObsNetwork,
    BurchimRewardNetwork,
    BurchimDiscountNetwork,
    BurchimPolicyNetwork,
    BurchimValueNetwork,
) = _load_burchim_components()


def _detach_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach() for key, value in state.items()}


def _stack_states(states: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        key: torch.stack([state[key] for state in states], dim=1)
        for key in states[0]
    }


def _install_dreamer_relation_index_compatibility(builder: nn.Module) -> None:
    """Fix diagonal relation indexing for this Dreamer observer instance.

    Both existing observer variants use a boolean advanced-index expression
    that is rejected by current PyTorch. This instance-local adapter preserves
    their exact relation computation and parameters while using explicit
    diagonal indices. The shared observer classes remain untouched.
    """
    def relation_head(
        observer: nn.Module,
        nodes: Tensor,
        positions_2d: Tensor | None = None,
    ) -> Tensor:
        batch, count, _ = nodes.shape
        if positions_2d is None:
            positions_2d = nodes.new_zeros(batch, count, 2)
        subject = observer.subject(nodes)[:, :, None, None, :]
        object_ = observer.object(nodes)[:, None, :, None, :]
        translation = observer.relation_translation[None, None, None, :, :]
        translation_score = -(
            subject + translation - object_
        ).square().mean(-1)
        sender = nodes[:, :, None, :].expand(-1, -1, count, -1)
        receiver = nodes[:, None, :, :].expand(-1, count, -1, -1)
        relative = positions_2d[:, None] - positions_2d[:, :, None]
        if hasattr(observer, "edge_seed"):
            edge = observer.edge_seed(
                torch.cat((sender, receiver, relative), dim=-1)
            )
            residual_input = torch.cat(
                (sender, receiver, edge, relative), dim=-1
            )
        else:
            residual_input = torch.cat(
                (sender, receiver, relative), dim=-1
            )
        logits = translation_score + observer.relation_residual(residual_input)
        diagonal = torch.arange(count, device=nodes.device)
        logits = logits.clone()
        logits[:, diagonal, diagonal, 0] += 20.0
        if observer.relation_count > 1:
            logits[:, diagonal, diagonal, 1:] -= 20.0
        return logits

    builder.relation_head = types.MethodType(relation_head, builder)


class DreamerV3WorldModel(nn.Module):
    """Burchim DreamerV3 DMC-S world model with an optional JointGV branch."""

    checkpoint_version = 10
    deter_dim = 512
    stoch_dim = 32
    classes = 32
    hidden_dim = 512
    dim_cnn = 32
    head_layers = 2
    free_nats = 1.0
    dynamics_scale = 0.5
    representation_scale = 0.1

    def __init__(
        self,
        action_dim: int,
        visual_kind: str,
        token_dim: int = 64,
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
        graph_history: int = 1,
        graph_pose_correction: bool = False,
        graph_pose_sigma: float = 0.15,
        global_correction: bool = False,
    ):
        """``global_correction`` selects the plug-in (P) path.

        False (default): the historical path -- ``_prior_step`` corrects the
        prior deter through ``graph_to_deter`` and the corrected prior feeds
        the posterior (joint training; online training uses this). True:
        ``forward_sequence_plugin`` / ``_plugin_rollout`` -- the RSSM runs
        exactly its graph-off computation and the shared plug-in's global
        correction head (``GraphDynamicsPlugin.correct_global``) corrects the
        prior deter from detached RSSM features. Graph history and pose
        correction are wired only on the plug-in path.
        """
        super().__init__()
        if visual_kind not in {"dreamerv3_native", "vae"}:
            raise ValueError(f"Unknown burchim visual kind: {visual_kind}")
        if not 0.0 < graph_correction_gate_init < 1.0:
            raise ValueError("graph_correction_gate_init must be in (0, 1)")
        if (graph_history != 1 or graph_pose_correction) and not global_correction:
            raise ValueError(
                "DreamerV3 graph_history > 1 and pose correction need the "
                "plug-in path (global_correction=True)"
            )
        self.global_correction = bool(global_correction)
        self.action_dim = int(action_dim)
        self.visual_kind = visual_kind
        self.visual_frozen = bool(freeze_visual)
        self.max_graph_correction = float(max_graph_correction)
        norm = {"class": "LayerNorm", "params": {"eps": 1e-3, "convert_float32": True}}

        if visual_kind == "dreamerv3_native":
            self.visual = BurchimReprNetwork(
                dim_input_cnn=3,
                dim_cnn=self.dim_cnn,
                dim_input_mlp=None,
                num_mlp_layers=self.head_layers,
                hidden_size=self.hidden_dim,
                cnn_norm=norm,
                mlp_norm=norm,
            )
            self.token_dim = 8 * self.dim_cnn
            self.token_count = 16
            embed_size = self.token_count * self.token_dim
        else:
            self.visual = VisualVAE(token_dim)
            self.token_dim = int(token_dim)
            self.token_count = self.visual.token_count
            embed_size = self.token_count * self.token_dim

        if self.visual_frozen:
            self.visual.requires_grad_(False)

        self.rssm = BurchimRSSM(
            num_actions=self.action_dim,
            deter_size=self.deter_dim,
            stoch_size=self.stoch_dim,
            hidden_size=self.hidden_dim,
            embed_size=embed_size,
            discrete=self.classes,
            learn_initial=True,
            norm=norm,
        )
        self.feat_dim = self.deter_dim + self.stoch_dim * self.classes
        if visual_kind == "dreamerv3_native":
            self.decoder = BurchimObsNetwork(
                dim_output_cnn=3,
                feat_size=self.feat_dim,
                dim_cnn=self.dim_cnn,
                dim_output_mlp=None,
                num_mlp_layers=self.head_layers,
                hidden_size=self.hidden_dim,
                norm=norm,
                mlp_norm=norm,
            )
            self.vae_token_decoder = None
        else:
            self.decoder = None
            self.vae_token_decoder = nn.Sequential(
                nn.Linear(self.feat_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim, eps=1e-3),
                nn.SiLU(),
                nn.Linear(
                    self.hidden_dim, self.token_count * self.token_dim
                ),
            )
        self.reward_head = BurchimRewardNetwork(
            hidden_size=self.hidden_dim,
            feat_size=self.feat_dim,
            num_mlp_layers=self.head_layers,
            norm=norm,
        )
        self.continuation_head = BurchimDiscountNetwork(
            hidden_size=self.hidden_dim,
            feat_size=self.feat_dim,
            num_mlp_layers=self.head_layers,
            norm=norm,
        )

        self.graph_plugin = GraphDynamicsPlugin(
            token_dim=self.token_dim,
            token_count=self.token_count,
            action_dim=self.action_dim,
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
            alignment_dim=self.feat_dim,
            alignment_mode="global",
            graph_history=graph_history,
            pose_correction=graph_pose_correction,
            pose_sigma=graph_pose_sigma,
            correction_dim=self.deter_dim if global_correction else None,
        )
        _install_dreamer_relation_index_compatibility(
            self.graph_plugin.builder
        )
        # DreamerV3 exposes one global RSSM feature, not a set of visual
        # dynamics tokens.  GraphDynamicsPlugin.forward_global therefore lets
        # the *real* Δs query active ΔG nodes and returns their pooled context.
        # This linear adapter only maps that node-space context into the RSSM
        # deterministic-state space; it does not fabricate pseudo tokens.
        self.graph_to_deter = nn.Sequential(
            nn.Linear(node_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.deter_dim),
        )
        nn.init.zeros_(self.graph_to_deter[-1].weight)
        nn.init.zeros_(self.graph_to_deter[-1].bias)
        graph_gate_logit = math.log(
            graph_correction_gate_init
            / (1.0 - graph_correction_gate_init)
        )
        self.graph_gate_logit = nn.Parameter(torch.tensor(graph_gate_logit))
        self.conditioning_strength = 1.0

    @property
    def graph_builder(self):
        return self.graph_plugin.builder

    @property
    def graph_dynamics(self):
        return self.graph_plugin.dynamics

    @property
    def visual_is_variational(self) -> bool:
        return self.visual_kind == "vae"

    def set_graph_conditioning_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("conditioning strength must be in [0, 1]")
        self.conditioning_strength = float(strength)
        self.graph_plugin.set_conditioning_strength(strength)

    def _encode(
        self, images: Tensor, sample: bool
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return spatial tokens, flat embed, mean, and log variance."""
        batch_shape = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:])
        if self.visual_kind == "dreamerv3_native":
            # Burchim preprocessing maps uint8 to float/255 - 0.5 before the
            # representation network. EpisodeSequences already yields [0, 1].
            feature = self.visual.cnn(flat - 0.5)
            tokens = feature.flatten(2).transpose(1, 2)
            mu = tokens
            logvar = torch.zeros_like(tokens)
        else:
            tokens, mu, logvar = self.visual.encode(flat, sample=sample)
        tokens = tokens.reshape(*batch_shape, self.token_count, self.token_dim)
        mu = mu.reshape_as(tokens)
        logvar = logvar.reshape_as(tokens)
        return tokens, tokens.flatten(-2), mu, logvar

    def encode_visual_tokens(self, images: Tensor) -> Tensor:
        return self._encode(images, sample=False)[0]

    def _decode_feature(self, feature: Tensor) -> tuple[Tensor, Any | None]:
        if self.visual_kind == "dreamerv3_native":
            distribution = self.decoder(feature)
            centered = distribution.mode()
            return (centered + 0.5).clamp(0.0, 1.0), distribution
        tokens = self.vae_token_decoder(feature).reshape(
            *feature.shape[:-1], self.token_count, self.token_dim
        )
        flat_tokens = tokens.reshape(-1, self.token_count, self.token_dim)
        images = self.visual.decode(flat_tokens).reshape(
            *feature.shape[:-1], 3, 64, 64
        )
        return images, None

    def _initial_posterior(self, embed: Tensor) -> dict[str, Tensor]:
        state = self.rssm.initial(
            batch_size=embed.shape[0], dtype=embed.dtype, device=embed.device
        )
        zero_action = embed.new_zeros(embed.shape[0], self.action_dim)
        post, _ = self.rssm(
            state, zero_action, embed, torch.ones(embed.shape[0], device=embed.device)
        )
        return post

    def reconstruct_observations(self, images: Tensor) -> dict[str, Any]:
        """Reconstruct frames without a temporal rollout or auxiliary heads.

        Stage-1 representation training still uses Dreamer's real posterior
        and decoder. Each frame is treated like the first observation of an
        episode: a zero action produces the initial prior, the observation
        embedding produces the posterior stochastic state, and the resulting
        RSSM feature is decoded. Consequently no future observation, reward,
        continuation, graph, actor, or critic computation is involved.
        """
        if images.ndim < 4 or images.shape[-3:] != (3, 64, 64):
            raise ValueError("Expected RGB observations ending in [3,64,64]")
        leading_shape = images.shape[:-3]
        flat_images = images.reshape(-1, *images.shape[-3:])
        _, embeds, _, _ = self._encode(flat_images, sample=False)
        initial = self.rssm.initial(
            batch_size=flat_images.shape[0],
            dtype=embeds.dtype,
            device=embeds.device,
            detach_learned=True,
        )
        zero_action = embeds.new_zeros(flat_images.shape[0], self.action_dim)
        # The frozen initial-prior path supplies the deterministic component
        # expected by the existing posterior and decoder. It is deliberately
        # not optimized during visual representation training.
        with torch.no_grad():
            prior = self.rssm.forward_img(initial, zero_action)
        posterior = self.rssm.forward_obs(prior["deter"], embeds)
        feature = self.rssm.get_feat(posterior)
        reconstruction, distribution = self._decode_feature(feature)
        return {
            "reconstruction": reconstruction.reshape(
                *leading_shape, *reconstruction.shape[-3:]
            ),
            "reconstruction_distribution": distribution,
            "posterior_features": feature.reshape(*leading_shape, -1),
        }

    @torch.no_grad()
    def open_loop_prediction(
        self,
        initial_image: Tensor,
        actions: Tensor,
        *,
        use_graph: bool = False,
    ) -> dict[str, Tensor]:
        """Predict recursively from one observed frame, optionally with Graph.

        ``actions[:, h]`` is the recorded action leading from horizon ``h``
        to ``h + 1``. Only the initial image enters the posterior; all future
        states are RSSM priors. In Graph mode, the initial graph is observed
        once and then recursively predicted alongside the RSSM state.
        """
        if initial_image.ndim != 4 or initial_image.shape[-3:] != (3, 64, 64):
            raise ValueError("Expected initial_image [B,3,64,64]")
        if actions.ndim != 3 or actions.shape[0] != initial_image.shape[0]:
            raise ValueError("Expected actions [B,H,A]")
        tokens, embeds, _, _ = self._encode(initial_image[:, None], sample=False)
        current = self._initial_posterior(embeds[:, 0])
        observed_graph = (
            self.graph_plugin.infer_sequence(tokens, initial_image[:, None])
            if use_graph
            else None
        )
        graph = observed_graph["nodes"][:, 0] if observed_graph is not None else None
        relations = (
            observed_graph["relation_logits"][:, 0]
            if observed_graph is not None
            else None
        )
        features: list[Tensor] = []
        images: list[Tensor] = []
        rewards: list[Tensor] = []
        continuations: list[Tensor] = []
        corrections: list[Tensor] = []
        base_prior_deters: list[Tensor] = []
        for step in range(actions.shape[1]):
            current, graph_output, _, correction, _, base_prior_deter = self._prior_step(
                current,
                actions[:, step],
                graph,
                relations,
                use_graph,
            )
            feature = self.rssm.get_feat(current)
            prediction, _ = self._decode_feature(feature)
            features.append(feature)
            images.append(prediction)
            rewards.append(self.reward_head(feature).mode().squeeze(-1))
            continuations.append(
                self.continuation_head(feature).base_dist.probs.squeeze(-1)
            )
            corrections.append(correction)
            base_prior_deters.append(base_prior_deter)
            if graph_output is not None:
                graph = graph_output.graph_next[:, 0]
                relations = graph_output.next_relation_logits[:, 0]
        result = {
            "features": torch.stack(features, dim=1),
            "images": torch.stack(images, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "continuations": torch.stack(continuations, dim=1),
            "graph_correction": torch.stack(corrections, dim=1),
            "base_prior_deter": torch.stack(base_prior_deters, dim=1),
        }
        if graph is not None:
            result["final_graph"] = graph
        return result

    def _prior_step(
        self,
        previous: dict[str, Tensor],
        action: Tensor,
        graph: Tensor | None,
        relations: Tensor | None,
        use_graph: bool,
    ) -> tuple[dict[str, Tensor], Any | None, Tensor, Tensor, Tensor, Tensor]:
        prior = self.rssm.forward_img(previous, action.clone())
        base_prior_deter = prior["deter"]
        if not use_graph:
            # The graph-off path returns immediately after the untouched
            # burchim RSSM prior. No graph/projection parameter is executed.
            zero_delta = action.new_zeros(action.shape[0], self.feat_dim)
            zero_correction = action.new_zeros(
                action.shape[0], self.deter_dim
            )
            zero_attention_entropy = action.new_zeros(action.shape[0])
            return (
                prior,
                None,
                zero_delta,
                zero_correction,
                zero_attention_entropy,
                base_prior_deter,
            )
        previous_feat = self.rssm.get_feat(previous)
        base_feat = self.rssm.get_feat(prior)
        # This is the actual DreamerV3 recurrent-state change.  Earlier
        # versions passed it through an unmotivated MLP and reshaped it into
        # 16 pseudo tokens; that bridge has intentionally been removed.
        base_delta = base_feat - previous_feat
        if graph is None or relations is None:
            raise ValueError("Graph state is required when graph conditioning is on")
        graph_output = self.graph_plugin.forward_global(
            base_delta[:, None],
            action[:, None],
            graph[:, None],
            relations[:, None],
        )
        attention_entropy = graph_output.alignment_attention_entropy
        if attention_entropy is None:
            attention_entropy = action.new_zeros(())
        raw = self.graph_to_deter(graph_output.context[:, 0])
        bounded = self.max_graph_correction * torch.tanh(
            raw / self.max_graph_correction
        )
        correction = (
            self.conditioning_strength
            * torch.sigmoid(self.graph_gate_logit)
            * bounded
        )
        deter = prior["deter"] + correction
        logits = self.rssm.mlp_img2(deter).reshape(
            deter.shape[0], self.stoch_dim, self.classes
        )
        stoch = self.rssm.get_dist({"logits": logits}).rsample()
        conditioned = {"stoch": stoch, "deter": deter, "logits": logits}
        return (
            conditioned,
            graph_output,
            base_delta,
            correction,
            attention_entropy.expand(action.shape[0]),
            base_prior_deter,
        )

    def forward_sequence(
        self,
        images: Tensor,
        actions: Tensor,
        *,
        use_graph: bool,
        sample_visual: bool,
    ) -> dict[str, Any]:
        batch, frames = images.shape[:2]
        if actions.shape[:2] != (batch, frames - 1):
            raise ValueError("Expected one action between adjacent frames")
        tokens, embeds, mu, logvar = self._encode(images, sample_visual)
        current = self._initial_posterior(embeds[:, 0])
        initial_posterior = current
        observed_graph = (
            self.graph_plugin.infer_sequence(tokens, images) if use_graph else None
        )
        posts: list[dict[str, Tensor]] = []
        priors: list[dict[str, Tensor]] = []
        graph_outputs: list[Any] = []
        base_deltas: list[Tensor] = []
        graph_corrections: list[Tensor] = []
        graph_pool_entropies: list[Tensor] = []
        base_prior_deters: list[Tensor] = []
        for step in range(actions.shape[1]):
            graph = (
                observed_graph["nodes"][:, step]
                if observed_graph is not None
                else None
            )
            relations = (
                observed_graph["relation_logits"][:, step]
                if observed_graph is not None
                else None
            )
            (
                prior,
                graph_output,
                base_delta,
                graph_correction,
                graph_pool_entropy,
                base_prior_deter,
            ) = self._prior_step(
                current, actions[:, step], graph, relations, use_graph
            )
            post = self.rssm.forward_obs(prior["deter"], embeds[:, step + 1])
            priors.append(prior)
            posts.append(post)
            base_deltas.append(base_delta)
            graph_corrections.append(graph_correction)
            graph_pool_entropies.append(graph_pool_entropy)
            base_prior_deters.append(base_prior_deter)
            if graph_output is not None:
                graph_outputs.append(graph_output)
            current = post

        post_states = _stack_states(posts)
        prior_states = _stack_states(priors)
        post_feat = self.rssm.get_feat(post_states)
        prior_feat = self.rssm.get_feat(prior_states)
        reconstructed, reconstruction_dist = self._decode_feature(post_feat)
        predicted, _ = self._decode_feature(prior_feat)
        reward_dist = self.reward_head(post_feat)
        continuation_dist = self.continuation_head(post_feat)

        post_detached = _detach_state(post_states)
        prior_detached = _detach_state(prior_states)
        dynamics_kl = torch.distributions.kl_divergence(
            self.rssm.get_dist(post_detached), self.rssm.get_dist(prior_states)
        )
        representation_kl = torch.distributions.kl_divergence(
            self.rssm.get_dist(post_states), self.rssm.get_dist(prior_detached)
        )
        dynamics_kl_raw = dynamics_kl.mean()
        representation_kl_raw = representation_kl.mean()
        dynamics_kl = dynamics_kl.clamp_min(self.free_nats).mean()
        representation_kl = representation_kl.clamp_min(self.free_nats).mean()

        zero = images.new_zeros(())
        # Upstream Burchim applies the same RSSM KL objective to each replay
        # row, including an is_first row whose prior is the reset state.
        # Transition-level JointGV losses below are masked at internal resets
        # because they compare G_{t-1} with G_t and have no reset analogue.
        output: dict[str, Any] = {
            "loss_family": "dreamerv3",
            # Read by forward_sequence_plugin to rebuild the plug-in's input
            # (the state before each step); not used by any loss.
            "initial_posterior": initial_posterior,
            "prior_states": prior_states,
            "all_tokens": tokens,
            "state": current,
            "z": tokens[:, :-1],
            "z_mu": mu,
            "z_logvar": logvar,
            "visual_is_variational": self.visual_is_variational,
            # Dreamer dynamics live in global RSSM feature space.  These
            # fields are diagnostics only; Dreamer's transition objective is
            # the categorical RSSM KL below.
            "target_tokens": post_feat.detach(),
            "predicted_tokens": prior_feat,
            "base_predicted_tokens": prior_feat,
            "base_delta": torch.stack(base_deltas, dim=1),
            "predicted_image": predicted,
            "vae_reconstruction": reconstructed,
            "reconstruction_distribution": reconstruction_dist,
            "reward_distribution": reward_dist,
            "continuation_distribution": continuation_dist,
            "reward_prediction": reward_dist.mode().squeeze(-1),
            "continuation_logit": continuation_dist.base_dist.logits.squeeze(-1),
            "dynamics_kl": dynamics_kl,
            "dynamics_kl_raw": dynamics_kl_raw,
            "representation_kl": representation_kl,
            "representation_kl_raw": representation_kl_raw,
            "ot_loss": zero,
            "alignment_mode": "global",
            # Online Dreamer actor training starts imagination from every
            # posterior state in the replay sequence, exactly as the pinned
            # implementation does.  The graph-on path carries the matching
            # observed graph state into the same recursive imagination.
            "posterior_states": post_states,
        }
        if use_graph:
            stacked_correction = torch.stack(graph_corrections, dim=1)
            stacked_prior_deter = torch.stack(base_prior_deters, dim=1)
            correction_rms = stacked_correction.detach().square().mean().sqrt()
            prior_deter_rms = (
                stacked_prior_deter.detach().square().mean().sqrt()
            )
            output.update({
                "graph": torch.cat([item.graph for item in graph_outputs], dim=1),
                "graph_next": torch.cat([item.graph_next for item in graph_outputs], dim=1),
                "delta_g": torch.cat([item.delta_g for item in graph_outputs], dim=1),
                "relation_logits": torch.cat([item.relation_logits for item in graph_outputs], dim=1),
                "next_relation_logits": torch.cat([item.next_relation_logits for item in graph_outputs], dim=1),
                "graph_context": torch.cat([item.context for item in graph_outputs], dim=1),
                "ot_loss": torch.stack([item.ot_loss for item in graph_outputs]).mean(),
                "ot_variance_loss": torch.stack([item.ot_variance_loss for item in graph_outputs]).mean(),
                "ot_cross": torch.stack([item.ot_cross for item in graph_outputs]).mean(),
                "ot_self_z": torch.stack([item.ot_self_z for item in graph_outputs]).mean(),
                "ot_self_g": torch.stack([item.ot_self_g for item in graph_outputs]).mean(),
                "ot_z_proj_rms": torch.stack([item.ot_z_proj_rms for item in graph_outputs]).mean(),
                "ot_g_proj_rms": torch.stack([item.ot_g_proj_rms for item in graph_outputs]).mean(),
                "ot_z_dispersion": torch.stack([item.ot_z_dispersion for item in graph_outputs]).mean(),
                "ot_g_dispersion": torch.stack([item.ot_g_dispersion for item in graph_outputs]).mean(),
                "ot_z_pair_cos": torch.stack([item.ot_z_pair_cos for item in graph_outputs]).mean(),
                "ot_g_pair_cos": torch.stack([item.ot_g_pair_cos for item in graph_outputs]).mean(),
                "delta_z_rms": torch.stack([item.delta_z_rms for item in graph_outputs]).mean(),
                "delta_g_rms": torch.stack([item.delta_g_rms for item in graph_outputs]).mean(),
                "delta_z_dispersion": torch.stack([item.delta_z_dispersion for item in graph_outputs]).mean(),
                "delta_g_dispersion": torch.stack([item.delta_g_dispersion for item in graph_outputs]).mean(),
                "delta_z_pair_cos": torch.stack([item.delta_z_pair_cos for item in graph_outputs]).mean(),
                "delta_g_pair_cos": torch.stack([item.delta_g_pair_cos for item in graph_outputs]).mean(),
                "graph_context_rms": torch.stack([item.graph_context_rms for item in graph_outputs]).mean(),
                "graph_correction": stacked_correction,
                "prior_deter_rms": prior_deter_rms,
                "graph_correction_ratio": (
                    correction_rms / prior_deter_rms.clamp_min(1e-8)
                ),
                "graph_correction_gate": torch.sigmoid(self.graph_gate_logit).detach(),
                "graph_pool_attention_entropy": torch.stack(graph_pool_entropies, dim=1).mean().detach(),
                "target_graph": observed_graph["nodes"][:, 1:].detach(),
                "observed_graph_state": observed_graph.get(
                    "state",
                    self.graph_builder.teacher_state(observed_graph["nodes"]),
                ),
                "graph_state_prediction": self.graph_builder.teacher_state(torch.cat([item.graph_next for item in graph_outputs], dim=1)),
                "observed_graph_presence": observed_graph["presence_logits"],
                "graph_presence_prediction": self.graph_builder.presence(torch.cat([item.graph_next for item in graph_outputs], dim=1)).squeeze(-1),
                "observed_relation_logits": observed_graph["relation_logits"],
                "graph_attention": observed_graph["attention"],
                "graph_positions_2d": observed_graph["positions_2d"],
                "posterior_graph": observed_graph["nodes"][:, 1:],
                "posterior_relations": observed_graph["relation_logits"][:, 1:],
            })
        return output

    # ------------------------------------------------------------------
    # Plug-in (P) path. The RSSM runs exactly its graph-off computation;
    # the shared GraphDynamicsPlugin reads detached RSSM features and
    # corrects the prior deter through its global correction head.
    # ------------------------------------------------------------------

    def _stoch_mode(self, logits: Tensor) -> Tensor:
        return self.rssm.get_dist({"logits": logits}).mode()

    def _frozen_prior_logits(self, deter: Tensor) -> Tensor:
        """RSSM prior logits of ``deter`` with the prior head held fixed.

        ``mlp_img2`` belongs to the backbone. Plug-in losses must reach the
        correction through it without updating it, so it is evaluated with
        detached copies of its parameters (same values, no gradient).
        """
        head = self.rssm.mlp_img2
        state = {name: value.detach() for name, value in head.named_parameters()}
        state.update(head.named_buffers())
        logits = torch.func.functional_call(head, state, (deter,))
        return logits.reshape(*deter.shape[:-1], self.stoch_dim, self.classes)

    def _image_step(
        self, state: dict[str, Tensor], action: Tensor, deterministic: bool
    ) -> dict[str, Tensor]:
        """Graph-off RSSM prior step; mode stochastic state if deterministic."""
        prior = self.rssm.forward_img(state, action.clone())
        if deterministic:
            prior = {**prior, "stoch": self._stoch_mode(prior["logits"])}
        return prior

    def _observe_step(
        self, deter: Tensor, embed: Tensor, deterministic: bool
    ) -> dict[str, Tensor]:
        post = self.rssm.forward_obs(deter, embed)
        if deterministic:
            post = {**post, "stoch": self._stoch_mode(post["logits"])}
        return post

    def _start_posterior(
        self, embed: Tensor, deterministic: bool
    ) -> dict[str, Tensor]:
        post = self._initial_posterior(embed)
        if deterministic:
            post = {**post, "stoch": self._stoch_mode(post["logits"])}
        return post

    def _plugin_correct(
        self,
        prior_deter: Tensor,
        prior_feat: Tensor,
        base_delta: Tensor,
        context: Tensor,
        graph_next: Tensor,
        strength: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Corrected prior (deter, logits) and the correction itself.

        The correction reads the backbone's prediction detached (prior
        feature, its change from the previous state) and adds to the
        detached prior deter; the stochastic prior is recomputed from the
        corrected deter by the fixed prior head.
        """
        plugin = self.graph_plugin
        pose = plugin.pose_vector(graph_next) if plugin.pose_correction else None
        correction = plugin.correct_global(
            base_delta.detach(),
            context,
            prior_feat.detach(),
            pose=pose,
            strength=strength,
        )
        deter = prior_deter.detach() + correction
        return deter, self._frozen_prior_logits(deter), correction

    def forward_sequence_plugin(
        self,
        images: Tensor,
        actions: Tensor,
        *,
        sample_visual: bool,
        graph_multistep_horizon: int = 0,
        correction_unroll_horizon: int = 0,
    ) -> dict[str, Any]:
        """Plug-in (P) teacher-forced pass: graph-off world model + plug-in.

        The world model is ``forward_sequence(use_graph=False)`` itself, so
        its outputs, losses and gradients are exactly those of a graph-off
        run. The plug-in then reads detached RSSM features: graph dynamics
        and alignment see the prior's global change Δs, and the correction
        of the prior deter is trained on DreamerV3's own dynamics target,
        the (unclamped) KL from the detached posterior to the corrected
        prior (``plugin_corrected_kl``).
        """
        output = self.forward_sequence(
            images, actions, use_graph=False, sample_visual=sample_visual
        )
        plugin = self.graph_plugin
        posterior = output["posterior_states"]
        prior = output["prior_states"]
        prior_feat = output["predicted_tokens"].detach()
        # State before each step: the initial posterior, then the posterior
        # of the previous frame.
        previous_feat = torch.cat(
            (
                self.rssm.get_feat(output["initial_posterior"])[:, None],
                output["target_tokens"][:, :-1],
            ),
            dim=1,
        ).detach()
        base_delta = prior_feat - previous_feat
        # Detached: a proxy observer reads the encoder's tokens, and graph
        # losses must not reach the encoder (the explicit observer reads RGB).
        observed_graph = plugin.infer_sequence(
            output["all_tokens"].detach(), images
        )
        graph_output = plugin.forward_global(
            base_delta,
            actions,
            observed_graph["nodes"][:, :-1],
            observed_graph["relation_logits"][:, :-1],
        )
        graph_next = graph_output.graph_next
        _, logits, correction = self._plugin_correct(
            prior["deter"], prior_feat, base_delta, graph_output.context, graph_next
        )
        corrected_kl = torch.distributions.kl_divergence(
            self.rssm.get_dist(_detach_state(posterior)),
            self.rssm.get_dist({"logits": logits}),
        )
        correction_rms = correction.detach().square().mean().sqrt()
        prior_deter_rms = prior["deter"].detach().square().mean().sqrt()
        history = plugin.graph_history
        time = actions.shape[1]
        output.update({
            "base_delta": base_delta,
            "graph": graph_output.graph,
            "graph_next": graph_next,
            "delta_g": graph_output.delta_g,
            "relation_logits": graph_output.relation_logits,
            "next_relation_logits": graph_output.next_relation_logits,
            "graph_context": graph_output.context,
            "ot_loss": graph_output.ot_loss,
            "ot_variance_loss": graph_output.ot_variance_loss,
            "graph_correction": correction,
            "prior_deter_rms": prior_deter_rms,
            "graph_correction_ratio": (
                correction_rms / prior_deter_rms.clamp_min(1e-8)
            ),
            "graph_correction_gate": plugin.last_correction_gate,
            "graph_pool_attention_entropy": (
                graph_output.alignment_attention_entropy
            ),
            "plugin_corrected_kl": corrected_kl.mean(),
            "target_graph": observed_graph["nodes"][:, 1:].detach(),
            "observed_graph_state": observed_graph.get(
                "state",
                self.graph_builder.teacher_state(observed_graph["nodes"]),
            ),
            "graph_state_prediction": self.graph_builder.teacher_state(
                graph_next
            ),
            "observed_graph_presence": observed_graph["presence_logits"],
            "graph_presence_prediction": self.graph_builder.presence(
                graph_next
            ).squeeze(-1),
            "observed_relation_logits": observed_graph["relation_logits"],
            "graph_attention": observed_graph["attention"],
            "graph_positions_2d": observed_graph["positions_2d"],
            "posterior_graph": observed_graph["nodes"][:, 1:],
            "posterior_relations": observed_graph["relation_logits"][:, 1:],
            # Steps before this index have a padded (zero-velocity) graph
            # history window; graph-dynamics supervision skips them.
            "graph_prediction_start": (
                min(history - 1, time - 1) if history > 1 else 0
            ),
        })
        for name in (
            "ot_cross",
            "ot_self_z",
            "ot_self_g",
            "ot_z_proj_rms",
            "ot_g_proj_rms",
            "ot_z_dispersion",
            "ot_g_dispersion",
            "ot_z_pair_cos",
            "ot_g_pair_cos",
            "delta_z_rms",
            "delta_g_rms",
            "delta_z_dispersion",
            "delta_g_dispersion",
            "delta_z_pair_cos",
            "delta_g_pair_cos",
            "graph_context_rms",
        ):
            output[name] = getattr(graph_output, name)
        if plugin.pose_correction:
            output["graph_position_observed"] = plugin.node_position(
                observed_graph["nodes"]
            )
            output["graph_position_prediction"] = plugin.node_position(
                graph_next
            )
        if graph_multistep_horizon > 0:
            output.update(
                plugin.graph_multistep(
                    observed_graph["nodes"],
                    observed_graph["relation_logits"],
                    actions,
                    graph_multistep_horizon,
                )
            )
        if correction_unroll_horizon > 0:
            output.update(
                self._plugin_correction_unroll(
                    output["initial_posterior"],
                    posterior,
                    observed_graph,
                    actions,
                    correction_unroll_horizon,
                )
            )
        return output

    def _plugin_correction_unroll(
        self,
        initial_posterior: dict[str, Tensor],
        posterior: dict[str, Tensor],
        observed_graph: dict[str, Tensor],
        actions: Tensor,
        horizon: int,
    ) -> dict[str, Tensor]:
        """Train the correction on the model's own drifted open-loop rollout.

        Mirrors ``_plugin_rollout``: from the posterior of the first frame
        with a full graph history window (mode stochastic state), RSSM and
        graph run K steps on their own outputs with the corrected prior fed
        back, under no_grad, so only the correction learns. Each step's
        target is the teacher-forced posterior of that frame: the KL to the
        corrected prior plus 0.5 * ||deter - posterior deter||^2 (a unit
        Gaussian summed over dimensions, like the KL). The deter error
        matters only here: in the one-step case the posterior deter equals
        the prior deter, but along a rollout the drifted deter carries the
        error forward. Full strength, as in the token path.
        """
        plugin = self.graph_plugin
        history = plugin.graph_history
        time = actions.shape[1]
        start = min(history - 1, time - 1)
        horizon = min(horizon, time - start)
        with torch.no_grad():
            offsets = torch.arange(
                start - history + 1, start + 1, device=actions.device
            ).clamp_min(0)
            window = observed_graph["nodes"][:, offsets].detach()
            graph = window[:, -1]
            relations = observed_graph["relation_logits"][:, start].detach()
            source = (
                initial_posterior
                if start == 0
                else {key: value[:, start - 1] for key, value in posterior.items()}
            )
            state = _detach_state(source)
            state["stoch"] = self._stoch_mode(state["logits"])
        corrected_losses: list[Tensor] = []
        base_losses: list[Tensor] = []
        for step in range(horizon):
            index = start + step
            action = actions[:, index]
            # posterior[:, t] is the frame after action t.
            target_logits = posterior["logits"][:, index].detach()
            target_deter = posterior["deter"][:, index].detach()
            target = self.rssm.get_dist({"logits": target_logits})
            with torch.no_grad():
                prior = self._image_step(state, action, deterministic=True)
                prior_feat = self.rssm.get_feat(prior)
                base_delta = prior_feat - self.rssm.get_feat(state)
                graph_output = plugin.forward_global(
                    base_delta[:, None],
                    action[:, None],
                    graph[:, None],
                    relations[:, None],
                    graph_window=window[:, None] if history > 1 else None,
                )
                base_losses.append(
                    torch.distributions.kl_divergence(
                        target, self.rssm.get_dist(prior)
                    )
                    + 0.5 * (prior["deter"] - target_deter).square().sum(-1)
                )
            # Outside no_grad: the pose payload belongs to the correction.
            deter, logits, _ = self._plugin_correct(
                prior["deter"],
                prior_feat,
                base_delta,
                graph_output.context[:, 0],
                graph_output.graph_next[:, 0],
                strength=1.0,
            )
            corrected_losses.append(
                torch.distributions.kl_divergence(
                    target, self.rssm.get_dist({"logits": logits})
                )
                + 0.5 * (deter - target_deter).square().sum(-1)
            )
            logits = logits.detach()
            state = {
                "deter": deter.detach(),
                "logits": logits,
                "stoch": self._stoch_mode(logits),
            }
            graph = graph_output.graph_next[:, 0]
            relations = graph_output.next_relation_logits[:, 0]
            if history > 1:
                window = torch.cat((window[:, 1:], graph[:, None]), dim=1)
        return {
            "correction_unroll_loss": torch.stack(corrected_losses, 1).mean(),
            "correction_unroll_base_loss": torch.stack(base_losses, 1).mean(),
        }

    @torch.no_grad()
    def _plugin_rollout(
        self,
        initial_image: Tensor,
        actions: Tensor,
        *,
        use_graph: bool,
        decode: bool,
        context_images: Tensor | None,
        context_actions: Tensor | None,
        visual_context: bool,
        deterministic: bool,
    ) -> dict[str, Tensor]:
        """Open-loop prior imagination of a plug-in model, graph on or off.

        Same protocol as the DINO-WM rollout. The RSSM starts from the
        posterior of ``initial_image``; with ``visual_context`` it first
        observes the ``context_images`` (and ``context_actions`` between
        them) through its posterior, the DreamerV3 equivalent of giving the
        backbone real history. Context frames always seed the graph history
        window. Deterministic rollouts use the mode stochastic state, so at
        correction strength 0 the graph-on rollout equals the graph-off one.
        """
        plugin = self.graph_plugin
        history = plugin.graph_history
        if visual_context and context_images is not None and context_images.shape[1]:
            if (
                context_actions is None
                or context_actions.shape[1] != context_images.shape[1]
            ):
                raise ValueError("visual_context needs one action per context frame")
            frames = torch.cat((context_images, initial_image[:, None]), dim=1)
            _, embeds, _, _ = self._encode(frames, sample=False)
            state = self._start_posterior(embeds[:, 0], deterministic)
            for index in range(1, frames.shape[1]):
                prior = self._image_step(
                    state, context_actions[:, index - 1], deterministic
                )
                state = self._observe_step(
                    prior["deter"], embeds[:, index], deterministic
                )
        else:
            _, embeds, _, _ = self._encode(initial_image[:, None], sample=False)
            state = self._start_posterior(embeds[:, 0], deterministic)
        graph = relations = window = None
        if use_graph:
            if history == 1:
                tokens = self._encode(initial_image[:, None], sample=False)[0]
                observation = plugin.observe(
                    tokens[:, 0], initial_image, sample=False
                )
                graph = observation.nodes
                relations = observation.relation_logits
            else:
                seed_images = initial_image[:, None]
                if context_images is not None:
                    seed_images = torch.cat(
                        (context_images[:, -(history - 1):], seed_images), dim=1
                    )
                seed_frames = seed_images.shape[1]
                seed_tokens = self._encode(seed_images, sample=False)[0]
                observed = plugin.infer_sequence(seed_tokens, seed_images)
                offsets = torch.arange(
                    seed_frames - history, seed_frames, device=actions.device
                ).clamp_min(0)
                window = observed["nodes"][:, offsets]
                graph = window[:, -1]
                relations = observed["relation_logits"][:, -1]
        features: list[Tensor] = []
        images: list[Tensor] = []
        rewards: list[Tensor] = []
        continuations: list[Tensor] = []
        graphs: list[Tensor] = []
        relation_steps: list[Tensor] = []
        for step in range(actions.shape[1]):
            action = actions[:, step]
            prior = self._image_step(state, action, deterministic)
            if use_graph:
                prior_feat = self.rssm.get_feat(prior)
                base_delta = prior_feat - self.rssm.get_feat(state)
                graph_output = plugin.forward_global(
                    base_delta[:, None],
                    action[:, None],
                    graph[:, None],
                    relations[:, None],
                    graph_window=window[:, None] if window is not None else None,
                )
                deter, logits, _ = self._plugin_correct(
                    prior["deter"],
                    prior_feat,
                    base_delta,
                    graph_output.context[:, 0],
                    graph_output.graph_next[:, 0],
                )
                stoch = (
                    self._stoch_mode(logits)
                    if deterministic
                    else self.rssm.get_dist({"logits": logits}).sample()
                )
                prior = {"stoch": stoch, "deter": deter, "logits": logits}
                graph = graph_output.graph_next[:, 0]
                relations = graph_output.next_relation_logits[:, 0]
                if window is not None:
                    window = torch.cat((window[:, 1:], graph[:, None]), dim=1)
                graphs.append(graph)
                relation_steps.append(relations)
            feature = self.rssm.get_feat(prior)
            features.append(feature)
            if decode:
                images.append(self._decode_feature(feature)[0])
            rewards.append(self.reward_head(feature).mode().squeeze(-1))
            continuations.append(
                self.continuation_head(feature).base_dist.probs.squeeze(-1)
            )
            state = prior
        result = {
            "tokens": torch.stack(features, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "continuations": torch.stack(continuations, dim=1),
        }
        if decode:
            result["images"] = torch.stack(images, dim=1)
        if use_graph:
            result["graphs"] = torch.stack(graphs, dim=1)
            result["relations"] = torch.stack(relation_steps, dim=1)
        return result

    def forward_online_sequence(
        self,
        images: Tensor,
        previous_actions: Tensor,
        firsts: Tensor,
        *,
        use_graph: bool,
    ) -> dict[str, Any]:
        """Burchim-aligned replay forward over ``L`` observations.

        Each replay row contains the current observation and the action that
        led to it.  The first action is zero at an episode boundary.  Online
        replay windows never cross boundaries, so the first posterior is
        initialized from observation zero and the remaining ``L-1`` priors
        correspond exactly to ``previous_actions[:, 1:]``.
        """
        if images.ndim != 5 or previous_actions.ndim != 3:
            raise ValueError("Expected images [B,L,C,H,W] and actions [B,L,A]")
        if images.shape[:2] != previous_actions.shape[:2]:
            raise ValueError("Online image/action sequence lengths differ")
        if firsts.shape != images.shape[:2]:
            raise ValueError("Online is_first sequence shape differs")
        tokens, embeds, mu, logvar = self._encode(images, sample=False)
        observed_graph = (
            self.graph_plugin.infer_sequence(tokens, images, firsts=firsts)
            if use_graph
            else None
        )
        initial = self.rssm.initial(
            batch_size=images.shape[0],
            dtype=embeds.dtype,
            device=embeds.device,
        )
        initial_prior = self.rssm.forward_img(
            initial, previous_actions[:, 0].clone().zero_()
        )
        current = self.rssm.forward_obs(
            initial_prior["deter"], embeds[:, 0]
        )
        posts = [current]
        priors = [initial_prior]
        graph_outputs: list[Any] = []
        base_deltas: list[Tensor] = []
        graph_corrections: list[Tensor] = []
        base_prior_deters: list[Tensor] = []
        pool_entropies: list[Tensor] = []
        for step in range(1, images.shape[1]):
            graph = (
                observed_graph["nodes"][:, step - 1]
                if observed_graph is not None
                else None
            )
            relations = (
                observed_graph["relation_logits"][:, step - 1]
                if observed_graph is not None
                else None
            )
            (
                prior,
                graph_output,
                base_delta,
                correction,
                entropy,
                base_prior_deter,
            ) = self._prior_step(
                current,
                previous_actions[:, step],
                graph,
                relations,
                use_graph,
            )
            reset = firsts[:, step].to(embeds.dtype)
            if bool(reset.any()):
                reset_state = self.rssm.initial(
                    batch_size=images.shape[0],
                    dtype=embeds.dtype,
                    device=embeds.device,
                )
                reset_prior = self.rssm.forward_img(
                    reset_state, previous_actions[:, step].clone().zero_()
                )
                for key in prior:
                    mask = reset.reshape(
                        reset.shape + (1,) * (prior[key].ndim - reset.ndim)
                    )
                    prior[key] = (
                        prior[key] * (1.0 - mask) + reset_prior[key] * mask
                    )
            current = self.rssm.forward_obs(prior["deter"], embeds[:, step])
            priors.append(prior)
            posts.append(current)
            base_deltas.append(base_delta)
            graph_corrections.append(correction)
            base_prior_deters.append(base_prior_deter)
            pool_entropies.append(entropy)
            if graph_output is not None:
                graph_outputs.append(graph_output)
        post_states = _stack_states(posts)
        prior_states = _stack_states(priors)
        post_feat = self.rssm.get_feat(post_states)
        prior_feat = self.rssm.get_feat(prior_states)
        reconstruction, reconstruction_dist = self._decode_feature(post_feat)
        reward_dist = self.reward_head(post_feat)
        continuation_dist = self.continuation_head(post_feat)
        dynamics_kl_raw = torch.distributions.kl_divergence(
            self.rssm.get_dist(_detach_state(post_states)),
            self.rssm.get_dist(prior_states),
        )
        representation_kl_raw = torch.distributions.kl_divergence(
            self.rssm.get_dist(post_states),
            self.rssm.get_dist(_detach_state(prior_states)),
        )
        output: dict[str, Any] = {
            "loss_family": "dreamerv3_online",
            "posterior_states": post_states,
            "posterior_features": post_feat,
            "prior_features": prior_feat,
            "reconstruction": reconstruction,
            "reconstruction_distribution": reconstruction_dist,
            "reward_distribution": reward_dist,
            "continuation_distribution": continuation_dist,
            "dynamics_kl": dynamics_kl_raw.clamp_min(self.free_nats).mean(),
            "dynamics_kl_raw": dynamics_kl_raw.mean(),
            "representation_kl": representation_kl_raw.clamp_min(
                self.free_nats
            ).mean(),
            "representation_kl_raw": representation_kl_raw.mean(),
            "z_mu": mu,
            "z_logvar": logvar,
            "alignment_mode": "global",
            "transition_mask": 1.0 - firsts[:, 1:],
        }
        if use_graph and observed_graph is not None:
            output.update({
                "posterior_graph": observed_graph["nodes"],
                "posterior_relations": observed_graph["relation_logits"],
                "graph_outputs": graph_outputs,
                "graph": torch.cat([x.graph for x in graph_outputs], 1),
                "graph_next": torch.cat(
                    [x.graph_next for x in graph_outputs], 1
                ),
                "target_graph": observed_graph["nodes"][:, 1:].detach(),
                "ot_loss": torch.stack(
                    [x.ot_loss for x in graph_outputs]
                ).mean(),
                "alignment_by_step": torch.stack(
                    [x.ot_loss for x in graph_outputs]
                ),
                "ot_variance_loss": torch.stack(
                    [x.ot_variance_loss for x in graph_outputs]
                ).mean(),
                "alignment_variance_by_step": torch.stack(
                    [x.ot_variance_loss for x in graph_outputs]
                ),
                "graph_correction": torch.stack(
                    graph_corrections, dim=1
                ),
                "base_prior_deter": torch.stack(
                    base_prior_deters, dim=1
                ),
                "graph_pool_attention_entropy": torch.stack(
                    pool_entropies, dim=1
                ).mean(),
            })
        else:
            zero = images.new_zeros(())
            output["ot_loss"] = zero
            output["ot_variance_loss"] = zero
        return output

    def imagine_behavior(
        self,
        policy: nn.Module,
        start_state: dict[str, Tensor],
        horizon: int,
        *,
        use_graph: bool,
        start_graph: Tensor | None = None,
        start_relations: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Differentiable Dreamer imagination for online actor training.

        Policy inputs are detached, matching burchim ``RSSM.imagine``.  The
        sampled action remains reparameterized, so the dynamics-gradient
        actor objective still differentiates through RSSM transitions.  For
        JointGV, graph state and relation state recurse alongside the RSSM.
        """
        if horizon < 1:
            raise ValueError("imagination horizon must be positive")
        current = start_state
        graph = start_graph
        relations = start_relations
        states = [current]
        features = [self.rssm.get_feat(current)]
        actions: list[Tensor] = []
        rewards: list[Tensor] = []
        continuations: list[Tensor] = []
        graphs: list[Tensor] = []
        for _ in range(horizon):
            action = policy(features[-1].detach()).rsample()
            next_state, graph_output, _, _, _, _ = self._prior_step(
                current, action, graph, relations, use_graph
            )
            next_feature = self.rssm.get_feat(next_state)
            actions.append(action)
            states.append(next_state)
            features.append(next_feature)
            rewards.append(self.reward_head(next_feature).mode())
            continuations.append(
                self.continuation_head(next_feature).base_dist.probs
            )
            if graph_output is not None:
                graph = graph_output.graph_next[:, 0]
                relations = graph_output.next_relation_logits[:, 0]
                graphs.append(graph)
            current = next_state
        result = {
            "features": torch.stack(features, dim=1),
            "actions": torch.stack(actions, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "continuations": torch.stack(continuations, dim=1),
        }
        if graphs:
            result["graphs"] = torch.stack(graphs, dim=1)
        return result

    @torch.no_grad()
    def policy_observe(
        self,
        image: Tensor,
        previous_state: dict[str, Tensor] | None,
        previous_action: Tensor | None,
        *,
        use_graph: bool,
        previous_graph: Tensor | None = None,
        previous_relations: Tensor | None = None,
        reset_mask: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor | None, Tensor | None]:
        """Update the recurrent state from one real environment image."""
        tokens, embeds, _, _ = self._encode(image[:, None], sample=False)
        embed = embeds[:, 0]
        if previous_state is None:
            initial = self.rssm.initial(
                batch_size=embed.shape[0],
                dtype=embed.dtype,
                device=embed.device,
            )
            zero_action = embed.new_zeros(embed.shape[0], self.action_dim)
            prior = self.rssm.forward_img(initial, zero_action)
            state = self.rssm.forward_obs(prior["deter"], embed)
        else:
            if previous_action is None:
                raise ValueError("previous_action is required after reset")
            prior, _, _, _, _, _ = self._prior_step(
                previous_state,
                previous_action,
                previous_graph,
                previous_relations,
                use_graph,
            )
            if reset_mask is not None and bool(reset_mask.any()):
                initial = self.rssm.initial(
                    batch_size=embed.shape[0],
                    dtype=embed.dtype,
                    device=embed.device,
                )
                zero_action = embed.new_zeros(embed.shape[0], self.action_dim)
                reset_prior = self.rssm.forward_img(initial, zero_action)
                mask = reset_mask.to(embed.dtype)[:, None]
                for key in prior:
                    expanded = mask.reshape(
                        mask.shape + (1,) * (prior[key].ndim - mask.ndim)
                    )
                    prior[key] = (
                        prior[key] * (1.0 - expanded)
                        + reset_prior[key] * expanded
                    )
            state = self.rssm.forward_obs(prior["deter"], embed)
        graph = relations = None
        if use_graph:
            observation = self.graph_plugin.observe(
                tokens[:, 0],
                image,
                previous_slots=previous_graph,
                sample=False,
                reset_mask=reset_mask,
            )
            graph = observation.nodes
            relations = observation.relation_logits
        return state, self.rssm.get_feat(state), graph, relations

    @torch.no_grad()
    def rollout(
        self,
        initial_image: Tensor,
        actions: Tensor,
        *,
        use_graph: bool,
        decode: bool,
        context_images: Tensor | None = None,
        context_actions: Tensor | None = None,
        visual_context: bool = False,
        deterministic: bool = True,
    ) -> dict[str, Tensor]:
        """Open-loop imagination. Plug-in models use ``_plugin_rollout``;
        historical models keep this path, which samples the stochastic state
        and ignores the context and ``deterministic`` arguments."""
        if self.global_correction:
            return self._plugin_rollout(
                initial_image,
                actions,
                use_graph=use_graph,
                decode=decode,
                context_images=context_images,
                context_actions=context_actions,
                visual_context=visual_context,
                deterministic=deterministic,
            )
        tokens, embeds, _, _ = self._encode(initial_image[:, None], False)
        current = self._initial_posterior(embeds[:, 0])
        if use_graph:
            observation = self.graph_plugin.observe(
                tokens[:, 0], initial_image, sample=False
            )
            graph, relations = observation.nodes, observation.relation_logits
        else:
            graph = relations = None
        token_rollout: list[Tensor] = []
        image_rollout: list[Tensor] = []
        graph_rollout: list[Tensor] = []
        relation_rollout: list[Tensor] = []
        rewards: list[Tensor] = []
        continuations: list[Tensor] = []
        for step in range(actions.shape[1]):
            prior, graph_output, _, _, _, _ = self._prior_step(
                current, actions[:, step], graph, relations, use_graph
            )
            feature = self.rssm.get_feat(prior)
            image = None
            if decode:
                image, _ = self._decode_feature(feature)
            reward = self.reward_head(feature).mode().squeeze(-1)
            continuation = self.continuation_head(feature).base_dist.probs.squeeze(-1)
            token_rollout.append(feature)
            if image is not None:
                image_rollout.append(image)
            rewards.append(reward)
            continuations.append(continuation)
            if graph_output is not None:
                graph = graph_output.graph_next[:, 0]
                relations = graph_output.next_relation_logits[:, 0]
                graph_rollout.append(graph)
                relation_rollout.append(relations)
            current = prior
        result = {
            "tokens": torch.stack(token_rollout, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "continuations": torch.stack(continuations, dim=1),
        }
        if decode:
            result["images"] = torch.stack(image_rollout, dim=1)
        if use_graph:
            result["graphs"] = torch.stack(graph_rollout, dim=1)
            result["relations"] = torch.stack(relation_rollout, dim=1)
        return result
