from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from .backbones import MDNOutput, mdn_nll


@dataclass
class JointLoss:
    total: Tensor
    transition: Tensor
    base_transition: Tensor
    reconstruction: Tensor
    vae_reconstruction: Tensor
    graph: Tensor
    graph_supervision: Tensor
    ot: Tensor
    ot_variance: Tensor
    vae_kl: Tensor
    visual_latent_rms: Tensor
    visual_latent_absmax: Tensor
    dynamics_kl: Tensor
    dynamics_kl_raw: Tensor
    representation_kl: Tensor
    representation_kl_raw: Tensor
    reward: Tensor
    continuation: Tensor
    # K-step self-fed graph terms; only the generic path computes them, so
    # other backbones keep reporting zero without changing their call sites.
    graph_multistep_embedding: Tensor = field(
        default_factory=lambda: torch.zeros(())
    )
    graph_multistep_state: Tensor = field(
        default_factory=lambda: torch.zeros(())
    )
    # K-step self-fed visual token loss (backbone's own prediction).
    visual_multistep: Tensor = field(
        default_factory=lambda: torch.zeros(())
    )
    # Node 2D position head vs graph_node_pixel (pose correction only).
    graph_position: Tensor = field(
        default_factory=lambda: torch.zeros(())
    )
    # Correction-unroll token error with the correction (trained) and,
    # for reference, of the same steps' uncorrected predictions (logged
    # only, never weighted).
    correction_unroll: Tensor = field(
        default_factory=lambda: torch.zeros(())
    )
    correction_unroll_base: Tensor = field(
        default_factory=lambda: torch.zeros(())
    )
    # Plug-in mode: token + image error of the corrected prediction, which
    # trains only the correction (the backbone's own terms stay graph-off).
    plugin_correction: Tensor = field(
        default_factory=lambda: torch.zeros(())
    )
    # Logged only: single-step token error of the graph-corrected prediction,
    # defined exactly like ``transition`` (which, in plug-in mode, is the
    # backbone's own prediction). Joint mode: equal to ``transition``.
    corrected_transition: Tensor = field(
        default_factory=lambda: torch.zeros(())
    )

    @property
    def mdn(self) -> Tensor:
        return self.transition

    @property
    def base_mdn(self) -> Tensor:
        return self.base_transition

    @property
    def kl(self) -> Tensor:
        return self.vae_kl


def _graph_supervision(
    output: dict[str, Any], targets: dict[str, Tensor] | None
) -> Tensor:
    graph_next = output["graph_next"]
    zero = graph_next.new_zeros(())
    if targets is None or "node_state" not in targets:
        return zero
    node_state = targets["node_state"]
    node_mask = targets["node_mask"].to(node_state.dtype)
    relation = targets["relation"].long()
    observed_state = output["observed_graph_state"]
    predicted_state = output["graph_state_prediction"]
    observed_presence = output["observed_graph_presence"]
    predicted_presence = output["graph_presence_prediction"]
    observed_relations = output["observed_relation_logits"]
    predicted_relations = output["next_relation_logits"]
    # Steps before this index had a padded (fabricated zero-velocity) graph
    # history window; the dynamics predictions there are not supervised.
    # Observer terms below are per-frame and keep every step.
    start = int(output.get("graph_prediction_start", 0))
    if node_state.shape[-2:] != observed_state.shape[-2:]:
        raise ValueError(
            "MuJoCo node_state must match graph_nodes and teacher_state_dim"
        )

    def masked_smooth_l1(
        prediction: Tensor, target: Tensor, mask: Tensor
    ) -> Tensor:
        values = F.smooth_l1_loss(
            prediction, target, reduction="none"
        ).mean(-1)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    state_loss = masked_smooth_l1(
        observed_state, node_state, node_mask
    ) + masked_smooth_l1(
        predicted_state[:, start:],
        node_state[:, 1 + start :],
        node_mask[:, 1 + start :],
    )
    presence_loss = F.binary_cross_entropy_with_logits(
        observed_presence, node_mask
    ) + F.binary_cross_entropy_with_logits(
        predicted_presence[:, start:], node_mask[:, 1 + start :]
    )
    pixel_loss = zero
    if "node_pixel" in targets and "graph_positions_2d" in output:
        pixel_target = targets["node_pixel"]
        visible = targets.get("node_visible", node_mask).to(node_state.dtype)
        pixel_values = F.smooth_l1_loss(
            output["graph_positions_2d"], pixel_target, reduction="none"
        ).mean(-1)
        pixel_loss = (pixel_values * visible).sum() / visible.sum().clamp_min(1.0)
    pair_mask = (node_mask[..., :, None] * node_mask[..., None, :]).bool()
    diagonal = torch.arange(pair_mask.shape[-1], device=pair_mask.device)
    pair_mask[:, :, diagonal, diagonal] = False
    if pair_mask.any():
        observed_relation_loss = F.cross_entropy(
            observed_relations[pair_mask], relation[pair_mask]
        )
    else:
        observed_relation_loss = zero
    next_pair_mask = pair_mask[:, 1 + start :]
    if next_pair_mask.any():
        predicted_relation_loss = F.cross_entropy(
            predicted_relations[:, start:][next_pair_mask],
            relation[:, 1 + start :][next_pair_mask],
        )
    else:
        predicted_relation_loss = zero
    return (
        state_loss
        + pixel_loss
        + 0.25 * presence_loss
        + observed_relation_loss
        + predicted_relation_loss
    )


def compute_sequence_loss(
    model,
    output: dict[str, Any],
    images: Tensor,
    beta_kl: float = 1e-3,
    dynamics_kl_weight: float = 0.1,
    base_transition_weight: float = 0.25,
    base_mdn_weight: float | None = None,
    graph_weight: float = 0.2,
    ot_weight: float = 0.1,
    ot_variance_weight: float = 0.1,
    prediction_weight: float = 1.0,
    vae_reconstruction_weight: float = 0.5,
    graph_targets: dict[str, Tensor] | None = None,
    targets: dict[str, Tensor] | None = None,
    graph_supervision_weight: float = 1.0,
    reward_weight: float = 1.0,
    continuation_weight: float = 0.1,
    motion_weighted_transition: bool = False,
    graph_multistep_embedding_weight: float = 0.0,
    graph_multistep_state_weight: float = 0.0,
    visual_multistep_weight: float = 0.0,
    graph_position_weight: float = 0.0,
    correction_unroll_weight: float = 0.0,
    plugin_correction_weight: float = 0.0,
) -> JointLoss:
    if output.get("loss_family") == "dreamerv3":
        zero = output["dynamics_kl"].new_zeros(())
        dynamics_kl = output["dynamics_kl"]
        dynamics_kl_raw = output["dynamics_kl_raw"]
        representation_kl = output["representation_kl"]
        representation_kl_raw = output["representation_kl_raw"]
        if output["reconstruction_distribution"] is not None:
            reconstruction = -output["reconstruction_distribution"].log_prob(
                images[:, 1:] - 0.5
            ).mean()
        else:
            reconstruction = F.mse_loss(
                output["vae_reconstruction"], images[:, 1:]
            )
        reward = zero
        continuation = zero
        merged_targets = targets if targets is not None else graph_targets
        if merged_targets is not None and "reward" in merged_targets:
            reward = -output["reward_distribution"].log_prob(
                merged_targets["reward"].unsqueeze(-1)
            ).mean()
        if merged_targets is not None and "continuation" in merged_targets:
            continuation = -output["continuation_distribution"].log_prob(
                merged_targets["continuation"].unsqueeze(-1)
            ).mean()
        graph_multistep_embedding = graph_multistep_state = zero
        graph_position = zero
        if "graph_next" in output:
            # graph_prediction_start is 0 except on the plug-in path with
            # graph history, where early steps have a padded window.
            graph_start = int(output.get("graph_prediction_start", 0))
            graph = F.mse_loss(
                output["graph_next"][:, graph_start:],
                output["target_graph"][:, graph_start:],
            )
            graph_supervision = _graph_supervision(output, merged_targets)
            graph_multistep_embedding, graph_multistep_state = _graph_multistep(
                output, merged_targets
            )
            graph_position = _graph_position(output, merged_targets)
        else:
            graph = zero
            graph_supervision = zero
        # Plug-in (P) path only: the corrected prior's KL to the posterior
        # (one step) and along the correction unroll; both reach only the
        # correction. Absent on the historical path, where they stay zero.
        plugin_correction = output.get("plugin_corrected_kl", zero)
        correction_unroll = output.get("correction_unroll_loss", zero)
        correction_unroll_base = output.get(
            "correction_unroll_base_loss", zero
        ).detach()
        if output.get("visual_is_variational", False):
            z_mu, z_logvar = output["z_mu"], output["z_logvar"]
            vae_kl = -0.5 * (
                1 + z_logvar - z_mu.square() - z_logvar.exp()
            ).mean()
        else:
            vae_kl = zero
        visual_latent_rms = output["z_mu"].detach().square().mean().sqrt()
        visual_latent_absmax = output["z_mu"].detach().abs().amax()
        transition = (
            0.5 * dynamics_kl + 0.1 * representation_kl
        )
        ot_variance = output.get("ot_variance_loss", zero)
        total = (
            reconstruction
            + transition
            + reward
            + continuation
            + graph_weight * graph
            + graph_supervision_weight * graph_supervision
            + ot_weight * output["ot_loss"]
            + ot_variance_weight * ot_variance
            + beta_kl * vae_kl
            + graph_multistep_embedding_weight * graph_multistep_embedding
            + graph_multistep_state_weight * graph_multistep_state
            + graph_position_weight * graph_position
            + correction_unroll_weight * correction_unroll
            + plugin_correction_weight * plugin_correction
        )
        return JointLoss(
            total=total,
            transition=transition,
            base_transition=transition,
            reconstruction=reconstruction,
            vae_reconstruction=zero,
            graph=graph,
            graph_supervision=graph_supervision,
            ot=output["ot_loss"],
            ot_variance=ot_variance,
            vae_kl=vae_kl,
            visual_latent_rms=visual_latent_rms,
            visual_latent_absmax=visual_latent_absmax,
            dynamics_kl=dynamics_kl,
            dynamics_kl_raw=dynamics_kl_raw,
            representation_kl=representation_kl,
            representation_kl_raw=representation_kl_raw,
            reward=reward,
            continuation=continuation,
            graph_multistep_embedding=graph_multistep_embedding,
            graph_multistep_state=graph_multistep_state,
            graph_position=graph_position,
            correction_unroll=correction_unroll,
            correction_unroll_base=correction_unroll_base,
            plugin_correction=plugin_correction,
            # Raw one-step KL of the prediction that is rolled out: the
            # corrected prior on the plug-in path, otherwise the prior.
            corrected_transition=(
                plugin_correction.detach()
                if "plugin_corrected_kl" in output
                else dynamics_kl_raw.detach()
            ),
        )
    if base_mdn_weight is not None:
        base_transition_weight = base_mdn_weight
    target_tokens = output["target_tokens"].detach()
    mdn = output.get("mdn")
    if isinstance(mdn, MDNOutput):
        transition = mdn_nll(
            mdn, target_tokens, output["conditioned_mean"]
        )
        base_transition = mdn_nll(mdn, target_tokens)
    elif motion_weighted_transition:
        # A uniform per-token average lets a large, nearly-static background
        # (most patches) swamp the few patches covering a small, fast-moving
        # foreground object -- the model can reach a low aggregate loss while
        # barely tracking that object's motion. Reweight by each token's own
        # ground-truth temporal change (unrelated to the model's prediction,
        # so no gradient should flow through it) so patches that are actually
        # moving drive more of the gradient than patches that are not; +1e-3
        # keeps static tokens contributing a little rather than exactly zero,
        # and dividing by the per-frame mean keeps the loss on the same scale
        # as the unweighted version.
        source_tokens = output["z"].detach()
        motion = (target_tokens - source_tokens).norm(dim=-1) + 1e-3
        weight = (motion / motion.mean(dim=-1, keepdim=True)).detach()
        transition = (
            F.smooth_l1_loss(
                output["predicted_tokens"], target_tokens, reduction="none"
            ).mean(dim=-1)
            * weight
        ).mean()
        base_transition = (
            F.smooth_l1_loss(
                output["base_predicted_tokens"], target_tokens, reduction="none"
            ).mean(dim=-1)
            * weight
        ).mean()
    else:
        transition = F.smooth_l1_loss(
            output["predicted_tokens"], target_tokens
        )
        base_transition = F.smooth_l1_loss(
            output["base_predicted_tokens"], target_tokens
        )
    reconstruction = F.mse_loss(
        output["predicted_image"], images[:, 1:]
    )
    vae_reconstruction = F.mse_loss(
        output["vae_reconstruction"], images
    )
    graph_start = int(output.get("graph_prediction_start", 0))
    graph = F.mse_loss(
        output["graph_next"][:, graph_start:],
        output["target_graph"][:, graph_start:],
    )
    merged_targets = targets if targets is not None else graph_targets
    # Skip the observer-state/presence/relation readouts whenever the result
    # would be zero-weighted anyway (graph conditioning disabled, or a
    # dataset with no graph targets in the first place -- ``_graph_supervision``
    # already short-circuits that second case internally, so this check only
    # ever skips work that was always going to be discarded).
    graph_supervision = (
        _graph_supervision(output, merged_targets)
        if graph_supervision_weight != 0.0
        else output["graph_next"].new_zeros(())
    )
    z_mu, z_logvar = output["z_mu"], output["z_logvar"]
    if output.get("visual_is_variational", True):
        vae_kl = -0.5 * (
            1 + z_logvar - z_mu.square() - z_logvar.exp()
        ).mean()
    else:
        # DreamerV3-native visual embeddings are deterministic. Their
        # stochastic regularizer belongs to the discrete RSSM and is reported
        # separately as ``dynamics_kl``.
        vae_kl = transition.new_zeros(())
    visual_latent_rms = z_mu.detach().square().mean().sqrt()
    visual_latent_absmax = z_mu.detach().abs().amax()
    dynamics_kl = output.get(
        "dynamics_kl", transition.new_zeros(())
    )
    dynamics_kl_raw = output.get(
        "dynamics_kl_raw", transition.new_zeros(())
    )
    representation_kl = output.get(
        "representation_kl", transition.new_zeros(())
    )
    representation_kl_raw = output.get(
        "representation_kl_raw", transition.new_zeros(())
    )
    reward = transition.new_zeros(())
    continuation = transition.new_zeros(())
    ot_variance = output.get(
        "ot_variance_loss", transition.new_zeros(())
    )
    if merged_targets is not None and "reward" in merged_targets:
        reward = F.mse_loss(
            output["reward_prediction"], merged_targets["reward"]
        )
    if merged_targets is not None and "continuation" in merged_targets:
        continuation = F.binary_cross_entropy_with_logits(
            output["continuation_logit"],
            merged_targets["continuation"],
        )
    graph_multistep_embedding, graph_multistep_state = _graph_multistep(
        output, merged_targets
    )
    visual_multistep = _visual_multistep(output, motion_weighted_transition)
    graph_position = _graph_position(output, merged_targets)
    correction_unroll, correction_unroll_base = _correction_unroll(
        output, motion_weighted_transition
    )
    plugin_correction = transition.new_zeros(())
    corrected_transition = transition.detach()
    if "plugin_corrected_tokens" in output:
        # Same token and image terms the joint path puts on its corrected
        # prediction; the tokens are built from a detached prediction.
        corrected_tokens_error = _motion_weighted_token_error(
            output["plugin_corrected_tokens"],
            target_tokens,
            output["z"].detach(),
            motion_weighted_transition,
        )
        corrected_transition = corrected_tokens_error.detach()
        plugin_correction = corrected_tokens_error + prediction_weight * F.mse_loss(
            output["plugin_corrected_image"], images[:, 1:]
        )
    total = (
        transition
        + base_transition_weight * base_transition
        + prediction_weight * reconstruction
        + vae_reconstruction_weight * vae_reconstruction
        + graph_weight * graph
        + graph_supervision_weight * graph_supervision
        + ot_weight * output["ot_loss"]
        + ot_variance_weight * ot_variance
        + beta_kl * vae_kl
        + dynamics_kl_weight * dynamics_kl
        + reward_weight * reward
        + continuation_weight * continuation
        + graph_multistep_embedding_weight * graph_multistep_embedding
        + graph_multistep_state_weight * graph_multistep_state
        + visual_multistep_weight * visual_multistep
        + graph_position_weight * graph_position
        + correction_unroll_weight * correction_unroll
        + plugin_correction_weight * plugin_correction
    )
    return JointLoss(
        total=total,
        transition=transition,
        base_transition=base_transition,
        reconstruction=reconstruction,
        vae_reconstruction=vae_reconstruction,
        graph=graph,
        graph_supervision=graph_supervision,
        ot=output["ot_loss"],
        ot_variance=ot_variance,
        vae_kl=vae_kl,
        visual_latent_rms=visual_latent_rms,
        visual_latent_absmax=visual_latent_absmax,
        dynamics_kl=dynamics_kl,
        dynamics_kl_raw=dynamics_kl_raw,
        representation_kl=representation_kl,
        representation_kl_raw=representation_kl_raw,
        reward=reward,
        continuation=continuation,
        graph_multistep_embedding=graph_multistep_embedding,
        graph_multistep_state=graph_multistep_state,
        visual_multistep=visual_multistep,
        graph_position=graph_position,
        correction_unroll=correction_unroll,
        correction_unroll_base=correction_unroll_base,
        plugin_correction=plugin_correction,
        corrected_transition=corrected_transition,
    )


def _motion_weighted_token_error(
    predicted: Tensor, target: Tensor, source: Tensor, motion_weighted: bool
) -> Tensor:
    """Smooth L1 token error, optionally weighted like ``transition``."""
    if not motion_weighted:
        return F.smooth_l1_loss(predicted, target)
    motion = (target - source).norm(dim=-1) + 1e-3
    weight = (motion / motion.mean(dim=-1, keepdim=True)).detach()
    return (
        F.smooth_l1_loss(predicted, target, reduction="none").mean(dim=-1)
        * weight
    ).mean()


def _correction_unroll(
    output: dict[str, Any], motion_weighted: bool
) -> tuple[Tensor, Tensor]:
    """Token error along the correction unroll: corrected and uncorrected.

    Same definition as the single-step ``transition`` loss; motion weights
    use the real previous token. Only the corrected term carries gradient
    (into the correction alone -- the unroll runs backbone/graph without
    gradient).
    """
    zero = output["predicted_tokens"].new_zeros(())
    if "correction_unroll_prediction" not in output:
        return zero, zero
    target = output["correction_unroll_target"]
    source = output["correction_unroll_source"]
    corrected = _motion_weighted_token_error(
        output["correction_unroll_prediction"], target, source, motion_weighted
    )
    base = _motion_weighted_token_error(
        output["correction_unroll_base"].detach(), target, source, motion_weighted
    )
    return corrected, base.detach()


def _graph_position(
    output: dict[str, Any], targets: dict[str, Tensor] | None
) -> Tensor:
    """Node 2D position head vs graph_node_pixel, over visible nodes.

    Observed-frame embeddings, one-step dynamics predictions (from
    ``graph_prediction_start``) and, when present, K-step predictions.
    """
    zero = output["graph_next"].new_zeros(())
    if (
        "graph_position_observed" not in output
        or targets is None
        or "node_pixel" not in targets
    ):
        return zero
    pixel = targets["node_pixel"]
    visible = targets["node_visible"].to(pixel.dtype)

    def masked(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        values = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    start = int(output.get("graph_prediction_start", 0))
    loss = masked(output["graph_position_observed"], pixel, visible) + masked(
        output["graph_position_prediction"][:, start:],
        pixel[:, 1 + start :],
        visible[:, 1 + start :],
    )
    if "graph_multistep_position_prediction" in output:
        predicted = output["graph_multistep_position_prediction"]
        multistep_start = int(output["graph_multistep_start"])
        stop = multistep_start + 1 + predicted.shape[1]
        loss = loss + masked(
            predicted,
            pixel[:, multistep_start + 1 : stop],
            visible[:, multistep_start + 1 : stop],
        )
    return loss


def _visual_multistep(
    output: dict[str, Any], motion_weighted: bool
) -> Tensor:
    """Token error along the visual K-step self-fed unroll.

    Same definition as the single-step ``transition`` loss: smooth L1 per
    token, optionally weighted by each token's own ground-truth motion
    (target minus the *real* previous token), normalized per frame.
    """
    if "visual_multistep_prediction" not in output:
        return output["predicted_tokens"].new_zeros(())
    predicted = output["visual_multistep_prediction"]
    target = output["visual_multistep_target"]
    if not motion_weighted:
        return F.smooth_l1_loss(predicted, target)
    motion = (target - output["visual_multistep_source"]).norm(dim=-1) + 1e-3
    weight = (motion / motion.mean(dim=-1, keepdim=True)).detach()
    return (
        F.smooth_l1_loss(predicted, target, reduction="none").mean(dim=-1)
        * weight
    ).mean()


def _graph_multistep(
    output: dict[str, Any], targets: dict[str, Tensor] | None
) -> tuple[Tensor, Tensor]:
    """Embedding and MuJoCo-state errors along the K-step self-fed unroll.

    Mirrors the single-step pair: ``graph`` (embedding vs observed next
    embedding) and ``graph_supervision``'s state term (decoded state vs
    MuJoCo state). Padding nodes are masked out when MuJoCo targets exist.
    """
    reference = output["graph_next"]
    zero = reference.new_zeros(())
    if "graph_multistep_prediction" not in output:
        return zero, zero
    predicted = output["graph_multistep_prediction"]
    horizon = predicted.shape[1]
    start = int(output["graph_multistep_start"])
    has_state = targets is not None and "node_state" in targets
    if has_state:
        mask = targets["node_mask"][:, start + 1 : start + 1 + horizon].to(
            predicted.dtype
        )
    else:
        mask = predicted.new_ones(predicted.shape[:3])
    denominator = mask.sum().clamp_min(1.0)
    embedding_error = (
        predicted - output["graph_multistep_target"]
    ).square().mean(-1)
    embedding = (embedding_error * mask).sum() / denominator
    if not has_state:
        return embedding, zero
    state_error = F.smooth_l1_loss(
        output["graph_multistep_state_prediction"],
        targets["node_state"][:, start + 1 : start + 1 + horizon],
        reduction="none",
    ).mean(-1)
    state = (state_error * mask).sum() / denominator
    return embedding, state


def compute_loss(
    model,
    output: dict[str, Any],
    image: Tensor,
    next_image: Tensor,
    **kwargs: float,
) -> JointLoss:
    """Compatibility one-step objective; sequence training is preferred."""
    images = torch.stack((image, next_image), dim=1)
    with torch.no_grad():
        target, target_mu, target_logvar = model.vae.encode(
            next_image, sample=False
        )
        target_graph = model.graph_builder(
            target,
            previous_slots=output["graph"],
            sample=False,
        )
    expanded = dict(output)
    for key in (
        "predicted_tokens",
        "base_predicted_tokens",
        "base_delta",
        "graph",
        "graph_next",
        "delta_g",
        "relation_logits",
        "next_relation_logits",
        "graph_context",
        "predicted_image",
        "reward_prediction",
        "continuation_logit",
    ):
        value = output.get(key)
        if isinstance(value, Tensor):
            expanded[key] = value[:, None]
    conditioned_mean = output.get("conditioned_mean")
    if isinstance(conditioned_mean, Tensor):
        expanded["conditioned_mean"] = conditioned_mean[:, None]
    expanded["target_tokens"] = target[:, None]
    expanded["target_graph"] = target_graph.nodes[:, None].detach()
    expanded["z_mu"] = torch.stack((output["z_mu"], target_mu), dim=1)
    expanded["z_logvar"] = torch.stack(
        (output["z_logvar"], target_logvar), dim=1
    )
    expanded["vae_reconstruction"] = torch.stack(
        (output["vae_reconstruction"], model.vae.decode(target)), dim=1
    )
    return compute_sequence_loss(
        model,
        expanded,
        images,
        graph_supervision_weight=0.0,
        **kwargs,
    )
