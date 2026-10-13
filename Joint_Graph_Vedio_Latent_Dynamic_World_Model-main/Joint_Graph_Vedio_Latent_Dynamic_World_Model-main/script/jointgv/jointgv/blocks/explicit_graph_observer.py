"""RGB-to-physical-graph observer initialized from the official RelTR model.

RelTR itself is imported from ``benchmark/reltr`` and kept unmodified.  Its
ResNet-50, transformer encoder, entity decoder and relation decoder produce
visual entity/relation tokens.  This module only adds the task-specific layer
that RelTR cannot provide: persistent robot-role binding, temporal recurrence,
physical-state decoding and the compact relation vocabulary used by JointGV.

MuJoCo state is a training target only.  At inference time the observer reads
RGB frames and its previous role slots; it never reads simulator state.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .graph_builder import GraphObservation


DEFAULT_ROLE_NAMES = (
    "base",
    "link_1",
    "link_2",
    "end_effector",
    "target",
    "object_1",
    "object_2",
    "object_3",
)


@dataclass
class VisualFeatures:
    """Visual tokens and their normalized image-plane centers."""

    tokens: Tensor
    positions: Tensor
    dense_tokens: Tensor | None = None
    dense_positions: Tensor | None = None


def _repository_root() -> Path:
    # .../script/jointgv/jointgv/blocks/file.py -> repository root
    return Path(__file__).resolve().parents[4]


def _load_official_reltr_modules():
    """Import the vendored official RelTR package without copying its code."""

    root = (_repository_root() / "benchmark" / "reltr").resolve()
    if not (root / "models" / "reltr.py").is_file():
        raise FileNotFoundError(
            "Missing official RelTR submodule. Run: "
            "git submodule update --init --recursive"
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    existing = sys.modules.get("models")
    if existing is not None:
        origin = Path(getattr(existing, "__file__", "")).resolve()
        if root not in origin.parents:
            raise RuntimeError(
                "A different top-level 'models' package was imported before "
                "RelTR. Start a fresh Python process before constructing the "
                "Explicit Graph Observer."
            )
    backbone_module = importlib.import_module("models.backbone")
    # The official 2022 code asks torchvision to download ImageNet weights at
    # construction time.  RelTR's own checkpoint contains those weights, so
    # disable that redundant network request before calling build_model.
    backbone_module.is_main_process = lambda: False
    model_module = importlib.import_module("models")
    misc_module = importlib.import_module("util.misc")
    return model_module, misc_module


def _reltr_args() -> SimpleNamespace:
    """Architecture arguments matching the official VG checkpoint exactly."""

    return SimpleNamespace(
        lr_backbone=0.0,
        dataset="vg",
        backbone="resnet50",
        dilation=False,
        position_embedding="sine",
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_entities=100,
        num_triplets=200,
        pre_norm=False,
        aux_loss=True,
        device="cpu",
        set_cost_class=1,
        set_cost_bbox=5,
        set_cost_giou=2,
        set_iou_threshold=0.7,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        rel_loss_coef=1,
        eos_coef=0.1,
        return_interm_layers=True,
    )


class SpatialVisualBackbone(nn.Module):
    """Official RelTR feature extractor or a debug-only lightweight CNN."""

    def __init__(
        self,
        name: str = "reltr_resnet50",
        pretrained: bool = True,
        frozen: bool = True,
        reltr_checkpoint: str | Path = "Models/reltr/checkpoint0149.pth",
        image_size: int = 224,
        inference_batch_size: int = 8,
    ) -> None:
        super().__init__()
        self.name = name
        self.frozen = bool(frozen)
        self.input_size = int(image_size)
        self.inference_batch_size = int(inference_batch_size)
        if self.input_size < 64 or self.inference_batch_size < 1:
            raise ValueError("RelTR image and inference batch sizes are invalid")

        if name == "debug_conv":
            self.feature_dim = 96
            self.dense_feature_dim = self.feature_dim
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2),
                nn.GroupNorm(4, 32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
                nn.Conv2d(64, self.feature_dim, 3, padding=1),
                nn.GroupNorm(8, self.feature_dim),
                nn.SiLU(),
            )
            self.reltr_checkpoint = None
        elif name == "reltr_resnet50":
            model_module, misc_module = _load_official_reltr_modules()
            model, _criterion, _postprocessors = model_module.build_model(
                _reltr_args()
            )
            self.encoder = model
            self._nested_tensor_from_tensor_list = (
                misc_module.nested_tensor_from_tensor_list
            )
            self.feature_dim = int(model.hidden_dim)
            # ResNet-50 layer3 is a 14x14 map for 224x224 input.  Keep it as
            # frozen official-backbone output; the robot-specific projection
            # is trained in ExplicitGraphObserver.
            self.dense_feature_dim = 1024
            checkpoint_path = Path(reltr_checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = _repository_root() / checkpoint_path
            self.reltr_checkpoint = str(checkpoint_path)
            if pretrained:
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Missing official RelTR checkpoint: {checkpoint_path}. "
                        "Download Visual Genome checkpoint0149.pth as shown "
                        "in the README."
                    )
                payload = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                state = payload.get("model", payload)
                model.load_state_dict(state, strict=True)
        else:
            raise ValueError(f"Unknown graph visual backbone: {name}")

        if self.frozen:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        else:
            # Official build_backbone freezes ResNet when lr_backbone == 0.
            # An explicit --train-backbone request must override that default
            # for both ResNet and transformer parameters.
            self.encoder.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    @staticmethod
    def _grid_positions(feature: Tensor) -> Tensor:
        height, width = feature.shape[-2:]
        y = torch.linspace(-1.0, 1.0, height, device=feature.device)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        position = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        return position.to(dtype=feature.dtype)[None].expand(
            feature.shape[0], -1, -1
        )

    def _forward_debug(self, images: Tensor) -> VisualFeatures:
        feature = F.adaptive_avg_pool2d(self.encoder(images), (16, 16))
        tokens = feature.flatten(2).transpose(1, 2)
        positions = self._grid_positions(feature)
        return VisualFeatures(
            tokens=tokens,
            positions=positions,
            dense_tokens=tokens,
            dense_positions=positions,
        )

    def _forward_reltr(self, images: Tensor) -> VisualFeatures:
        resized = F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = resized.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = resized.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        normalized = (resized - mean) / std
        samples = self._nested_tensor_from_tensor_list(normalized)
        features, position_encoding = self.encoder.backbone(samples)
        source, mask = features[-1].decompose()
        hidden, triplets, _so_masks, _memory = self.encoder.transformer(
            self.encoder.input_proj(source),
            mask,
            self.encoder.entity_embed.weight,
            self.encoder.triplet_embed.weight,
            position_encoding[-1],
            self.encoder.so_embed.weight,
        )
        entity = hidden[-1]
        subject, object_ = torch.split(
            triplets[-1], self.feature_dim, dim=-1
        )
        tokens = torch.cat((entity, subject, object_), dim=1)
        # Reuse the official pretrained box heads to attach an image location
        # to every entity, subject and object query.
        centers = torch.cat(
            (
                self.encoder.entity_bbox_embed(entity)[..., :2].sigmoid(),
                self.encoder.sub_bbox_embed(subject)[..., :2].sigmoid(),
                self.encoder.obj_bbox_embed(object_)[..., :2].sigmoid(),
            ),
            dim=1,
        )
        dense_source, _dense_mask = features[-2].decompose()
        return VisualFeatures(
            tokens=tokens,
            positions=centers * 2.0 - 1.0,
            dense_tokens=dense_source.flatten(2).transpose(1, 2),
            dense_positions=self._grid_positions(dense_source),
        )

    def _forward_chunk(self, images: Tensor) -> VisualFeatures:
        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            if self.name == "debug_conv":
                return self._forward_debug(images)
            return self._forward_reltr(images)

    def forward(self, images: Tensor) -> VisualFeatures:
        chunks = [
            self._forward_chunk(images[start : start + self.inference_batch_size])
            for start in range(0, images.shape[0], self.inference_batch_size)
        ]
        return VisualFeatures(
            tokens=torch.cat([item.tokens for item in chunks], dim=0),
            positions=torch.cat([item.positions for item in chunks], dim=0),
            dense_tokens=torch.cat(
                [item.dense_tokens for item in chunks], dim=0
            ),
            dense_positions=torch.cat(
                [item.dense_positions for item in chunks], dim=0
            ),
        )


class ExplicitGraphObserver(nn.Module):
    """Official RelTR + persistent role binding + physical graph heads."""

    requires_images = True

    def __init__(
        self,
        node_dim: int,
        nodes: int,
        relation_count: int = 4,
        teacher_state_dim: int = 13,
        backbone: str = "reltr_resnet50",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        reltr_checkpoint: str | Path = "Models/reltr/checkpoint0149.pth",
        reltr_image_size: int = 224,
        reltr_batch_size: int = 8,
        role_names: Sequence[str] | None = None,
        attention_heads: int = 4,
        temporal_max_correction: float = 0.25,
        max_position_residual: float = 0.25,
    ) -> None:
        super().__init__()
        if node_dim % attention_heads:
            raise ValueError("node_dim must be divisible by attention_heads")
        if not 0.0 < temporal_max_correction <= 1.0:
            raise ValueError("temporal_max_correction must be in (0, 1]")
        if not 0.0 < max_position_residual <= 1.0:
            raise ValueError("max_position_residual must be in (0, 1]")
        self.nodes = int(nodes)
        self.node_dim = int(node_dim)
        self.relation_count = int(relation_count)
        self.teacher_state_dim = int(teacher_state_dim)
        names = list(role_names or DEFAULT_ROLE_NAMES)
        if len(names) < nodes:
            names.extend(f"object_{index}" for index in range(len(names), nodes))
        self.role_names = tuple(names[:nodes])
        self.visual_backbone = SpatialVisualBackbone(
            backbone,
            pretrained=pretrained_backbone,
            frozen=freeze_backbone,
            reltr_checkpoint=reltr_checkpoint,
            image_size=reltr_image_size,
            inference_batch_size=reltr_batch_size,
        )
        self.backbone_name = backbone
        self.backbone_frozen = bool(freeze_backbone)
        self.reltr_checkpoint = str(reltr_checkpoint)
        self.reltr_image_size = int(reltr_image_size)
        self.reltr_batch_size = int(reltr_batch_size)
        self.temporal_max_correction = float(temporal_max_correction)
        self.max_position_residual = float(max_position_residual)
        self.temporal_strength = 1.0

        self.visual_projection = nn.Linear(
            self.visual_backbone.feature_dim, node_dim
        )
        self.dense_visual_projection = nn.Linear(
            self.visual_backbone.dense_feature_dim, node_dim
        )
        self.role_queries = nn.Parameter(torch.randn(nodes, node_dim) * 0.02)
        self.temporal_query = nn.Linear(node_dim, node_dim, bias=False)
        self.temporal_norm = nn.LayerNorm(node_dim)
        self.temporal_gate = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1),
        )
        # History starts as a small hint (~12% gate), not as the primary
        # localization signal. Every frame is re-anchored to current RGB.
        nn.init.zeros_(self.temporal_gate[-1].weight)
        nn.init.constant_(self.temporal_gate[-1].bias, -2.0)
        self.cross_attention = nn.MultiheadAttention(
            node_dim, attention_heads, batch_first=True
        )
        self.query_norm = nn.LayerNorm(node_dim)
        self.visual_norm = nn.LayerNorm(node_dim)
        self.dense_visual_norm = nn.LayerNorm(node_dim)
        self.position_residual = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 2),
        )
        nn.init.zeros_(self.position_residual[-1].weight)
        nn.init.zeros_(self.position_residual[-1].bias)

        # One IMP-inspired node-edge refinement round after RelTR perception.
        self.edge_seed = nn.Sequential(
            nn.Linear(2 * node_dim + 2, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.edge_update = nn.GRUCell(node_dim, node_dim)
        self.message = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.node_update = nn.GRUCell(node_dim, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)

        self.node_mu = nn.Linear(node_dim, node_dim)
        self.node_logvar = nn.Linear(node_dim, node_dim)
        self.presence = nn.Linear(node_dim, 1)
        self.teacher_state = nn.Linear(node_dim, teacher_state_dim)
        self.velocity_dim = max(0, teacher_state_dim - 7)
        self.motion_velocity = (
            nn.Sequential(
                nn.Linear(node_dim + 5, node_dim),
                nn.SiLU(),
                nn.Linear(node_dim, self.velocity_dim),
            )
            if self.velocity_dim
            else None
        )
        if self.motion_velocity is not None:
            # Preserve the former decoder at initialization/checkpoint upgrade;
            # training learns only a motion-conditioned residual at first.
            nn.init.zeros_(self.motion_velocity[-1].weight)
            nn.init.zeros_(self.motion_velocity[-1].bias)

        # VTransE-style compact physical relation head.
        self.subject = nn.Linear(node_dim, node_dim, bias=False)
        self.object = nn.Linear(node_dim, node_dim, bias=False)
        self.relation_translation = nn.Parameter(
            torch.randn(relation_count, node_dim) * 0.02
        )
        self.relation_residual = nn.Sequential(
            nn.Linear(3 * node_dim + 2, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, relation_count),
        )

    def set_temporal_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("temporal strength must be in [0, 1]")
        self.temporal_strength = float(strength)

    @staticmethod
    def _dense_attention(
        query: Tensor,
        dense_visual: Tensor,
        dense_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scores = torch.einsum(
            "bnd,btd->bnt", query, dense_visual
        ) / math.sqrt(float(query.shape[-1]))
        attention = scores.softmax(dim=-1)
        observed = torch.bmm(attention, dense_visual)
        positions = torch.einsum(
            "bnt,btd->bnd", attention, dense_positions
        )
        return observed, attention, positions

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
        relative = positions_2d[:, None] - positions_2d[:, :, None]
        edge = self.edge_seed(torch.cat((sender, receiver, relative), dim=-1))
        residual = self.relation_residual(
            torch.cat((sender, receiver, edge, relative), dim=-1)
        )
        logits = translation_score + residual
        diagonal = torch.eye(count, device=nodes.device, dtype=torch.bool)
        logits = logits.clone()
        logits[:, diagonal, 0] += 20.0
        if self.relation_count > 1:
            logits[:, diagonal, 1:] -= 20.0
        return logits

    def _imp_refine(self, nodes: Tensor, positions: Tensor) -> Tensor:
        batch, count, dim = nodes.shape
        sender = nodes[:, :, None, :].expand(-1, -1, count, -1)
        receiver = nodes[:, None, :, :].expand(-1, count, -1, -1)
        relative = positions[:, None] - positions[:, :, None]
        seed = self.edge_seed(torch.cat((sender, receiver, relative), dim=-1))
        edge = self.edge_update(
            seed.reshape(-1, dim), torch.zeros_like(seed).reshape(-1, dim)
        ).reshape(batch, count, count, dim)
        messages = self.message(torch.cat((sender, edge), dim=-1))
        diagonal = torch.eye(count, device=nodes.device, dtype=nodes.dtype)
        messages = messages * (1.0 - diagonal)[None, :, :, None]
        aggregate = messages.sum(dim=1) / max(1, count - 1)
        refined = self.node_update(
            aggregate.reshape(-1, dim), nodes.reshape(-1, dim)
        ).reshape(batch, count, dim)
        return self.node_norm(refined)

    def _observe_visual(
        self,
        visual_features: VisualFeatures,
        previous_slots: Tensor | None = None,
        sample: bool = False,
        reset_mask: Tensor | None = None,
    ) -> GraphObservation:
        visual = self.visual_projection(visual_features.tokens)
        dense_tokens = (
            visual_features.dense_tokens
            if visual_features.dense_tokens is not None
            else visual_features.tokens
        )
        dense_positions = (
            visual_features.dense_positions
            if visual_features.dense_positions is not None
            else visual_features.positions
        )
        dense_visual = self.dense_visual_norm(
            self.dense_visual_projection(dense_tokens)
        )
        batch = visual.shape[0]
        role_queries = self.role_queries[None].expand(batch, -1, -1)
        anchor_query = self.query_norm(role_queries)
        visual = self.visual_norm(visual)
        # RelTR itself is image-only. This first pass deliberately ignores
        # history so every role is localized afresh from the current RGB.
        anchor_observed, _anchor_semantic_attention = self.cross_attention(
            anchor_query,
            visual,
            visual,
            need_weights=True,
            average_attn_weights=True,
        )
        anchor_dense, anchor_dense_attention, anchor_dense_position = (
            self._dense_attention(
                anchor_query, dense_visual, dense_positions
            )
        )
        anchor_fused = anchor_observed + anchor_dense
        if previous_slots is not None:
            previous = self.temporal_norm(previous_slots)
            gate = torch.sigmoid(
                self.temporal_gate(torch.cat((anchor_fused, previous), dim=-1))
            )
            correction = (
                self.temporal_strength
                * self.temporal_max_correction
                * gate
                * torch.tanh(self.temporal_query(previous))
            )
            if reset_mask is not None:
                keep = (1.0 - reset_mask.to(correction.dtype))[:, None, None]
                gate = gate * keep
                correction = correction * keep
            corrected_query = anchor_query + correction
            semantic_observed, _semantic_attention = self.cross_attention(
                corrected_query,
                visual,
                visual,
                need_weights=True,
                average_attn_weights=True,
            )
            dense_observed, attention, dense_position = self._dense_attention(
                corrected_query, dense_visual, dense_positions
            )
            observed = semantic_observed + dense_observed
            refinement_query = corrected_query
        else:
            observed = anchor_fused
            attention = anchor_dense_attention
            dense_position = anchor_dense_position
            refinement_query = anchor_query
            gate = anchor_query.new_zeros(batch, self.nodes, 1)
            correction = anchor_query.new_zeros(batch, self.nodes, self.node_dim)
        attention = attention.clamp_min(1e-8)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        positions = (
            dense_position
            + self.max_position_residual
            * torch.tanh(self.position_residual(observed))
        ).clamp(-1.0, 1.0)
        refined = self._imp_refine(observed + refinement_query, positions)
        mu = self.node_mu(refined)
        logvar = self.node_logvar(refined).clamp(-8, 5)
        nodes = (
            mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            if sample
            else mu
        )
        return GraphObservation(
            nodes=nodes,
            node_mu=mu,
            node_logvar=logvar,
            relation_logits=self.relation_head(nodes, positions),
            positions_2d=positions,
            presence_logits=self.presence(nodes).squeeze(-1),
            attention=attention,
            temporal_gate=gate.squeeze(-1),
            identity_features=observed,
            temporal_correction=correction,
        )

    def decode_state_sequence(
        self,
        nodes: Tensor,
        positions_2d: Tensor,
        firsts: Tensor | None = None,
    ) -> Tensor:
        """Decode physical state with a consecutive-position velocity residual."""
        state = self.teacher_state(nodes)
        if self.motion_velocity is None or nodes.ndim < 4:
            return state
        position_delta = torch.zeros_like(state[..., :3])
        pixel_delta = torch.zeros_like(positions_2d)
        position_delta[:, 1:] = state[:, 1:, :, :3] - state[:, :-1, :, :3]
        pixel_delta[:, 1:] = positions_2d[:, 1:] - positions_2d[:, :-1]
        motion = self.motion_velocity(
            torch.cat((nodes, position_delta, pixel_delta), dim=-1)
        )
        # The first observation has no finite difference. Keep the original
        # single-frame estimate there; subsequent frames receive a learned
        # residual explicitly conditioned on visual/physical displacement.
        temporal_mask = torch.ones_like(motion[..., :1])
        temporal_mask[:, 0] = 0.0
        if firsts is not None:
            temporal_mask = temporal_mask * (
                1.0 - firsts.to(temporal_mask.dtype)[..., None, None]
            )
        decoded = state.clone()
        decoded[..., 7:] = state[..., 7:] + temporal_mask * motion
        return decoded

    def forward(
        self,
        images: Tensor,
        previous_slots: Tensor | None = None,
        sample: bool = False,
        reset_mask: Tensor | None = None,
    ) -> GraphObservation:
        return self._observe_visual(
            self.visual_backbone(images),
            previous_slots,
            sample,
            reset_mask=reset_mask,
        )

    def forward_sequence(
        self,
        images: Tensor,
        sample: bool = False,
        firsts: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch, frames = images.shape[:2]
        flat = self.visual_backbone(images.flatten(0, 1))
        tokens = flat.tokens.reshape(batch, frames, *flat.tokens.shape[1:])
        positions = flat.positions.reshape(
            batch, frames, *flat.positions.shape[1:]
        )
        dense_tokens = flat.dense_tokens.reshape(
            batch, frames, *flat.dense_tokens.shape[1:]
        )
        dense_positions = flat.dense_positions.reshape(
            batch, frames, *flat.dense_positions.shape[1:]
        )
        previous: Tensor | None = None
        observations: list[GraphObservation] = []
        for step in range(frames):
            observation = self._observe_visual(
                VisualFeatures(
                    tokens[:, step],
                    positions[:, step],
                    dense_tokens[:, step],
                    dense_positions[:, step],
                ),
                previous,
                sample=sample,
                reset_mask=(None if firsts is None else firsts[:, step]),
            )
            observations.append(observation)
            previous = observation.node_mu
        result = {
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
                "temporal_gate",
                "identity_features",
                "temporal_correction",
            )
        }
        result["state"] = self.decode_state_sequence(
            result["nodes"], result["positions_2d"], firsts=firsts
        )
        return result
