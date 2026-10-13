"""DINO-WM backbone: frozen DINOv2 patch tokens + a causal token predictor.

Mirrors the official DINO-WM (Zhou et al., 2025) design -- a fixed pretrained
visual representation plus a causal Transformer that predicts the next
frame's patch tokens -- through JointGV's existing pluggable
``LatentBackbone`` interface. The visual side lives in
``jointgv.blocks.dino_visual_codec.DinoPatchVisualCodec``; this module is
only the transition. Both plug into the same generic training/rollout path
already used by the David Ha MDN-LSTM backbone (see ``world_model.py``), so
the shared Graph Dynamics + OT plugin, reward/continuation heads, and
recursive-rollout evaluation protocol apply unchanged -- a graph-off vs.
graph-on comparison under this backbone differs only in
``--disable-graph-conditioning``.

Unlike the official repo's ``ViTPredictor`` (which hardcodes its causal-mask
buffer to CUDA), this predictor builds its mask per call from the input's own
device, and unlike the official repo's fixed ``num_hist``-frame window, it
runs full (capacity-bounded) causal attention so the same module handles both
long teacher-forced training windows and single-step autoregressive rollout
through one consistent interface: ``state`` is the raw token/action history,
carried and grown by the generic ``forward_sequence`` calling convention in
``world_model.py``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .base import BackboneOutput, CorrectionFunction, LatentBackbone


DinoWMState = tuple[Tensor, Tensor] | None


class _CausalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = max(int(dim * mlp_ratio), dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, attention_bias: Tensor) -> Tensor:
        normed = self.norm1(tokens)
        attended, _ = self.attention(
            normed, normed, normed, attn_mask=attention_bias, need_weights=False
        )
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class DinoWMPredictor(nn.Module):
    """Frame-causal Transformer over (patch tokens + one action token)/frame.

    Position ``p`` of frame ``t`` in the output directly predicts patch ``p``
    of frame ``t+1`` -- patch identity is preserved across the stack, never
    reshuffled -- so a plain elementwise loss against the next frame's
    encoded tokens is exact teacher forcing, the same
    ``target_tokens = z[:, 1:]`` convention the MDN-LSTM backbone already
    uses. The action token's own output position is discarded, matching how
    the official implementation excludes its action/proprio positions from
    the prediction loss.

    Temporal position uses a fixed sinusoidal encoding rather than a learned
    table, so a rollout longer than any training ``--sequence-length`` still
    gets a well-defined position signal for frames it never trained on,
    instead of an untrained embedding row. A separate small learned
    ``slot_embedding`` (time-independent, always fully trained) marks which
    patch slot or the action token a position is, since DINOv2's own patch
    tokens already carry spatial identity and only the slot *type* needs to
    be added.
    """

    def __init__(
        self,
        token_dim: int,
        action_dim: int,
        patch_count: int,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 40,
    ):
        super().__init__()
        if token_dim % heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by "
                f"dino_predictor_heads ({heads})"
            )
        if token_dim % 2 != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be even for the sinusoidal "
                "frame-position encoding"
            )
        if max_frames < 1:
            raise ValueError("dino_max_frames must be positive")
        self.token_dim = token_dim
        self.patch_count = patch_count
        self.tokens_per_frame = patch_count + 1
        # A compute/memory cap on the attention window (enforced by
        # DinoWMBackbone's sliding-window eviction before frames this long
        # ever reach forward()), not a position-embedding capacity limit --
        # frame position now uses a fixed encoding that is well-defined for
        # any frame count.
        self.max_frames = max_frames
        self.action_projection = nn.Linear(action_dim, token_dim)
        # Learned per-slot embedding distinguishes "which patch position, or
        # the action token" -- every training batch exercises all
        # ``tokens_per_frame`` slots regardless of sequence length, so this
        # table is always fully trained. It intentionally does not depend on
        # time; DINOv2's own patch tokens already carry spatial identity, so
        # this only needs to add a per-slot-type marker on top.
        self.slot_embedding = nn.Parameter(
            torch.randn(1, 1, self.tokens_per_frame, token_dim) * 0.02
        )
        # Fixed sinusoidal frame-position encoding: unlike a learned table,
        # it is mathematically defined for every frame index, so a rollout
        # longer than any training --sequence-length still gets a
        # meaningful signal instead of an untrained embedding row.
        self.register_buffer(
            "_frame_position_inv_freq",
            1.0 / (10000 ** (torch.arange(0, token_dim, 2).float() / token_dim)),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [
                _CausalBlock(token_dim, heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(token_dim)
        # Runtime switch (not architecture, not saved in model_config):
        # recompute block activations in backward instead of storing them.
        # Same outputs and gradients, less memory, slower backward.
        self.gradient_checkpointing = False

    def _frame_position_encoding(
        self, frames: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        position = torch.arange(frames, device=device, dtype=torch.float32)
        angles = position.unsqueeze(1) * self._frame_position_inv_freq.unsqueeze(0)
        encoding = torch.zeros(frames, self.token_dim, device=device)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles)
        return encoding.to(dtype).view(1, frames, 1, self.token_dim)

    @staticmethod
    def _causal_bias(
        frames: int,
        tokens_per_frame: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        frame_index = torch.arange(frames, device=device)
        allowed = frame_index.unsqueeze(1) >= frame_index.unsqueeze(0)
        allowed = allowed.repeat_interleave(
            tokens_per_frame, dim=0
        ).repeat_interleave(tokens_per_frame, dim=1)
        bias = torch.zeros(allowed.shape, device=device, dtype=dtype)
        bias.masked_fill_(~allowed, float("-inf"))
        return bias

    def forward(self, patch_tokens: Tensor, actions: Tensor) -> Tensor:
        """patch_tokens: [B,F,P,D]; actions: [B,F,action_dim] -> [B,F,P,D]."""
        batch, frames, patches, dim = patch_tokens.shape
        if patches != self.patch_count:
            raise ValueError(
                f"Expected {self.patch_count} DINO patch tokens, got {patches}"
            )
        if frames > self.max_frames:
            raise ValueError(
                f"DINO-WM predictor received {frames} frames but "
                f"dino_max_frames={self.max_frames}"
            )
        action_tokens = self.action_projection(actions).unsqueeze(2)
        sequence = torch.cat((patch_tokens, action_tokens), dim=2)
        sequence = sequence + self.slot_embedding
        sequence = sequence + self._frame_position_encoding(
            frames, sequence.device, sequence.dtype
        )
        flat = sequence.reshape(batch, frames * self.tokens_per_frame, dim)
        bias = self._causal_bias(
            frames, self.tokens_per_frame, flat.device, flat.dtype
        )
        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        for block in self.blocks:
            if use_checkpoint:
                flat = checkpoint(block, flat, bias, use_reentrant=False)
            else:
                flat = block(flat, bias)
        flat = self.output_norm(flat)
        out = flat.reshape(batch, frames, self.tokens_per_frame, dim)
        return out[:, :, : self.patch_count, :]


class DinoWMBackbone(LatentBackbone):
    """Adapter exposing the causal patch-token predictor through the common API."""

    name = "dino_wm"

    def __init__(
        self,
        token_count: int,
        token_dim: int,
        action_dim: int,
        dino_predictor_depth: int = 6,
        dino_predictor_heads: int = 8,
        dino_mlp_ratio: float = 4.0,
        dino_dropout: float = 0.1,
        dino_max_frames: int = 40,
        **_: object,
    ):
        super().__init__()
        self.predictor = DinoWMPredictor(
            token_dim,
            action_dim,
            token_count,
            depth=dino_predictor_depth,
            heads=dino_predictor_heads,
            mlp_ratio=dino_mlp_ratio,
            dropout=dino_dropout,
            max_frames=dino_max_frames,
        )
        self.max_frames = dino_max_frames

    def _windowed_predict(
        self, tokens: Tensor, actions: Tensor, state: DinoWMState
    ) -> tuple[Tensor, DinoWMState]:
        if state is None:
            history_tokens = tokens.new_zeros(
                tokens.shape[0], 0, tokens.shape[2], tokens.shape[3]
            )
            history_actions = actions.new_zeros(actions.shape[0], 0, actions.shape[2])
        else:
            history_tokens, history_actions = state
        full_tokens = torch.cat((history_tokens, tokens), dim=1)
        full_actions = torch.cat((history_actions, actions), dim=1)
        if full_tokens.shape[1] > self.max_frames:
            # Sliding-window fallback for rollouts longer than the positional
            # embedding capacity. Teacher-forced training never hits this:
            # train.py rejects --sequence-length > --dino-max-frames upfront.
            full_tokens = full_tokens[:, -self.max_frames :]
            full_actions = full_actions[:, -self.max_frames :]
        predicted_all = self.predictor(full_tokens, full_actions)
        new_frames = tokens.shape[1]
        predicted = predicted_all[:, -new_frames:]
        return predicted, (full_tokens, full_actions)

    def forward_sequence(
        self,
        tokens: Tensor,
        actions: Tensor,
        state: DinoWMState = None,
        target_tokens: Tensor | None = None,
        deterministic: bool = False,
    ) -> BackboneOutput:
        del target_tokens, deterministic
        predicted, new_state = self._windowed_predict(tokens, actions, state)
        base_delta = predicted - tokens
        return BackboneOutput(
            predicted_tokens=predicted,
            base_predicted_tokens=predicted,
            base_delta=base_delta,
            state=new_state,
            aux={},
        )

    def apply_graph_condition(
        self,
        output: BackboneOutput,
        source_tokens: Tensor,
        graph_context: Tensor,
        correction: CorrectionFunction,
        enabled: bool,
    ) -> BackboneOutput:
        del source_tokens
        correction_value = (
            correction(output.base_delta, graph_context)
            if enabled
            else torch.zeros_like(output.base_delta)
        )
        output.predicted_tokens = output.base_predicted_tokens + correction_value
        output.aux["correction"] = correction_value
        return output
