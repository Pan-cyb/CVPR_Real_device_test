"""Model-predictive control with a learned world model.

``CEMPlanner`` + ``LatentRollout`` support two costs for generic-path
backbones (DINO-WM): ``goal_latent_cost`` preserves DINO-WM's goal-image
protocol, while ``predicted_reward_cost`` ranks candidates by a learned
reward head.  The latter is used by visible-target tasks such as RM65-B: the
target is already in the current image, so a synthetic full-pose goal image
would incorrectly select one inverse-kinematics solution.

The reward head is either the world model's own (trained in train.py on the
backbone's one-step predictions) or a ``TrackRewardHead`` fitted afterwards
on one planning track's own imagination (train_reward_heads.py), so the
backbone-only and graph-corrected planners are each scored by a head that
matches their predictions.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 24          # predicted frames per plan (~one swing period)
    candidates: int = 256
    elites: int = 32
    iterations: int = 4
    init_std: float = 0.5
    min_std: float = 0.05
    cost_steps: int = 8        # cost averaged over the last K predicted frames


class CEMPlanner:
    """Cross-entropy method over action sequences in [-1, 1]^A.

    Warm-started from the previous plan shifted by the executed steps; the
    current mean is always one of the evaluated candidates, so a replan can
    never score worse than keeping the previous plan.
    """

    def __init__(
        self,
        action_dim: int,
        config: CEMConfig,
        device: torch.device,
        seed: int = 0,
    ):
        if not 1 <= config.elites <= config.candidates:
            raise ValueError("Require 1 <= elites <= candidates")
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.reset()

    def reset(self) -> None:
        self.mean = torch.zeros(
            self.config.horizon, self.action_dim, device=self.device
        )
        self.last_cost = float("nan")

    def shift(self, steps: int) -> None:
        """Drop the ``steps`` executed actions; pad the tail with zeros."""
        steps = min(steps, self.config.horizon)
        self.mean = torch.cat(
            (
                self.mean[steps:],
                torch.zeros(steps, self.action_dim, device=self.device),
            )
        )

    def plan(self, cost_fn: Callable[[Tensor], Tensor]) -> Tensor:
        """cost_fn: [N, H, A] sequences -> [N] costs (lower is better)."""
        config = self.config
        mean = self.mean
        std = torch.full_like(mean, config.init_std)
        for _ in range(config.iterations):
            noise = torch.randn(
                config.candidates,
                config.horizon,
                self.action_dim,
                device=self.device,
                generator=self.generator,
            )
            samples = (mean[None] + std[None] * noise).clamp(-1.0, 1.0)
            samples[0] = mean
            costs = cost_fn(samples).to(self.device)
            order = torch.topk(costs, config.elites, largest=False).indices
            elites = samples[order]
            mean = elites.mean(0)
            std = elites.std(0, unbiased=False).clamp_min(config.min_std)
            self.last_cost = float(costs[order[0]])
        self.mean = mean
        return mean


class LatentRollout:
    """Batched open-loop imagination of a generic-path world model (DINO-WM).

    Same computation as ``JointGraphVideoWorldModel.rollout`` with visual
    context (current frame, real context frames and actions, graph history,
    correction), except that the context is encoded once per replan and
    broadcast to every candidate, and that the DINO-WM predictor may be
    limited to the last ``history_window`` frames (DINO-WM's ``num_hist``);
    ``None`` keeps ``rollout``'s full history. ``use_graph=False`` is the
    backbone alone (correction strength 0 gives the same tokens).
    ``reward_head`` (a ``TrackRewardHead``) replaces the world model's own
    reward head in :meth:`rewards` / :meth:`predicted_return`.
    """

    def __init__(
        self,
        model,
        *,
        use_graph: bool,
        strength: float = 1.0,
        history_window: int | None = 4,
        amp: bool = False,
        reward_head: "TrackRewardHead | None" = None,
    ):
        # The broadcast of the backbone history assumes DINO-WM's state, a
        # (tokens, actions) pair with the batch first.
        if model.dreamerv3_core is not None or model.backbone_name != "dino_wm":
            raise ValueError("LatentRollout supports the dino_wm backbone")
        if history_window is not None and history_window < 2:
            raise ValueError("history_window must be at least 2")
        self.model = model
        self.use_graph = use_graph
        # Correction strength is model state shared by every LatentRollout
        # on the same model; it is set again before each rollout.
        self.strength = strength if use_graph else 0.0
        self.history_window = history_window
        self.amp = amp
        self.reward_head = reward_head
        self._start: dict[str, Any] | None = None

    def _backbone(self, tokens: Tensor, actions: Tensor, state: Any):
        backbone = self.model.backbone
        if self.history_window is not None and state is not None:
            history_tokens, history_actions = state
            keep = self.history_window - 1
            state = (history_tokens[:, -keep:], history_actions[:, -keep:])
        context = (
            torch.autocast(device_type=tokens.device.type, dtype=torch.float16)
            if self.amp and tokens.device.type == "cuda"
            else nullcontext()
        )
        with context:
            transition = backbone.forward_sequence(
                tokens, actions, state=state, deterministic=True
            )
        if self.amp:
            transition.predicted_tokens = transition.predicted_tokens.float()
            transition.base_predicted_tokens = (
                transition.base_predicted_tokens.float()
            )
            transition.base_delta = transition.base_delta.float()
            if isinstance(transition.state, tuple):
                transition.state = tuple(value.float() for value in transition.state)
        return transition

    @torch.no_grad()
    def prepare(self, frames: Tensor, actions: Tensor) -> None:
        """frames [C+1, 3, H, W] in [0, 1], oldest to current; actions [C, A]
        with actions[i] leading from frame i to frame i+1.

        A batch of B starts ([B, C+1, 3, H, W], [B, C, A]) is also accepted;
        :meth:`rollout` then takes exactly one sequence per start instead of
        broadcasting a single start to every candidate.
        """
        model = self.model
        if frames.dim() == 4:
            frames, actions = frames[None], actions[None]
        if actions.shape[1] != frames.shape[1] - 1:
            raise ValueError("Need one action between consecutive frames")
        batch, length = frames.shape[:2]
        tokens_all = model.vae.encode(frames.flatten(0, 1), sample=False)[0]
        tokens_all = tokens_all.reshape(batch, length, *tokens_all.shape[1:])
        start: dict[str, Any] = {
            "tokens": tokens_all[:, -1],
            # Frame before the current one (the current one if there is none).
            "previous": tokens_all[:, -2] if length > 1 else tokens_all[:, -1],
            "state": None,
        }
        if length > 1:
            start["state"] = self._backbone(
                tokens_all[:, :-1], actions, None
            ).state
        start["graph"] = start["relations"] = start["window"] = None
        if self.use_graph:
            history = model.graph_history
            if history == 1:
                observation = model.graph_plugin.observe(
                    tokens_all[:, -1], frames[:, -1], sample=False
                )
                start["graph"] = observation.nodes
                start["relations"] = observation.relation_logits
            else:
                seed = frames[:, -history:]
                seed_tokens = tokens_all[:, -history:]
                observed = model.graph_plugin.infer_sequence(seed_tokens, seed)
                frames_seen = seed.shape[1]
                offsets = torch.arange(
                    frames_seen - history, frames_seen, device=frames.device
                ).clamp_min(0)
                start["window"] = observed["nodes"][:, offsets]
                start["graph"] = start["window"][:, -1]
                start["relations"] = observed["relation_logits"][:, -1]
        self._start = start

    @torch.no_grad()
    def initial(self, count: int) -> dict[str, Any]:
        """Imagination carry for ``count`` sequences from the prepared start(s)."""
        if self._start is None:
            raise RuntimeError("call prepare() first")
        starts = self._start["tokens"].shape[0]
        if starts not in (1, count):
            raise ValueError(
                f"{starts} prepared starts need one sequence each, got {count}"
            )

        def expand(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.expand(count, *value.shape[1:])

        start = self._start
        state = start["state"]
        return {
            "tokens": expand(start["tokens"]),
            "previous": expand(start["previous"]),
            "state": None if state is None else tuple(expand(value) for value in state),
            "graph": expand(start["graph"]),
            "relations": expand(start["relations"]),
            "window": expand(start["window"]),
        }

    @torch.no_grad()
    def step(self, carry: dict[str, Any], actions: Tensor) -> dict[str, Any]:
        """One imagined step: actions [N, A] -> new carry; its "tokens" entry
        [N, P, D] is the predicted next frame and "previous" the frame it was
        predicted from. Used by :meth:`rollout` and by policies that pick each
        action from the imagined state."""
        model = self.model
        model.set_graph_conditioning_strength(self.strength)
        tokens, graph = carry["tokens"], carry["graph"]
        previous = tokens
        relations, window = carry["relations"], carry["window"]
        step_actions = actions[:, None]
        transition = self._backbone(tokens[:, None], step_actions, carry["state"])
        if self.use_graph:
            transition, graph_output = model._condition(
                transition,
                tokens[:, None],
                step_actions,
                graph[:, None],
                relations[:, None],
                True,
                graph_window=window[:, None] if window is not None else None,
            )
            tokens = transition.aux.get(
                "plugin_corrected_tokens", transition.predicted_tokens
            )[:, 0]
            graph = graph_output.graph_next[:, 0]
            relations = graph_output.next_relation_logits[:, 0]
            if window is not None:
                window = torch.cat((window[:, 1:], graph[:, None]), dim=1)
        else:
            tokens = transition.base_predicted_tokens[:, 0]
        return {
            "tokens": tokens,
            "previous": previous,
            "state": model.backbone.feedback(transition.state, tokens),
            "graph": graph,
            "relations": relations,
            "window": window,
        }

    @torch.no_grad()
    def rollout(self, actions: Tensor) -> Tensor:
        """actions [N, H, A] -> predicted tokens [N, H, P, D]."""
        carry = self.initial(actions.shape[0])
        predictions = []
        for step in range(actions.shape[1]):
            carry = self.step(carry, actions[:, step])
            predictions.append(carry["tokens"])
        return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def predicted_return(
        self,
        actions: Tensor,
        discount: float = 0.99,
        reward_min: float | None = None,
        reward_max: float | None = None,
    ) -> Tensor:
        """Discounted learned return for candidate action sequences.

        The visual/graph rollout is identical to :meth:`rollout`; reward is
        read from its predicted tokens.  Both graph-on and backbone-only
        controllers therefore use the same task objective.  A
        fixed-horizon CEM task uses the known discount only: multiplying by a
        learned continuation would let continuation error suppress a valid
        late-horizon target acquisition even though RM65-B never terminates
        inside the planning horizon.
        """
        rewards = self.rewards(self.rollout(actions))
        if reward_min is not None or reward_max is not None:
            rewards = rewards.clamp(min=reward_min, max=reward_max)
        weight = rewards.new_ones(rewards.shape[0])
        value = rewards.new_zeros(rewards.shape[0])
        for step in range(rewards.shape[1]):
            value = value + weight * rewards[:, step]
            weight = weight * discount
        return value

    def start_tokens(self) -> Tensor:
        """Encoded current frame(s) of the last :meth:`prepare`, [B, P, D]."""
        if self._start is None:
            raise RuntimeError("call prepare() first")
        return self._start["tokens"]

    @torch.no_grad()
    def rewards(self, predicted: Tensor) -> Tensor:
        """Per-step reward [N, H] of this rollout's predicted tokens [N, H, P, D]."""
        if self.reward_head is None:
            return self.model.reward_head(predicted.mean(dim=-2)).squeeze(-1)
        start = self.start_tokens()
        start = start.expand(predicted.shape[0], *start.shape[1:])
        previous = torch.cat((start[:, None], predicted[:, :-1]), dim=1)
        return self.reward_head(predicted, previous)


class TrackRewardHead(nn.Module):
    """Reward of one imagined frame, fitted to one planning track.

    Fitted after world-model training (train_reward_heads.py) on the
    open-loop imagination of a single track (backbone alone or graph
    corrected), so each reward-CEM planner is scored by a head matched to
    its own predictions. The input is the predicted patch grid and its change
    from the previous frame: the RM65-B gate depends on where the TCP is
    relative to the target (spatial layout) and on swing and TCP speed
    (motion), which the world model's pooled single-frame head cannot see.
    Every track uses the same architecture.
    """

    def __init__(
        self, token_dim: int, grid: int, hidden: int = 64, dropout: float = 0.1
    ):
        super().__init__()
        self.config = {
            "token_dim": token_dim,
            "grid": grid,
            "hidden": hidden,
            "dropout": dropout,
        }
        self.token_dim = token_dim
        self.grid = grid
        channels = 2 * token_dim
        # Per-channel input standardisation, set from the training features.
        self.register_buffer("input_mean", torch.zeros(channels))
        self.register_buffer("input_std", torch.ones(channels))
        groups = 8 if hidden % 8 == 0 else 1

        def block(inputs: int, stride: int) -> list[nn.Module]:
            return [
                nn.Conv2d(inputs, hidden, 3, stride=stride, padding=1),
                nn.GroupNorm(groups, hidden),
                nn.SiLU(),
            ]

        self.net = nn.Sequential(
            *block(channels, 1),
            *block(hidden, 2),
            *block(hidden, 2),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 16, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, 1),
        )

    def features(self, current: Tensor, previous: Tensor) -> Tensor:
        """[..., P, D] pair -> unstandardised input [..., P, 2D]."""
        return torch.cat((current, current - previous), dim=-1).float()

    def forward(self, current: Tensor, previous: Tensor) -> Tensor:
        """current, previous [..., P, D] (frame and the one before) -> [...]."""
        if current.shape[-2] != self.grid * self.grid:
            raise ValueError(
                f"expected {self.grid}x{self.grid} tokens, got {current.shape[-2]}"
            )
        lead = current.shape[:-2]
        x = (self.features(current, previous) - self.input_mean) / self.input_std
        x = x.reshape(-1, self.grid, self.grid, 2 * self.token_dim)
        return self.net(x.permute(0, 3, 1, 2)).reshape(lead)


def load_track_reward_head(
    path: str | Path, device: torch.device
) -> tuple[TrackRewardHead, dict[str, Any]]:
    """A head saved by train_reward_heads.py and the track it was fitted on."""
    saved = torch.load(path, map_location=device, weights_only=False)
    head = TrackRewardHead(**saved["config"]).to(device)
    head.load_state_dict(saved["state_dict"])
    return head.eval(), saved["track"]


def limit_history(model, window: int) -> None:
    """Make ``model.rollout`` slide the DINO-WM predictor over the last
    ``window`` frames, like DINO-WM's num_hist and ``LatentRollout``'s
    ``history_window``; 0 keeps the full history (up to dino_max_frames).
    A window no longer than the training --sequence-length keeps every frame
    position inside the range the predictor was trained on."""
    if not window:
        return
    backbone = getattr(model, "backbone", None)
    if model.backbone_name != "dino_wm" or not hasattr(backbone, "max_frames"):
        raise SystemExit("--history-window needs a dino_wm checkpoint")
    if not 2 <= window <= backbone.max_frames:
        raise SystemExit(
            f"--history-window must be in [2, dino_max_frames={backbone.max_frames}]"
        )
    backbone.max_frames = window


def goal_latent_cost(
    rollout: LatentRollout,
    goal_tokens: Tensor,
    cost_steps: int,
    chunk_size: int = 64,
) -> Callable[[Tensor], Tensor]:
    """Mean squared distance between predicted and goal patch features over
    the last ``cost_steps`` predicted frames (arrive and stay)."""

    def cost(sequences: Tensor) -> Tensor:
        values = []
        for start in range(0, sequences.shape[0], chunk_size):
            predicted = rollout.rollout(sequences[start : start + chunk_size])
            tail = predicted[:, -cost_steps:]
            values.append((tail - goal_tokens).square().mean(dim=(1, 2, 3)))
        return torch.cat(values)

    return cost


def predicted_reward_cost(
    rollout: LatentRollout,
    discount: float = 0.99,
    chunk_size: int = 64,
    reward_min: float | None = None,
    reward_max: float | None = None,
) -> Callable[[Tensor], Tensor]:
    """CEM cost that maximizes learned discounted task reward.

    Returned values are negated because :class:`CEMPlanner` minimizes costs.
    Optional reward bounds stop CEM exploiting out-of-distribution head
    values beyond the environment's known label range. The candidate set is
    chunked only for memory; it does not change scores.
    """

    def cost(sequences: Tensor) -> Tensor:
        values = []
        for start in range(0, sequences.shape[0], chunk_size):
            chunk = sequences[start : start + chunk_size]
            values.append(
                -rollout.predicted_return(
                    chunk,
                    discount=discount,
                    reward_min=reward_min,
                    reward_max=reward_max,
                )
            )
        return torch.cat(values)

    return cost


@torch.no_grad()
def score_action_sequences(
    model,
    image: Tensor,
    actions: Tensor,
    *,
    use_graph: bool = True,
    discount: float = 0.99,
    chunk_size: int = 32,
) -> Tensor:
    """Score [C,H,A] sequences using learned rewards and continuations."""
    scores: list[Tensor] = []
    for start in range(0, actions.shape[0], chunk_size):
        chunk = actions[start : start + chunk_size]
        initial = image.expand(chunk.shape[0], -1, -1, -1)
        rollout = model.rollout(
            initial,
            chunk,
            use_graph=use_graph,
            deterministic=True,
            decode=False,
        )
        rewards = rollout["rewards"]
        continuations = rollout["continuations"]
        weight = rewards.new_ones(rewards.shape[0])
        score = rewards.new_zeros(rewards.shape[0])
        for step in range(rewards.shape[1]):
            score = score + weight * rewards[:, step]
            weight = weight * discount * continuations[:, step]
        scores.append(score)
    return torch.cat(scores)


@torch.no_grad()
def cem_action(
    model,
    image: Tensor,
    action_min: Tensor,
    action_max: Tensor,
    *,
    horizon: int = 8,
    candidates: int = 64,
    elites: int = 8,
    iterations: int = 3,
    use_graph: bool = True,
    discount: float = 0.99,
    chunk_size: int = 32,
) -> Tensor:
    """Cross-entropy-method planning shared by both backbones."""
    if image.shape[0] != 1:
        raise ValueError("CEM expects exactly one current observation")
    if not 1 <= elites <= candidates:
        raise ValueError("Require 1 <= elites <= candidates")
    center = (action_min + action_max) / 2
    scale = (action_max - action_min) / 2
    mean = center.expand(horizon, -1).clone()
    std = scale.expand(horizon, -1).clone()
    for _ in range(iterations):
        noise = torch.randn(
            candidates,
            horizon,
            action_min.numel(),
            device=image.device,
            dtype=image.dtype,
        )
        sequences = (mean[None] + std[None] * noise).clamp(
            action_min, action_max
        )
        scores = score_action_sequences(
            model,
            image,
            sequences,
            use_graph=use_graph,
            discount=discount,
            chunk_size=chunk_size,
        )
        elite = sequences[scores.topk(elites).indices]
        mean = elite.mean(0)
        std = elite.std(0, unbiased=False).clamp_min(0.05 * scale)
    return mean[0].clamp(action_min, action_max)
