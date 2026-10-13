"""Online DMC replay and DreamerV3 actor--critic utilities.

The defaults in this module mirror the pinned burchim DreamerV3-PyTorch DMC
configuration.  Replay storage is episode-deduplicated, but sampling remains
uniform over every legal length-L sliding window, which is distributionally
equivalent to the upstream replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from dm_control import suite
from torch import Tensor, nn

from .backbones.dreamerv3 import BurchimPolicyNetwork, BurchimValueNetwork
from .loss_balance import GradientRatioOTController


BURCHIM_DMC_DEFAULTS: dict[str, Any] = {
    "image_size": 64,
    "action_repeat": 2,
    "num_envs": 4,
    "replay_capacity": 1_000_000,
    "prefill_windows": 100,
    "batch_size": 16,
    "sequence_length": 64,
    "imagination_horizon": 15,
    "epochs": 50,
    "steps_per_epoch": 5000,
    "env_step_period": 512,
    "model_lr": 1e-4,
    "actor_lr": 3e-5,
    "critic_lr": 3e-5,
    "model_eps": 1e-8,
    "actor_eps": 1e-5,
    "critic_eps": 1e-5,
    "model_grad_clip": 1000.0,
    "actor_grad_clip": 100.0,
    "critic_grad_clip": 100.0,
    "gamma": 0.997,
    "lambda_td": 0.95,
    "entropy_weight": 3e-4,
    "critic_ema_decay": 0.02,
    "critic_slow_reg_scale": 1.0,
    "return_norm_decay": 0.99,
    "return_quantile_low": 0.05,
    "return_quantile_high": 0.95,
    "return_norm_min_range": 1.0,
    "eval_episodes": 10,
}


@dataclass
class ReplayEpisode:
    images: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    firsts: np.ndarray

    def __len__(self) -> int:
        return int(self.images.shape[0])


class SlidingWindowReplay:
    """Uniform replay over legal sliding windows without RGB duplication."""

    state_version = 1

    def __init__(self, sequence_length: int, capacity: int) -> None:
        self.sequence_length = int(sequence_length)
        # As upstream, capacity counts replay *windows*, not unique RGB frames.
        self.capacity = int(capacity)
        self.episodes: list[ReplayEpisode] = []
        self.streams: dict[int, dict[str, list[Any]]] = {}
        self.total_frames = 0
        self._revision = 0
        self._eligible_cache: dict[
            tuple[int, bool, float | None, str, float | None], np.ndarray
        ] = {}

    def _touch(self) -> None:
        self._revision += 1
        self._eligible_cache.clear()

    @property
    def num_windows(self) -> int:
        complete = sum(
            max(0, len(episode) - self.sequence_length + 1)
            for episode in self.episodes
        )
        active = sum(
            max(0, len(stream["images"]) - self.sequence_length + 1)
            for stream in self.streams.values()
        )
        return complete + active

    def start(self, stream_id: int, image: np.ndarray) -> None:
        action_dim = 0
        existing = self.streams.get(stream_id)
        if existing and existing["actions"]:
            action_dim = int(np.asarray(existing["actions"][-1]).shape[-1])
        self.streams[stream_id] = {
            "images": [np.asarray(image, dtype=np.uint8)],
            "actions": [np.zeros(action_dim, dtype=np.float32)],
            "rewards": [np.float32(0.0)],
            "dones": [np.float32(0.0)],
            "firsts": [np.float32(1.0)],
        }
        self.total_frames += 1
        self._touch()

    def start_with_action_dim(
        self, stream_id: int, image: np.ndarray, action_dim: int
    ) -> None:
        self.streams[stream_id] = {
            "images": [np.asarray(image, dtype=np.uint8)],
            "actions": [np.zeros(action_dim, dtype=np.float32)],
            "rewards": [np.float32(0.0)],
            "dones": [np.float32(0.0)],
            "firsts": [np.float32(1.0)],
        }
        self.total_frames += 1
        self._touch()

    def append(
        self,
        stream_id: int,
        image: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        stream = self.streams[stream_id]
        stream["images"].append(np.asarray(image, dtype=np.uint8))
        stream["actions"].append(np.asarray(action, dtype=np.float32))
        stream["rewards"].append(np.float32(reward))
        stream["dones"].append(np.float32(done))
        stream["firsts"].append(np.float32(0.0))
        self.total_frames += 1
        self._touch()

    @staticmethod
    def _pack(stream: dict[str, list[Any]]) -> ReplayEpisode:
        return ReplayEpisode(
            images=np.stack(stream["images"]),
            actions=np.stack(stream["actions"]),
            rewards=np.asarray(stream["rewards"], dtype=np.float32),
            dones=np.asarray(stream["dones"], dtype=np.float32),
            firsts=np.asarray(stream["firsts"], dtype=np.float32),
        )

    def finish(self, stream_id: int) -> None:
        # Keep the per-environment stream continuous. Upstream Burchim
        # windows can cross time-limit resets and rely on is_first to reset
        # RSSM inside the sampled sequence.
        self._enforce_capacity()

    def append_reset(
        self, stream_id: int, image: np.ndarray, action_dim: int
    ) -> None:
        stream = self.streams[stream_id]
        stream["images"].append(np.asarray(image, dtype=np.uint8))
        stream["actions"].append(np.zeros(action_dim, dtype=np.float32))
        stream["rewards"].append(np.float32(0.0))
        stream["dones"].append(np.float32(0.0))
        stream["firsts"].append(np.float32(1.0))
        self.total_frames += 1
        self._touch()
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        while self.num_windows > self.capacity and self.episodes:
            oldest = self.episodes[0]
            if len(oldest) <= self.sequence_length:
                self.episodes.pop(0)
                self.total_frames -= len(oldest)
                continue
            self.episodes[0] = ReplayEpisode(
                images=oldest.images[1:],
                actions=oldest.actions[1:],
                rewards=oldest.rewards[1:],
                dones=oldest.dones[1:],
                firsts=oldest.firsts[1:],
            )
            self.total_frames -= 1
        while self.num_windows > self.capacity and self.streams:
            stream_id = max(
                self.streams,
                key=lambda key: len(self.streams[key]["images"]),
            )
            stream = self.streams[stream_id]
            if len(stream["images"]) <= self.sequence_length:
                break
            for key in stream:
                stream[key].pop(0)
            self.total_frames -= 1

    @staticmethod
    def _source_length(
        source: ReplayEpisode | dict[str, list[Any]],
    ) -> int:
        if isinstance(source, ReplayEpisode):
            return len(source)
        return len(source["images"])

    @staticmethod
    def _source_slice(
        source: ReplayEpisode | dict[str, list[Any]],
        key: str,
        start: int,
        stop: int,
    ) -> np.ndarray:
        if isinstance(source, ReplayEpisode):
            return np.asarray(getattr(source, key)[start:stop])
        values = source[key][start:stop]
        if key in {"images", "actions"}:
            return np.stack(values)
        return np.asarray(values, dtype=np.float32)

    def _sources(
        self,
    ) -> tuple[
        list[ReplayEpisode | dict[str, list[Any]]],
        np.ndarray,
    ]:
        # Do not call _pack() for active streams here. Online streams grow
        # throughout training, and packing every full stream on every replay
        # sample made each update O(total replay frames). Sample directly from
        # the list-backed streams so work remains O(batch * sequence_length).
        sources: list[ReplayEpisode | dict[str, list[Any]]] = (
            list(self.episodes) + list(self.streams.values())
        )
        counts = np.asarray(
            [
                max(0, self._source_length(source) - self.sequence_length + 1)
                for source in sources
            ],
            dtype=np.int64,
        )
        keep = counts > 0
        return [
            source for source, valid in zip(sources, keep) if valid
        ], counts[keep]

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        *,
        avoid_internal_resets: bool = False,
        positive_fraction: float = 0.0,
        high_action_zero_fraction: float = 0.0,
        ordinary_zero_fraction: float = 0.0,
        high_action_threshold: float = 0.6,
        positive_threshold: float = 1e-6,
    ) -> dict[str, Tensor]:
        fractions = (
            positive_fraction,
            high_action_zero_fraction,
            ordinary_zero_fraction,
        )
        if any(not 0.0 <= fraction <= 1.0 for fraction in fractions):
            raise ValueError("replay fractions must be in [0, 1]")
        if sum(fractions) > 1.0 + 1e-9:
            raise ValueError("replay fractions must sum to at most 1")
        if high_action_threshold <= 0.0:
            raise ValueError("high_action_threshold must be positive")
        episodes, counts = self._sources()
        total = int(counts.sum())
        if total < 1:
            raise RuntimeError("Replay has no complete sequence window")
        offsets = np.cumsum(counts)
        uniform_pool = self._eligible_indices(
            episodes,
            counts,
            avoid_internal_resets=avoid_internal_resets,
            positive_threshold=None,
            reward_group="any",
            high_action_threshold=None,
        )
        if len(uniform_pool) < 1:
            raise RuntimeError("Replay has no eligible sequence window")
        positive_count = int(round(int(batch_size) * positive_fraction))
        high_action_zero_count = int(
            round(int(batch_size) * high_action_zero_fraction)
        )
        ordinary_zero_count = int(
            round(int(batch_size) * ordinary_zero_fraction)
        )
        assigned = positive_count + high_action_zero_count + ordinary_zero_count
        if assigned > int(batch_size):
            overflow = assigned - int(batch_size)
            ordinary_zero_count = max(0, ordinary_zero_count - overflow)
            assigned = positive_count + high_action_zero_count + ordinary_zero_count
        uniform_count = int(batch_size) - assigned
        selected_parts: list[np.ndarray] = []
        # 0=uniform, 1=positive, 2=high-action zero, 3=ordinary zero.
        # Keep the stratum label through shuffling so downstream diagnostics
        # can report the hard-negative reward error without re-inferring it.
        selected_group_parts: list[np.ndarray] = []
        if positive_count:
            positive_pool = self._eligible_indices(
                episodes,
                counts,
                avoid_internal_resets=avoid_internal_resets,
                positive_threshold=positive_threshold,
                reward_group="positive",
                high_action_threshold=None,
            )
            if len(positive_pool) < 1:
                raise RuntimeError(
                    "Replay has no positive-reward sequence window; collect "
                    "more random prefill before stratified sampling"
                )
            selected_parts.append(
                np.random.choice(positive_pool, size=positive_count, replace=True)
            )
            selected_group_parts.append(
                np.full(positive_count, 1, dtype=np.int64)
            )
        if high_action_zero_count:
            high_action_zero_pool = self._eligible_indices(
                episodes,
                counts,
                avoid_internal_resets=avoid_internal_resets,
                positive_threshold=positive_threshold,
                reward_group="zero",
                high_action_threshold=high_action_threshold,
            )
            if len(high_action_zero_pool) < 1:
                raise RuntimeError(
                    "Replay has no high-action zero-reward sequence window; "
                    "collect more exploratory prefill or lower the threshold"
                )
            selected_parts.append(
                np.random.choice(
                    high_action_zero_pool,
                    size=high_action_zero_count,
                    replace=True,
                )
            )
            selected_group_parts.append(
                np.full(high_action_zero_count, 2, dtype=np.int64)
            )
        if ordinary_zero_count:
            ordinary_zero_pool = self._eligible_indices(
                episodes,
                counts,
                avoid_internal_resets=avoid_internal_resets,
                positive_threshold=positive_threshold,
                reward_group="zero",
                high_action_threshold=-high_action_threshold,
            )
            if len(ordinary_zero_pool) < 1:
                raise RuntimeError(
                    "Replay has no ordinary zero-reward sequence window; "
                    "collect more replay or raise the high-action threshold"
                )
            selected_parts.append(
                np.random.choice(
                    ordinary_zero_pool,
                    size=ordinary_zero_count,
                    replace=True,
                )
            )
            selected_group_parts.append(
                np.full(ordinary_zero_count, 3, dtype=np.int64)
            )
        if uniform_count:
            selected_parts.append(
                np.random.choice(uniform_pool, size=uniform_count, replace=True)
            )
            selected_group_parts.append(
                np.zeros(uniform_count, dtype=np.int64)
            )
        selected = np.concatenate(selected_parts)
        selected_groups = np.concatenate(selected_group_parts)
        permutation = np.random.permutation(len(selected))
        selected = selected[permutation]
        selected_groups = selected_groups[permutation]
        rows: dict[str, list[np.ndarray]] = {
            key: [] for key in ("images", "actions", "rewards", "dones", "firsts")
        }
        for flat_index in selected:
            episode_index = int(np.searchsorted(offsets, flat_index, side="right"))
            previous = 0 if episode_index == 0 else int(offsets[episode_index - 1])
            start = int(flat_index - previous)
            stop = start + self.sequence_length
            episode = episodes[episode_index]
            for key in rows:
                rows[key].append(
                    self._source_slice(episode, key, start, stop)
                )
        return {
            "images": torch.from_numpy(np.stack(rows["images"])).to(device),
            "actions": torch.from_numpy(np.stack(rows["actions"])).to(device),
            "rewards": torch.from_numpy(np.stack(rows["rewards"])).to(device),
            "dones": torch.from_numpy(np.stack(rows["dones"])).to(device),
            "firsts": torch.from_numpy(np.stack(rows["firsts"])).to(device),
            "replay_group": torch.from_numpy(selected_groups).to(device),
        }

    def _eligible_indices(
        self,
        sources: list[ReplayEpisode | dict[str, list[Any]]],
        counts: np.ndarray,
        *,
        avoid_internal_resets: bool,
        positive_threshold: float | None,
        reward_group: str,
        high_action_threshold: float | None,
    ) -> np.ndarray:
        if reward_group not in {"any", "positive", "zero"}:
            raise ValueError(f"unknown replay reward group: {reward_group}")
        key = (
            self._revision,
            avoid_internal_resets,
            positive_threshold,
            reward_group,
            high_action_threshold,
        )
        cached = self._eligible_cache.get(key)
        if cached is not None:
            return cached
        eligible: list[np.ndarray] = []
        flat_offset = 0
        for source, count_value in zip(sources, counts):
            count = int(count_value)
            starts = np.arange(count, dtype=np.int64)
            valid = np.ones(count, dtype=bool)
            if avoid_internal_resets:
                firsts = self._source_slice(
                    source, "firsts", 0, self._source_length(source)
                ).astype(bool)
                prefix = np.concatenate(([0], np.cumsum(firsts, dtype=np.int64)))
                valid &= (prefix[starts + self.sequence_length] - prefix[starts + 1]) == 0
            if reward_group != "any":
                if positive_threshold is None:
                    raise ValueError("grouped replay requires positive_threshold")
                rewards = self._source_slice(
                    source, "rewards", 0, self._source_length(source)
                ) > positive_threshold
                prefix = np.concatenate(([0], np.cumsum(rewards, dtype=np.int64)))
                positive = (
                    prefix[starts + self.sequence_length]
                    - prefix[starts + 1]
                ) > 0
                valid &= positive if reward_group == "positive" else ~positive
            if high_action_threshold is not None:
                actions = self._source_slice(
                    source, "actions", 0, self._source_length(source)
                )
                action_magnitude = np.abs(actions).mean(axis=-1)
                action_prefix = np.concatenate(
                    ([0.0], np.cumsum(action_magnitude, dtype=np.float64))
                )
                window_action_mean = (
                    action_prefix[starts + self.sequence_length]
                    - action_prefix[starts + 1]
                ) / max(1, self.sequence_length - 1)
                if high_action_threshold > 0.0:
                    valid &= window_action_mean >= high_action_threshold
                else:
                    valid &= window_action_mean < -high_action_threshold
            eligible.append(flat_offset + starts[valid])
            flat_offset += count
        result = (
            np.concatenate(eligible)
            if eligible
            else np.empty(0, dtype=np.int64)
        )
        self._eligible_cache[key] = result
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.state_version,
            "sequence_length": self.sequence_length,
            "capacity": self.capacity,
            "episodes": self.episodes,
            "streams": self.streams,
            "total_frames": self.total_frames,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 0)) != self.state_version:
            raise ValueError("Unsupported online replay checkpoint")
        if int(state["sequence_length"]) != self.sequence_length:
            raise ValueError("Replay sequence length differs from checkpoint")
        self.episodes = list(state["episodes"])
        self.streams = dict(state["streams"])
        self.total_frames = int(state["total_frames"])
        self._touch()


class OnlineDMCBatch:
    """Four synchronous DMC pixel environments following burchim defaults."""

    def __init__(
        self,
        domain: str,
        task: str,
        num_envs: int,
        seed: int,
        image_size: int = 64,
        action_repeat: int = 2,
        camera: int = 0,
    ) -> None:
        self.domain = domain
        self.task = task
        self.num_envs = int(num_envs)
        self.image_size = int(image_size)
        self.action_repeat = int(action_repeat)
        self.camera = int(camera)
        self.envs = [
            suite.load(
                domain,
                task,
                task_kwargs={"random": int(seed) + index},
            )
            for index in range(self.num_envs)
        ]
        spec = self.envs[0].action_spec()
        self.action_dim = int(np.prod(spec.shape))
        self.action_min = np.asarray(spec.minimum, dtype=np.float32)
        self.action_max = np.asarray(spec.maximum, dtype=np.float32)
        self.timesteps: list[Any] = []
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int64)

    def _render(self, index: int) -> np.ndarray:
        return self.envs[index].physics.render(
            height=self.image_size,
            width=self.image_size,
            camera_id=self.camera,
        ).transpose(2, 0, 1).copy()

    def reset_all(self) -> list[np.ndarray]:
        self.timesteps = [env.reset() for env in self.envs]
        self.episode_returns.fill(0.0)
        self.episode_steps.fill(0)
        return [self._render(index) for index in range(self.num_envs)]

    def random_actions(self) -> np.ndarray:
        return np.random.uniform(
            self.action_min, self.action_max, (self.num_envs, self.action_dim)
        ).astype(np.float32)

    def step(
        self, actions: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[dict[str, float]]]:
        images: list[np.ndarray] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        lasts = np.zeros(self.num_envs, dtype=np.bool_)
        completed: list[dict[str, float]] = []
        for index, (env, action) in enumerate(zip(self.envs, actions)):
            reward = 0.0
            timestep = None
            for _ in range(self.action_repeat):
                timestep = env.step(np.clip(action, self.action_min, self.action_max))
                reward += float(timestep.reward or 0.0)
                if timestep.last():
                    break
            assert timestep is not None
            self.timesteps[index] = timestep
            self.episode_returns[index] += reward
            self.episode_steps[index] += self.action_repeat
            rewards[index] = reward
            lasts[index] = bool(timestep.last())
            images.append(self._render(index))
            if lasts[index]:
                completed.append({
                    "stream": float(index),
                    "return": float(self.episode_returns[index]),
                    "steps": float(self.episode_steps[index]),
                })
        return images, rewards, lasts, completed

    def reset_one(self, index: int) -> np.ndarray:
        self.timesteps[index] = self.envs[index].reset()
        self.episode_returns[index] = 0.0
        self.episode_steps[index] = 0
        return self._render(index)


class DreamerBehavior(nn.Module):
    """Pinned burchim policy/value networks plus slow critic and quantiles."""

    def __init__(self, action_dim: int, feature_dim: int) -> None:
        super().__init__()
        norm = {
            "class": "LayerNorm",
            "params": {"eps": 1e-3, "convert_float32": True},
        }
        self.actor = BurchimPolicyNetwork(
            num_actions=action_dim,
            hidden_size=512,
            num_mlp_layers=2,
            feat_size=feature_dim,
            discrete=False,
            norm=norm,
        )
        self.critic = BurchimValueNetwork(
            hidden_size=512,
            num_mlp_layers=2,
            feat_size=feature_dim,
            norm=norm,
        )
        import copy

        self.slow_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.register_buffer("return_low", torch.tensor(0.0))
        self.register_buffer("return_high", torch.tensor(0.0))

    @torch.no_grad()
    def update_slow_critic(self, decay: float = 0.02) -> None:
        for target, source in zip(
            self.slow_critic.parameters(), self.critic.parameters()
        ):
            target.mul_(1.0 - decay).add_(source, alpha=decay)

    @torch.no_grad()
    def update_return_scale(
        self, returns: Tensor, decay: float = 0.99
    ) -> tuple[Tensor, Tensor]:
        low = torch.quantile(returns.detach(), 0.05)
        high = torch.quantile(returns.detach(), 0.95)
        self.return_low.mul_(decay).add_(low, alpha=1.0 - decay)
        self.return_high.mul_(decay).add_(high, alpha=1.0 - decay)
        scale = (self.return_high - self.return_low).clamp_min(1.0)
        return self.return_low.detach(), scale.detach()


def lambda_returns(
    rewards: Tensor,
    values: Tensor,
    discounts: Tensor,
    lambda_td: float = 0.95,
) -> Tensor:
    """Pinned burchim backwards TD(lambda) recurrence."""
    intermediate = rewards + discounts * (1.0 - lambda_td) * values
    result = [values[:, -1]]
    for step in reversed(range(intermediate.shape[1])):
        result.append(
            intermediate[:, step]
            + discounts[:, step] * lambda_td * result[-1]
        )
    return torch.stack(list(reversed(result))[:-1], dim=1)


__all__ = [
    "BURCHIM_DMC_DEFAULTS",
    "DreamerBehavior",
    "GradientRatioOTController",
    "OnlineDMCBatch",
    "SlidingWindowReplay",
    "lambda_returns",
]
