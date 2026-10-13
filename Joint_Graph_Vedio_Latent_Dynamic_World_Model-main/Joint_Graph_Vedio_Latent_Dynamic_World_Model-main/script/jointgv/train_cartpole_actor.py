"""Actor-critic in a frozen DINO-WM imagination for the RM65-B cart-pole.

One planning track (``--mode base``: backbone alone, ``--mode graph``:
backbone + JointGV correction) of one frozen P checkpoint imagines
``--horizon`` steps ahead from real starts drawn anywhere in the training
episodes (not only episode starts). At every imagined step the actor picks
the action from the imagined frame and its change from the previous frame,
and a reward head from train_reward_heads.py scores the result: either one
head fitted on real frames and shared by both tracks (MODE real, the main
comparison: only the imagination differs) or the track's own head fitted on
its imagination (the per-track reference). The actor learns with REINFORCE
against a learned value baseline (lambda returns, Dreamer-style return
normalisation, entropy bonus); no gradient flows through the world model, so
memory stays small. Backbone and graph actors share every setting and seed.

Every ``--eval-every`` updates the deterministic actor controls the real
MuJoCo cart-pole from its own 64x64 camera frames (encoded by the same frozen
DINO encoder) at 25 Hz, from the control protocol's start (lean uniform in
+/-7 deg, at rest): success = upright for ``--eval-steps`` steps with no
workspace-guard command. Progress and every evaluation are printed live; the
best actor (by success, then survival) is kept as ``actor_best.pt``.

    python script/jointgv/train_cartpole_actor.py --mode graph \\
        --checkpoint runs/.../checkpoints/epoch_0050.pt \\
        --reward-head runs/rm65b_cartpole_reward_heads_v3_ep50/graph.pt --amp \\
        --output runs/rm65b_cartpole_actor_v3_graph
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Windows" else "egl")

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jointgv import JointGraphVideoWorldModel  # noqa: E402
from jointgv.dmc_protocol import resolve_camera  # noqa: E402
from jointgv.envs import load_environment  # noqa: E402
from jointgv.envs.rm65b_cartpole import (  # noqa: E402
    CONTROL_TIMESTEP,
    HINGE_PRESETS,
    hinge_parameters,
    hinge_task_kwargs,
)
from jointgv.online import lambda_returns  # noqa: E402
from jointgv.planning import LatentRollout, load_track_reward_head  # noqa: E402

DOMAIN, TASK = "rm65b", "cartpole"


class GridEncoder(nn.Module):
    """Patch-token grid of a frame + its change from the previous frame ->
    feature vector (same input as the per-track reward head)."""

    def __init__(self, token_dim: int, grid: int, hidden: int = 64, features: int = 256):
        super().__init__()
        self.token_dim, self.grid = token_dim, grid
        channels = 2 * token_dim
        self.register_buffer("input_mean", torch.zeros(channels))
        self.register_buffer("input_std", torch.ones(channels))

        def block(inputs: int, stride: int) -> list[nn.Module]:
            return [nn.Conv2d(inputs, hidden, 3, stride=stride, padding=1),
                    nn.GroupNorm(8, hidden), nn.SiLU()]

        self.net = nn.Sequential(
            *block(channels, 1), *block(hidden, 2), *block(hidden, 2),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(hidden * 16, features), nn.LayerNorm(features), nn.SiLU(),
        )

    @staticmethod
    def raw(current: Tensor, previous: Tensor) -> Tensor:
        return torch.cat((current, current - previous), dim=-1).float()

    def forward(self, current: Tensor, previous: Tensor) -> Tensor:
        x = (self.raw(current, previous) - self.input_mean) / self.input_std
        x = x.reshape(-1, self.grid, self.grid, 2 * self.token_dim).permute(0, 3, 1, 2)
        return self.net(x)


class Actor(nn.Module):
    """Tanh-squashed Gaussian policy over the normalised pivot command."""

    def __init__(self, token_dim: int, grid: int, action_dim: int):
        super().__init__()
        self.encoder = GridEncoder(token_dim, grid)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2 * action_dim))

    def distribution(self, current: Tensor, previous: Tensor) -> torch.distributions.Normal:
        mean, log_std = self.head(self.encoder(current, previous)).chunk(2, dim=-1)
        return torch.distributions.Normal(mean, log_std.clamp(-5.0, 0.0).exp())

    @staticmethod
    def _log_prob(dist: torch.distributions.Normal, raw: Tensor) -> Tensor:
        # log|d tanh/dx| = 2 (log 2 - x - softplus(-2x)), stable when saturated.
        log_det = 2.0 * (math.log(2.0) - raw - nn.functional.softplus(-2.0 * raw))
        return (dist.log_prob(raw) - log_det).sum(-1)

    def sample(self, current: Tensor, previous: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """action in [-1, 1], its log-probability (score-function term) and a
        one-sample estimate of the entropy of the squashed action.

        The entropy is of the action actually sent, after tanh: saturating at
        +/-1 lowers it, so the bonus cannot be raised by a wider pre-squash
        Gaussian (which only drives the actor to random bang-bang). Its
        estimate is reparameterised, so it has a nonzero gradient."""
        dist = self.distribution(current, previous)
        raw = dist.rsample()
        action = torch.tanh(raw.detach())
        return action, self._log_prob(dist, raw.detach()), -self._log_prob(dist, raw)

    @torch.no_grad()
    def act(self, current: Tensor, previous: Tensor) -> Tensor:
        return torch.tanh(self.distribution(current, previous).mean)


class Critic(nn.Module):
    def __init__(self, token_dim: int, grid: int):
        super().__init__()
        self.encoder = GridEncoder(token_dim, grid)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, current: Tensor, previous: Tensor) -> Tensor:
        return self.head(self.encoder(current, previous)).squeeze(-1)


class StartSampler:
    """Real starts (context frames + current frame) anywhere before the fall
    with the pole within ``max_lean_deg`` of upright. The lean of frame s is
    read from the recorded reward of the step into it (reward = action repeat
    x (1 - |angle| / 90 deg)); leaner starts are unrecoverable on the rail."""

    def __init__(self, root: Path, context: int, max_lean_deg: float, rng: np.random.Generator):
        self.images, self.actions, self.index = [], [], []
        # Simulated pole hinges (damping, friction) in the data; episodes that
        # predate the record used the default hinge.
        self.hinges: set[tuple[float, float]] = set()
        paths = sorted(root.glob("*.npz"))
        for path in tqdm(paths, desc=f"loading {root}", dynamic_ncols=True):
            with np.load(path, allow_pickle=False) as episode:
                images, actions = episode["image"], episode["action"]
                rewards, repeat = episode["reward"], int(episode["action_repeat"])
                fall = int(episode["fall_step"]) if "fall_step" in episode.files else -1
                self.hinges.add(
                    (round(float(episode["cartpole_hinge_damping"]), 9),
                     round(float(episode["cartpole_hinge_friction"]), 9))
                    if "cartpole_hinge_damping" in episode.files
                    else HINGE_PRESETS["damper"]
                )
            last = len(actions) if fall < 0 else min(len(actions), fall)
            lean = 90.0 * (1.0 - rewards / repeat)      # lean[s - 1] = lean of frame s
            starts = [s for s in range(context, last + 1) if lean[s - 1] <= max_lean_deg]
            if not starts:
                continue
            episode_id = len(self.images)
            self.images.append(images)
            self.actions.append(actions.astype(np.float32))
            self.index.extend((episode_id, s) for s in starts)
        if not self.index:
            raise SystemExit(f"no usable starts in {root}")
        self.episodes = len(paths)
        self.context, self.rng = context, rng

    def sample(self, count: int, device: torch.device) -> tuple[Tensor, Tensor]:
        picks = self.rng.integers(0, len(self.index), size=count)
        frames, actions = [], []
        for pick in picks:
            episode, s = self.index[pick]
            frames.append(self.images[episode][s - self.context : s + 1])
            actions.append(self.actions[episode][s - self.context : s])
        frames = torch.from_numpy(np.stack(frames)).to(device).permute(0, 1, 4, 2, 3).float() / 255.0
        return frames, torch.from_numpy(np.stack(actions)).to(device)


@torch.no_grad()
def encode(model, image: np.ndarray, device: torch.device) -> Tensor:
    # MuJoCo renders bottom-up into a flipped view (negative strides).
    frame = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    frame = frame.permute(2, 0, 1).float()[None] / 255.0
    return model.vae.encode(frame, sample=False)[0]


@torch.no_grad()
def evaluate(
    actor: Actor, model, args, device: torch.device, seed: int, episodes: int
) -> dict[str, float]:
    camera = resolve_camera(DOMAIN, -1)
    successes, survival, decide = [], [], []
    for episode in range(episodes):
        env = load_environment(
            DOMAIN, TASK, task_kwargs={"random": seed + episode, **hinge_task_kwargs(args.hinge)}
        )
        env.reset()
        current = encode(model, env.physics.render(args.size, args.size, camera_id=camera), device)
        previous = current
        steps, fell = 0, False
        for _ in range(args.eval_steps):
            started = time.perf_counter()
            action = actor.act(current, previous)[0].cpu().numpy()
            decide.append(time.perf_counter() - started)
            for _ in range(args.action_repeat):
                if env.step(action).last():
                    fell = True
                    break
            if fell:
                break
            steps += 1
            started = time.perf_counter()
            previous, current = current, encode(
                model, env.physics.render(args.size, args.size, camera_id=camera), device
            )
            decide[-1] += time.perf_counter() - started
        successes.append(steps == args.eval_steps and not fell and env.task.unsafe_commands == 0)
        survival.append(steps * args.action_repeat * CONTROL_TIMESTEP)
    return {
        "success_rate": float(np.mean(successes)),
        "survival_s": float(np.mean(survival)),
        "ms_per_step": 1000.0 * float(np.mean(decide)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("base", "graph"), required=True)
    parser.add_argument("--reward-head", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/rm65b_cartpole_train_v3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=64, help="Imagined trajectories per update.")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument(
        "--context-frames", type=int, default=1,
        help="Real frames before each start, as the reward head was fitted (1 keeps the "
        "17-step padded falls in its 1 + 16-step windows).",
    )
    parser.add_argument("--start-max-lean", type=float, default=15.0, help="deg; leaner starts are skipped.")
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--lambda-td", type=float, default=0.95)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument(
        "--entropy", type=float, default=3e-4,
        help="Bonus on the squashed-action entropy (DreamerV3 actor entropy scale).",
    )
    parser.add_argument("--slow-critic-decay", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=150, help="150 x 0.04 s = 6 s.")
    parser.add_argument("--eval-seed", type=int, default=41000)
    parser.add_argument("--test-episodes", type=int, default=50, help="0 skips the final test.")
    parser.add_argument("--test-seed", type=int, default=42000)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hinge", default="damper",
        help="Pole hinge of the evaluation plant: damper, bearing606 or DAMPING,FRICTION; "
        "must match the hinge the --data episodes were simulated with.",
    )
    parser.add_argument("--device", default="cuda", help="cuda (default) or cpu; never falls back silently.")
    args = parser.parse_args()
    if min(args.updates, args.batch, args.horizon, args.eval_every, args.eval_episodes, args.context_frames) < 1:
        parser.error("--updates, --batch, --horizon, --eval-every, --eval-episodes and --context-frames must be positive")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(
            "CUDA is not visible in this terminal (check $env:CUDA_VISIBLE_DEVICES); "
            "open a fresh terminal, or pass --device cpu on purpose"
        )
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = JointGraphVideoWorldModel(**checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    reward_head, track = load_track_reward_head(args.reward_head, device)
    reward_head.requires_grad_(False)
    # A head fitted on real frames (mode "real") is shared by both tracks and
    # never saw imagination, so only the encoder (checkpoint) must match.
    shared_head = track["mode"] == "real"
    problems = []
    if track.get("checkpoint_epoch") != checkpoint.get("epoch"):
        problems.append(f"fitted on epoch {track.get('checkpoint_epoch')}, checkpoint is epoch {checkpoint.get('epoch')}")
    if not shared_head:
        if track["mode"] != args.mode:
            problems.append(f"fitted for mode {track['mode']}")
        if track["horizon"] < args.horizon or track["context_frames"] != args.context_frames:
            problems.append("fitted with a different horizon / context")
        if bool(track.get("amp")) != args.amp:
            problems.append(f"fitted with amp={bool(track.get('amp'))}, pass the same --amp")
        if track["history_window"] != (args.history_window or None):
            problems.append(f"fitted with history window {track['history_window']}")
    if problems:
        parser.error("--reward-head does not match: " + "; ".join(problems))
    rollout = LatentRollout(
        model, use_graph=args.mode == "graph",
        history_window=args.history_window or None, amp=args.amp,
    )
    try:
        hinge = hinge_parameters(args.hinge)
    except ValueError as error:
        parser.error(str(error))
    sampler = StartSampler(args.data, args.context_frames, args.start_max_lean, rng)
    expected = (round(hinge[0], 9), round(hinge[1], 9))
    if sampler.hinges != {expected}:
        parser.error(
            f"--hinge {args.hinge} {expected} does not match the hinge(s) of {args.data}: "
            f"{sorted(sampler.hinges)}"
        )
    frames, actions = sampler.sample(min(256, len(sampler.index)), device)
    with torch.no_grad():
        tokens = model.vae.encode(frames.flatten(0, 1), sample=False)[0]
        tokens = tokens.reshape(*frames.shape[:2], *tokens.shape[1:])
    token_count, token_dim = tokens.shape[-2:]
    grid = int(round(math.sqrt(token_count)))
    if grid * grid != token_count:
        raise SystemExit(f"{token_count} tokens are not a square patch grid")
    action_dim = int(checkpoint["model_config"]["action_dim"])
    actor = Actor(token_dim, grid, action_dim).to(device)
    critic = Critic(token_dim, grid).to(device)

    # Input standardisation from real start frames (shared by both nets).
    with torch.no_grad():
        raw = GridEncoder.raw(tokens[:, -1], tokens[:, -2]).flatten(0, -2)
        for net in (actor.encoder, critic.encoder):
            net.input_mean.copy_(raw.mean(0))
            net.input_std.copy_(raw.std(0).clamp_min(1e-4))
    slow_critic = copy.deepcopy(critic).requires_grad_(False)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)
    return_low = return_high = None

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        | {"checkpoint_epoch": checkpoint.get("epoch"), "starts": len(sampler.index),
           "reward_head_mode": track["mode"], "hinge_parameters": list(hinge)},
        indent=2), encoding="utf-8")
    metrics_path = args.output / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    print(
        f"[{device}] {args.mode} track of {args.checkpoint} (epoch {checkpoint.get('epoch')}); "
        f"reward head {args.reward_head} ({'shared, real frames' if shared_head else 'own, ' + track['mode'] + ' imagination'}); "
        f"{len(sampler.index)} real starts within "
        f"{args.start_max_lean:g} deg from {sampler.episodes} episodes; "
        f"imagination {args.batch} x {args.horizon} steps per update; evaluation hinge {args.hinge} "
        f"(damping {hinge[0]:g}, friction {hinge[1]:g})"
    )

    best = (-1.0, -1.0)
    window: list[dict[str, float]] = []
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        frames, actions = sampler.sample(args.batch, device)
        rollout.prepare(frames, actions)
        carry = rollout.initial(args.batch)
        currents, previouses = [carry["tokens"]], [carry["previous"]]
        log_probs, entropies, rewards, taken = [], [], [], []
        for _ in range(args.horizon):
            action, log_prob, entropy = actor.sample(carry["tokens"], carry["previous"])
            carry = rollout.step(carry, action.detach())
            with torch.no_grad():
                rewards.append(reward_head(carry["tokens"], carry["previous"]))
            previouses.append(carry["previous"])
            currents.append(carry["tokens"])
            log_probs.append(log_prob)
            entropies.append(entropy)
            taken.append(action.detach())
        current = torch.stack(currents, 1).flatten(0, 1)
        previous = torch.stack(previouses, 1).flatten(0, 1)
        rewards_t = torch.stack(rewards, 1)[..., None]
        with torch.no_grad():
            target_values = slow_critic(current, previous).reshape(args.batch, args.horizon + 1, 1)
            discounts = torch.full_like(rewards_t, args.gamma)
            returns = lambda_returns(rewards_t, target_values[:, 1:], discounts, args.lambda_td)
            low, high = torch.quantile(returns, 0.05), torch.quantile(returns, 0.95)
            return_low = low if return_low is None else 0.99 * return_low + 0.01 * low
            return_high = high if return_high is None else 0.99 * return_high + 0.01 * high
            scale = (return_high - return_low).clamp_min(1.0)
        values = critic(current, previous).reshape(args.batch, args.horizon + 1)
        critic_loss = 0.5 * (values[:, :-1] - returns[..., 0]).square().mean()
        advantage = ((returns[..., 0] - values[:, :-1].detach()) / scale).detach()
        log_prob_t, entropy_t = torch.stack(log_probs, 1), torch.stack(entropies, 1)
        actor_loss = -(advantage * log_prob_t).mean() - args.entropy * entropy_t.mean()
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
        actor_opt.step()
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
        critic_opt.step()
        with torch.no_grad():
            for slow, fast in zip(slow_critic.parameters(), critic.parameters()):
                slow.mul_(1.0 - args.slow_critic_decay).add_(fast, alpha=args.slow_critic_decay)
        taken_t = torch.stack(taken, 1)
        window.append({
            "imagined_reward": float(rewards_t.mean()), "return": float(returns.mean()),
            "actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
            "entropy": float(entropy_t.detach().mean()), "action_abs": float(taken_t.abs().mean()),
        })
        if update % args.log_every == 0 or update == 1:
            mean = {k: float(np.mean([w[k] for w in window])) for k in window[0]}
            elapsed = time.perf_counter() - started
            print(
                f"update {update:5d}/{args.updates}  reward {mean['imagined_reward']:.3f}  "
                f"return {mean['return']:.2f}  actor {mean['actor_loss']:+.3f}  "
                f"critic {mean['critic_loss']:.3f}  entropy {mean['entropy']:+.2f}  "
                f"|a| {mean['action_abs']:.2f}  {elapsed / update:.2f}s/update"
                + (f"  gpu {torch.cuda.max_memory_reserved(device) / 2**30:.1f}GB" if device.type == "cuda" else ""),
                flush=True,
            )
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"update": update, **mean}) + "\n")
            window = []
        if update % args.eval_every == 0 or update == args.updates:
            actor.eval()
            result = evaluate(actor, model, args, device, args.eval_seed, args.eval_episodes)
            actor.train()
            key = (result["success_rate"], result["survival_s"])
            improved = key > best
            if improved:
                best = key
                torch.save({"actor": actor.state_dict(), "update": update, "eval": result,
                            "token_dim": token_dim, "grid": grid, "action_dim": action_dim,
                            "mode": args.mode, "checkpoint": str(args.checkpoint), "hinge": args.hinge},
                           args.output / "actor_best.pt")
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "update": update,
                        "token_dim": token_dim, "grid": grid, "action_dim": action_dim,
                        "mode": args.mode, "checkpoint": str(args.checkpoint), "hinge": args.hinge},
                       args.output / "actor_latest.pt")
            print(
                f"== eval @ update {update}: success {result['success_rate']:.0%} "
                f"({args.eval_episodes} episodes, +/-7 deg at rest), survival "
                f"{result['survival_s']:.2f}s / {args.eval_steps * args.action_repeat * CONTROL_TIMESTEP:.1f}s, "
                f"{result['ms_per_step']:.1f} ms per step (encode + actor)"
                + ("  |--> saved actor_best.pt" if improved else ""),
                flush=True,
            )
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"update": update, "eval": result}) + "\n")
    print(f"best: success {best[0]:.0%}, survival {best[1]:.2f}s -> {args.output / 'actor_best.pt'}")
    # actor_best.pt was picked on the --eval-seed episodes; report it on
    # episodes the selection never saw.
    if args.test_episodes > 0:
        saved = torch.load(args.output / "actor_best.pt", map_location=device, weights_only=False)
        actor.load_state_dict(saved["actor"])
        actor.eval()
        result = evaluate(actor, model, args, device, args.test_seed, args.test_episodes)
        print(
            f"== test (actor_best.pt from update {saved['update']}, {args.test_episodes} "
            f"fresh episodes from seed {args.test_seed}): success {result['success_rate']:.0%}, "
            f"survival {result['survival_s']:.2f}s, {result['ms_per_step']:.1f} ms per step",
            flush=True,
        )
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"test": result, "best_update": saved["update"]}) + "\n")


if __name__ == "__main__":
    main()
