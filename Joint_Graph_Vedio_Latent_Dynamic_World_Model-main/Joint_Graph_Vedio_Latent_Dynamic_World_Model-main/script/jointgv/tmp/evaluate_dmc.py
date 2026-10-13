from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MUJOCO_GL", "glfw" if platform.system() == "Windows" else "egl"
)

import numpy as np
import torch
from torch import Tensor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a JointGV checkpoint on a live DMC task."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--domain", default="reacher")
    parser.add_argument("--task", default="easy")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Maximum agent steps; 0 evaluates the complete DMC episode.",
    )
    parser.add_argument(
        "--allow-partial-episode", action="store_true",
        help="Acknowledge that --steps > 0 is not a reportable evaluation.",
    )
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--camera", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--planner", choices=("cem", "random", "actor", "q_guided_actor"),
        default="cem",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--q-candidates", type=int, default=16)
    parser.add_argument("--q-horizon", type=int, default=1)
    parser.add_argument("--q-selection", choices=("argmax",), default="argmax")
    mean_group = parser.add_mutually_exclusive_group()
    mean_group.add_argument(
        "--q-include-mean-action", dest="q_include_mean_action",
        action="store_true",
        help="Add the deterministic actor mode to sampled candidates (default).",
    )
    mean_group.add_argument(
        "--no-q-include-mean-action", dest="q_include_mean_action",
        action="store_false",
    )
    parser.set_defaults(q_include_mean_action=True)
    parser.add_argument(
        "--q-temperature", type=float, default=1.0,
        help="Scale actor samples around the actor mode; must be positive.",
    )
    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument("--enable-graph-conditioning", action="store_true")
    graph_group.add_argument("--disable-graph-conditioning", action="store_true")
    parser.add_argument(
        "--allow-unverified-checkpoint", action="store_true",
        help="Only for smoke tests with legacy or truncated training data.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/dmc_evaluation"))
    return parser


def _expand_batch(value: Tensor, count: int) -> Tensor:
    if value.shape[0] != 1:
        raise ValueError("Q-guided evaluation expects one real environment")
    return value.expand(count, *value.shape[1:]).clone()


def _expand_state(state: dict[str, Tensor], count: int) -> dict[str, Tensor]:
    return {key: _expand_batch(value, count) for key, value in state.items()}


@torch.no_grad()
def _score_q_candidates(
    core: Any,
    behavior: Any,
    state: dict[str, Tensor],
    candidates: Tensor,
    action_min: Tensor,
    action_max: Tensor,
    *,
    horizon: int,
    gamma: float,
    use_graph: bool,
    graph: Tensor | None,
    relations: Tensor | None,
) -> dict[str, Tensor]:
    """Score actor candidates from the current real posterior state."""
    count = candidates.shape[0]
    imagined_state = _expand_state(state, count)
    imagined_graph = None if graph is None else _expand_batch(graph, count)
    imagined_relations = (
        None if relations is None else _expand_batch(relations, count)
    )
    weight = candidates.new_ones(count)
    scores = candidates.new_zeros(count)
    first_reward = first_continuation = first_value = None
    feature = None
    for step in range(horizon):
        if step == 0:
            action = candidates
        else:
            assert feature is not None
            action = behavior.actor(feature).mode().clamp(action_min, action_max)
        prior, graph_output, _, _, _, _ = core._prior_step(
            imagined_state, action, imagined_graph, imagined_relations, use_graph
        )
        feature = core.rssm.get_feat(prior)
        reward = core.reward_head(feature).mode().squeeze(-1)
        continuation = core.continuation_head(
            feature
        ).base_dist.probs.squeeze(-1)
        value = behavior.critic(feature).mode().squeeze(-1)
        if step == 0:
            first_reward = reward
            first_continuation = continuation
            first_value = value
        scores = scores + weight * reward
        weight = weight * gamma * continuation
        imagined_state = prior
        if graph_output is not None:
            imagined_graph = graph_output.graph_next[:, 0]
            imagined_relations = graph_output.next_relation_logits[:, 0]
    assert feature is not None
    scores = scores + weight * behavior.critic(feature).mode().squeeze(-1)
    assert first_reward is not None
    assert first_continuation is not None
    assert first_value is not None
    return {
        "scores": scores,
        "predicted_reward": first_reward,
        "predicted_continuation": first_continuation,
        "predicted_value": first_value,
    }


@torch.no_grad()
def _actor_or_q_action(
    model: Any,
    behavior: Any,
    image: Tensor,
    previous_state: dict[str, Tensor] | None,
    previous_action: Tensor | None,
    action_min: Tensor,
    action_max: Tensor,
    *,
    planner: str,
    q_candidates: int,
    q_horizon: int,
    q_include_mean_action: bool,
    q_temperature: float,
    gamma: float,
    use_graph: bool,
    previous_graph: Tensor | None,
    previous_relations: Tensor | None,
) -> tuple[
    Tensor, dict[str, Tensor], Tensor | None, Tensor | None, dict[str, float]
]:
    core = model.dreamerv3_core
    if core is None:
        raise ValueError("actor planners require a DreamerV3 checkpoint")
    state, feature, graph, relations = core.policy_observe(
        image,
        previous_state,
        previous_action,
        use_graph=use_graph,
        previous_graph=previous_graph,
        previous_relations=previous_relations,
    )
    distribution = behavior.actor(feature)
    actor_action = distribution.mode().clamp(action_min, action_max)
    if planner == "actor":
        return actor_action, state, graph, relations, {}

    samples = distribution.sample((q_candidates,))[:, 0]
    samples = actor_action[0] + q_temperature * (samples - actor_action[0])
    candidates = samples.clamp(action_min, action_max)
    if q_include_mean_action:
        candidates = torch.cat((candidates, actor_action), dim=0)
    scored = _score_q_candidates(
        core,
        behavior,
        state,
        candidates,
        action_min,
        action_max,
        horizon=q_horizon,
        gamma=gamma,
        use_graph=use_graph,
        graph=graph,
        relations=relations,
    )
    scores = scored["scores"]
    selected_index = int(scores.argmax())
    selected = candidates[selected_index : selected_index + 1]
    diagnostics = {
        "candidate_score_mean": float(scores.mean()),
        "candidate_score_max": float(scores.max()),
        "candidate_score_min": float(scores.min()),
        "candidate_score_std": float(scores.std(unbiased=False)),
        "selected_score": float(scores[selected_index]),
        "selected_action_norm": float(selected.norm(dim=-1)[0]),
        "actor_mean_action_norm": float(actor_action.norm(dim=-1)[0]),
        "selected_candidate_rank": 1.0,
        "predicted_reward_selected": float(
            scored["predicted_reward"][selected_index]
        ),
        "predicted_value_selected": float(scored["predicted_value"][selected_index]),
        "predicted_continuation_selected": float(
            scored["predicted_continuation"][selected_index]
        ),
    }
    return selected, state, graph, relations, diagnostics


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative (0 means natural ending)")
    if args.steps > 0 and not args.allow_partial_episode:
        parser.error(
            "--steps > 0 truncates official evaluation; add "
            "--allow-partial-episode only for a smoke test"
        )
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.q_candidates < 1:
        parser.error("--q-candidates must be positive")
    if args.q_horizon < 1:
        parser.error("--q-horizon must be positive")
    if args.q_temperature <= 0.0:
        parser.error("--q-temperature must be positive")

    # Keep --help usable on local machines without the server's DMC runtime.
    from dm_control import suite

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jointgv import JointGraphVideoWorldModel
    from jointgv.dmc_protocol import (
        DREAMERV3_DMC_VISION,
        resolve_camera,
        validate_dreamerv3_vision,
    )
    from jointgv.online import BURCHIM_DMC_DEFAULTS, DreamerBehavior
    from jointgv.planning import cem_action

    try:
        validate_dreamerv3_vision(args.size, args.action_repeat, args.camera)
    except ValueError as error:
        parser.error(str(error))
    render_camera = resolve_camera(args.domain, args.camera)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not args.allow_unverified_checkpoint:
        checkpoint_protocol = checkpoint.get("data_protocol")
        if checkpoint_protocol is None:
            # Full online checkpoints created before staged training did not
            # persist data_protocol, but do contain both the Dreamer model and
            # behavior state. Accept that exact legacy shape for evaluation;
            # no training-stage metadata is required.
            legacy_full_dreamer = (
                isinstance(checkpoint.get("model"), dict)
                and isinstance(checkpoint.get("behavior"), dict)
                and checkpoint.get("model_config", {}).get("backbone")
                == "dreamerv3"
            )
            if not legacy_full_dreamer:
                raise ValueError(
                    "Checkpoint was not trained from verified official DMC "
                    "vision data. Recollect and retrain, or use "
                    "--allow-unverified-checkpoint only for a smoke test."
                )
            print(
                "warning: accepting legacy full DreamerV3 online checkpoint "
                "without data_protocol metadata"
            )
        elif checkpoint_protocol != DREAMERV3_DMC_VISION:
            raise ValueError(
                "Checkpoint was not trained from verified official DMC vision "
                "data. Recollect and retrain, or use --allow-unverified-checkpoint "
                "only for a smoke test."
            )
        if int(checkpoint.get("incomplete_dmc_files", 0)) != 0:
            raise ValueError(
                "Checkpoint used truncated DMC episodes and cannot be used "
                "for an official evaluation."
            )

    probe_env = suite.load(
        args.domain, args.task, task_kwargs={"random": args.seed}
    )
    probe_spec = probe_env.action_spec()
    action_dim = int(np.prod(probe_spec.shape))
    model_config = dict(checkpoint["model_config"])
    model_config.setdefault("action_dim", action_dim)
    model = JointGraphVideoWorldModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    behavior = None
    if args.planner in {"actor", "q_guided_actor"}:
        if model.dreamerv3_core is None:
            raise ValueError("actor planners require a DreamerV3 checkpoint")
        behavior_state = checkpoint.get("behavior")
        if not isinstance(behavior_state, dict):
            raise ValueError(
                "q_guided_actor requires a trained actor and critic checkpoint."
                if args.planner == "q_guided_actor"
                else "actor requires a trained actor checkpoint."
            )
        behavior = DreamerBehavior(action_dim, model.dreamerv3_core.feat_dim).to(device)
        try:
            behavior.load_state_dict(behavior_state)
        except (KeyError, RuntimeError) as error:
            message = (
                "q_guided_actor requires a trained actor and critic checkpoint."
                if args.planner == "q_guided_actor"
                else "actor requires a trained actor checkpoint."
            )
            raise ValueError(message) from error
        behavior.eval()

    if args.planner == "q_guided_actor":
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.planner in {"actor", "q_guided_actor"}:
        use_graph = bool(args.enable_graph_conditioning)
    else:
        # Preserve the historical CEM/random default (Graph ON unless disabled).
        use_graph = not args.disable_graph_conditioning
    if args.disable_graph_conditioning:
        use_graph = False
    gamma = float(
        checkpoint.get("config", {}).get("gamma", BURCHIM_DMC_DEFAULTS["gamma"])
    )
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"evaluation output: {args.output}")
    print(
        f"planner={args.planner} seed={args.seed} graph={use_graph} "
        f"q_candidates={args.q_candidates} q_horizon={args.q_horizon}"
    )
    rng = np.random.default_rng(args.seed)
    returns: list[float] = []
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for episode_id in range(args.episodes):
        env = probe_env if episode_id == 0 else suite.load(
            args.domain, args.task,
            task_kwargs={"random": args.seed + episode_id},
        )
        timestep = env.reset()
        spec = env.action_spec()
        action_min = torch.tensor(
            np.asarray(spec.minimum).copy(), device=device, dtype=torch.float32
        )
        action_max = torch.tensor(
            np.asarray(spec.maximum).copy(), device=device, dtype=torch.float32
        )
        images = [env.physics.render(args.size, args.size, camera_id=render_camera)]
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        discounts: list[float] = []
        is_last: list[bool] = []
        is_terminal: list[bool] = []
        physics_states = [env.physics.get_state().copy()]
        geom_xpos = [env.physics.data.geom_xpos.copy()]
        environment_steps = 0
        recurrent_state = None
        previous_action = None
        graph_state = None
        relation_state = None
        episode_q_rows: list[dict[str, float]] = []
        episode_started = time.perf_counter()
        while args.steps == 0 or len(actions) < args.steps:
            step_diagnostics: dict[str, float] = {}
            if args.planner == "cem":
                current = (
                    torch.from_numpy(images[-1].copy()).permute(2, 0, 1)
                    .float().div(255.0).unsqueeze(0).to(device)
                )
                action_tensor = cem_action(
                    model, current, action_min, action_max,
                    horizon=args.horizon, candidates=args.candidates,
                    elites=args.elites, iterations=args.iterations,
                    use_graph=use_graph, chunk_size=args.chunk_size,
                )
                action = action_tensor.cpu().numpy()
            elif args.planner == "random":
                action = rng.uniform(spec.minimum, spec.maximum).astype(np.float32)
            else:
                assert behavior is not None
                current = (
                    torch.from_numpy(images[-1].copy()).permute(2, 0, 1)
                    .float().div(255.0).unsqueeze(0).to(device)
                )
                (
                    action_tensor, recurrent_state, graph_state,
                    relation_state, step_diagnostics,
                ) = _actor_or_q_action(
                    model, behavior, current, recurrent_state, previous_action,
                    action_min, action_max, planner=args.planner,
                    q_candidates=args.q_candidates, q_horizon=args.q_horizon,
                    q_include_mean_action=args.q_include_mean_action,
                    q_temperature=args.q_temperature, gamma=gamma,
                    use_graph=use_graph, previous_graph=graph_state,
                    previous_relations=relation_state,
                )
                previous_action = action_tensor
                action = action_tensor[0].cpu().numpy()
            repeated_reward = 0.0
            repeated_discount = 1.0
            for _ in range(args.action_repeat):
                timestep = env.step(action)
                environment_steps += 1
                repeated_reward += float(timestep.reward or 0.0)
                repeated_discount *= float(
                    1.0 if timestep.discount is None else timestep.discount
                )
                if timestep.last():
                    break
            actions.append(np.asarray(action, dtype=np.float32))
            rewards.append(repeated_reward)
            discounts.append(repeated_discount)
            is_last.append(bool(timestep.last()))
            is_terminal.append(bool(timestep.last() and repeated_discount == 0.0))
            images.append(env.physics.render(args.size, args.size, camera_id=render_camera))
            physics_states.append(env.physics.get_state().copy())
            geom_xpos.append(env.physics.data.geom_xpos.copy())
            if step_diagnostics:
                episode_q_rows.append(step_diagnostics)
                step_rows.append({
                    "episode": episode_id, "step": len(actions) - 1,
                    **step_diagnostics,
                })
            if timestep.last():
                break
        episode_return = float(sum(rewards))
        returns.append(episode_return)
        episode_row: dict[str, Any] = {
            "episode": episode_id, "episode_return": episode_return,
            "episode_length": len(actions), "planner": args.planner,
            "q_candidates": args.q_candidates, "q_horizon": args.q_horizon,
            "q_selection": args.q_selection,
            "runtime_seconds": time.perf_counter() - episode_started,
        }
        if episode_q_rows:
            episode_row.update({
                "candidate_score_mean": _mean(episode_q_rows, "candidate_score_mean"),
                "candidate_score_max": max(row["candidate_score_max"] for row in episode_q_rows),
                "candidate_score_min": min(row["candidate_score_min"] for row in episode_q_rows),
                "candidate_score_std": _mean(episode_q_rows, "candidate_score_std"),
                "selected_score_mean": _mean(episode_q_rows, "selected_score"),
                "selected_score_max": max(row["selected_score"] for row in episode_q_rows),
                "selected_score_min": min(row["selected_score"] for row in episode_q_rows),
                "selected_action_norm_mean": _mean(episode_q_rows, "selected_action_norm"),
                "actor_action_norm_mean": _mean(episode_q_rows, "actor_mean_action_norm"),
                "selected_candidate_rank_mean": _mean(episode_q_rows, "selected_candidate_rank"),
                "q_value_mean": _mean(episode_q_rows, "predicted_value_selected"),
                "predicted_reward_mean": _mean(episode_q_rows, "predicted_reward_selected"),
                "predicted_continuation_mean": _mean(
                    episode_q_rows, "predicted_continuation_selected"
                ),
            })
        episode_rows.append(episode_row)
        path = args.output / f"{args.domain}_{args.task}_{episode_id:05d}.npz"
        np.savez_compressed(
            path,
            image=np.asarray(images, dtype=np.uint8),
            action=np.asarray(actions, dtype=np.float32),
            reward=np.asarray(rewards, dtype=np.float32),
            discount=np.asarray(discounts, dtype=np.float32),
            is_last=np.asarray(is_last, dtype=np.bool_),
            is_terminal=np.asarray(is_terminal, dtype=np.bool_),
            physics_state=np.asarray(physics_states, dtype=np.float64),
            geom_xpos=np.asarray(geom_xpos, dtype=np.float64),
            domain=np.asarray(args.domain), task=np.asarray(args.task),
            episode_seed=np.asarray(args.seed + episode_id, dtype=np.int64),
            planner=np.asarray(args.planner), graph_conditioning=np.asarray(use_graph),
            checkpoint=np.asarray(str(args.checkpoint)),
            camera=np.asarray(render_camera, dtype=np.int64),
            requested_camera=np.asarray(args.camera, dtype=np.int64),
            render_size=np.asarray(args.size, dtype=np.int64),
            action_repeat=np.asarray(args.action_repeat, dtype=np.int64),
            environment_steps=np.asarray(environment_steps, dtype=np.int64),
            episode_complete=np.asarray(bool(timestep.last())),
            observation_mode=np.asarray("image_only"),
            protocol=np.asarray(DREAMERV3_DMC_VISION),
            dm_control_version=np.asarray(version("dm-control")),
            mujoco_version=np.asarray(version("mujoco")),
        )
        print(
            f"[{episode_id + 1}/{args.episodes}] "
            f"return={episode_return:.3f} saved={path}"
        )

    _write_csv(args.output / "per_episode.csv", episode_rows)
    _write_csv(args.output / "per_step.csv", step_rows)
    summary: dict[str, Any] = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "episodes": args.episodes, "seed": args.seed,
        "planner": args.planner, "q_candidates": args.q_candidates,
        "q_horizon": args.q_horizon, "q_selection": args.q_selection,
        "q_include_mean_action": args.q_include_mean_action,
        "q_temperature": args.q_temperature, "gamma": gamma,
        "graph_conditioning": use_graph,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint": str(args.checkpoint),
    }
    if step_rows:
        summary.update({
            "candidate_score_mean": _mean(step_rows, "candidate_score_mean"),
            "candidate_score_max": max(row["candidate_score_max"] for row in step_rows),
            "candidate_score_min": min(row["candidate_score_min"] for row in step_rows),
            "candidate_score_std": _mean(step_rows, "candidate_score_std"),
            "selected_score_mean": _mean(step_rows, "selected_score"),
            "selected_score_max": max(row["selected_score"] for row in step_rows),
            "selected_score_min": min(row["selected_score"] for row in step_rows),
            "selected_action_norm_mean": _mean(step_rows, "selected_action_norm"),
            "actor_action_norm_mean": _mean(step_rows, "actor_mean_action_norm"),
            "selected_candidate_rank_mean": _mean(step_rows, "selected_candidate_rank"),
            "q_value_mean": _mean(step_rows, "predicted_value_selected"),
            "predicted_reward_mean": _mean(step_rows, "predicted_reward_selected"),
            "predicted_continuation_mean": _mean(
                step_rows, "predicted_continuation_selected"
            ),
        })
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"mean_return={summary['mean_return']:.3f} "
        f"std_return={summary['std_return']:.3f} device={device}"
    )


if __name__ == "__main__":
    main()
