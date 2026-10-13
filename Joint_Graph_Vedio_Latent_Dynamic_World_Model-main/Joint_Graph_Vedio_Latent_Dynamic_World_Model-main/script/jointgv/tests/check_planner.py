"""Pre-flight check for the CEM-MPC planner (CPU, a few seconds).

Uses a tiny random DINO-WM model (debug_conv encoder, proxy observer, graph
history 4, pose correction) and asserts:

1. LatentRollout with full history reproduces model.rollout exactly
   (visual context, graph on and off), for several candidates at once.
2. With history_window=W it equals model.rollout with the predictor capped
   at W frames (DINO-WM's sliding window).
3. CEM cuts a known quadratic cost below 10 % of the zero plan's, never
   loses the evaluated mean on a replan, and warm-starts by shifting.
4. The rm65b target is visible, fixed and non-trivial within an episode, and
   control-state save/restore makes simulate-ahead exact.
5. A batch of different starts imagines exactly what each start imagines on
   its own (train_reward_heads.py relies on it), and a per-track
   TrackRewardHead scores the frame/previous-frame pairs of the rollout and
   survives a save/load round trip.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv import JointGraphVideoWorldModel  # noqa: E402
from jointgv.envs import load_environment  # noqa: E402
from jointgv.envs.rm65b_pendulum import MIN_TARGET_DISTANCE  # noqa: E402
from jointgv.planning import (  # noqa: E402
    CEMConfig,
    CEMPlanner,
    LatentRollout,
    TrackRewardHead,
    limit_history,
    load_track_reward_head,
)


def build_model() -> JointGraphVideoWorldModel:
    torch.manual_seed(0)
    model = JointGraphVideoWorldModel(
        action_dim=3,
        backbone="dino_wm",
        token_dim=32,
        node_dim=32,
        ot_dim=16,
        dino_model="debug_conv",
        dino_image_size=32,
        dino_predictor_depth=2,
        dino_predictor_heads=4,
        dino_max_frames=16,
        graph_history=4,
        graph_pose_correction=True,
        max_graph_correction=1.0,
    )
    # A zero-initialised correction would make graph on == graph off.
    torch.nn.init.normal_(model.graph_plugin.pose_correction_mlp[-1].weight, std=0.05)
    return model.eval()


def main() -> None:
    model = build_model()
    frames = torch.rand(4, 3, 64, 64)          # 3 context frames + current
    context_actions = torch.rand(3, 3) * 2 - 1
    candidates = torch.rand(5, 6, 3) * 2 - 1   # N=5 sequences, H=6

    # 1. Full history: identical to model.rollout, graph on and off.
    for use_graph in (True, False):
        planner_rollout = LatentRollout(model, use_graph=use_graph, history_window=None)
        planner_rollout.prepare(frames, context_actions)
        mine = planner_rollout.rollout(candidates)
        model.set_graph_conditioning_strength(1.0 if use_graph else 0.0)
        with torch.no_grad():
            reference = model.rollout(
                frames[-1:].expand(5, -1, -1, -1),
                candidates,
                use_graph=use_graph,
                decode=False,
                context_images=frames[None, :-1].expand(5, -1, -1, -1, -1),
                context_actions=context_actions[None].expand(5, -1, -1),
                visual_context=True,
            )["tokens"]
        difference = float((mine - reference).abs().max())
        assert difference < 1e-5, f"graph={use_graph}: differs by {difference}"
        print(f"[1] graph {'on ' if use_graph else 'off'}: batched planner rollout = model.rollout (max diff {difference:.1e})")
    graph_on = LatentRollout(model, use_graph=True, history_window=None)
    graph_on.prepare(frames, context_actions)
    graph_off = LatentRollout(model, use_graph=False, history_window=None)
    graph_off.prepare(frames, context_actions)
    assert not torch.allclose(graph_on.rollout(candidates), graph_off.rollout(candidates))
    with torch.no_grad():
        tokens = graph_on.rollout(candidates)
        rewards = model.reward_head(tokens.mean(dim=-2)).squeeze(-1)
        discounts = 0.99 ** torch.arange(rewards.shape[1])
        expected_return = (rewards * discounts).sum(dim=1)
    assert torch.allclose(
        graph_on.predicted_return(candidates), expected_return, atol=1e-5
    ), "fixed-horizon reward return differs from the direct calculation"

    # 2. history_window=4 == predictor capped at 4 frames.
    windowed = LatentRollout(model, use_graph=True, history_window=4)
    windowed.prepare(frames, context_actions)
    mine = windowed.rollout(candidates)
    saved = model.backbone.max_frames
    limit_history(model, 4)   # evaluate_rollout/evaluate_ablation --history-window
    try:
        model.set_graph_conditioning_strength(1.0)
        with torch.no_grad():
            reference = model.rollout(
                frames[-1:].expand(5, -1, -1, -1),
                candidates,
                use_graph=True,
                decode=False,
                context_images=frames[None, :-1].expand(5, -1, -1, -1, -1),
                context_actions=context_actions[None].expand(5, -1, -1),
                visual_context=True,
            )["tokens"]
    finally:
        model.backbone.max_frames = saved
    difference = float((mine - reference).abs().max())
    assert difference < 1e-5, f"windowed rollout differs by {difference}"
    print(f"[2] history window 4 = predictor sliding window of 4 frames (max diff {difference:.1e})")

    # 3. CEM on a quadratic with a known optimum.
    target = torch.linspace(-0.6, 0.6, 8)[:, None].repeat(1, 3)
    planner = CEMPlanner(
        3,
        CEMConfig(horizon=8, candidates=128, elites=16, iterations=6),
        torch.device("cpu"),
    )
    best = planner.plan(lambda s: (s - target).square().mean(dim=(1, 2)))
    zero_cost = float(target.square().mean())
    first_cost = planner.last_cost
    error = float((best - target).abs().mean())
    assert first_cost < 0.1 * zero_cost, (
        f"CEM cost {first_cost:.4f} vs zero plan {zero_cost:.4f}"
    )
    again = planner.plan(lambda s: (s - target).square().mean(dim=(1, 2)))
    assert planner.last_cost <= first_cost + 1e-9, "warm start lost the previous mean"
    planner.shift(3)
    assert torch.equal(planner.mean[:5], again[3:]) and not planner.mean[5:].any()
    print(
        f"[3] CEM on a known quadratic: cost {first_cost:.4f} vs zero plan "
        f"{zero_cost:.4f} (mean error {error:.3f}); warm start and shift work"
    )

    # 4. Visible target and simulate-ahead on rm65b.
    env = load_environment("rm65b", "pendulum", {"random": 5})
    env.reset()
    physics, task = env.physics, env.task
    assert env.action_spec().shape == (3,), "RM65-B policy should control J2/J3/J5"
    target = physics.target_position().copy()
    assert physics.target_distance() >= MIN_TARGET_DISTANCE - 1e-6
    image = physics.render(64, 64, camera_id=0)   # the collection camera
    assert image.shape == (64, 64, 3)
    red = (
        (image[..., 0] > 100)
        & (image[..., 0] > 1.5 * image[..., 1])
        & (image[..., 0] > 1.5 * image[..., 2])
    )
    assert int(red.sum()) >= 8, "red target is occluded, clipped or too small at 64x64"
    start_state = physics.get_state().copy()
    physics.set_state(physics.goal_state(task.target_pose()))
    physics.forward()
    assert physics.target_distance() < 1e-4, (
        "camera-projected target does not match its sampled TCP pose"
    )
    assert task.get_reward(physics) > 0.99, "stable target pose did not open hard gate"
    task.unsafe_commands = 1
    assert task.get_reward(physics) == 0.0, "unsafe guard did not suppress reward"
    task.unsafe_commands = 0
    physics.set_state(start_state)
    physics.forward()
    rng = np.random.default_rng(0)
    for _ in range(10):
        env.step(task.exploration_action(physics, rng, 2))
    assert np.allclose(physics.target_position(), target), "target moved within episode"
    saved_control, saved_state = task.control_state(), physics.get_state().copy()
    action = np.asarray([0.5, -0.3, 0.2])

    def advance() -> np.ndarray:
        for _ in range(4):
            task.before_step(action, physics)
            physics.step(10)
        return physics.tcp_position()

    first = advance()
    task.set_control_state(saved_control)
    physics.set_state(saved_state)
    physics.forward()
    second = advance()
    assert np.allclose(first, second), "simulate-ahead is not repeatable"
    assert np.allclose(physics.target_position(), target)
    print("[4] visible target is fixed and non-trivial; simulate-ahead is repeatable")

    # 5. Batched starts and per-track reward heads.
    batch_frames = torch.rand(3, 4, 3, 64, 64)
    batch_context = torch.rand(3, 3, 3) * 2 - 1
    batch_actions = torch.rand(3, 6, 3) * 2 - 1
    for use_graph in (True, False):
        batched = LatentRollout(model, use_graph=use_graph, history_window=4)
        batched.prepare(batch_frames, batch_context)
        together = batched.rollout(batch_actions)
        for index in range(3):
            alone = LatentRollout(model, use_graph=use_graph, history_window=4)
            alone.prepare(batch_frames[index], batch_context[index])
            difference = float(
                (alone.rollout(batch_actions[index : index + 1])[0] - together[index])
                .abs()
                .max()
            )
            assert difference < 1e-5, f"batched start {index} differs by {difference}"
    torch.manual_seed(1)
    grid = int(model.token_count ** 0.5)
    head = TrackRewardHead(model.token_dim, grid, hidden=16).eval()
    head.input_std.uniform_(0.5, 2.0)
    scored = LatentRollout(model, use_graph=True, history_window=4, reward_head=head)
    scored.prepare(frames, context_actions)
    with torch.no_grad():
        tokens = scored.rollout(candidates)
        start = scored.start_tokens().expand(5, -1, -1)
        previous = torch.cat((start[:, None], tokens[:, :-1]), dim=1)
        manual = torch.stack(
            [head(tokens[:, k], previous[:, k]) for k in range(tokens.shape[1])], dim=1
        )
    assert torch.allclose(scored.rewards(tokens), manual, atol=1e-6)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "head.pt"
        torch.save(
            {"config": head.config, "state_dict": head.state_dict(), "track": {"mode": "graph"}},
            path,
        )
        loaded, track = load_track_reward_head(path, torch.device("cpu"))
    assert track["mode"] == "graph"
    with torch.no_grad():
        assert torch.allclose(loaded(tokens, previous), head(tokens, previous))
    print("[5] batched starts = single starts; per-track reward head scores and reloads")
    print("all planner checks passed")


if __name__ == "__main__":
    main()
