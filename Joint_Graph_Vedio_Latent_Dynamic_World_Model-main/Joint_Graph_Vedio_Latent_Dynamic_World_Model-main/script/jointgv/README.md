# JointGV Implementation

This directory contains the active, runnable code. Graph Dynamics is a
backbone-independent plugin under `jointgv/blocks`; selectable latent
transitions live under `jointgv/backbones`.

| Location | Contents |
|---|---|
| `train.py`, `train_online_dmc.py` | the two training entry points (offline / online) |
| `utils/` | data collection (`collect_dmc.py`) |
| `pretrain/` | graph observer and DINO decoder pretraining |
| `evaluation/` | rollout tables, single-episode rollouts, online actor evaluation |
| `viz/` | GIFs, plots, episode viewer, observer/decoder debuggers, architecture figure |
| `tests/` | pre-flight and smoke checks |
| `tmp/` | not needed by the graph plugin: LIBERO tools, VAE and graph-dynamics pretraining, drift diagnostics, legacy CEM |

Run scripts from the repository root, e.g.
`python script/jointgv/evaluation/evaluate_ablation.py ...`.

> Primary control training is now `train_online_dmc.py`: online DMC
> collection, replay, world-model update, actor imagination, critic update,
> and actor evaluation follow the pinned Burchim protocol. The fixed-data
> `collect_dmc.py` + `train.py` + CEM path is preserved by Git tag
> `v0.3.0-offline` and remains useful for observer pretraining/diagnostics.

Minimal matched online pair:

```powershell
python script/jointgv/train_online_dmc.py --domain reacher --task easy `
  --backbone dreamerv3 --disable-graph-conditioning `
  --output runs/online_reacher_dreamerv3_baseline

python script/jointgv/train_online_dmc.py --domain reacher --task easy `
  --backbone dreamerv3 --graph-observer explicit `
  --graph-observer-checkpoint runs/reacher_easy_reltr_observer_v3/best.pt `
  --output runs/online_reacher_dreamerv3_jointgv
```

### DreamerV3 Stage 1: visual reconstruction

Stage 1 treats every replay frame as an independent initial observation and
trains only the native Dreamer visual encoder, RSSM observation-posterior MLP,
and image decoder. Temporal transition, reward, continuation, Graph, actor,
critic, and imagination parameters are frozen and excluded from the loss.
The default `--training-stage joint` retains the previous training behavior.

```powershell
python script/jointgv/train_online_dmc.py --domain reacher --task hard `
  --backbone dreamerv3 --training-stage visual `
  --epochs 20 --steps-per-epoch 1000 `
  --output runs/online_reacher_hard_dreamerv3_visual
```

The run stores one immutable 12-frame set in
`diagnostics/reconstruction/fixed_frames.pt`, paired original/reconstruction
grids in `diagnostics/reconstruction/epoch_XXXX.png`, and composable
`checkpoints/visual_latest.pt` and `checkpoints/visual_best.pt` checkpoints.
Stage 1 explicitly disables Graph even when the historical graph-on default is
used, so `--disable-graph-conditioning` is optional for this stage.

### DreamerV3 Stage 2: world model

Stage 2 must start from a Stage-1 visual checkpoint. It freezes the visual
encoder, observation-posterior MLP, decoder, Graph path, actor, and critics;
only the RSSM temporal prior, reward head, and continuation head are trained.
Graph is forced off. The temporal loss contains dynamics KL, reward, and
continuation terms; reconstruction and representation KL are monitoring-only.

```powershell
python script/jointgv/train_online_dmc.py --domain reacher --task hard `
  --backbone dreamerv3 --training-stage world `
  --stage-checkpoint runs/reacher_hard_dreamerv3_stage1_visual_20ep/visual_best.pt `
  --epochs 3 --steps-per-epoch 500 --batch-size 8 `
  --seed 0 --smoke `
  --output runs/reacher_hard_dreamerv3_stage2_world_smoke
```

Fixed replay probes drive teacher-forced one-step prediction, open-loop
H=1/5/10/20/50 image comparisons, and reward-curve diagnostics. Metrics are
appended to `diagnostics/world_metrics.jsonl` and `.csv`. Checkpoints are
written as `world_latest.pt` and `world_best.pt`; the best score is
`reward_mae_all + horizon_10_mse`.

### DreamerV3 Stage 3: Graph prediction comparison

Stage 3 starts from a Stage-2 `world` checkpoint and keeps actor, critics,
visual encoder, posterior, and decoder frozen. Graph OFF continues the same
world-only baseline. Graph ON additionally trains the existing graph dynamics,
alignment, and graph-to-RSSM correction path; a pretrained explicit observer
remains frozen. The run reuses the Stage-2 fixed probe file when it is adjacent
to the source checkpoint and writes an epoch-0 Graph-OFF baseline before any
Stage-3 update.

```powershell
# Graph OFF control.
python script/jointgv/train_online_dmc.py --domain reacher --task hard `
  --backbone dreamerv3 --training-stage graph `
  --disable-graph-conditioning `
  --stage-checkpoint runs/reacher_hard_dreamerv3_stage2_world_20ep/checkpoints/world_best.pt `
  --epochs 3 --steps-per-epoch 500 --batch-size 8 --seed 0 --smoke `
  --output runs/reacher_hard_dreamerv3_stage3_graphoff_smoke

# Frozen explicit observer + trainable Graph conditioning.
python script/jointgv/train_online_dmc.py --domain reacher --task hard `
  --backbone dreamerv3 --training-stage graph `
  --graph-observer explicit `
  --graph-observer-checkpoint pretrained_graph_dynamics/best_long_rollout.pt `
  --stage-checkpoint runs/reacher_hard_dreamerv3_stage2_world_20ep/checkpoints/world_best.pt `
  --epochs 3 --steps-per-epoch 500 --batch-size 8 --seed 0 --smoke `
  --output runs/reacher_hard_dreamerv3_stage3_explicit_graph_smoke
```

Stage-3 metrics add correction RMS/max, correction-to-prior RMS ratio, gate,
conditioning strength, active/total node counts, padding-node audit, and graph
losses while retaining the Stage-2 metric columns. `graph_best.pt` uses
`horizon_10_mse + reward_mae_top10_reward`; it never uses control return.

Do not pass offline `--data` or CEM options to this entry point. It collects
policy experience itself and stores resumable replay in `replay_latest.pt`.
Replay sampling slices only selected windows rather than stacking the full
growing stream on every update. Runtime diagnostics report seconds/update,
replay size, and peak CUDA allocation/reservation each epoch.

## Available Experiment Matrix

The design exposes four backbone configurations with Graph off/on, giving eight
runnable configurations.

| Backbone family | Graph off | JointGV | Status |
|---|---|---|---|
| David Ha 2018 | `david_ha` + `--disable-graph-conditioning` | `david_ha` | Runnable |
| DreamerV3 native vision | `dreamerv3` + `--disable-graph-conditioning` | `dreamerv3` | Runnable |
| DreamerV3 + VisualVAE | `dreamerv3_vae` + `--disable-graph-conditioning` | `dreamerv3_vae` | Runnable |
| DINO-WM frozen DINOv2 + causal patch predictor | `dino_wm` + `--disable-graph-conditioning` | `dino_wm` | Runnable |

Graph is on unless `--disable-graph-conditioning` is supplied.

```text
BackboneAdapter
  ├─ VAE tokens ─ DavidHaMDNBackbone ─────┐
  ├─ VAE tokens ─ DreamerV3WorldModel ─────┼─ base Δvisual
  └─ native CNN ─ DreamerV3WorldModel ─────┘
                              │
ExplicitGraphObserver ─ GraphDynamics ─ ΔG
                              │
                  Sinkhorn transport + correction
                              │
                  next tokens / reward / continuation
                              │
                       recursive CEM rollout
```

The reportable path uses a separately pretrained RGB-only Explicit Graph
Observer. It directly imports the pinned official RelTR ResNet-50 + relation
transformer and initializes it from the Visual Genome checkpoint. The role
localizer fuses RelTR semantic queries with the official ResNet layer3 dense
map; dense soft-argmax plus a bounded residual grounds each role without
depending only on RelTR boxes. JointGV adds temporal recurrence, physical
decoding, a compact VTransE-style relation head, and one IMP refinement round.
The same representative MuJoCo geom supplies 3D and pixel teacher coordinates,
but teacher values are never inference inputs. The old visual-token proxy is
still selectable with
`--graph-observer proxy` for compatibility.

Pretrain and visually inspect the observer before world-model training:

```powershell
python script/jointgv/pretrain/pretrain_graph_observer.py `
  --data data/dmc_reacher_easy_train_v3 `
  --validation-data data/dmc_reacher_easy_val_v3 `
  --output runs/reacher_easy_reltr_observer_v3 `
  --epochs 30 --lr 3e-4 `
  --batch-size 4 `
  --sequence-length 32 --stride 16 `
  --graph-nodes 8 --node-dim 64 `
  --backbone reltr_resnet50 `
  --reltr-checkpoint Models/reltr/checkpoint0149.pth `
  --reltr-image-size 224 --reltr-batch-size 32 `
  --temporal-max-correction 0.25 `
  --max-position-residual 0.25 `
  --identity-weight 0.05 --motion-weight 1.0 `
  --link-pixel-weight 2.0 --end-effector-pixel-weight 5.0 `
  --grounding-epochs 10 --temporal-warmup-epochs 10 `
  --recursive-validation-every 5 `
  --recursive-validation-episodes 0 `
  --recursive-frame-batch-size 64 `
  --workers 0

python script/jointgv/viz/visualize_graph_observer.py `
  --checkpoint runs/reacher_easy_reltr_observer_v3/best.pt `
  --episode data/dmc_reacher_easy_val_v3/reacher_easy_00000.npz `
  --output runs/reacher_easy_reltr_observer_v3/debug_episode `
  --batch-size 8 --max-frames 200 `
  --fps 20 --scale 6 --device cuda

python script/jointgv/viz/view_episode.py `
  runs/reacher_easy_reltr_observer_v3/debug_episode/comparison_episode.npz `
  --fps 20 --scale 1
```

This is the recommended 24 GB GPU configuration. It trains 32-frame tracks and
runs continuous full-episode validation every five epochs. This runbook uses
the manually visual-verified `best.pt`; a fully warmed
`best_long_rollout.pt` remains an available alternative. Epochs 1-10 train
absolute RGB grounding with temporal correction disabled, and epochs 11-20
linearly warm up temporal identity binding. A
long-rollout checkpoint first becomes eligible at full temporal strength
(epoch 20 here); the world-model loader rejects earlier observer checkpoints.
A pre-v3 checkpoint cannot be resumed because both teacher coordinates and the
localizer changed. Training, recursive validation, and visualization report
per-role pixel MAE, temporal gate, and actual correction magnitude. For a
lower-memory GPU, use `--batch-size 2 --reltr-batch-size 16`.

## Local Smoke Commands (Not Reportable)

The commands in this first block only verify that collection, training, and
evaluation execute locally. `dmc_reacher_tiny`, batch size 4, sequence length
8, and stride 4 must not be compared with a full JointGV run. DreamerV3 itself
uses length 64 below because that setting is fixed for this adapter.

```powershell
# Collect rendered DMC episodes.
python script/jointgv/utils/collect_dmc.py --domain reacher --task easy `
  --episodes 10 --action-repeat 2 --camera -1 `
  --output data/dmc_reacher_tiny

# David Ha 2018 MDN-LSTM + Graph Dynamics.
python script/jointgv/train.py --backbone david_ha `
  --data data/dmc_reacher_tiny --output runs/reacher_david_graph `
  --epochs 5 --batch-size 4 --sequence-length 8 --stride 4 `
  --rnn-dim 128 --workers 0

# Pretrain the Gaussian VAE used by dreamerv3_vae.
python script/jointgv/tmp/pretrain_visual_vae.py `
  --data data/dmc_reacher_tiny --output runs/reacher_visual_vae `
  --epochs 5 --batch-size 4 --sequence-length 8 --stride 8

# VisualVAE + DreamerV3 RSSM.
python script/jointgv/train.py --backbone dreamerv3_vae `
  --visual-checkpoint runs/reacher_visual_vae/best.pt --freeze-visual `
  --data data/dmc_reacher_tiny --output runs/reacher_dreamerv3_vae `
  --epochs 5 --batch-size 4 --sequence-length 64 --stride 32 `
  --workers 0

# Native visual encoder + the same RSSM and Graph Dynamics.
python script/jointgv/train.py --backbone dreamerv3 `
  --data data/dmc_reacher_tiny --output runs/reacher_dreamerv3 `
  --epochs 5 --batch-size 4 --sequence-length 64 --stride 32 `
  --workers 0

# Closed-loop learned-world-model control.
python script/jointgv/tmp/evaluate_dmc.py `
  --checkpoint runs/reacher_dreamerv3/best.pt `
  --domain reacher --task easy --episodes 5 `
  --action-repeat 2 --camera -1 `
  --planner cem --horizon 8 --candidates 64 --elites 8 `
  --iterations 3 --output runs/reacher_dreamerv3/eval

# View the rendered result.
python script/jointgv/viz/view_episode.py `
  runs/reacher_dreamerv3/eval/reacher_easy_00000.npz `
  --fps 20 --scale 6
```

These are the official DreamerV3 `dmc_vision` environment settings. Collection
and evaluation run to the natural DMC ending when `--steps` is omitted.
`--steps N` intentionally truncates an episode for a local smoke test, and the
saved `episode_complete` field will then be false. It must be explicitly
acknowledged with `--allow-partial-episode`.

`train.py` rejects legacy DMC files, non-official environment settings, and
partial episodes by default. Use `--allow-partial-dmc` only for a local smoke
test. Likewise, `evaluate_dmc.py` rejects unverified checkpoints unless
`--allow-unverified-checkpoint` is explicitly supplied.

Add `--disable-graph-conditioning` for the corresponding backbone-only
ablation. Add `--disable-mujoco-supervision` to remove privileged graph labels.

## Matched DreamerV3 Experiment

Keep every argument identical except `--disable-graph-conditioning` and the
output path. See `../../docs/DREAMERV3_ALIGNMENT.md` for the precise version
and world-model/CEM scope.

JointGV:

```powershell
python script/jointgv/train.py `
  --data data/dmc_reacher_easy_train_v3 `
  --validation-data data/dmc_reacher_easy_val_v3 `
  --backbone dreamerv3 `
  --graph-observer explicit `
  --graph-observer-checkpoint runs/reacher_easy_reltr_observer_v3/best.pt `
  --epochs 100 --lr 1e-4 `
  --batch-size 16 --sequence-length 64 --stride 32 `
  --graph-warmup-epochs 10 --ot-warmup-epochs 20 `
  --ot-weight 0.1 --ot-variance-weight 0.1 `
  --dynamic-ot-weight --ot-target-grad-ratio 0.1 `
  --ot-balance-update-every 50 --ot-balance-ema 0.95 `
  --ot-weight-min 0.1 --ot-weight-max 1000 `
  --max-graph-correction 0.5 `
  --graph-correction-gate-init 0.25 `
  --rollout-eval-start 20 --rollout-eval-every 5 `
  --rollout-horizon 32 --rollout-windows-per-episode 1 `
  --workers 0 `
  --output runs/reacher_easy_dreamerv3_jointgv
```

Native DreamerV3 baseline:

```powershell
python script/jointgv/train.py `
  --data data/dmc_reacher_easy_train_v3 `
  --validation-data data/dmc_reacher_easy_val_v3 `
  --backbone dreamerv3 `
  --disable-graph-conditioning `
  --epochs 100 --lr 1e-4 `
  --batch-size 16 --sequence-length 64 --stride 32 `
  --graph-warmup-epochs 10 --ot-warmup-epochs 20 `
  --rollout-eval-start 20 --rollout-eval-every 5 `
  --rollout-horizon 32 --rollout-windows-per-episode 1 `
  --workers 0 `
  --output runs/reacher_easy_dreamerv3_baseline
```

If GPU memory requires a smaller batch size, change it in both commands
together. The graph-off DreamerV3 path returns directly from the unmodified
RSSM prior and does not execute the Graph/OT/correction modules.

Explicit-observer world-model runs use checkpoint version 13 and require an
observer checkpoint with schema version 3. Older explicit checkpoints predate
either active-node masking or the geom-centric dense localizer and cannot be
resumed. Padding roles never enter GN message passing, graph deltas, Sinkhorn
transport, variance regularization, or collapse diagnostics. Proxy-observer formats
remain compatibility-only. Monitor `dynamics_kl_raw` and `representation_kl_raw`
alongside their free-nats-clamped values.

Run `python script/jointgv/tests/smoke_test.py` for the CUDA/gradient check. See the
repository root README for model semantics, complete data fields, open-loop
rollout visualization, experiment caveats, and LIBERO conversion.

## Sinkhorn Choice

Sinkhorn divergence compares the geometry of the empirical `Δz` and `ΔG`
distributions and returns a soft graph-node-to-video-token transport plan.
Wasserstein distance is the underlying optimal-transport idea; Sinkhorn is its
entropy-regularized differentiable approximation with self-transport bias
removed. Unlike InfoNCE, it directly supplies the correspondence used by the
conditioning block. The projectors are bias-free and centered across each
token/node set before unit normalization by default. Pass
`--disable-ot-centering` to retain the common motion component; safe per-vector
unit normalization remains enabled. A lightweight variance-floor loss prevents
zero-set collapse. DreamerV3 does not expose a genuine latent token set, so it
does not run Sinkhorn: its real global `Delta s` queries active `Delta G`
nodes, and normalized half-squared L2 (equivalently cosine distance) aligns it
with the attention-pooled graph change. David Ha and DINO-WM retain full
token-set Sinkhorn. Use `--ot-weight` and
`--ot-variance-weight` to recalibrate the two objectives after inspecting the
saved diagnostics.

For the recommended first dynamic-weight experiment, add
`--dynamic-ot-weight`. The trainer then treats `--ot-weight` as an initial
value and periodically sets the full-strength coefficient from the EMA ratio
of raw backbone and raw OT gradient norms on their genuinely shared
parameters. `--ot-target-grad-ratio 0.1` targets a 10% weighted OT gradient at
full warm-up; `--ot-balance-update-every 50`, `--ot-balance-ema 0.95`, and
`--ot-weight-min/--ot-weight-max` control measurement cost, smoothing, and
safety bounds. The epoch OT warm-up still multiplies the dynamic value. This
balances optimization pressure rather than incomparable pixel-NLL and OT
scalar units. Controller EMA, current weight, update count, and minibatch step
are checkpointed and restored by `--resume`.

Graph correction magnitude and its initial learned gate are independent. For
example, `--max-graph-correction 0.5 --graph-correction-gate-init 0.25`
starts with a per-component cap of `0.125` before graph warm-up scaling. The
trainer reports `prior_deter_rms` and `graph_correction_ratio` for DreamerV3,
where the latter is `RMS(correction) / RMS(unconditioned prior deter)`.
