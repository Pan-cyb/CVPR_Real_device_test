# DreamerV3 alignment contract

JointGV pins `benchmark/burchim-dreamerv3-pytorch` at the submodule commit
recorded by Git. This repository is the implementation base for the primary
DreamerV3 comparison.

## Shared burchim core

Both Dreamer visual variants reuse the pinned burchim components:

- categorical RSSM with deter 512, stochastic 32×32, hidden 512;
- learned initial state and 1% categorical unimix from the pinned RSSM;
- reward and continuation heads;
- free nats 1 and separate dynamics/representation KL scales 0.5/0.1;
- Adam (`lr=1e-4`, `eps=1e-8`) and gradient clipping at 1000;
- DMC 64×64 vision protocol with action repeat 2 and sequence length 64.

The online trainer exposes `--dreamer-behavior-update-period` as an opt-in
compute tradeoff. A value above 1 leaves world-model updates, environment
collection, and action repeat unchanged, while running actor/critic imagination
less often. The default value 1 preserves the pinned full-frequency schedule.
Graph-off/on comparisons must match this period, and reduced-frequency runs
must be reported separately from historical full-frequency checkpoints.

`dreamerv3` uses the pinned representation network and observation network.
`dreamerv3_vae` changes only that visual boundary to the
JointGV Gaussian VisualVAE and a feature-to-token decoder bridge. It does not
use a separate or compact RSSM.

## JointGV boundary

With graph conditioning disabled, the transition returns the unmodified
burchim image prior and does not execute graph modules. With it enabled,
JointGV reads the RSSM feature change, predicts action-conditioned graph
dynamics, and adds a bounded/gated correction only to the prior deterministic
state. The categorical prior is then recomputed from the corrected
deterministic state.

DreamerV3 exposes one global recurrent feature
`s_t = concat(stoch_t, deter_t)`, not a set of object/patch dynamics tokens.
The v13 adapter therefore uses the genuine
`Delta s_t = s_t^prior - s_t`, with no MLP that fabricates a token axis. A
bias-free projection of `Delta s_t` queries bias-free projections of the
active `Delta G_t` nodes. Softmax attention pools the raw graph-node changes;
normalized half-squared L2 between the global query and pooled graph key is
the alignment loss (equivalently `1 - cosine`). This is called **global delta
alignment**, not Sinkhorn OT. The pooled graph context is mapped to the RSSM
deterministic-state dimension, bounded, gated, and added to the prior.

Backbones with a genuine latent token set, including David Ha's spatial VAE
adapter and DINO-WM patch tokens, continue to use full Sinkhorn correspondence.

This is therefore a burchim DreamerV3 world-model integration with a modified
transition prior, not a claim that the complete actor-critic training loop is
unchanged. Online evaluation currently uses CEM over the learned reward model.

## Removed implementation

The historical RSSM-to-16-pseudo-token projection and its checkpoints were
removed in v13 because the invented set axis had no DreamerV3 semantics.
The historical compact DreamerV3-style RSSM, deterministic compact visual
codec, CLI choices, and compatibility loading were removed. Pre-v7 Dreamer
checkpoints belong to that implementation and must not be mixed with the
burchim-backed experiments.

## Required comparison

Use the same data split and hyperparameters for each Graph off/on pair:

1. David Ha MDN-LSTM;
2. burchim DreamerV3 with native vision;
3. burchim DreamerV3 with VisualVAE.

For each pair, report held-out open-loop rollout error and online DMC return.
