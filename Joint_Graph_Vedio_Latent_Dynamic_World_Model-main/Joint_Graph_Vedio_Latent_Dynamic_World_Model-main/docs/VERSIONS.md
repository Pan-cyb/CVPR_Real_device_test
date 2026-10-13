# Training protocol versions

## `v0.3.0-offline`

Commit `8fe1fb050af782aa50dfca9714d9caf5283bf258` preserves the fixed-dataset
world-model training pipeline. Episodes are collected first with
`collect_dmc.py`, the world model is trained by `train.py`, and closed-loop
control uses CEM rather than a learned actor. The annotated Git tag is the
rollback point for reproducing those experiments.

## Online actor--critic (`main` after `v0.3.0-offline`)

The primary DMC comparison is online: four environments continuously append
experience to replay. DreamerV3 updates world model, actor, and critic from the
start. DreamerV3 defaults match the pinned Burchim commit
`a6ee1d86cf3e387694d58a49aa3e81a5b010d8c5`: 64x64 RGB, action repeat 2,
batch 16, sequence length 64, imagination horizon 15, 100-window random
prefill, 5000 updates per epoch, 50 epochs, and 10 actor-evaluation episodes.

`collect_dmc.py` remains necessary only to produce MuJoCo teacher data for
pretraining/debugging the RGB Explicit Graph Observer. It is not used to
prefill the online control replay.

The online replay checkpoint format remains version 1, but active streams are
now sampled by direct window slices. Earlier code repacked every complete
growing stream on each batch, causing epoch time to rise with replay size even
though the requested batch shape was constant. Epoch metrics expose replay
frames/windows, seconds per update, and CUDA peak memory for regression checks.
