# Architecture

`BT4 Primitive Mixer (dynamic_adjacency_gating)` is a controlled architecture study, not a new
primitive. It holds the BT4-style residual tower fixed and swaps only the
per-block spatial-mixing operator.

## Implementation Binding

- Registered model name: `bt4_dynamic_adjacency_gating_mixer` (alias of `bt4_primitive_mixer`
  with `mixer=dynamic_adjacency_gating`)
- Tower / source implementation file: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
- Mixer source: `src/chess_nn_playground/models/architecture/bt4_mixers/dynamic_adjacency_gating.py`
- Idea-local wrapper: `ideas/registry/a037_bt4_dynamic_adjacency_gating_mixer/model.py`
- Source primitive idea: `p032_dynamic_adjacency_gating`

## What this is

The `bt4_primitive_mixer` tower mirrors `lc0_bt4_classifier`: a stem conv, N
residual blocks each `mixer -> SqueezeExcite -> +residual -> ReLU`, then the
value head. The original lc0_bt4 block mixes spatially with a pair of 3x3
convs. Here that mixer is replaced by the `dynamic_adjacency_gating` primitive, adapted to the
shape-preserving `(B, C, 8, 8) -> (B, C, 8, 8)` mixer contract.

Because the tower, optimizer protocol, data contract, and training
hyperparameters are identical across every `a###_bt4_*_mixer` idea and the
`conv` / `attention` baselines, the only variable is the mixer. That makes
this the cleanest available test of "is the `dynamic_adjacency_gating` primitive a better
spatial mixer than conv or attention?".

## Contract

- Input: `(B, 18, 8, 8)` simple_18 board tensor only.
- Output: `dict` with `logits` of shape `(B,)` for the one-logit puzzle_binary
  BCE-with-logits trainer.
