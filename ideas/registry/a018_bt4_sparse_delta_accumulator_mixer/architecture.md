# Architecture

`BT4 Primitive Mixer (sparse_delta_accumulator)` is a controlled architecture study, not a new
primitive. It holds the BT4-style residual tower fixed and swaps only the
per-block spatial-mixing operator.

## Implementation Binding

- Registered model name: `bt4_sparse_delta_accumulator_mixer` (alias of `bt4_primitive_mixer`
  with `mixer=sparse_delta_accumulator`)
- Tower: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
- Mixer: `src/chess_nn_playground/models/architecture/bt4_mixers/sparse_delta_accumulator.py`
- Source primitive idea: `p013_sparse_delta_accumulator`

## What this is

The `bt4_primitive_mixer` tower mirrors `lc0_bt4_classifier`: a stem conv, N
residual blocks each `mixer -> SqueezeExcite -> +residual -> ReLU`, then the
value head. The original lc0_bt4 block mixes spatially with a pair of 3x3
convs. Here that mixer is replaced by the `sparse_delta_accumulator` primitive, adapted to the
shape-preserving `(B, C, 8, 8) -> (B, C, 8, 8)` mixer contract.

Because the tower, optimizer protocol, data contract, and training
hyperparameters are identical across every `a###_bt4_*_mixer` idea and the
`conv` / `attention` baselines, the only variable is the mixer. That makes
this the cleanest available test of "is the `sparse_delta_accumulator` primitive a better
spatial mixer than conv or attention?".

## Contract

- Input: `(B, 18, 8, 8)` simple_18 board tensor only.
- Output: `dict` with `logits` of shape `(B,)` for the one-logit puzzle_binary
  BCE-with-logits trainer.
