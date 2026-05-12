# Architecture

`Causal Piece-Derivative Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and returns one
puzzle logit per position. Instead of pooling a single board
embedding, the model asks *which pieces or squares are causally
critical to the puzzle decision* by running a small bank of
deterministic interventions through a lightweight shared delta
encoder.

## Mechanism

A compact convolutional trunk turns the 18-plane board into a
per-square feature map. The trunk feeds two heads:

- a `base_logit` puzzle head over the globally pooled trunk features,
- a per-square `gating_head` that scores all 64 squares and picks the
  top-`candidate_k` candidates.

For each candidate square `i` and intervention type `t`, the model
estimates the puzzle logit that would result if the intervention were
applied. To stay cheap (the packet warns the model is "more expensive
per batch") the trunk is *not* re-run per intervention. Instead a
shared delta encoder consumes:

- the candidate square's trunk feature vector,
- the globally pooled trunk feature vector,
- a learnable intervention-type embedding (`remove_piece`,
  `hide_square`, `neutralize_side`),
- per-square coordinate features (rank, file, centred rank/file,
  edge distance),
- candidate occupancy and own / opponent indicators read from the 12
  piece planes,
- the side-to-move scalar.

The delta encoder is an `MLP(LayerNorm → Linear → GELU)` stack of
`delta_layers` layers ending in a scalar head. It outputs

```
delta_logit_{i, t} = delta_encoder(candidate_features, intervention_t, ...)
sensitivity_{i, t} = base_logit - delta_logit_{i, t}
```

Per-candidate sensitivity is the mean of `sensitivity_{i, t}` over
intervention types (so the `full_remove_only` ablation is on the same
numerical scale).

The criticality readout follows the packet exactly:

```
criticality_stats = [
    max(|sensitivity_per_candidate|),
    top1 - top2 of |sensitivity_per_candidate|,
    entropy(softmax(|sensitivity_per_candidate|)),
    sum(sensitivity_per_candidate),
    sum(sensitivity_per_candidate * (own_indicator - opp_indicator))
        / sum(|own_indicator - opp_indicator|),
]
puzzle_logit = base_logit + criticality_mlp(criticality_stats)
```

Inputs to the model are limited to the `simple_18` board tensor.
Engine, verification, source, and CRTK metadata are never used.

## Trunk and heads

The trunk is `depth` blocks of `Conv3x3 → BatchNorm → ReLU` from
18 input planes to `channels`. The base puzzle head is
`LayerNorm → Linear → GELU → Dropout → Linear` over the mean-pooled
trunk features. The gating head is `Conv1x1 → GELU → Conv1x1` over
the trunk feature map, emitting a single per-square score.

## Candidate selection

Default mode (`uses_learned_candidates=True`): the candidates are the
top-`candidate_k` squares of `gating_logits + 0.5 * occupancy`. Adding
the occupancy bias keeps the gating head focused on actual pieces
while the network is still warming up; the gating head can override
this bias once it learns useful structure.

Random ablation mode (`random_candidates`): the candidates are a fixed
random permutation of the 64 squares (seeded once at module init), so
the gating signal is removed without disturbing the rest of the
forward pass.

## Delta encoder

The shared delta encoder is run once over the `(B, K, T)` candidate
× intervention grid via broadcasting, so every intervention is
processed in a single matrix multiply. It receives the candidate
trunk features and a learnable embedding of the intervention type.
The packet's three intervention types are:

- `remove_piece` — pretend the piece on this candidate square is
  removed,
- `hide_square` — pretend the candidate square's piece-channel group
  is hidden from the trunk,
- `neutralize_side` — pretend the candidate square's side-to-move
  ownership is neutral.

The output of the delta encoder is a scalar `delta_logit_{i, t}`. The
model never re-runs the trunk; only the small delta encoder pays the
cost of the interventions.

## Readout

```
sensitivity_per_candidate_i = mean_t sensitivity_{i, t}
criticality_stats           = [max, top2_gap, entropy, signed_sum, own_vs_enemy_split]
criticality_residual        = criticality_mlp(criticality_stats)
puzzle_logit                = base_logit + criticality_residual
```

The `criticality_mlp` is `LayerNorm → Linear → GELU → Dropout →
Linear` and emits a scalar residual. Setting the `no_delta_readout`
ablation drops the residual so the puzzle logit collapses to
`base_logit` — i.e. the trunk-only baseline. When `num_classes > 1`
the puzzle logit is written into the last column of a zero-padded
logits tensor so the BCE-with-logits trainer contract still holds.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for
the repository `puzzle_binary` BCE-with-logits trainer. All
tensors are finite per batch:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` when
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `base_logit`: `(B,)` trunk-only logit.
- `criticality_residual`: `(B,)` criticality-MLP output added to
  `base_logit`.
- `candidate_indices`: `(B, K)` selected square indices.
- `candidate_gating_scores`: `(B, K)` raw gating scores at the
  candidate squares.
- `candidate_own_indicator`, `candidate_opp_indicator`,
  `candidate_occupancy`: `(B, K)` square statistics from the 12
  piece planes.
- `delta_logits`: `(B, K, T)` per-(candidate, intervention) estimated
  logit after the intervention.
- `sensitivities`: `(B, K, T)` `base_logit - delta_logit`.
- `sensitivity_per_candidate`: `(B, K)` mean sensitivity over
  intervention types.
- `criticality_max`, `criticality_top2_gap`, `criticality_entropy`,
  `criticality_signed_sum`, `criticality_own_vs_enemy_split`: `(B,)`
  the five readout statistics.
- `gating_distribution_entropy`: `(B,)` entropy of the per-square
  gating distribution (normalised by `log 64`).
- `trunk_features`: `(B, channels, 8, 8)` CNN stem output.
- `ablation_active`, `uses_learned_candidates`, `uses_delta_readout`,
  `uses_all_interventions`, `candidate_k_levels`,
  `num_intervention_types`: `(B,)` flags exposing the running
  ablation.

## Ablations

The packet's required ablations are exposed via `model.ablation`:

- `"none"` — main model (`candidate_k`, all three interventions,
  learned gating, criticality residual on).
- `"random_candidates"` — replace the gating-driven top-k with a
  fixed random permutation.
- `"no_delta_readout"` — drop the criticality residual; falls back
  to the trunk-only `base_logit`.
- `"full_remove_only"` — restrict to the `remove_piece` intervention
  type only.
- `"candidate_k_4"` — collapse `candidate_k` to 4 to test the
  cost/performance trade-off the packet calls out.

## Implementation Binding

- Registered model name: `causal_piece_derivative_network`
- Source implementation file: `src/chess_nn_playground/models/causal_piece_derivative_network.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i179_causal_piece_derivative_network/model.py`
