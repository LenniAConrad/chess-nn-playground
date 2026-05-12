# Architecture

`Disproof-Ledger Puzzle Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and returns one
puzzle logit per position. The packet thesis is that a puzzle is not
only "this position is sharp" — it is also "and *no* disproof
condition holds." A near-puzzle typically satisfies the first part but
fails one of the disproof conditions: the king can escape, a defender
can recapture, the line is blocked, the threat is too slow, the target
is protected enough, or the side to move lacks tempo. The model
therefore learns a positive-evidence head and a typed *disproof
ledger* head, and computes the puzzle logit as `positive_evidence -
sum_d softplus(disproof_entry_d)`.

## Mechanism

A compact convolutional trunk turns the 18-plane board into a
per-square feature map. Two heads share that trunk:

- **Positive-evidence head.** Global-average-pools the trunk feature
  map, then runs a small `LayerNorm → Linear → GELU → Dropout →
  Linear` MLP that returns one scalar `positive_evidence` per batch
  row. This is the packet's `positive_evidence = pos_head(h)`.
- **Disproof head.** A `Conv1x1 → GELU → Conv1x1` stack emits a
  `(B, D, 8, 8)` per-square *disproof field*. The default `D = 8`
  matches the packet's first config; the first six channels are
  semantically tied to the named disproof reasons listed in the
  thesis (`king_can_escape`, `defender_can_recapture`,
  `line_is_blocked`, `threat_is_too_slow`, `target_is_protected`,
  `side_lacks_tempo`), and any extra channels are unnamed slack
  channels.

The disproof field is mean-pooled over the 64 squares to form a
vector of raw `disproof_entries ∈ R^{B×D}`. Softplus turns each entry
into a non-negative `disproof_strength` so a channel can only
*increase* disproof, never invert into positive evidence:

```
disproof_strengths = softplus(disproof_entries)        # (B, D), >= 0
disproof_strength_total = disproof_strengths.sum(-1)   # (B,)
puzzle_logit = positive_evidence - disproof_strength_total
```

## Why subtraction matters

Subtracting a non-negative ledger from positive evidence is not the
same as adding a generic negative head. It enforces the asymmetry the
packet calls for: every disproof channel is by construction unable to
*help* the position look like a puzzle, and the readout literal-mindedly
reads "evidence minus disproof." This is why the head differs from
the negative-class disentangling head, which is allowed to put any
sign on its output.

## Sparsity and the near-puzzle auxiliary

The packet specifies two training-side regularisers:

- **L1 sparsity** on the softplus disproof entries so a small number of
  clear disproofs dominate. `disproof_l1` (= `disproof_strength_total`)
  is exposed as a per-batch tensor that the trainer multiplies by
  `disproof_sparsity` and adds to the loss.
- **Near-puzzle auxiliary** so positions that source-classify as
  near-puzzle should light up at least one disproof channel.
  `max_disproof_strength` and `max_disproof_channel` are exposed for
  this purpose, weighted by `near_disproof_aux_weight` in the trainer.

Both regularisers are model-side flags (`uses_disproof_sparsity`,
`uses_near_disproof_aux`) and the ablations turn them off.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for
the repository `puzzle_binary` BCE-with-logits trainer (or
`(B, num_classes)` when `num_classes > 1`, with the puzzle scalar
written into the last column of a zero-padded tensor).
All tensors are finite per batch:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` when
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `positive_evidence`: `(B,)` raw positive logit.
- `disproof_field`: `(B, D, 8, 8)` per-square disproof field.
- `disproof_entries`: `(B, D)` mean-pooled raw disproof entry.
- `disproof_strengths`: `(B, D)` softplus of entries (>= 0).
- `disproof_strength_total`: `(B,)` sum of channel strengths.
- `disproof_l1`: `(B,)` L1 of softplus strengths (= total) for the
  sparsity penalty.
- `max_disproof_strength`: `(B,)` max channel strength.
- `max_disproof_channel`: `(B,)` argmax channel index, as a float
  tensor for the diagnostics columns.
- `trunk_features`: `(B, channels, 8, 8)`.
- `ablation_active`, `uses_disproof_subtraction`,
  `uses_disproof_sparsity`, `uses_near_disproof_aux`,
  `num_disproof_channels`: `(B,)` flags exposing the running
  ablation.

## Ablations

The packet's required ablations are exposed via `model.ablation`:

- `"none"` — main model: full subtraction, sparsity active, near-aux
  active.
- `"no_disproof_subtraction"` — drop the subtraction so
  `puzzle_logit = positive_evidence`. Tests whether the ledger
  contributes more than its parameters.
- `"dense_disproof_no_sparsity"` — keep the ledger but flip the
  sparsity flag off so the trainer applies no L1 pressure.
- `"no_near_aux"` — flip the near-puzzle auxiliary flag off so the
  trainer does not require near-puzzles to light at least one
  disproof channel.

## Implementation Binding

- Registered model name: `disproof_ledger_puzzle_network`
- Source implementation file: `src/chess_nn_playground/models/disproof_ledger_puzzle_network.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i181_disproof_ledger_puzzle_network/model.py`
