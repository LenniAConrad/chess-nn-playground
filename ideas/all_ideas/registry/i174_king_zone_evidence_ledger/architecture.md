# Architecture

`King-Zone Evidence Ledger` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and returns one
puzzle logit per position, plus diagnostics that expose each
king's ledger energy, ledger attention spread, and king-ring
attention pressure.

## Mechanism

The architecture follows the packet thesis verbatim: maintain a
small bank of learned evidence ledger slots around each king, and
let the puzzle logit be a function of the comparison between the
own-king and opponent-king ledgers.

```
own_king_slots:  K x D
opp_king_slots:  K x D
global_slots:    K x D

slot = slot + gated_pool(board_features,
                         piece_features,
                         king_relative_features)

puzzle_logit = MLP([
    own_king_ledger,
    opp_king_ledger,
    ledger_difference,
    ledger_product,
    global_board_pool,
])
```

Inputs to the model are limited to the `simple_18` board tensor.
Engine, verification, source, and CRTK metadata are never used. The
two king squares are read off planes 5 (white king) and 11 (black
king) and re-keyed to *own* / *opp* using the side-to-move plane
(12) so the network is consistent under the existing trainer
contract for `puzzle_binary`.

## Trunk

A stack of `depth` `Conv3x3 → BatchNorm → GELU [→ Dropout2d]`
layers turns the 18-plane board into a per-square feature map of
width `channels`. The trunk only consumes board planes; the piece
"tokens" mentioned in the packet are realised as the standard
12-piece planes inside `simple_18`.

## King anchors and king-relative features

For each board we recover

- `own_king_anchor`: argmax of plane 5 (white king) when the side
  to move is white, else argmax of plane 11.
- `opp_king_anchor`: the other king.

For a king anchor `k` and every square `q` we compute five
king-relative coordinate channels:

1. signed rank delta `(q.rank − k.rank) / 7`
2. signed file delta `(q.file − k.file) / 7`
3. Chebyshev distance `max(|Δrank|, |Δfile|) / 7`
4. Manhattan distance `(|Δrank| + |Δfile|) / 14`
5. in-king-ring indicator `1[Chebyshev(q, k) ≤ king_ring_radius]`

These five channels are concatenated to the trunk feature map to
form `F̂(s, k) ∈ R^{(C + 5) × 8 × 8}` for the own and opponent
ledgers. The global ledger sees the trunk concatenated with five
zero channels so the three banks share one feature width.

## Evidence ledgers

Each ledger bank holds `num_slots` learnable initial vectors of
width `slot_dim` (`slots0`). The forward pass flattens
`F̂(s, k)` to `(B, 64, F)` and runs `ledger_layers` rounds of the
update

```
queries  = W_q · slots                          # (B, K, F)
scores   = queries · features^T / sqrt(F)        # (B, K, 64)
α        = softmax(scores, dim=-1)               # (B, K, 64)
pooled   = α · features                          # (B, K, F)
update   = W_v · pooled                          # (B, K, slot_dim)
gate     = sigmoid(W_g · slots)                  # (B, K, slot_dim)
slots    = LayerNorm(slots + gate * Dropout(update))
```

This is the packet's `slot = slot + gated_pool(features)` rule
realised as slot-conditioned soft attention. The non-linearities
are kept compact so the ledger remains a bottleneck of width
`K · D` rather than an unconstrained per-square head.

## Readout

The three slot tensors `S_o, S_x, S_g ∈ R^{B × K × D}` are
flattened to width `K · D` and concatenated as the packet
specifies:

```
readout = [
    flat(S_o), flat(S_x),
    flat(S_o − S_x), flat(S_o ⊙ S_x),
    flat(S_g),
]
```

A `LayerNorm → Linear → GELU → Dropout → Linear` head emits
`num_classes` logits — the puzzle logit when `num_classes == 1`.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for
the repository `puzzle_binary` BCE-with-logits trainer. All
tensors are finite per batch:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` when
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `trunk_features`: `(B, channels, 8, 8)` features after the
  trunk.
- `own_king_ledger`, `opp_king_ledger`, `global_ledger`:
  `(B, num_slots, slot_dim)` ledger slot banks after the last
  update layer.
- `ledger_difference`, `ledger_product`:
  `(B, num_slots, slot_dim)` element-wise comparisons fed to the
  head.
- `own_king_energy`, `opp_king_energy`, `global_energy`,
  `own_minus_opp_energy`: `(B,)` mean-square ledger activations.
- `own_attention`, `opp_attention`, `global_attention`:
  `(B, num_slots, 64)` final-layer slot attention weights.
- `own_attention_entropy`, `opp_attention_entropy`,
  `global_attention_entropy`: `(B,)` mean Shannon entropy of slot
  attention.
- `own_king_ring_pressure`, `opp_king_ring_pressure`: `(B,)` mean
  attention mass landing inside the king ring (`Chebyshev ≤
  king_ring_radius`).
- `own_anchor_rank`, `own_anchor_file`, `opp_anchor_rank`,
  `opp_anchor_file`: `(B,)` real king square coordinates.
- `own_anchor_rank_used`, `own_anchor_file_used`,
  `opp_anchor_rank_used`, `opp_anchor_file_used`: `(B,)` anchor
  coordinates actually fed to the ledger (differs from the real
  king square only under `random_king_anchor`).
- `side_to_move`: `(B,)` `1.0` if white to move else `0.0`.
- `num_slots_levels`, `slot_dim_levels`, `ledger_layers_levels`:
  `(B,)` scalar tags carrying the configured ledger geometry.
- `ablation_active`, `uses_king_relative`, `uses_random_anchor`,
  `uses_global_only`: `(B,)` flags exposing the running ablation.

## Ablations

The packet's required ablations are exposed via `ablation`:

- `"none"` — main model.
- `"no_king_relative"` — drop the five king-relative coordinate
  channels so the gated pool sees only board features. Tests
  king anchoring.
- `"random_king_anchor"` — replace the real king anchors with
  deterministic per-batch random anchors. Tests real king
  semantics.
- `"global_slots_only"` — drop the per-king ledgers and feed only
  the global slot ledger to the head. Tests king-specific ledger
  value.
- `"slot_count_sweep"` — no-op structural flag. The sweep itself
  is driven by the `num_slots` config value; tagging a run with
  this ablation marks it as a sweep entry without changing the
  main model.

## Implementation Binding

- Registered model name: `king_zone_evidence_ledger`
- Source implementation file: `src/chess_nn_playground/models/king_zone_evidence_ledger.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i174_king_zone_evidence_ledger/model.py`
