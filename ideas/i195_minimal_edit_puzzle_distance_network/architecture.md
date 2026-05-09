# Architecture

`Minimal-Edit Puzzle Distance Network` is a bespoke puzzle_binary
classifier built around a single, sharp idea: *measure how far the
input board is from the closest puzzle prototype, where distance is a
soft per-square edit cost*. A near-puzzle should be one small edit away
from a real puzzle prototype; a non-puzzle should be many edits away.

## Inputs

- Board tensor only: `(B, 18, 8, 8)` simple_18 contract.
- CRTK / source / engine metadata is reporting-only and never enters
  the model.

## Pipeline

1. **Symbol encoder** `S = encoder(x)`. A compact convolutional square
   encoder (`depth` repetitions of
   `Conv2d(_, channels, 3, padding=1) -> Norm -> GELU -> Dropout2d`)
   feeds a `1x1` projection to `num_symbols` channels followed by a
   per-square softmax. The output `S(x) in R^(B, num_symbols, 8, 8)`
   is a per-square distribution over a learned symbol alphabet.
2. **Learnable puzzle prototype bank** `P_k`,
   `k = 1..num_prototypes`. Each prototype is a per-square
   distribution over the same symbol alphabet
   (`P_k = softmax(prototype_logits_k)`). The bank is the model's
   compact memory of "what a puzzle position looks like".
3. **Per-square soft edit cost.** For every square `(r, f)` and every
   prototype `k`, the agreement is the inner product of the two
   distributions:
   `agreement_k(r, f) = <S(x)[:, r, f], P_k[:, r, f]> in [0, 1]`.
   The edit cost is its complement,
   `cost_k(r, f) = 1 - agreement_k(r, f)`. This is the natural soft
   relaxation of "this square needs an edit": cost is `0` when the
   input symbol matches the prototype symbol and `1` when they
   maximally disagree.
4. **Per-prototype edit distance.**
   `D_k(x) = sum_{r, f} cost_k(r, f)`. This is a soft Hamming-style
   distance: the total number of soft edits required to turn `x` into
   prototype `k`.
5. **Soft minimum edit distance** (the canonical "minimal-edit puzzle
   distance"):
   `D_min(x) = -T * logsumexp(-D_k(x) / T)`, with temperature
   `edit_temperature = T`. The accompanying soft assignment
   `pi_k = softmax(-D_k / T)` measures which prototype is closest.
6. **Diagnostics.** From the same edit-cost tensor the forward pass
   exposes the hard min distance, the nearest prototype index, the
   assignment entropy, the per-square min cost map (weighted by `pi`),
   and per-prototype distance summary statistics.
7. **Classifier head.** A small MLP
   `LayerNorm -> Linear(hidden_dim) -> GELU -> Dropout -> Linear(1)`
   reads an 11-dim feature pack assembled from the soft min distance,
   per-prototype distance summary statistics, the assignment entropy,
   and per-square cost summary scalars to produce one puzzle logit.
   Positions close to a prototype (small `D_min`) are pushed toward
   the puzzle class; positions many edits away are pushed toward
   non-puzzle.

## Tensor Contract

```
input x:                            (B, 18, 8, 8)
trunk features feats:               (B, channels, 8, 8)
symbol distribution S(x):           (B, num_symbols, 8, 8)
prototype bank P:                   (num_prototypes, num_symbols, 8, 8)
per-square cost:                    (B, num_prototypes, 8, 8)
per-prototype distance D:           (B, num_prototypes)
soft min edit distance D_min:       (B,)
hard min edit distance:             (B,)
nearest_prototype_index:            (B,)
prototype_assignment pi:            (B, num_prototypes)
assignment_entropy:                 (B,)
per_square_min_cost:                (B, 8, 8)
per_square_min_cost_mean / _max:    (B,)
mean_per_square_cost / max_:        (B,)
trunk_energy:                       (B,)
logits:                             (B,)
```

## Why "Minimal Edit" Rather Than a Generic Distance

Closeness in pixel/feature space is not closeness in tactical content.
The thesis is more specific: *one small edit* — moving a single piece,
inserting a single attacker, removing a single defender — should
suffice to convert a near-puzzle into a real puzzle. Modeling this
explicitly requires (a) a per-square comparison so the cost can be
attributed to a small number of squares, and (b) a learnable bank of
puzzle prototypes the input is compared against. The classifier head
reads `D_min` so positions that need only a few squares' worth of edit
to match a prototype score high — exactly the "near-puzzle" regime the
benchmark targets.

## Material Distinctness

This architecture is materially distinct from:

- The shared `ResearchPacketProbe` scaffold: no per-square edit cost,
  no learnable prototype bank, no soft-min distance head.
- `RayGrammarEditDistanceNetwork` (i217): that model runs a
  Needleman-Wunsch DP over 1-D rays against template strings; this
  one computes a full-board, per-square Hamming-style edit cost
  against 2-D puzzle prototypes.
- `SymmetricDifferenceTwinEncoder` (i116): that model compares the
  board to a deterministic safe transform of itself; this one
  compares the board to a *learnable* puzzle prototype bank.

Removing the prototype bank, the per-square edit cost, or the
soft-min head would change the model's computation in observable ways
and is exactly what the central ablations switch off.

## Central Ablations (config switches)

| Ablation         | Config knob              | Effect                                                                                  |
|------------------|--------------------------|-----------------------------------------------------------------------------------------|
| `narrow_trunk`   | `channels: 32`           | Halves the encoder latent width.                                                        |
| `shallow_trunk`  | `depth: 1`               | Single-conv encoder; tests how much depth the symbol encoder needs.                     |
| `wide_head`      | `hidden_dim: 192`        | Doubles the head width.                                                                 |
| `few_prototypes` | `num_prototypes: 4`      | Smaller puzzle prototype bank.                                                          |
| `many_prototypes`| `num_prototypes: 64`     | Larger bank — more flexible memory of puzzle templates.                                 |
| `cool_min`       | `edit_temperature: 0.25` | Sharper softmin (closer to argmin) over prototypes.                                     |
| `warm_min`       | `edit_temperature: 4.0`  | Softer min — averages distances across prototypes.                                      |
| `no_dropout`     | `dropout: 0.0`           | Removes regularization on encoder and head.                                             |
| `no_bn`          | `use_batchnorm: false`   | Replaces BN with GroupNorm(1, ...); useful for tiny batches.                            |

## Implementation Binding

- Registered model name: `minimal_edit_puzzle_distance_network`
- Source implementation file:
  `src/chess_nn_playground/models/minimal_edit_puzzle_distance_network.py`
- Idea-local wrapper:
  `ideas/i195_minimal_edit_puzzle_distance_network/model.py`

The wrapper is a thin adapter over
`build_minimal_edit_puzzle_distance_network_from_config`; it does not
touch `ResearchPacketProbe`. The shared probe wrapper has been
removed.
