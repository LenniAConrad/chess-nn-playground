# Architecture

`Hall-Defect Obligation Matroid Network` implements the source packet's overload certificate as an exact Hall-defect zeta profile over current-board defender-obligation set systems.

## Implementation Binding

- Registered model name: `hall_defect_obligation_matroid_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/hall_defect_obligation_matroid.py`
- Idea-local wrapper: `ideas/registry/i053_hall_defect_obligation_matroid_network/model.py`

## Input Contract

The deterministic branch supports the repo's `simple_18` tensor with shape `(B, 18, 8, 8)`. It decodes channels `0..5` as white pawn, knight, bishop, rook, queen, king planes; channels `6..11` as the corresponding black planes; and channel `12` as side-to-move. Castling and en-passant planes remain available to the learned board adapter but are not interpreted by the Hall rule branch. Unknown encodings fail closed with a clear `ValueError`.

The repo puzzle-binary trainer uses one BCE logit, so `output["logits"]` has shape `(B,)`. The model still computes internal non-puzzle/puzzle scores and exposes them as `two_class_logits` with shape `(B, 2)`.

## Pseudo-Legal Control

`SafeBoardDecoder` and `PseudoLegalAttackGenerator` build current-board contact tables only. Pawns use color-aware diagonal attacks; knights and kings use fixed leaper offsets; bishops, rooks, and queens scan along rays and stop at blockers. The implementation does not generate legal moves, count replies, check mate or stalemate, call an engine, or inspect future move trees.

The decoded rule tensors include:

- `pieces`: `(B, 2, 6, 8, 8)`;
- `piece_slots`: `(B, 2, 16, 10)`;
- `controls`: `(B, 2, 16, 64)`;
- `attack_count`: `(B, 2, 64)`.

## Obligation Sets

`ObligationSetBuilder` constructs two side-relative roles for every board:

- role `0`: obligations of the side not to move under pressure from the side to move;
- role `1`: obligations of the side to move under pressure from the opponent.

For each role it builds the six packet strata:

- `attacked_all_nonking_assets`;
- `attacked_value_ge_3`;
- `attacked_value_ge_5`;
- `attacked_queen_or_rook`;
- `king_ring_radius_1_contested`;
- `king_ring_radius_2_contested`.

Attacked asset obligations are own non-king pieces currently attacked by the opposing pseudo-legal controls. King-ring obligations are current squares in the Chebyshev radius-1 or radius-2 king zone that are attacked by the opposing side. Obligation weights are deterministic current-board values: piece values for assets and fixed king-zone weights with a small attack-count increment for contested king squares.

For each role and stratum, defenders are own pieces that pseudo-legally control at least one obligation square. The defended piece itself is excluded as its own defender. If more than `D_max` defenders exist, the model keeps defenders by descending positive obligation degree, piece value, center proximity, and stable square order, and records the discarded count. Each obligation neighborhood is encoded as a bitmask over the selected defenders.

## Hall Zeta Defect Layer

`HallZetaDefectLayer` receives neighborhood masks `(B, 2, 6, O_max)` and builds histograms over all `2^D_max` defender subsets. It applies the subset zeta transform:

```text
Z_count(T) = sum_{m subset T} count(m)
Z_weight(T) = sum_{m subset T} weight(m)
```

The layer emits cardinal Hall defects:

```text
max_T Z_count(T) - |T|
```

and weighted defects for the configured penalties:

```text
max_T Z_weight(T) - lambda * |T|
```

The resulting token for each role/stratum includes obligation counts, defender counts, truncation counts, edge density, degree summaries, weight summaries, the cardinal defect, the cardinal argmax subset size and mass, weighted defects, and weighted argmax subset sizes.

## Token And Board Fusion

`HallDefectTokenEncoder` applies a shared token MLP to the twelve role/stratum Hall tokens, then concatenates mean pooling, max pooling, and the role difference summary. `BoardContextAdapter` is intentionally small: a `1x1` convolution followed by a `3x3` convolution, global mean pooling, and global max pooling. The fusion MLP receives the Hall embedding, board embedding, side-to-move scalar, and low-dimensional nuisance summaries and returns internal two-class scores.

## Ablation Modes

The implementation supports the packet's central controls:

- `edge_ablation_mode: degree_rewire` replaces each neighborhood with a deterministic degree-matched defender subset;
- `count_only` removes zeta max/cut semantics and keeps count/degree/weight nuisance tokens;
- `weight_shuffle` rotates obligation weights inside each role/stratum;
- `complete_neighborhood` gives every obligation all selected defenders.

## Outputs

Forward returns a dictionary containing:

- `logits`: BCE-compatible puzzle logits, shape `(B,)`;
- `two_class_logits`: internal class scores, shape `(B, 2)`;
- diagnostics including `hall_cardinal_defect`, `hall_mean_cardinal_defect`, `hall_weighted_defect`, `hall_defect_energy`, `sparse_certificate_energy`, `overload_role_gap`, `defense_gap`, `obligation_count`, `defender_count`, `defender_truncation_count`, `hall_edge_density`, `zero_defender_obligation_count`, `board_context_energy`, `mechanism_energy`, and `proposal_profile_strength`;
- when `return_aux=True`, decoded piece slots, contact tables, obligation masks, weights, defender masks, bitmasks, Hall tokens, histograms, and argmax subset-size diagnostics.
