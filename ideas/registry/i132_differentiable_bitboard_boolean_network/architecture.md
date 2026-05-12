# Architecture

`Differentiable Bitboard Boolean Network` learns soft bitboard predicates
from the simple_18 board tensor and combines them through differentiable
Boolean operations to produce one ``puzzle_binary`` logit. Predicates are
real-valued maps in ``(0, 1)`` over the 8x8 grid, so they behave like
soft bitboards; chess-shaped shifts and Boolean complement build a
literal bank, and a learned soft-AND clause layer feeds a learned
soft-OR disjunct layer (a differentiable disjunctive-normal-form
readout).

## Input And Predicate Bank

- Input is the repo `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- A small convolutional trunk (``BoardConvStem``-style ``ConvNormAct``
  blocks of width ``channels``) produces a feature map, and a final
  ``1x1`` convolution emits ``num_predicates`` logits that are passed
  through a sigmoid to yield the soft bitboard bank
  ``P in (0, 1) ^ (num_predicates, 8, 8)``.

## Literal Expansion (Bitboard Algebra)

Each predicate ``p`` is expanded into a literal bank that mirrors how
chess rules are written with bitboards:

- the predicate itself ``p``,
- its Boolean complement ``not p = 1 - p``,
- ``num_shifts`` chess-shape shifted copies ``shift_k(p)``.

Shifts reuse the deterministic ``build_shift_maps`` table from the
``BitboardShiftAlgebraNetwork`` (king and knight directions, padded with
zeros at board edges). The default ``num_shifts = 6`` selects the first
six entries of ``SHIFT_NAMES`` (north / south / east / west /
diagonal-NE / diagonal-NW). The total literal count is
``num_predicates * (2 + num_shifts)``.

## Differentiable Boolean Operations

The literal bank is fed into a single-layer DNF reasoner with two stages:

- **Soft AND clauses.** ``num_clauses`` clauses each carry a learnable
  per-literal selector
  ``s_{c,l} = sigmoid(theta_{c,l} - clause_bias)``. With
  ``g(lit, s) = s * lit + (1 - s)`` (the standard noisy-AND gate), the
  clause activation is

  ```text
  clause_c(b, h, w) = prod_l g(lit_l(b, h, w), s_{c,l})
                    = exp( sum_l log(s_{c,l} * lit_l(b, h, w) + (1 - s_{c,l})) )
  ```

  When ``s = 0`` the literal contributes a no-op factor of 1, when
  ``s = 1`` it contributes the literal itself. ``clause_bias_init`` is
  set to ``4.0`` so most selectors start near 0 and clauses begin life
  close to 1 (no commitments).

- **Soft OR disjuncts.** ``num_disjuncts`` disjuncts apply the
  De Morgan dual on top of the clause field. Each disjunct ``d`` has
  a learnable selector
  ``v_{d,c} = sigmoid(phi_{d,c} - disjunct_bias)`` and computes

  ```text
  log(1 - disjunct_d(b, h, w)) = sum_c log(1 - v_{d,c} * clause_c(b, h, w))
  disjunct_d(b, h, w)          = 1 - exp( log(1 - disjunct_d(b, h, w)) )
  ```

  ``disjunct_bias_init = 1.0`` keeps initial disjuncts moderately active.
  Boolean NOT enters through the literal bank, so the layer realises a
  full Boolean (AND, OR, NOT) algebra over soft bitboards.

## Pooling And Head

- Spatial mean and max over each disjunct map produce a
  ``2 * num_disjuncts`` summary vector.
- A LayerNorm + two-layer GELU MLP with dropout maps the summary to one
  ``puzzle_binary`` logit (fine labels ``0`` and ``1`` map to non-puzzle,
  fine label ``2`` maps to puzzle).
- The forward pass returns a dict whose ``logits`` tensor has shape
  ``(batch,)`` alongside per-sample diagnostics (predicate / clause /
  disjunct mean and max activations, binary entropies, the active
  fraction at threshold ``0.5``, and global selector strengths) of
  shape ``(batch,)`` for ablation analysis.

## Implementation Binding

- Registered model name: `differentiable_bitboard_boolean_network`.
- Source implementation: `src/chess_nn_playground/models/trunk/differentiable_bitboard_boolean_network.py`.
- Idea-local wrapper: `ideas/registry/i132_differentiable_bitboard_boolean_network/model.py`.
