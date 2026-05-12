# Math Thesis

Differentiable Chess Fact Lattice (DCFL).

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-28_0857_tuesday_new_york_diff_ai.md`.

## Working thesis

DCFL is a neural classifier whose bottleneck is an explicit, differentiable
abstract interpretation. A board tensor is mapped to interval-valued
abstract chess facts on a complete lattice, those facts are propagated by
sound, monotone transfer functions, and the resulting abstract state is read
out by a small puzzle head. The lattice and operators are smooth so that
gradients flow from puzzle loss back into the symbolic abstraction.

## Lattice

Let `L` be the complete lattice of interval-valued chess facts:

- `L_occ = [0, 1]^{C × 8 × 8}` for piece occupancy intervals,
- `L_attack`, `L_defense`, `L_king_zone`, `L_tension` analogously,
- joined by the product order so that `(occ, attack, defense, king_zone,
  tension, conflict)` lives in a complete sub-lattice
  `L = L_occ × L_attack × L_defense × L_king_zone × L_tension × L_conflict`.

`bottom` is the all-zero interval state; `top` is the all-`[0,1]` state.
Meet `\sqcap` is implemented as componentwise product, join `\sqcup` as
componentwise noisy-or `1 - (1-a)(1-b)`. Interval complement is
`[1-u, 1-l]`. These operators are differentiable lattice morphisms.

## Abstraction map

`alpha : concrete board -> L` reads the simple_18 board tensor and projects
the 12 piece planes plus side-to-move plane to interval-valued occupancy
`[occ_lo, occ_hi]`, with interval width 0 on observed pieces and a safety
gap on uncertain channels.

## Transfer functions

The transfer step `T : L -> L` is a sound differentiable approximation of
the chess-fact deduction step:

- `attack_T` uses leaper kernels (pawn, knight, king) and ray-based slider
  transfer (bishop, rook, queen) with smooth blocker propagation
  `clear *= (1 - blocker)`. A learned monotone gate
  `sigmoid(piece_attack_gate)` modulates per-color, per-type intensities.
- `defense_T` is the meet of friendly occupancy with friendly attack mass.
- `king_zone_T` is a soft 1-ring noisy-or with a damped sigmoid 2-ring.
- `tension_T` produces interval channels for opponent attack, friendly
  defense, attack-defense imbalance, value-at-risk
  `occ \sqcap value \sqcap opp_attack \sqcap (1 - defense)`,
  king pressure, line exposure, contested squares, and loose pieces.
- `conflict_T` collects scalar imbalance and value-at-risk per side.

`T` is monotone with respect to the product order because every operator is
a noisy-or, product, ReLU, or interval complement composed with monotone
shifts.

## Widened fixpoint

The forward pass iterates `T` for `transfer_passes` rounds with a soft join
and widening operator:

- soft join: `lower = -tau * logsumexp(-a/tau, -b/tau)`,
  `upper = tau * logsumexp(a/tau, b/tau)`,
- widening: each step adds a decaying `epsilon` to the upper and subtracts
  it from the lower, then clamps to `[0, 1]`.

This is the smooth analogue of the discrete widening `\nabla` from abstract
interpretation and provides a finite-step over-approximation of the
post-fixed point of `T`.

## Readout and head

The fixed point is concretized into lower / upper / width slices of all
interval channels plus conflict channels, conditioned on the side-to-move
plane, encoded by a `1x1 -> 3x3` convolutional readout, mean+max pooled, and
mapped by a two-layer MLP to one BCE puzzle logit. The model preserves the
repo board-tensor contract: `(batch, 18, 8, 8)` in, `(batch,)` logits out.

## Falsifiers

The thesis is falsifiable by ablations exposed on the bespoke model:

- `use_intervals=False` collapses each interval to its midpoint (point
  abstraction). DCFL should beat this when puzzle structure depends on
  uncertainty propagation.
- `use_meet_channels=False` zeroes the meet diagnostics (value-at-risk, king
  pressure, line exposure, loose piece). DCFL should beat this when the
  meet operator is what carries puzzle signal.
- `use_ray_transfer=False` removes slider blocker propagation. DCFL should
  beat this when ray-blocked tactics matter.
- `use_king_zone=False` removes king-zone transfer. DCFL should beat this
  when king pressure matters.
- The `pool_control` variant is a CNN-pool baseline at matched compute.
