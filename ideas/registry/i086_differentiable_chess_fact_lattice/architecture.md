# Architecture

`Differentiable Chess Fact Lattice` (DCFL) implements the math thesis as a
neural classifier whose bottleneck is an explicit, differentiable abstract
interpretation over chess facts. The model is board-only for the
`puzzle_binary` contract and returns one BCE puzzle logit plus
abstract-interpretation diagnostics.

## Mechanism

The forward path follows the math thesis:

1. **Concrete-to-abstract abstraction (`alpha`).** The simple_18 current-board
   tensor is read as 12 piece occupancy planes (white/black × P,N,B,R,Q,K)
   plus a side-to-move plane. Each occupancy plane is lifted to an
   interval-valued abstract occupancy `[occ_lo, occ_hi]` over the lattice
   `L = [0,1]^{12 \times 8 \times 8}`.
2. **Transfer functions over chess facts.** The interpreter computes
   interval-valued abstract states for:
   - **Attack** per piece type and color (leaper kernels for pawn / knight /
     king; ray-based slider transfer for bishop / rook / queen with
     differentiable blocker propagation `clear *= (1 - blocker)`),
   - **Defense** as the meet of friendly occupancy with friendly attack mass,
   - **King zone** as a soft 1-ring noisy-or with a damped 2-ring,
   - **Tension / conflict** channels combining opponent attack mass, friendly
     defense, attack-defense imbalance, value-at-risk, king pressure, line
     exposure, contested squares, and loose pieces.
   All operators are differentiable lattice morphisms: noisy-or for join,
   product for meet, and interval complements `1 - [u, l]`.
3. **Widened fixpoint loop.** The transfer step is iterated `transfer_passes`
   times. Each iteration uses a soft join
   `lower = -tau * logsumexp(-old/tau, -new/tau)`,
   `upper = tau * logsumexp(old/tau, new/tau)`,
   followed by a widening operator with a step-decayed epsilon. This is the
   smooth analogue of the discrete widening `\nabla` from abstract
   interpretation and provides a finite-step over-approximation.
4. **Readout from abstract features.** The abstract state is flattened into
   lower / upper / width slices of all interval channels plus conflict
   channels, concatenated with the side-to-move plane, and processed by a
   1x1 + 3x3 convolutional readout. Mean and max pooling produce the input to
   the puzzle head.
5. **Puzzle head.** A two-layer MLP returns one logit. The model preserves
   the repo board-tensor contract: `(batch, 18, 8, 8)` in, `(batch,)` BCE
   logits out, with auxiliary diagnostics in the output dictionary.

## Diagnostics

The output dictionary carries the abstract-interpretation diagnostics used by
the trainer / report:

- `interval_width_mean`, `widening_width` — fixpoint approximation tightness,
- `conflict_energy` — squared mass of unresolved attack/defense conflicts,
- `attack_mass`, `defense_mass` — pooled abstract attack and defense,
- `king_zone_pressure`, `value_at_risk`, `line_exposure` — meet-channel
  diagnostics from the tension lattice,
- `board_consistency_error` — abstraction sanity error,
- `monotonicity_penalty` — `relu(-piece_attack_gate)` mean for the lattice
  monotonicity constraint,
- `abstract_feature_energy`, `mechanism_energy` — pooled feature energies,
- `proposal_profile_strength`, `proposal_keyword_count` — packet diagnostics.

## Ablations

The model exposes the falsifiers from the math thesis as constructor flags:

- `use_intervals=False` collapses each interval to its midpoint (point
  abstraction), removing the over-approximation.
- `use_meet_channels=False` zeroes the value-at-risk, king-pressure,
  line-exposure, and loose-piece meet channels (no interaction features).
- `use_ray_transfer=False` disables slider blocker propagation (leapers only).
- `use_king_zone=False` removes the soft king-zone transfer.
- A `variant: "pool_control"` config builds the lightweight
  `PoolControlClassifier` baseline used by the lattice-vs-pool ablation.

## Implementation Binding

- Registered model name: `differentiable_chess_fact_lattice`.
- Source implementation file:
  `src/chess_nn_playground/models/trunk/differentiable_chess_fact_lattice.py`.
- Idea-local wrapper: `ideas/registry/i086_differentiable_chess_fact_lattice/model.py`.
