# Math Thesis

Barrier-Cut Puzzle Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md`.

Batch candidate rank: `1`.

## Working Thesis

A true puzzle exists because the defender cannot maintain a barrier
between attacking force and a valuable target — king, queen,
promotion square, pinned defender, mating square. A near-puzzle may
contain pressure on the same target, but the defender's barrier
still holds: the attack flux dissipates against the barrier before
it reaches a valuable target.

This bespoke architecture turns that thesis into an explicit
differentiable barrier-cut computation on the 8x8 board:

1. The encoder predicts three non-negative per-square fields from the
   board tensor:
   - `A(x)` — attacker pressure mass.
   - `D(x)` — defender barrier capacity.
   - `T(x)` — target value (king, queen, promotion square, pinned
     defender, mating square).
2. An iterative diffusion propagates `A` across the board for
   `barrier_steps` rounds. At each step the attack potential is
   absorbed elementwise by the defender field (`min(u, decay_scale * D)`)
   before being smoothed by a learnable 3x3 kernel whose entries
   are non-negative and sum to one.
3. The reachable target value
   ```
   reachable_target_value = sum_{r, f} u_T(r, f) * T(r, f)
   ```
   is the canonical barrier-defect signal: how much attack mass
   reaches a valuable target after the barrier has done its work.
4. The defense-gap field `relu(u_T - D)` highlights squares where
   the barrier is locally insufficient and pooled summaries of it
   feed the puzzle classifier.

The classifier reads `reachable_target_value`, the defense-gap
summary scalars, the per-step absorbed mass, and pooled trunk
features, and emits one puzzle logit. High reachable-target value
or large defense gap drive the position toward the puzzle class; a
barrier that absorbs all attack mass before it reaches any target
drives it toward non-puzzle.

## Implementation

This idea is implemented as a bespoke architecture, not a shared
`ResearchPacketProbe` wrapper. See the `Implementation Binding`
section of `architecture.md` for the registered model name, the
source implementation file, and the idea-local wrapper.
