# Architecture

`Barrier-Cut Puzzle Network` is a bespoke `puzzle_binary` classifier
built around an explicit barrier / min-cut interpretation of tactical
positions: a true puzzle exists when the defender cannot maintain a
barrier between an attacking force and a valuable target (king,
queen, promotion square, pinned defender, mating square). A near-
puzzle exerts pressure but the defender's barrier still holds — the
attack flux does not reach a valuable target.

## Inputs

- Board tensor only: `(B, 18, 8, 8)` simple_18 contract.
- CRTK / source / engine metadata is reporting-only and never enters
  the model.

## Pipeline

1. **Compact convolutional trunk.** `feats = trunk(x)` runs `depth`
   `Conv2d(_, channels, 3, padding=1) -> Norm -> GELU -> Dropout2d`
   blocks (`Norm` is BatchNorm2d when `use_batchnorm = true`,
   GroupNorm(1, ...) otherwise).
2. **Three per-square fields.** A `1x1` projection to three channels
   followed by a per-channel `softplus` produces three non-negative
   per-square fields:
   - `attack_field A(x) in R_+^(B, 8, 8)` — attacker pressure mass.
   - `defense_field D(x) in R_+^(B, 8, 8)` — local defender barrier
     capacity (how much attack flux a square absorbs).
   - `target_field T(x) in R_+^(B, 8, 8)` — value of a target sitting
     on a square (king, queen, promotion square, pinned defender,
     mating square).
3. **Iterative barrier-cut diffusion.** The attack potential is
   propagated for `barrier_steps` rounds; at each round the
   defender field acts as a per-square cut that absorbs flow before
   the rest is smoothed across neighbours by a learnable 3x3
   diffusion kernel `K = softmax(kernel_logits)` (entries are
   non-negative and sum to 1 so mass is conserved up to the
   defender absorption term):
   ```
   absorbed_t = min(u_t, decay_scale * D)
   u_{t+1}    = K * relu(u_t - absorbed_t)
   ```
   `u_0 = A`. Strong-barrier squares cut attack flow; weak-barrier
   squares let it leak through.
4. **Reachable target value.** The canonical barrier-defect signal:
   ```
   reachable_target_value = sum_{r, f} u_T(r, f) * T(r, f)
   ```
   How much attack mass actually arrives at a valuable target after
   the barrier has done its work.
5. **Defense-gap field.** Per-square locally insufficient barrier:
   ```
   defense_gap(r, f) = max(0, u_T(r, f) - D(r, f))
   ```
   Pooled summary scalars (`defense_gap_mean`, `defense_gap_max`)
   tell the head where the barrier is locally failing.
6. **Diagnostics.** From the same fields the forward pass exposes
   the per-step absorbed mass, the global field totals
   (`attack_total_mass`, `defense_total_capacity`,
   `target_total_value`), the field maxima, and trunk-energy
   summaries.
7. **Classifier head.** A small MLP
   `LayerNorm -> Linear(hidden_dim) -> GELU -> Dropout -> Linear(1)`
   reads a 14-dim feature pack assembled from the reachable target
   value, the field totals and maxima, the per-step absorbed mass
   total, the defense-gap summary scalars, the final attack
   summary scalars, and the trunk-energy scalars to produce one
   puzzle logit. High reachable-target value or large defense gap
   pushes the position toward the puzzle class; a barrier that
   absorbs the attack mass before it reaches any target pushes it
   toward non-puzzle.

## Tensor Contract

```
input x:                          (B, 18, 8, 8)
trunk feats:                      (B, channels, 8, 8)
attack / defense / target field:  (B, 8, 8) each
final attack potential u_T:       (B, 8, 8)
reachable target value:           (B,)
barrier_absorbed_mass per step:   (B, barrier_steps)
defense_gap field:                (B, 8, 8)
defense_gap_mean / _max:          (B,)
attack_total_mass:                (B,)
defense_total_capacity:           (B,)
target_total_value:               (B,)
trunk_energy:                     (B,)
logits:                           (B,)
```

## Why a Barrier Cut Rather Than a Generic Mechanism Probe

The thesis is structural: defenders fail because they cannot hold a
barrier between an attacking force and a valuable target. Modelling
that explicitly requires three things — an attacker mass field, a
defender barrier field, and a target value field — and a transport
that lets attack flow leak across squares while the defender field
absorbs it. The classifier reads `reachable_target_value` and
`defense_gap` so positions whose barrier is locally insufficient and
whose attack mass arrives at a valuable target score high — exactly
the regime the puzzle benchmark targets.

## Material Distinctness

This architecture is materially distinct from:

- The shared `ResearchPacketProbe` scaffold: no attack/defense/target
  fields, no barrier diffusion, no min-cut head.
- Sheaf / Hodge / Sinkhorn architectures (i010-i040 family): those
  compute global tension, curl, or transport statistics; this one
  runs an explicit iterative attack diffusion damped by a learnable
  defender barrier.
- `KingEscapePercolationNetwork` (i007): that model percolates the
  king out of attacker squares; this one percolates an attacker
  potential field into target squares through a defender barrier.

Removing the three-field encoder, the iterative barrier-cut
diffusion, or the reachable-target / defense-gap head would change
the model's computation in observable ways and is exactly what the
central ablations switch off.

## Central Ablations (config switches)

| Ablation         | Config knob              | Effect                                                                                  |
|------------------|--------------------------|-----------------------------------------------------------------------------------------|
| `narrow_trunk`   | `channels: 32`           | Halves the encoder latent width.                                                        |
| `shallow_trunk`  | `depth: 1`               | Single-conv trunk; tests how much depth the field encoder needs.                        |
| `wide_head`      | `hidden_dim: 192`        | Doubles the head width.                                                                 |
| `short_barrier`  | `barrier_steps: 1`       | One diffusion round; tests how much iteration the cut needs.                            |
| `long_barrier`   | `barrier_steps: 8`       | Eight diffusion rounds; flow can reach distant targets through weak barriers.           |
| `weak_barrier`   | `decay_scale: 0.25`      | Defender absorbs less attack mass per step; attack reaches further.                     |
| `strong_barrier` | `decay_scale: 4.0`       | Defender absorbs more attack mass per step; only severe attacks break through.          |
| `no_dropout`     | `dropout: 0.0`           | Removes regularization on encoder and head.                                             |
| `no_bn`          | `use_batchnorm: false`   | Replaces BN with GroupNorm(1, ...); useful for tiny batches.                            |

## Implementation Binding

- Registered model name: `barrier_cut_puzzle_network`
- Source implementation file:
  `src/chess_nn_playground/models/trunk/barrier_cut_puzzle_network.py`
- Idea-local wrapper:
  `ideas/registry/i198_barrier_cut_puzzle_network/model.py`

The wrapper is a thin adapter over
`build_barrier_cut_puzzle_network_from_config`; it does not touch
`ResearchPacketProbe`. The shared probe wrapper has been removed.
