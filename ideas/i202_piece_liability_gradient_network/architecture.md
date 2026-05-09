# Architecture

`Piece Liability Gradient Network` is a bespoke `puzzle_binary`
classifier that turns the piece-liability thesis from `math_thesis.md`
into an explicit differentiable pipeline. It is no longer a
`ResearchPacketProbe` wrapper.

## Thesis Recap

In many puzzles, one piece is not merely attacked; it is *liable* --
it cannot move, defend, capture, or stay without losing something.
Near-puzzles may attack pieces, but the liability does not propagate.
This network detects puzzles by (i) scoring how good each affordance
(`move`, `defend`, `capture`, `stay`) is for every piece on the board,
(ii) collapsing those scores into a per-square liability field, and
(iii) propagating liability through learned spatial relations so that
"my defender is liable" can lift a defended piece's liability too.

## Inputs

- Board tensor only: `(B, 18, 8, 8)` simple_18 contract.
- The first `num_piece_planes = 12` planes are the piece bitboards
  (`P, N, B, R, Q, K, p, n, b, r, q, k`). Their per-square sum forms a
  binary `piece_mask` so liability is restricted to occupied squares.
- CRTK / source / verification / engine metadata is reporting-only and
  never enters the model.

## Pipeline

1. **Compact convolutional trunk.** `feats = trunk(x)` runs `depth`
   `Conv2d(_, channels, 3, padding=1) -> Norm -> GELU -> Dropout2d`
   blocks (`Norm` is BatchNorm2d when `use_batchnorm = true`,
   GroupNorm(1, ...) otherwise). The trunk emits
   `(B, channels, 8, 8)`.
2. **Piece presence mask.** `piece_mask[b, s] = clip(sum_{p<12} x[b, p, h, w], 0, 1)`
   gives a per-square indicator of an occupied square. Liability lives
   only on occupied squares.
3. **Action affordances.** A `1x1` convolution produces an
   `A`-channel tensor `affordance[b, a, s] in R` of action values for
   each affordance type at each square. Default `num_affordances = 4`
   follows the thesis: `move`, `defend`, `capture`, `stay`. Higher is
   better for the piece sitting on the square.
4. **Initial liability.** A piece is liable when *every* affordance is
   bad. A soft minimum across affordances drives the score:

   ```text
   soft_min[b, s] = -tau * logsumexp(-affordance[b, :, s] / tau)
   L_0[b, s]      = sigmoid(-soft_min[b, s] / lambda) * piece_mask[b, s]
   ```

   so `L_0[b, s] in [0, 1]`, near 1 when even the best affordance is
   bad and the square is occupied. Crucially, a piece can be attacked
   (poor `capture` and `stay` scores) yet still have a good `move` or
   `defend` value -- the soft minimum stays low, matching the
   "near-puzzles attack but liability does not propagate" requirement.
5. **Liability propagation rounds.** `propagation_rounds = K`
   iterations propagate liability through learned spatial relation
   kernels. `relation_count = R` row-stochastic kernels
   `relations[r, s, s'] = softmax_{s'} relation_logits[r, s, s']`
   model "if my defender is liable, so am I" / "if my retreat
   square's defender is liable, so is the retreat" propagators. A
   per-round, per-relation gate `gate[t, r] in [0, 1]` controls how
   much liability flows along each relation each round:

   ```text
   L_propagated[b, r, s] = sum_{s'} relations[r, s, s'] * L_t[b, s']
   delta[b, s]           = sum_r gate[t, r] * L_propagated[b, r, s]
   L_{t+1}[b, s]         = L_t[b, s] + (1 - L_t[b, s]) * delta[b, s] * piece_mask[b, s]
   ```

   The probabilistic-OR update keeps every liability score bounded in
   `[0, 1]` and never lets liability grow on empty squares.
6. **Liability gradient and aggregate features.** The final liability
   field `L_K` is summarised by `max_liability`, `mean_liability`
   (over occupied squares), `top_k_liability` (mean of the top
   `liability_top_k` values), and the propagation magnitude
   `liability_gradient = L_K - L_0`. The pooled trunk summary
   `(mean, max, energy)` is concatenated to give the head input.
7. **Classifier head.** A
   `LayerNorm -> Linear(hidden_dim) -> GELU -> Dropout -> Linear(num_classes)`
   MLP returns one puzzle logit. Strong propagation amplifies a
   liability into the puzzle class; mere local attacks without
   propagation stay near the non-puzzle side.

## Tensor Contract

```text
input x:                     (B, 18, 8, 8)
trunk feats:                 (B, channels, 8, 8)
piece_mask:                  (B, S)
action_affordances:          (B, A, S)
initial_liability L_0:       (B, S)
relation_kernels:            (R, S, S)            # row-stochastic
propagation_gates:           (K, R)               # in [0, 1]
liability_trajectory:        (B, K + 1, S)
final_liability L_K:         (B, S)
liability_gradient:          (B, S)               # L_K - L_0
trunk_energy:                (B,)
logits:                      (B,)
```

with `S = 64`, `A = num_affordances`, `R = relation_count`,
`K = propagation_rounds`.

## Why a Liability Gradient Rather Than a Generic Mechanism Probe

The thesis is structural: the puzzle signal is that *one* piece's
inability to escape *propagates* into the position. Modelling this
needs three specific objects -- per-square action-affordance
decomposition, a soft-min collapse that distinguishes "attacked but
escapable" from "no escape", and an iterated propagation that turns a
local liability into a global one. The shared `ResearchPacketProbe`
exposes none of these; it cannot return `liability_trajectory` or
`relation_kernels` because it never builds an action-affordance head
or a propagation layer.

## Material Distinctness

This architecture is materially distinct from:

- The shared `ResearchPacketProbe` scaffold: no action-affordance
  head, no piece-presence mask, no soft-minimum liability collapse,
  no row-stochastic propagation kernels, no iterated liability
  trajectory.
- `tactical_threat_sheaf_network` and friends: sheaf models score
  threats per square but never decompose into the four affordances
  and never iterate "liability of my defender flows back to me"
  through a learned, gated propagation kernel.
- `neural_clause_resolution_puzzle_network` (i201): clause resolution
  reasons over typed predicates and shared variables; this network
  has no predicate vocabulary, no clause heads, and reasons directly
  over per-square action values plus a liability propagation field.
- `pinned_mobility_nullspace_network`: pinned-mobility models look
  for null-space mobility patterns but do not perform iterative
  propagation of a per-square liability value.

## Central Ablations (config switches)

| Ablation                    | Config knob                     | Effect                                                                                                  |
|-----------------------------|---------------------------------|---------------------------------------------------------------------------------------------------------|
| `no_propagation`            | `relation_count: 1`             | Single relation kernel collapses propagation; tests whether learned liability flow matters.            |
| `one_round_only`            | `propagation_rounds: 1`         | Single propagation step; isolates the multi-round liability-gradient effect.                           |
| `no_action_decomposition`   | `num_affordances: 1`            | Collapses move/defend/capture/stay into one channel; tests the affordance decomposition.               |
| `narrow_trunk`              | `channels: 32`                  | Halves the encoder latent width.                                                                        |
| `shallow_trunk`             | `depth: 1`                      | Single-conv trunk; tests how much depth the affordance head needs.                                     |
| `tiny_top_k`                | `liability_top_k: 1`            | Only the single most liable square contributes to the head feature.                                    |
| `wide_top_k`                | `liability_top_k: 16`           | Aggregates many liability sites; tests whether one liable piece is enough.                             |
| `cool_affordance`           | `affordance_temperature: 0.25`  | Sharper soft-min (closer to hard-min); tests sensitivity to the soft-min temperature.                  |
| `warm_affordance`           | `affordance_temperature: 4.0`   | Softer soft-min (closer to mean); tests sensitivity to the soft-min temperature.                       |
| `no_dropout`                | `dropout: 0.0`                  | Removes regularization on encoder and head.                                                             |
| `no_bn`                     | `use_batchnorm: false`          | Replaces BN with GroupNorm(1, ...).                                                                     |

## Implementation Binding

- Registered model name: `piece_liability_gradient_network`
- Source implementation file:
  `src/chess_nn_playground/models/piece_liability_gradient_network.py`
- Idea-local wrapper:
  `ideas/i202_piece_liability_gradient_network/model.py`

The wrapper is a thin adapter over
`build_piece_liability_gradient_network_from_config`; it does not
touch `ResearchPacketProbe`. The shared probe wrapper has been
removed.
