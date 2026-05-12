# Architecture

`Neural Board Cellular Automaton` is a bespoke recurrent local-update
architecture: one shared 3x3 update rule `f` is applied to a cell-state
board representation for several discrete time steps, and the puzzle
classifier reads from both the evolving board state and the per-step
update energies.

## Inputs

- Board tensor only: `(B, 18, 8, 8)` simple_18 contract.
- CRTK / source / engine metadata is reporting-only and never enters
  the model.

## Pipeline

1. **Lift to cell state.** A 1x1 convolution maps the 18 input planes
   to `channels` cell-state planes. There is no spatial mixing in this
   step, so each square's initial cell state is a learned linear
   combination of its own piece/state encoding.
2. **Shared local update rule `f`.** A single small convnet, owned by
   one `_LocalUpdateRule` module, is reused at every CA step:
   1. `BatchNorm2d` (or `GroupNorm` if BN is disabled).
   2. `depth` repetitions of `Conv2d(_, hidden_dim, 3, padding=1)` then
      `GELU` (and optional `Dropout2d`). The 3x3 kernel is what makes
      the rule **local**: each update only sees its 3x3
      Moore-neighborhood.
   3. A final `Conv2d(hidden_dim, channels, 1)` projects back into the
      cell-state space. Its weights and bias are zero-initialized so an
      untrained network produces `f(h) = 0` and the CA dynamics start
      out as a stable identity fixed point.
3. **Iterated relaxation.** The forward pass runs `steps` cellular-
   automaton iterations of the residual update
   `h_{t+1} = h_t + step_size * f(h_t)`, with a learnable, sigmoid-
   bounded scalar `step_size = sigmoid(theta) * max_step_size` shared
   across all cells and steps. **Crucially, `f` is the same module at
   every step**: the weights are tied across `t = 0..steps-1`, which is
   what distinguishes the CA from a stack of independent residual
   blocks.
4. **Energy trajectory.** At each step the model records the per-sample
   update energy `||delta_t||^2 / N` and state energy `||h_t||^2 / N`
   (where `N = channels * 8 * 8`). These trajectories carry the
   relaxation signal: easy positions damp out quickly while
   tactically-rich positions sustain non-trivial update energy.
5. **Classifier head.** The final state `h_T` is mean-pooled spatially
   to give `(B, channels)`. The energy summary
   `[update_mean, update_sum, update_last, state_mean, state_last]`
   is concatenated with the pooled state and passed through
   `LayerNorm -> Linear(hidden_dim) -> GELU -> Dropout -> Linear(1)`
   to produce one puzzle logit. Per-step trajectories, the final
   spatial state, the pooled features, and the realized step size are
   returned as diagnostics.

## Tensor Contract

```
input:                       (B, 18, 8, 8)
embedded cell state:         (B, channels, 8, 8)
update_energy_per_step:      (B, steps)
state_energy_per_step:       (B, steps + 1)   # includes h_0
final_state:                 (B, channels, 8, 8)
pooled_features:             (B, channels)
update_energy:               (B,)
update_energy_mean:          (B,)
final_step_update_energy:    (B,)
final_state_energy:          (B,)
step_size:                   (B,)             # broadcast scalar
logits:                      (B,)
```

## Central Ablations (config switches)

| Ablation         | Config knob              | Effect                                                                              |
|------------------|--------------------------|-------------------------------------------------------------------------------------|
| `single_step`    | `steps: 1`               | Disables iterative relaxation. Tests whether tying `f` across multiple steps helps. |
| `more_steps`     | `steps: 12`              | Doubles the relaxation horizon to test depth-via-time vs static depth.              |
| `narrow_rule`    | `hidden_dim: 48`         | Halves the local rule's hidden width.                                               |
| `bigger_state`   | `channels: 96`           | Wider cell state per square.                                                        |
| `frozen_step`    | `max_step_size: 0.5`     | Tightens the per-step relaxation magnitude.                                         |

## Implementation Binding

- Registered model name: `neural_board_cellular_automaton`
- Source implementation file:
  `src/chess_nn_playground/models/neural_board_cellular_automaton.py`
- Idea-local wrapper:
  `ideas/registry/i115_neural_board_cellular_automaton/model.py`

The wrapper is a thin adapter over
`build_neural_board_cellular_automaton_from_config`; it does not touch
`ResearchPacketProbe`. The shared probe wrapper has been removed.
