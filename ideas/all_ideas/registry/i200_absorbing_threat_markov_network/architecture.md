# Architecture

`Absorbing Threat Markov Network` is a bespoke `puzzle_binary`
classifier that turns the absorbing-Markov-chain thesis from
`math_thesis.md` into an explicit differentiable computation. It is no
longer a `ResearchPacketProbe` wrapper.

## Thesis Recap

Puzzle detection is modelled as a probabilistic process over a small
set of named tactical states. The chain has six transient states
(`attack_pressure`, `defender_available`, `line_open`,
`king_constrained`, `target_hanging`, `counterplay`) and two absorbing
states (`proof_absorb`, `disproof_absorb`). A real puzzle should
concentrate probability mass into `proof_absorb` quickly; a
near-puzzle should leak mass into `disproof_absorb`.

## Inputs

- Board tensor only: `(B, 18, 8, 8)` simple_18 contract.
- CRTK / source / verification / engine metadata is reporting-only and
  never enters the model.

## Pipeline

1. **Compact convolutional trunk.** `feats = trunk(x)` runs `depth`
   `Conv2d(_, channels, 3, padding=1) -> Norm -> GELU -> Dropout2d`
   blocks (`Norm` is BatchNorm2d when `use_batchnorm = true`,
   GroupNorm(1, ...) otherwise). The trunk emits
   `(B, channels, 8, 8)`.
2. **Board context.** The trunk feature map is pooled by mean, max
   and energy over squares. The concatenation `(B, 3 * channels)` is
   projected by a linear layer to `(B, state_dim)`; this is the
   board-context vector that conditions both the initial distribution
   and the transition matrix. The pooled scalars `(mean, max, energy)`
   are also kept and fed to the head.
3. **Named tactical state tokens.** A learnable embedding table
   `state_embeddings ∈ R^(state_count, state_dim)` represents the
   tokens `attack_pressure`, `defender_available`, `line_open`,
   `king_constrained`, `target_hanging`, `counterplay`,
   `proof_absorb`, `disproof_absorb`. The last two rows are the
   absorbing states.
4. **Initial distribution `π_0` over transient states.** Each
   transient state token reads a board-conditioned mass via 1×1
   spatial attention `state_attention(feats)`; softmax over squares
   yields per-token attention weights, which produce a per-token
   summary scalar. A bilinear alignment between `board_context` and
   the transient state tokens adds a global scoring channel. Softmax
   over transient states gives `π_0 ∈ Δ^(transient_count)`; the
   absorbing entries of `π_0` are zero so the chain starts in a
   transient state.
5. **Board-conditioned transition matrix `P`.** Transient rows come
   from a board-modulated bilinear form on the state embeddings:

   ```text
   logits[b, i, j] = sum_d state_emb[i, d] * (board_proj[b, d] * state_emb[j, d]) + bias[i, j]
   ```

   Row softmax over `j` yields a row-stochastic distribution over all
   `state_count` states. The two absorbing rows are forced to identity
   (`P[proof_absorb, proof_absorb] = 1`,
   `P[disproof_absorb, disproof_absorb] = 1`) so probability mass
   cannot leak out. The full matrix is row-stochastic by construction.
6. **Power iteration.** With `transition_steps = T`,
   `π_t = π_{t-1} P` is iterated `T` times. The state-distribution
   tape `(π_0, π_1, …, π_T)` is retained for diagnostics.
7. **Absorption readout.** From `π_T` we read the absorption
   probabilities `prob_proof = π_T[proof_absorb]` and
   `prob_disproof = π_T[disproof_absorb]`. The soft expected
   pre-absorption step count is

   ```text
   E[steps] = sum_{t < T} (1 - π_t[proof_absorb] - π_t[disproof_absorb])
   ```

   which approaches the canonical `π_0 N 1` absorption time as `T`
   grows.
8. **Classifier head.** A `LayerNorm -> Linear(hidden_dim) -> GELU ->
   Dropout -> Linear(num_classes)` MLP reads a feature pack assembled
   from `π_T`, `[prob_proof, prob_disproof, prob_proof - prob_disproof]`,
   `expected_steps`, the transient initial distribution, and the
   pooled board summary `(mean, max, energy)`. It returns one puzzle
   logit. Mass concentrated in `proof_absorb` pushes the position
   toward the puzzle class; mass leaking into `disproof_absorb`
   pushes it toward non-puzzle.

## Tensor Contract

```text
input x:                        (B, 18, 8, 8)
trunk feats:                    (B, channels, 8, 8)
board_context:                  (B, state_dim)
state_embeddings:               (state_count, state_dim)
initial_distribution π_0:       (B, state_count)
transition_matrix P:            (B, state_count, state_count)
state_distributions:            (B, transition_steps + 1, state_count)
final_distribution π_T:         (B, state_count)
prob_proof:                     (B,)
prob_disproof:                  (B,)
proof_minus_disproof:           (B,)
expected_steps:                 (B,)
transient_initial:              (B, state_count - 2)
trunk_energy:                   (B,)
logits:                         (B,)
```

## Why an Absorbing Markov Chain Rather Than a Generic Mechanism Probe

The thesis is structural: a tactical position is closer to a puzzle if
its *flow of probability* under a learned tactical-process matrix
concentrates into a proof state quickly, and closer to a non-puzzle if
that flow leaks toward a disproof state. Modelling this requires three
specific objects — named tactical states, a row-stochastic
board-conditioned transition matrix with absorbing proof/disproof
rows, and an iteration that exposes absorption probabilities and
expected steps. The shared `ResearchPacketProbe` exposes none of
these; it cannot produce `prob_proof`, `prob_disproof`, or
`expected_steps` because it never builds a row-stochastic transition
matrix.

## Material Distinctness

This architecture is materially distinct from:

- The shared `ResearchPacketProbe` scaffold: no state tokens, no
  row-stochastic transition matrix, no absorbing-state masking, no
  power iteration over a Markov chain.
- `i007_neural_proof_number_search`: that model runs a learned
  proof-number search over an explicit AND/OR move tree; this one
  works on a small fixed set of named *tactical* states with a
  learned transition matrix and never enumerates moves.
- `response_minimax_classifier` and tactical-program-induction
  variants: those reason over move/reply tokens or program slots;
  this one reasons over abstract tactical states and absorption
  dynamics.

Removing the absorbing-state structure (`no_absorbing_states`),
forcing a symmetric transition kernel (`symmetric_transition`), or
collapsing the chain to a single step (`one_step_only`) eliminates the
features the head depends on, which is exactly what the markdown's
falsification table requires.

## Central Ablations (config switches)

| Ablation               | Config knob                   | Effect                                                                                                |
|------------------------|-------------------------------|-------------------------------------------------------------------------------------------------------|
| `narrow_trunk`         | `channels: 32`                | Halves the encoder latent width.                                                                      |
| `shallow_trunk`        | `depth: 1`                    | Single-conv trunk; tests how much depth the state activations need.                                   |
| `wide_head`            | `hidden_dim: 192`             | Doubles the head width.                                                                               |
| `tiny_state_dim`       | `state_dim: 32`               | Shrinks the state-token / board-projection latent.                                                    |
| `large_state_dim`      | `state_dim: 192`              | Widens the state-token / board-projection latent.                                                     |
| `few_states`           | `state_count: 4`              | Two transient states + two absorbing states; tests how much state granularity is needed.              |
| `extra_states`         | `state_count: 12`             | Adds extra unnamed transient states beyond the six named ones.                                        |
| `one_step_only`        | `transition_steps: 1`         | Tests whether iterative absorption matters versus a one-step kernel.                                  |
| `deeper_chain`         | `transition_steps: 8`         | Lets the chain converge closer to the analytic absorption probabilities.                              |
| `no_dropout`           | `dropout: 0.0`                | Removes regularization on encoder and head.                                                           |
| `no_bn`                | `use_batchnorm: false`        | Replaces BN with GroupNorm(1, ...); useful for tiny batches.                                          |

## Implementation Binding

- Registered model name: `absorbing_threat_markov_network`
- Source implementation file:
  `src/chess_nn_playground/models/absorbing_threat_markov_network.py`
- Idea-local wrapper:
  `ideas/all_ideas/registry/i200_absorbing_threat_markov_network/model.py`

The wrapper is a thin adapter over
`build_absorbing_threat_markov_network_from_config`; it does not touch
`ResearchPacketProbe`. The shared probe wrapper has been removed.
