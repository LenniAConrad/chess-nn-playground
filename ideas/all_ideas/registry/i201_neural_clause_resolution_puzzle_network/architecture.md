# Architecture

`Neural Clause-Resolution Puzzle Network` is a bespoke `puzzle_binary`
classifier that turns the typed-clause-proof thesis from
`math_thesis.md` into an explicit differentiable pipeline. It is no
longer a `ResearchPacketProbe` wrapper.

## Thesis Recap

Puzzlehood is treated as the existence of a short typed proof: a
small set of typed facts (`Attack`, `Defends`, `Pinned`, `LineOpen`,
`EscapeSquare`, `Tempo`) that compose, through soft Horn clauses with
shared variables, into a `PuzzleWitness` head predicate. A real
puzzle should admit such a derivation; near puzzles should fail at
either the conjunction or the variable-binding step.

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
2. **Initial fact base.** A `1×1` convolution gives a per-square truth
   value `F_0[b, p, s] ∈ [0, 1]` for each of `num_unary_predicates`
   typed predicates (default `attack`, `defends`, `pinned`,
   `line_open`, `escape_square`). A linear readout of the pooled trunk
   summary `(mean, max, energy)` produces a board-level truth value
   `G_0[b, g] ∈ [0, 1]` for each of `num_global_predicates` global
   predicates (default `tempo`).
3. **Predicate embeddings.** A learnable table
   `predicate_embeddings ∈ R^{P × predicate_dim}` represents both the
   unary and the global predicate types; the first `P_u` rows are
   unary, the last `P_g` rows are global. Soft predicate selectors are
   computed by softmax of a query–embedding inner product.
4. **Clause queries.** Each clause `c` carries a head query
   `head_query[c] ∈ R^{predicate_dim}` and `body_arity` body queries
   `body_query[c, k] ∈ R^{predicate_dim}`. The induced selectors are

   ```text
   head_sel[c, p]    = softmax_p(head_query[c] · predicate_embeddings[p])
   body_sel[c, k, p] = softmax_p(body_query[c, k] · predicate_embeddings[p])
   ```

   Each body slot also carries a soft mixture
   `body_rel[c, k, r] = softmax_r(clause_body_relation_logits[c, k, r])`
   over `relation_count` learned spatial relation kernels.
5. **Spatial relations and variable binding.** A bank
   `relations[r, s, s'] = softmax_{s'} relation_logits[r, s, s']`
   provides `R` row-stochastic kernels over square pairs. Composing
   `body_rel` and `relations` implements differentiable variable
   unification: a body predicate evaluated at the head's square `s`
   reads truth values at related squares `s'`. With
   `relation_count = 1` and a uniform kernel the binding collapses to
   the `no_variable_binding` ablation.
6. **Differentiable resolution rounds.** `resolution_rounds = K`
   iterations apply a soft Horn rule to every clause:

   ```text
   F_rel[b, r, p, s]    = sum_{s'} relations[r, s, s'] * F_unary[b, p, s']
   rel_mixed[b, c, k, p, s] = sum_r body_rel[c, k, r] * F_rel[b, r, p, s]
   body_score[b, c, k, s] = sum_{p < P_u} body_sel[c, k, p] * rel_mixed[b, c, k, p, s]
                          + sum_g        body_sel[c, k, P_u + g] * G[b, g]
   clause_activation[b, c, s] = sum_k log(body_score[b, c, k, s] + eps)
                              + clause_bias[c]
   clause_truth[b, c, s]      = sigmoid(clause_activation[b, c, s])
   ```

   The fact base is updated by a residual probabilistic-OR with a
   per-predicate gate `gate ∈ [0, 1]`:

   ```text
   delta_unary[b, p, s] = sum_c head_sel[c, p<P_u] * clause_truth[b, c, s]
   F_{t+1}[b, p, s]     = F_t[b, p, s] + (1 - F_t[b, p, s]) * gate[p] * delta_unary[b, p, s]
   ```

   with the analogous update for `G`. The trajectories
   `(F_0, …, F_K)` and `(G_0, …, G_K)` are retained for diagnostics,
   along with the per-round `clause_activations` tape.
7. **Classifier head.** A
   `LayerNorm -> Linear(hidden_dim) -> GELU -> Dropout -> Linear(num_classes)`
   MLP reads a feature pack assembled from the pooled final fact base
   `(mean, max)`, the final global facts, the pooled trunk summary
   `(mean, max, energy)` and a final-round clause-activation summary
   `(mean, max)`. It returns one puzzle logit. Coherent firing of a
   small set of clauses pushes the position toward the puzzle class;
   diffuse or contradictory firing pushes it toward non-puzzle.

## Tensor Contract

```text
input x:                     (B, 18, 8, 8)
trunk feats:                 (B, channels, 8, 8)
initial unary facts F_0:     (B, P_u, S)
initial global facts G_0:    (B, P_g)
relations:                   (R, S, S)        # row-stochastic
clause head selector:        (C, P)
clause body selector:        (C, A, P)
clause body relation:        (C, A, R)
unary fact trajectory:       (B, K + 1, P_u, S)
global fact trajectory:      (B, K + 1, P_g)
clause activations tape:     (B, K, C, S)
final unary facts F_K:       (B, P_u, S)
final global facts G_K:      (B, P_g)
predicate embeddings:        (P, predicate_dim)
trunk_energy:                (B,)
logits:                      (B,)
```

with `P_u = num_unary_predicates`, `P_g = num_global_predicates`,
`P = P_u + P_g`, `S = 64`, `C = clause_count`, `A = body_arity`,
`R = relation_count`, `K = resolution_rounds`.

## Why Soft Clause Resolution Rather Than a Generic Mechanism Probe

The thesis is structural: a tactical position is closer to a puzzle if
a small typed conjunction binds together through shared variables.
Modelling this requires three specific objects — typed predicates with
explicit head/body roles, soft variable unification through spatial
relation kernels, and an iteration that lets new facts derive from
old ones. The shared `ResearchPacketProbe` exposes none of these; it
cannot produce `clause_activations` or a per-clause head/body
predicate distribution because it never builds a clause-resolution
layer.

## Material Distinctness

This architecture is materially distinct from:

- The shared `ResearchPacketProbe` scaffold: no typed predicate
  embeddings, no head/body clause queries, no row-stochastic relation
  kernels, no soft Horn-clause resolution layer, no iterated fact
  trajectory.
- `tropical_constraint_circuit_network`: tropical clauses are fixed
  min/max constraints with no learned predicate types or shared
  variables; this network learns predicate types and runs soft
  unification over square variables.
- `tactical_program_induction_network`: program induction reasons over
  ordered program slots; this network reasons over a fixed predicate
  vocabulary with soft head/body roles.
- `differentiable_chess_fact_lattice`: the fact lattice composes a
  fixed taxonomy of facts; this network learns clause heads and body
  selectors and composes them through differentiable variable
  unification.

## Central Ablations (config switches)

| Ablation                  | Config knob                | Effect                                                                                                  |
|---------------------------|----------------------------|---------------------------------------------------------------------------------------------------------|
| `bag_of_facts`            | `resolution_rounds: 1`     | Single resolution step; tests whether iterated derivation matters versus a one-shot clause readout.     |
| `no_variable_binding`     | `relation_count: 1`        | Single relation kernel collapses spatial unification; tests whether learned variable binding matters.  |
| `one_round_only`          | `resolution_rounds: 1`     | Same as `bag_of_facts` from the markdown's perspective; isolates the multi-step derivation effect.     |
| `narrow_clauses`          | `clause_count: 8`          | Few clauses; tests whether the head needs a rich clause library.                                       |
| `tiny_predicate_dim`      | `predicate_dim: 16`        | Shrinks the predicate-embedding latent.                                                                 |
| `wide_predicate_dim`      | `predicate_dim: 128`       | Widens the predicate-embedding latent.                                                                  |
| `single_body_slot`        | `body_arity: 1`            | Removes the conjunction across body slots; tests the soft-AND structure.                                |
| `narrow_trunk`            | `channels: 32`             | Halves the encoder latent width.                                                                        |
| `shallow_trunk`           | `depth: 1`                 | Single-conv trunk; tests how much depth the fact channels need.                                         |
| `no_dropout`              | `dropout: 0.0`             | Removes regularization on encoder and head.                                                             |
| `no_bn`                   | `use_batchnorm: false`     | Replaces BN with GroupNorm(1, ...).                                                                     |

## Implementation Binding

- Registered model name: `neural_clause_resolution_puzzle_network`
- Source implementation file:
  `src/chess_nn_playground/models/neural_clause_resolution_puzzle_network.py`
- Idea-local wrapper:
  `ideas/all_ideas/registry/i201_neural_clause_resolution_puzzle_network/model.py`

The wrapper is a thin adapter over
`build_neural_clause_resolution_puzzle_network_from_config`; it does
not touch `ResearchPacketProbe`. The shared probe wrapper has been
removed.
