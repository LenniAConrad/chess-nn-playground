# Architecture

`Canonical-Orbit Straight-Through Operator` (p036) is an additive,
gated head over the existing i193 `ExchangeThenKingDualStreamNetwork`
trunk. The thesis (see `math_thesis.md`) is that the i193 trunk has to
relearn equivalent chess positions across the board-geometry symmetry
orbit; p036 quotients that orbit explicitly inside the model.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit, plus a per-sample
diagnostic dict mirroring the i248 contract.

## Mechanism

1. **i193 trunk forward**. The bespoke
   `ExchangeThenKingDualStreamNetwork` runs unchanged and emits the
   full diagnostic dict including `logits` (`base_logit`) and the
   dual-stream joint feature.

2. **Latent projection**. A single linear layer maps the joint feature
   `joint in R^{B, feature_dim}` into a per-square latent map
   `X in R^{B, 64, latent_dim}`.

3. **Group action**. The fixed C2 x C2 board-geometry group `G` is
   represented as a 4 x 64 permutation table. For each batch element we
   compute the four transformed copies `T_g X in R^{B, 4, 64, d}` by a
   single batched `gather`.

4. **Hash keys**. A fixed Gaussian projection (seeded
   `0xC0DEC0DE`) and quantised square-weighted reduction produce
   `keys in R^{B, 4, key_dim}` *under `torch.no_grad()`*. The argmin is
   non-differentiable by construction.

5. **Lexicographic argmin**. A small sequential pass over the four
   orbit elements selects the canonical index `chosen in {0,1,2,3}^B`
   by lexicographic order over the key columns, with deterministic
   tie-break favouring the lower group index (preferring `e`).

6. **Canonical representative**. `canonical[b] = T_{chosen[b]} X[b]`
   via a gather. Because the chosen index is computed in `no_grad`, the
   gradient with respect to `X` flows through the permutation only,
   i.e. through `T_{g^*}^{-1} = T_{g^*}` (each element of C2 x C2 is its
   own inverse).

7. **Readout**. Pool `canonical` (mean over squares) and the symmetry
   residual `residual = canonical - X` (per-channel RMS). Concatenate
   to a `(B, 2d)` readout vector and run through a LayerNorm + Linear +
   GELU + Dropout + Linear stack that emits `primitive_delta_raw`.

8. **Gate**. A second LayerNorm + Linear + GELU + Linear stack on the
   trunk joint feature produces `gate_logit`. The effective gate is
   `sigmoid(gate_logit)`, initialised near zero by `gate_init = -2.0`.

9. **Logit fusion**. `final_logit = base_logit + gate * delta_raw`.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shuffle_canonical` | In-batch permutation of the canonical representative. Primary falsifier: if A1 matches `none`, the canonical representative carries no signal beyond noise. |
| A2 | `identity_only` | Force `chosen = e`. Tests whether the orbit search is load-bearing versus a trivial "no-op" branch. |
| A3 | `fixed_choice` | Force `chosen = F` (file mirror). Tests whether the orbit choice needs to depend on the input. |
| A4 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A5 | `trunk_only` | Strongest control: zero delta and disable the head. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed by the model. The simple_18 board tensor is the only
external input; the canonical representative is computed in latent
space.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk forward | One pass through the dual-stream encoder |
| Latent projection | `O(feature_dim * 64 * latent_dim)` |
| Group gather | `O(4 * 64 * latent_dim)` |
| Hash keys (no_grad) | `O(4 * 64 * latent_dim * key_dim)` |
| Argmin | `O(4 * key_dim)` |
| Readout | `O(2 * latent_dim + head_hidden_dim)` |
| Head MLPs | Two small LayerNorm + GELU MLPs |

The per-sample primitive cost is dominated by the latent projection,
`O(feature_dim * 64 * latent_dim)`, which is comparable to one extra
linear layer of trunk width.

## Implementation Binding

- Registered model name: `canonical_orbit_st_operator`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/canonical_orbit_st_operator.py`.
- Idea-local wrapper:
  `ideas/registry/p036_canonical_orbit_st_operator/model.py`.
- Training config:
  `ideas/registry/p036_canonical_orbit_st_operator/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.
