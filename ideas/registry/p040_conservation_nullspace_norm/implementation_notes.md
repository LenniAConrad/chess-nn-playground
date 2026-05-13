# Implementation Notes

## Worktree

Implemented in the `cnp-primitive-gpt-orbit-boolean-algebra` worktree.

## Module

`src/chess_nn_playground/models/primitives/conservation_nullspace_norm.py`

Contains:

- `ConservationNullspaceNorm` nn.Module.
- `build_conservation_nullspace_norm_from_config`.
- `_build_charge_matrix()` -- the fixed `64 x 8` charge matrix.
- `ALLOWED_ABLATIONS` tuple.

## Rule-derived feature note

The per-square weights are computed from the simple_18 board tensor by
a 1x1 convolution + softplus, which makes them strictly positive and
rule-derived (the conv sees only the simple_18 channels, no CRTK
metadata). This satisfies rule 5 of the primitive batch ("rule-derived
chess features are allowed only when computed from legal board / FEN
state and explicitly documented"). The conv weights are *learned* --
the operator does not hard-code which board planes drive the weights
-- but the inputs are strictly board-derived.

## Why Cholesky for the SPD inverse

`A = C^T D C + epsilon I` is SPD by construction (`D > 0`, `epsilon > 0`,
`C^T D C >= 0`). `torch.linalg.cholesky` + `torch.cholesky_solve` gives
implicit gradients without us hand-rolling `dA^{-1} = -A^{-1} dA A^{-1}`.

## Stability

The `epsilon` regulariser (default `1e-3`) prevents `C^T D C` from being
ill-conditioned when the weight projection collapses. The
`max(1, sum_w - r)` denominator in the variance avoids divide-by-zero
when the active weight mass is small. The normalised residual is
clamped through `+ epsilon` inside the sqrt to avoid NaN gradients at
zero residual.

## Deferred internal proposals

The source packet contains four other proposals which were *not*
promoted in this batch:

| Proposal | Reason for deferral |
|---|---|
| `primitive_espa` (rank 1) | Duplicate of `p024_event_symmetric_interaction_accumulator`. |
| `primitive_isotypic_projector` (rank 3) | Finite-group isotypic decomposition; partial overlap with p036 (canonical-orbit operator) for the same C2 x C2 chess board-geometry group; deferred. |
| `primitive_green_solve` (rank 4) | Dense `O(N^3)` Green-function solve on the board graph; too heavy for scout scale without a sparse solver. |
| `primitive_matroid_base_pool` (rank 5) | Entropic matroid-base pool; overlaps with the matroid-rank envelope (external_31) and would need a separate matroid oracle infrastructure. |

## Input contract

- Input: `simple_18` current-board tensor, shape `(B, 18, 8, 8)`.
- Output: `dict` with `logits` of shape `(B,)` plus the diagnostics
  listed in `architecture.md` and the i193 trunk diagnostics (`trunk_*`).
- The model rejects non-simple_18 inputs and non-1 `num_classes`,
  invalid `latent_dim`, `epsilon`, or `ablation`.

## Trainer compatibility

Same `(B, 18, 8, 8)` contract as i193. Returns `dict["logits"]` shape
`(B,)`. The trainer reads only `dict["logits"]` for loss.
