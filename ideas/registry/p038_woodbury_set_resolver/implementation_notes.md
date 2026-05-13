# Implementation Notes

## Worktree

Implemented in the `cnp-primitive-gpt-orbit-boolean-algebra` worktree
alongside p036, p037, p039-p041 of the 2026-05-13 GPT primitive batch.

## Module

`src/chess_nn_playground/models/primitives/woodbury_set_resolver.py`

Contains:

- `WoodburySetResolver` nn.Module.
- `build_woodbury_set_resolver_from_config`.
- `_piece_tokens` helper for the piece-planes-plus-STM-plus-castling
  17-channel per-square token.
- `ALLOWED_ABLATIONS` tuple.

## Why Cholesky and not torch.linalg.solve

`A = lambda I + sum_i U_i U_i^T` is SPD by construction, so the
Cholesky factorisation exists and gives `O(r^3 / 3)` complexity. We
also need *two* solves (`P S` and the per-token leverage), so the
factorisation amortises across them. `torch.cholesky_solve` has a
defined backward that uses implicit differentiation of the SPD inverse,
so the gradient passes through correctly without us hand-rolling
`dA^{-1} = -A^{-1} dA A^{-1}`.

## Stability

The Tikhonov regulariser `lambda` (default `1e-2`) keeps `A` strictly
positive-definite even when the active piece set is small or collinear.
The packet's spec recommends `lambda >= 1e-3`; the default is one order
of magnitude larger for scout-scale robustness. `r` is bounded
practically by the `r^3` cost; default `r = 12`, and the cost is well
within the i193 trunk's per-sample budget.

## Token features

Per-square token = piece planes (12) + side-to-move (1) + castling
rights (4) = 17 channels. The construction is rule-exact -- nothing
about the FEN beyond what the simple_18 encoding already exposes is
consumed. CRTK metadata, source labels, verification flags, and engine
evaluations are not used.

## Deferred internal proposals

The source packet (`external_33_esp_permanent_woodbury_orbit_primitives.md`)
contains four other proposals which were *not* promoted in this batch:

| Proposal | Reason for deferral |
|---|---|
| `primitive_esp_set` (rank 1) | Duplicate of `p024_event_symmetric_interaction_accumulator`. |
| `primitive_permanent_roles` (rank 2) | Exchangeable permanent role assignment via DP; overlaps with the existing `permanent_ryser_network` trunk; needs a comparison before promotion. |
| `primitive_orbit_canonicalizer` (rank 4) | Duplicate of p036 in this batch. |
| `primitive_component_pool` (rank 5) | Connected-component pool over a dynamic mask; needs union-find / dynamic connectivity infrastructure; deferred. |

The rank-2 proposal in
`external_36_exterior_product_rank1_resolvent_primitives.md`
(``primitive_rank1_resolvent_pool``) is the same operator as this
implementation, so file 36 is covered by p038 *and* p041 (Truncated
Exterior Product Pool).

## Input contract

- Input: `simple_18` current-board tensor, shape `(B, 18, 8, 8)`.
- Output: `dict` with `logits` of shape `(B,)` plus the diagnostics
  listed in `architecture.md` and the i193 trunk diagnostics (`trunk_*`).
- The model rejects non-simple_18 inputs and non-1 `num_classes`,
  invalid `u_dim`, `v_dim`, `num_queries`, `lambda_reg`, or `ablation`.

## Trainer compatibility

Same `(B, 18, 8, 8)` contract as i193. Returns `dict["logits"]` shape
`(B,)`. The trainer reads only `dict["logits"]` for loss; diagnostics
are slice-report only.
