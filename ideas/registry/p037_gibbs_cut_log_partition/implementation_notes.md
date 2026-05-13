# Implementation Notes

## Worktree

This idea was implemented in the
`cnp-primitive-gpt-orbit-boolean-algebra` worktree alongside p036 and
p038-p041 of the 2026-05-13 GPT primitive batch.

## Module

`src/chess_nn_playground/models/primitives/gibbs_cut_log_partition.py`

Contains:

- `GibbsCutLogPartition` nn.Module.
- `build_gibbs_cut_log_partition_from_config`.
- `_build_state_bits`, `_build_within_row_xor`, `_build_between_row_xor`
  -- static state and XOR tables built once at construction time.
- `ALLOWED_ABLATIONS` tuple with the five named ablations.

## Why the grid is downsampled to 4x4

The chess board is 8x8 (`W = 8`, `2^W = 256` states), giving a
`256 x 256` transition matrix per row. The exact computation cost is
manageable (`8 * 256^2 * d_cut`) but slows training noticeably.
Projecting the trunk joint feature into a smaller latent grid keeps
the operator demonstrable on RTX 3070 within a scout-scale budget. The
hyperparameters `grid_h` and `grid_w` can be increased later if a
larger grid is justified by the falsifier.

## Numerical stability

`torch.logsumexp` is used throughout the row recurrence so the
log-partition stays in log space; no `exp`/`log` cycles. The
`softplus` projection of edge costs and source/sink penalties keeps
inputs non-negative without saturating gradients (unlike `exp`).

## Edge marginals

The packet's spec calls for the cut-edge marginals
`m_e = dy/dc_e` to be exposed. We surface a *summary* diagnostic
(`gibbs_cut_edge_energy`) -- the per-channel mean of the input edge
costs -- but do not materialise the full marginal tensor. The true
marginals are still available to the trainer through autograd if a
slice report ever needs them, because `y` is differentiable in `c_h`
and `c_v`.

## Deferred internal proposals

The source packet (`external_32_elementary_symmetric_gibbs_hodge_primitives.md`)
contains four other proposals which were *not* promoted in this batch:

| Proposal | Reason for deferral |
|---|---|
| `primitive_elem_sym_event` (rank 1) | Duplicate of `p024_event_symmetric_interaction_accumulator`, which already implements the elementary-symmetric polynomial recurrence over piece tokens. |
| `primitive_complementarity_contact` (rank 3) | Differentiable LCP / Fischer-Burmeister overlap with the existing `dykstra_lcp` family; needs an explicit comparison before promotion. |
| `primitive_hodge_cochain_projector` (rank 4) | Edge-cochain Hodge decomposition; substantial cell-complex / Laplacian infrastructure; deferred. |
| `primitive_signed_persistence` (rank 5) | Differentiable persistent homology; deferred as a topology layer (`torchph`-style libraries already exist). |

## Input contract

- Input: `simple_18` current-board tensor, shape `(B, 18, 8, 8)`.
- Output: `dict` with `logits` of shape `(B,)` plus the diagnostics
  listed in `architecture.md` and the i193 trunk diagnostics (`trunk_*`).
- The model rejects non-simple_18 inputs and non-1 `num_classes`.
- `grid_w` is bounded to `[2, 6]` so the per-row state count `2^W` stays
  small enough to avoid memory blow-ups.

## Trainer compatibility

The model exposes the same `(B, 18, 8, 8)` input contract as i193 and
returns a `dict` whose `"logits"` key has shape `(B,)`. The trainer
`idea_train_cli` reads only `dict["logits"]` for loss; the additional
diagnostic keys are surfaced to slice reports but do not affect the
training loop.
