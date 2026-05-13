# Implementation Notes

## Worktree

This idea was implemented in the
`cnp-primitive-gpt-orbit-boolean-algebra` worktree alongside p037-p041
of the 2026-05-13 GPT primitive batch. The batch ID range was
`p036-p041`; no conflicts with prior IDs.

## Module

`src/chess_nn_playground/models/primitives/canonical_orbit_st_operator.py`

Contains:

- `CanonicalOrbitSTOperator` nn.Module.
- `build_canonical_orbit_st_operator_from_config`.
- `_build_permutations()` -- the fixed 4 x 64 permutation table for the
  C2 x C2 board-geometry group.
- `ALLOWED_ABLATIONS` tuple with the five named ablations.

## Why the gradient does not need a custom autograd.Function

The source packet describes the operator as a "straight-through"
selector: forward picks one branch, backward uses the inverse of the
selected transform. For permutation actions in C2 x C2, the inverse is
the transform itself, and PyTorch's `tensor.gather(dim, index)` already
propagates gradients back through the gather. The discrete `chosen`
index is computed under `torch.no_grad()` so its gradient is implicitly
zero, matching the packet's "no gradient is defined through the
discrete hash choice" rule.

If a future variant uses non-involutive transforms (e.g. an 8-way
dihedral group with rotations of 90 and 270 degrees) the inverse must
be applied explicitly. For C2 x C2 that work is not required.

## Deterministic tie handling

The source packet flags "hash ties near symmetric boards can create
unstable branch choices; tie-breaking must be deterministic". This
implementation uses three layers of determinism:

1. The hash projection seed is fixed (`torch.Generator` seeded with
   `0xC0DEC0DE`); the projection is stored as a non-persistent buffer.
2. The hash quantum (default `1.0`) makes the key piecewise-constant in
   the input so small perturbations within a quantum produce the same
   key.
3. The lexicographic comparison favours the lower group-element index
   on exact ties, so the identity action `e` always wins a true tie.

## Diagnostics surfaced

- `cost_chosen_orbit_index` -- which of the four orbit elements was
  selected (float-encoded long).
- `cost_orbit_gap` -- l2 norm of `(worst_key - chosen_key)`; captures
  how strongly the input prefers the canonical orientation.
- `cost_orbit_ties` -- number of orbit elements tied with the chosen
  key (>= 1; >= 2 means a true tie).
- `cost_residual_norm` -- RMS of `canonical - latent`.
- `cost_canonical_norm` -- RMS of `canonical`.

These plus `primitive_gate*` and `trunk_*` are the slicing axes for the
falsifier.

## Deferred internal proposals

The source packet (`external_31_canonical_orbit_bdd_wmc_primitives.md`)
contains four other proposals which were *not* promoted in this batch:

| Proposal | Reason for deferral |
|---|---|
| `primitive_bdd_wmc` (BDD WMC layer) | Requires building and storing a reduced ordered BDD per learned predicate set; non-trivial infrastructure, deferred to a dedicated promotion. |
| `primitive_matroid_rank_envelope` | Matroid rank oracle infrastructure is a sister concept of the entropic-matroid pool in `external_35`; both should be designed together to avoid two ad-hoc rank oracles. |
| `primitive_tactical_lcp_projector` | Differentiable QP/LCP solve overlaps with the existing `dykstra_lcp` family; needs an explicit comparison before promoting. |
| `primitive_delta_cholesky_whiten` | Cholesky update / downdate with autograd is non-trivial; the closer relative `primitive_woodbury_resolver` is being implemented in this batch as p038. |

## Input contract

- Input: `simple_18` current-board tensor, shape `(B, 18, 8, 8)`.
- Output: `dict` with `logits` of shape `(B,)` plus the diagnostics
  listed above and the i193 trunk diagnostics (`trunk_*`).
- The model rejects non-simple_18 inputs and non-1 `num_classes`.

## Trainer compatibility

The model exposes the same `(B, 18, 8, 8)` input contract as i193 and
returns a `dict` whose `"logits"` key has shape `(B,)`. The trainer
`idea_train_cli` reads only `dict["logits"]` for loss; the additional
diagnostic keys are surfaced to slice reports but do not affect the
training loop.
