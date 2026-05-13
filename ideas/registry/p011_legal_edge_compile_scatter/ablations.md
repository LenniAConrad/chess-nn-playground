# Ablations — p011 Legal-Edge Compile Scatter

## Switches (model.ablation)

| Mode | What it tests |
|---|---|
| `none` | Full architecture (default). |
| `no_edge_gate` | Replace σ-gate with 1 wherever the typed adjacency is 1. Degenerates to a stricter p009 LMGConv (uniform per-edge weight). **Primary p011 falsifier**: tests whether the feature-conditioned gate carries lift beyond the rule-derived adjacency. |
| `random_typed_edges` | Replace typed adjacency with a random mask of identical density. Tests whether the rule structure is load-bearing. |
| `shared_type_weight` | Collapse the six `W_r` to one shared linear. Tests whether per-type weight tying matters. |
| `zero_delta` | Hold `primitive_delta = 0`. i193 baseline. |
| `disable_gate` | Hold `primitive_gate = 1`. |
| `trunk_only` | Strict no-op. |

## Falsification criteria

Promote p011 only if `model.ablation = none`:

- Aggregate PR AUC delta from i193 >= -0.005.
- CRTK class-1 matched-recall FP rate drops by >=3 percentage points.
- Wall-clock per epoch within 1.3x of i193.

Drop p011 if:

- `no_edge_gate` matches `none` (the σ-gate added no lift over the
  rule mask — operator is functionally p009 LMGConv with extra params).
- `random_typed_edges` matches `none` (geometry was not load-bearing).
- `zero_delta` matches `none` (delta head was noise).

## Deferred internal proposals from external_14

The source primitive packet
(`ideas/research/primitives/external_14_ray_occlusion_legal_edge_compile_scatter.md`)
ranks five proposals. Only **Content-Compiled Legal Edge Scatter** is
implemented here. The others:

- **Occlusion-Gated Ray Scan**: covered by p010 (Ray-Occlusion Semiring
  Scan) in this batch.
- **Delta-Apply Linear**: stateful NNUE-style incremental linear with
  an edit-script API. Deferred — engine-deployment primitive whose
  scout-scale falsifier needs a make/unmake search loop.
- **Rule-Automaton Selective Scan**: finite-state-automaton-controlled
  selective SSM. Deferred — interesting symbolic-control hybrid but
  requires automaton design choices that exceed the "minimal trainer
  surface area" budget of this batch.
- **Chess-Orbit Linear**: G-orbit-tied linear under the chess
  automorphism group. Deferred — flagged in the file's own framing as
  "underexplored finite chess-rule orbit primitive, not fully new
  equivariance".

If any prove relevant after p011's scout, promote under fresh `p###`.
