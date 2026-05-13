# Ablations — p009 Legal-Move-Graph Convolution

## Switches (model.ablation)

| Mode | What it tests |
|---|---|
| `none` | Full architecture (default). |
| `random_typed_edges` | Replace each per-type adjacency with a random mask of identical density. **Primary LMGConv falsifier**. |
| `shared_weight` | Collapse the six `W_r` to a single shared linear. Tests whether the typed channel is load-bearing — collapsing to a generic R-GCN-style message pass. |
| `no_normalization` | Disable degree normalisation. Tests whether the GraphSAGE-style mean is load-bearing (or hurting). |
| `zero_delta` | Hold `primitive_delta = 0`. i193 baseline. |
| `disable_gate` | Hold `primitive_gate = 1`. |
| `trunk_only` | Strict no-op. |

## Falsification criteria

Promote p009 only if `model.ablation = none`:

- Aggregate PR AUC delta from i193 >= -0.005.
- CRTK class-1 matched-recall FP rate drops by >=3 percentage points.
- Wall-clock per epoch within 1.2x of i193.

Drop p009 if:

- `random_typed_edges` matches `none` (geometry was not load-bearing).
- `shared_weight` matches `none` (per-type weight tying gave no lift).
- `zero_delta` matches `none` (delta head was noise).

## Deferred internal proposals from external_05

The source primitive packet
(`ideas/research/primitives/external_05_legal_move_graph_delta_accumulator.md`)
ranks five proposals. Only **LMGConv** is implemented here. The others
are deferred:

- **ΔAcc — Differentiable Delta Accumulator**: stateful Gated-DeltaNet-
  style accumulator for chess. Deferred — engine-deployment primitive
  whose scout-scale falsifier needs a search-trajectory benchmark this
  batch does not have. The "delta_accumulator" in the file's slug
  refers to this proposal; we keep the slug for traceability but did
  not implement it here.
- **CSE-Conv — Color-Swap × D4 Equivariant Conv**: G-CNN reframe.
  Deferred — reframing of established prior art, file's own ranking
  puts it in the medium-novelty tier.
- **MobFP — Mobility Fixed-Point Operator**: DEQ on the attack graph.
  Deferred — implicit-grad solver is brittle and likely slower than
  unrolled message-pass at scout scale (the file flags this as the
  expected outcome).
- **ADB — Attack-Defend Sparse Bilinear**: sparse-supported bilinear
  over the attack relation. Closely overlaps the p011 typed scatter
  operator; if ADB-specific bilinear interaction is wanted, it should
  be a future `p###` variant.

If any prove relevant after p009's scout, promote under fresh `p###`.
