# Math Thesis — DeltaState + SLG Diffusion (p018)

Source: `ideas/research/primitives/external_17_delta_state_slg_diffusion_fg_tp.md` (primitive_delta_state + primitive_slg_diffusion).

## Working thesis

DeltaState exposes the stateful triple-interface contract (forward / apply_delta / inverse_delta) and SLG Diffusion adds a single sheaf-Laplacian diffusion step over the rule-derived alignment-pair graph, with low-rank per-piece-type restriction maps F_ij = U_i V_j^T.

## Why this matters

The delta-accumulator family targets the missing O(|Δ|) inference-update
primitive in `torch.nn`. The static-position trainer evaluates the
analytical fixed point of the recurrence (a full forward over the active
piece-square set); the make/unmake inference path is documented in
`implementation_notes.md`. The accumulator embedding ``W ∈ R^{12·64 × d}``
is the shared parameter across forward and delta paths so gradients
flow consistently between them.

For this specific primitive the load-bearing structure is:
The combination of (a) stateful reversible delta semantics and (b) per-piece-type restriction maps over an input-determined legal-graph is not expressible as a composition of EmbeddingBag, GAT, or torch_geometric MessagePassing — the closest sheaf-NN (Bodnar et al. 2022) work uses non-input-determined adjacency.

## Falsifier

Run the matched-baseline (i193 alone) and the primitive-specific
ablations listed in `ablations.md`. The primitive is dropped if any
declared-load-bearing ablation matches the unablated run on the
declared target slice.

## Composition with other primitives

The delta-accumulator family is additive over independent gated heads
(see `ideas/research/primitives/PRIMITIVE_TRAINING_TODO.md`), so this
primitive composes orthogonally with TSDP (i248), PFCT (i246), TDCD
(i244), DHPE (i245), and CAIO (i247). Dropped primitives can be removed
without disturbing the trunk.
