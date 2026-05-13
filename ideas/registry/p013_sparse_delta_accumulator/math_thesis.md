# Math Thesis — Sparse-Delta Accumulator (p013)

Source: `ideas/research/primitives/external_07_sparse_delta_accumulator_segment_scatter.md` (primitive_sda (Sparse-Delta Accumulator)).

## Working thesis

SDA is the canonical generalisation of HalfKA's first-order accumulator: a persistent state h is updated by a signed-delta stream of feature ids with O(|Δ|·d) cost per ply, exposed as a differentiable autograd primitive. Static-position training uses the analytical fixed point Σ_i W[i] over the active piece-square indices.

## Why this matters

The delta-accumulator family targets the missing O(|Δ|) inference-update
primitive in `torch.nn`. The static-position trainer evaluates the
analytical fixed point of the recurrence (a full forward over the active
piece-square set); the make/unmake inference path is documented in
`implementation_notes.md`. The accumulator embedding ``W ∈ R^{12·64 × d}``
is the shared parameter across forward and delta paths so gradients
flow consistently between them.

For this specific primitive the load-bearing structure is:
No torch.nn op currently carries (state, Δ_add, Δ_remove) as a fused stateful primitive — EmbeddingBag is stateless and recomputes from scratch every forward. The defining contract is the stateful make/unmake autograd path, not the static sum.

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
