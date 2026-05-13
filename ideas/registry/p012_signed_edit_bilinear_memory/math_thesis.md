# Math Thesis — Signed-Edit Bilinear Memory (p012)

Source: `ideas/research/primitives/external_01_signed_edit_bilinear_memory_ray_scan.md` (primitive_signed_edit_bilinear_memory (SEBM)).

## Working thesis

SEBM maintains an exact O(|Δ|) state triple (s, u, p) over signed edits to the active piece-square feature set. The pair state p captures attacker+defender / blocker+slider / king-ring+intruder interactions in a single bilinear summary, generalising the first-order HalfKA accumulator without paying the all-pairs cost.

## Why this matters

The delta-accumulator family targets the missing O(|Δ|) inference-update
primitive in `torch.nn`. The static-position trainer evaluates the
analytical fixed point of the recurrence (a full forward over the active
piece-square set); the make/unmake inference path is documented in
`implementation_notes.md`. The accumulator embedding ``W ∈ R^{12·64 × d}``
is the shared parameter across forward and delta paths so gradients
flow consistently between them.

For this specific primitive the load-bearing structure is:
Differentiable signed-edit bilinear primitive: standard sum-only EmbeddingBag and append-only state-space scans cannot express the inverse-consistent insert/delete pair update used here. The pair state p is the cross-term s ⊙ u − Σ a_j ⊙ b_j, exactly the FM identity for the *same* (not partitioned) feature set.

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
