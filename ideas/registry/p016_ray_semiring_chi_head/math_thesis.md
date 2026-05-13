# Math Thesis — Ray-Semiring χ-Head (p016)

Source: `ideas/research/primitives/external_10_ray_semiring_exchange_and_chi_head.md` (primitive_chi_head (sign-graded χ-equivariant value head)).

## Working thesis

χ-head splits the accumulator into white-piece (h+) and black-piece (h-) channels and uses only the even × odd cross-bilinear ``Σ M^{+-}_{ij} h^+_i h^-_j`` as the value head, structurally guaranteeing ``f(τ x) = − f(x)`` for the colour-swap involution τ — the exact symmetry of chess evaluation.

## Why this matters

The delta-accumulator family targets the missing O(|Δ|) inference-update
primitive in `torch.nn`. The static-position trainer evaluates the
analytical fixed point of the recurrence (a full forward over the active
piece-square set); the make/unmake inference path is documented in
`implementation_notes.md`. The accumulator embedding ``W ∈ R^{12·64 × d}``
is the shared parameter across forward and delta paths so gradients
flow consistently between them.

For this specific primitive the load-bearing structure is:
Standard nn.Bilinear has no constraint on its weight tensor and must learn the colour antisymmetry from data; the χ-head bakes it into the operator's compute graph by parameter-tying the sign-graded subspace. This is the strongest novelty claim in external_10.

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
