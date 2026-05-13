# Math Thesis — DeltaCReLU + Involution Reynolds Head (p015)

Source: `ideas/research/primitives/external_09_delta_crelu_involution_graph_message.md` (primitive_delta_crelu + primitive_involution_reynolds_affine).

## Working thesis

DeltaCReLU is a saturation-aware accumulator: a ClippedReLU applied to the additive pre-activation state with per-channel tracking of the saturation regime. Combined with an involution split (h ± ι h) where ι is the chess colour-swap involution, the head exposes both the saturation summary and the Reynolds-symmetric / Reynolds-antisymmetric components of the state.

## Why this matters

The delta-accumulator family targets the missing O(|Δ|) inference-update
primitive in `torch.nn`. The static-position trainer evaluates the
analytical fixed point of the recurrence (a full forward over the active
piece-square set); the make/unmake inference path is documented in
`implementation_notes.md`. The accumulator embedding ``W ∈ R^{12·64 × d}``
is the shared parameter across forward and delta paths so gradients
flow consistently between them.

For this specific primitive the load-bearing structure is:
The composition EmbeddingBag + Hardtanh recomputes the full sum; DeltaCReLU's defining property is the *joint* gradient over the saturation tape, which a pure-functional composition cannot express. The involution split augments this with a structurally-enforced colour-flip equivariance (no augmentation needed).

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
