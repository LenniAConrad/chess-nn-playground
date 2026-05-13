# Math Thesis — Δ-Pair Accumulator (p014)

Source: `ideas/research/primitives/external_08_delta_pair_ray_selective_bispectrum.md` (primitive_delta_pair_accumulator (DPA)).

## Working thesis

DPA extends the first-order accumulator with an explicit pair term restricted to an input-dependent edge set E(S) ⊂ S × S. The chess instantiation uses the rule-derived alignment predicate (same rank, file, or diagonal) which is the cheapest deterministic-from-S generalisation of attacker→defender pairs.

## Why this matters

The delta-accumulator family targets the missing O(|Δ|) inference-update
primitive in `torch.nn`. The static-position trainer evaluates the
analytical fixed point of the recurrence (a full forward over the active
piece-square set); the make/unmake inference path is documented in
`implementation_notes.md`. The accumulator embedding ``W ∈ R^{12·64 × d}``
is the shared parameter across forward and delta paths so gradients
flow consistently between them.

For this specific primitive the load-bearing structure is:
The factorisation-machine identity (Rendle 2010) sums over ALL pairs and cannot recover an input-dependent strict subset E(S); EmbeddingBag is sum-only. DPA exposes the pair structure as a first-class operator with O(|S|+|E(S)|·k) cost rather than O(|S|²·k).

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
