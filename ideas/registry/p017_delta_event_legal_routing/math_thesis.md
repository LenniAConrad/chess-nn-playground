# Math Thesis — Delta-Event Legal-Move Routing (p017)

Source: `ideas/research/primitives/external_11_delta_event_legal_move_routing.md` (primitive_delta_event_accumulator + primitive_legal_move_routing).

## Working thesis

Delta-Event Accumulator + Legal-Move Routing: each active piece-square embedding is gated by a content-dependent routing weight α_i(S) derived from the piece's pseudo-legal mobility on the legal-move graph. The routing weight is rule-derived from S (piece type and source square), not from CRTK / Stockfish metadata.

## Why this matters

The delta-accumulator family targets the missing O(|Δ|) inference-update
primitive in `torch.nn`. The static-position trainer evaluates the
analytical fixed point of the recurrence (a full forward over the active
piece-square set); the make/unmake inference path is documented in
`implementation_notes.md`. The accumulator embedding ``W ∈ R^{12·64 × d}``
is the shared parameter across forward and delta paths so gradients
flow consistently between them.

For this specific primitive the load-bearing structure is:
Standard masked attention uses an input-independent mask; this primitive builds the routing weight inside the operator from the per-piece pseudo-legal target count, fusing edge generation and message routing into a single sparse-event aggregator.

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
