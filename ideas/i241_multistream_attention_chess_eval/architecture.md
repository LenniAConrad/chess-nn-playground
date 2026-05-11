# i241 — Multi-Stream Chess-Decomposed Transformer Evaluator

## Motivation

The architecture-scout sweep (see `docs/research_audit_2026-05-09.md`) trained
234 bespoke architectures at small scale on `puzzle_binary`. The largest
within-encoding architectural margin came from
**`i193_exchange_then_king_dual_stream`** — a small conv-based architecture
that splits its trunk into a tactical-exchange stream and a king-safety stream,
fuses them with a learned phase router, and reads a single puzzle logit out of
the fused embedding. At 157k parameters with `simple_18` input it scored
**0.8755 test PR AUC** — clear of i048 (0.8613, rank 2) and i011_vetoselect
(0.8584, rank 5) at far smaller parameter counts.

This is the strongest evidence in the whole sweep that **a chess-specific
inductive bias actually pulls weight** at small scale. Most "novel" priors in
the scout pool fail to beat a tuned plain CNN; this one wins by a margin
larger than the seed-noise band.

Two things are true about the i193 result:

1. The dual-stream decomposition is doing real work — it is the only
   architectural property in the entire scout pool that produces a
   reproducible (within the noise of a single seed) advantage.
2. The architecture is convolution-based, simple_18-encoded, and trained on
   173k puzzle samples. It is not by itself a candidate to beat LC0 BT4
   (which is a transformer trained on billions of self-play positions).

The proposal is to **lift i193's structural prior into a BT4-class
architecture**: keep the multi-stream decomposition, replace each stream's
convolutional encoder with a small transformer, add a third
positional/structural stream, and swap the puzzle-binary head for engine
value+policy heads.

## Architectural sketch

```
Input planes (lc0_bt4_112 + optional history)
        │
        ├──► Stream E (Exchange)
        │      • piece embeddings + attacker/defender attention bias
        │      • transformer blocks (N_E layers, d_E dim, h_E heads)
        │      • emits stream_E_embedding ∈ R^(64 × d_E)
        │
        ├──► Stream K (King)
        │      • piece embeddings + king-zone/check-ray attention bias
        │      • transformer blocks (N_K layers, d_K dim, h_K heads)
        │      • emits stream_K_embedding ∈ R^(64 × d_K)
        │
        └──► Stream P (Positional / Structural)
               • piece embeddings + standard positional encoding
               • transformer blocks (N_P layers, d_P dim, h_P heads)
               • emits stream_P_embedding ∈ R^(64 × d_P)

Fusion
        │
        ▼
   Phase-router MLP
        │  reads pooled (E, K, P) embeddings → soft mixture α ∈ Δ²
        │  ( α_E + α_K + α_P = 1 )
        ▼
   fused_embedding = α_E · E + α_K · K + α_P · P
        +   residual_head( concat(E, K, P) )

Heads
        │
        ├──► value:  (W, D, L) categorical  OR  centipawn regression
        ├──► policy: 1858-dim move logits, masked to legal moves
        └──► (training-time only) aux heads:
                • exchange-outcome classifier from E
                • king-attack-or-defend classifier from K
                • positional-eval regressor from P
```

### Stream specialization via attention bias

The key architectural lever is **attention bias matrices** that hand each
stream a chess-aware prior on which square pairs are relevant for its task:

- **Stream E** attention bias: a precomputed `(64,64)` matrix that has high
  entries for square pairs connected by attacker-defender relationships
  (capture moves, defended squares, x-rays). The attention layer then learns
  *which* of these tactical relations matter for the current position; it
  does not have to discover the geometry from scratch.
- **Stream K** attention bias: similar but biased to king-zone squares (the
  3×3 ring around each king), check-ray squares (queen, rook, bishop, knight
  rays into the king zone), and pawn-shield squares.
- **Stream P** attention bias: standard learned positional encoding
  (relative file/rank attention bias). No tactical prior.

This is the *attention analogue* of i193's "deterministic feature builder
that produces the per-stream input bias planes from precomputed geometric
attack and between-square tables" — but instead of biasing the input, it
biases attention.

### Phase router

The phase router produces a soft mixture over the three streams that depends
on the position. Reasonable behavior:

- In sharp tactical positions: α_E dominates.
- In king-attack positions: α_K dominates.
- In quiet positional positions: α_P dominates.

This is *learned*, not hand-coded. The router is a small MLP reading the
pooled stream embeddings; it emits softmax weights.

### Heads

- **Value**: WDL categorical head (3 logits, softmax). This is the LC0
  convention; supports MCTS evaluation directly.
- **Policy**: 1858-dim move logits, masked at inference to legal moves and
  softmaxed.
- **Aux**: per-stream classifiers/regressors active during training only,
  with loss weights ≤ 0.05 each. The aux losses give each stream a
  chess-aware gradient signal so the streams don't collapse into the same
  representation.

## Sizing target

| Component | base | scale_up | scale_xl (BT4-medium) |
|---|---|---|---|
| d per stream | 64 | 96 | 128 |
| heads per stream | 4 | 6 | 8 |
| blocks per stream | 4 | 6 | 8 |
| total params | ~3M | ~12M | ~50M |
| MFLOPs/position | ~50 | ~200 | ~800 |

For comparison, LC0 BT4-medium is ~50M parameters. The compute is comparable
to BT4 because the three streams can be ~`sqrt(3)≈1.7×` narrower than a
single-stream BT4 at matched total FLOPs.

## What this architecture has that BT4 does not

1. **Structural prior on chess concepts.** BT4 must learn the exchange /
   king-safety / positional decomposition implicitly. This architecture has
   it from initialization.
2. **Task-specific attention bias.** Each stream's attention is biased to
   the square pairs relevant to its task (attacker-defender for exchange,
   king-zone for king, none for positional).
3. **Stream-level supervision.** Training can supervise each stream
   independently with chess-aware aux losses. BT4 has no clean structural
   place to put aux supervision.
4. **Interpretable per-stream contributions.** At inference, the phase
   router's α weights and per-stream value contributions are individually
   inspectable. BT4 is monolithic.

## What this architecture lacks vs BT4

1. **Engineering hardening.** BT4 has had years of tuning at scale. This
   architecture has been *implied* by one scout result at 157k params.
2. **Cross-stream attention.** The streams interact only at the fusion
   head. Some tactics (e.g. king attacks built on tactical exchanges)
   involve genuine cross-stream interactions. Mitigation: residual head
   reads the concatenated stream embeddings, recovering most of this signal.

## Training plan (skeletal)

The architecture is only useful for engine play if trained on engine-scale
data. Three options ranked by realism:

1. **Stockfish-eval distillation** (cheapest realistic path): train on
   `(position, stockfish_eval, stockfish_best_move)` triples from a corpus
   of master games. Target ELO: ~3000-3200. Compute: ~weeks of GPU.
2. **Distillation from LC0 BT4 itself**: train the architecture to match
   BT4's value+policy outputs. Target ELO: approaches BT4. Compute: ~weeks.
   The interesting question is whether the structural prior lets the smaller
   architecture *exceed* its teacher on some metrics.
3. **Full self-play** (LC0-style): start with random weights, generate
   self-play games with MCTS, train on those, iterate. Target ELO:
   open-ended. Compute: months.

Heads-only puzzle_binary training (as a sanity check before engine training):
a few hours; should reproduce i193's PR AUC margin or better.

## Hypotheses to test

- **H1**: at matched training compute and parameter budget, this
  architecture beats a single-stream BT4-shaped trunk by ≥10 ELO.
  (Mechanism: structural prior compounds with scale.)
- **H2**: training-time aux supervision on the per-stream heads gives
  measurable ELO gain over training without aux.
  (Mechanism: prevents stream collapse / encourages specialization.)
- **H3**: the phase router's α weights show interpretable position-type
  dependence (sharp positions: α_E high; king-attack positions: α_K high;
  quiet positions: α_P high).
  (Mechanism: confirms the decomposition is actually being used.)

## Ablations (planned, see `ablations.md`)

- A1: remove Stream P (drop the positional stream)
- A2: remove Stream K (drop the king stream)
- A3: remove attention bias matrices (vanilla attention in each stream)
- A4: remove per-stream aux losses
- A5: single-stream baseline with same total params (no decomposition)
- A6: hard-routed streams (use a heuristic position-type detector instead
  of the learned phase router)

A1 and A2 isolate which stream is contributing what. A5 is the critical
control: does the multi-stream structure beat a same-size single-stream
transformer?

## Open questions

1. **Cross-stream attention?** Would adding a small cross-stream attention
   block (each stream attends to the other two) help, or destroy the
   specialization? Empirical question.
2. **Stream sharing?** Could the three streams share the embedding layer
   (just specialize the attention bias and the per-stream blocks)? This
   would halve parameter counts at the input.
3. **Number of streams?** Three is a guess. Could there be a "pawn
   structure" stream? An "endgame technique" stream? Test at small scale
   before committing.
4. **Engine-evaluation labels.** The closest available training signal is
   Stockfish eval distillation. Is there a more chess-natural training
   target — e.g. *outcome of the position assuming both sides play
   stream-aware sub-policies*?
