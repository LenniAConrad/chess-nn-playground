# Math Thesis

Material-Phase Low-Rank Adapter Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.

Batch candidate rank: `6`.

Working thesis: Chess positions vary greatly by material phase. Instead of one encoder for every position, condition low-rank adapter weights on material summaries while keeping a shared backbone. Concretely, a shared CNN encodes the board, a deterministic summary `s` of side-relative piece counts, side-to-move, castling/en-passant, and a smooth phase coordinate is computed from the simple_18 input, and selected hidden layers receive a per-sample LoRA-style update

```
h_out = W h + b + (1 / r) * B(s) A(s) h
```

with rank `r ≪ d`, where `A(s)` and `B(s)` are produced by linear generators from a phase embedding of `s` and `B(s)` is zero-initialised. The architecture tests whether tiny material-conditioned rank updates improve over a shared-backbone-only model without letting material become a shortcut: rank is held small by construction, and the report exposes per-sample adapter norms and material readouts so material-bucket evaluation, rank ablations, and shortcut probes are first-class diagnostics.
