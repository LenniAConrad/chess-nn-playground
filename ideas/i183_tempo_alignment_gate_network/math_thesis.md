# Math Thesis

Tempo-Alignment Gate Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.

Batch candidate rank: `7`.

Working thesis: Many near-puzzles are tactical-looking for the wrong
side or require a tempo that the side to move does not have. The
model should explicitly *gate* static tactical danger by side-to-move
tempo alignment instead of letting an undirected CNN absorb the
danger regardless of who is to move.

Let `H ∈ R^{B×C×8×8}` be a compact convolutional trunk over the
`simple_18` board. Define:

- An undirected static danger field
  `d(s) = relu(W_d · H[s] + b_d)`, the "tactical danger for
  somebody" signal.
- A signed side-of-attacker field
  `a(s) = W_a · [H[s] ; W(s) ; B(s)] + b_a`, where `W(s)` and
  `B(s)` are the white/black piece-occupancy summaries summed over
  planes 0-5 and 6-11. Positive means the local tactic is white's.
- A scalar tempo signal `stm ∈ {-1, +1}` from plane 12.

The per-square alignment is

```
alignment(s) = sigmoid(γ · stm · a(s) + β)
```

with learnable `γ, β`. A global tempo gate is

```
tempo_gate = sigmoid(MLP_t([mean_s H[s] ; sum_s W(s) ; sum_s B(s) ; stm]))
```

The gated per-square pressure is the *multiplicative* conjunction

```
g(s) = tempo_gate · alignment(s) · d(s)
```

so the signal collapses if either leg is small. The puzzle head reads
the pooled scalars `own_pressure = mean_s alignment·d`,
`opp_pressure = mean_s (1-alignment)·d`,
`alignment_gap = own - opp`, `gated_pressure = mean_s g`,
`tempo_gate`, `mean_danger`, `max_danger` and a counterfactual
contrast obtained by flipping the side-to-move plane and re-running
the trunk:

```
flip_contrast = gated_pressure(stm) - gated_pressure(-stm)
```

The puzzle logit is a small `LayerNorm → Linear → GELU → Dropout →
Linear` head over the concatenated scalars, trained with BCE.

The packet's required ablations are exposed: `no_tempo_gate` forces
`tempo_gate ≡ 1`, `no_alignment` forces `alignment ≡ 0.5`, and
`additive_gate` replaces `*` with `+` in the gating step. Each
ablation kills a specific part of the multiplicative conjunction the
markdown identifies as the architecture's defining property.
