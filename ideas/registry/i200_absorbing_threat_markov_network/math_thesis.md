# Math Thesis

Absorbing Threat Markov Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md`.

Batch candidate rank: `3`.

## Working Thesis

Puzzle detection can be treated as a probabilistic process over
tactical states:

```text
pressure -> threat -> forced response -> collapse/proof
pressure -> safe response -> disproof
```

A full proof tree is expensive. A compact absorbing Markov chain over
a small set of named tactical states — `attack_pressure`,
`defender_available`, `line_open`, `king_constrained`,
`target_hanging`, `counterplay`, with `proof_absorb` and
`disproof_absorb` as absorbing states — can approximate whether the
position tends toward proof (puzzle) or disproof (non-puzzle).

## Formal Object

For each board `x` we build:

- A learnable embedding table
  `E ∈ R^{K × d}` with `K = state_count`. The first
  `K − 2` rows are transient states; rows `K − 2` and `K − 1` are the
  absorbing `proof_absorb` and `disproof_absorb` states.
- An initial transient distribution `π_0(x) ∈ Δ^{K-2}`, embedded into
  `Δ^K` with zero mass on the absorbing rows.
- A board-conditioned row-stochastic transition matrix
  `P(x) ∈ R^{K × K}` whose two absorbing rows are identity. Concretely
  the transient logits are a board-modulated bilinear form
  `logits[i, j] = ∑_d E[i, d] · (W(x)[d] · E[j, d]) + b[i, j]`
  followed by row softmax over `j`.

## Iterated Absorption

Power iteration produces the trajectory `π_t = π_{t−1} P(x)` for
`t = 1, …, T = transition_steps`. From it we read:

- `prob_proof(x) = π_T[K − 2]`
- `prob_disproof(x) = π_T[K − 1]`
- `E[steps](x) = ∑_{t < T} (1 − π_t[K − 2] − π_t[K − 1])`

These quantities are differentiable in the encoder weights, the state
embeddings, and the board-projection weights. As `T → ∞`,
`(prob_proof, prob_disproof)` converges to the canonical absorption
probability vector `π_0 (I − Q)^{-1} R = π_0 N R` of the chain
restricted to its transient block; the soft `E[steps]` converges to
`π_0 N 1`.

## Decision Rule

A small head MLP maps the readout
`[π_T, prob_proof, prob_disproof, prob_proof − prob_disproof,
E[steps], π_0|transient, board_pool]` to one puzzle logit. Sharp
concentration in `proof_absorb` pushes the position toward the puzzle
class; mass leaking into `disproof_absorb` pushes it toward
non-puzzle.

This is a bespoke implementation of the markdown thesis; it is no
longer a `ResearchPacketProbe` scaffold.
