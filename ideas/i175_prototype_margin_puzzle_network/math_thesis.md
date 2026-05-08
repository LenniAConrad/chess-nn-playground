# Math Thesis

Prototype-Margin Puzzle Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.

Batch candidate rank: `6`.

Working thesis: The model should not merely say "puzzle-like." It should compare the board to separate learned prototypes — random non-puzzle positions, near-puzzle hard negatives, and real puzzles — and read the puzzle logit out of the margin between the puzzle similarity and the largest non-puzzle similarity.

## Construction

Encode the board to a latent `z` and learn three prototype banks
`P_random, P_near, P_puzzle ∈ R^{K x D}`. Define per-class
similarities by log-sum-exp over cosine scores at temperature `τ`:

```
sim_class(z) = logsumexp_k cos(z, P_class[k]) / τ
```

and the puzzle logit as the prototype margin

```
puzzle_logit = sim_puzzle - logsumexp([sim_random, sim_near]).
```

Near-puzzles get their own attractor bank, so the model is forced to
compete random-vs-near-vs-puzzle rather than averaging the two
negative populations into one logit.
