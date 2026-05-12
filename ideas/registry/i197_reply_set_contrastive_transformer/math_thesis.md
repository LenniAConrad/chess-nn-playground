# Math Thesis

Reply-Set Contrastive Transformer

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `12`.

Working thesis: A puzzle position should embed differently from its plausible reply positions. A near-puzzle may remain close to one or more safe replies. Use contrastive learning over the current position and a deterministic set of pseudo-reply positions.

Concretely, let `f_θ` be a board encoder and `x` the current board. Construct a pseudo-reply set `R(x) = {x_1, ..., x_K}` by translating the side-to-move's own piece planes along `K = 12` chess-relevant offsets (rook/bishop rays + knight jumps) and flipping the side-to-move plane. Define per-reply cosine similarities `s_k = cos(f_θ(x), f_θ(x_k))`. The contrastive features

```
g(s) = [min_k s_k, mean_k s_k, std_k s_k, top1_k s_k, top2_k s_k, sum_k max(s_k, 0)]
```

drop when the current position is far from every reply (a real puzzle) and remain large when at least one reply stays close (a near-puzzle). The puzzle head reads `g(s)` together with a token-attention summary and a defender-reply (king-ring) summary; it does **not** read raw per-reply embeddings, so the contrastive aggregation is the only path through which the reply set can change the prediction.
