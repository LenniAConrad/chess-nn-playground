# Math Thesis

Latent Reply Entropy Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `7`.

Working thesis: A forcing puzzle often reduces the opponent's viable reply distribution. A near-puzzle may have many replies that keep the position acceptable. The network can learn a reply entropy proxy without engine labels.

Let `R(B)` be a deterministic reply/resource set extracted from the current board: king escapes, captures, line blocks, target defenses, counter-threats, and quiet resources. Each reply token `r_i` is scored against the board context:

```text
s_i = safe_reply_score(r_i, board_context)
p_i = softmax(s_i / temperature)
```

The model then reads concentration statistics of the latent reply distribution:

```text
H = -sum_i p_i log p_i
top1 = max_i p_i
top2_gap = top1 - second_largest_i p_i
effective_reply_count = exp(H)
```

The final classifier receives the board context plus these entropy and concentration features:

```text
puzzle_logit = MLP([board_context, H, top1, top2_gap, effective_reply_count])
```

The intended signal is compression of viable replies. A true forcing puzzle should put high probability on very few defensive resources; a near-puzzle should retain a broader, higher-entropy reply distribution.
