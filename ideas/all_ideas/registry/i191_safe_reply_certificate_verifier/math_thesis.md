# Math Thesis

Safe-Reply Certificate Verifier

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `6`.

Working thesis: Instead of proving that a position is a puzzle, try to prove that it is not a puzzle. If the model can find a cheap safe-reply certificate, the puzzle logit should go down.

Let the current board tensor induce a finite certificate set

```text
C(B) = {move away king, capture attacker, block line, defend target, counter-threat, trade down}
```

where each member is represented by one or more board-local candidate squares or resources. A deterministic board program proposes certificate tokens `c_i` from simple_18 occupancy, side-to-move, attack maps, ray blockers, king zones, and material values. A neural verifier then estimates

```text
validity_i = sigmoid(v(c_i, B))
strength_i = softplus(s(c_i, B))
disproof_i = validity_i * strength_i
best_disproof = max_i disproof_i
```

The puzzle classifier is asymmetric: it first computes a positive puzzle logit from the board context, then subtracts the strongest safe-reply witness:

```text
puzzle_logit = positive_puzzle_logit - alpha * best_disproof
```

This makes non-puzzle evidence first-class. A board with many cheap replies can reduce the puzzle logit even when the generic tactical context looks sharp, while a true puzzle must survive the verifier failing to find a strong safe-reply certificate.
