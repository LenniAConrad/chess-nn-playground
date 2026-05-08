# Math Thesis

Exchange-Soundness Graph Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `2`.

Working thesis: Many false puzzle signals come from attacks that look
strong but lose material or fail tactically after exchanges. A puzzle
detector should know whether an apparent tactic is exchange-sound on
the side-to-move's attack/defense graph.

## Differentiable Static Exchange Evaluation

For a square `s`, let

- `v_target(s)` be the value of the opponent piece on `s` (zero off
  opponent squares),
- `v_a(s)` be the value of the cheapest available attacker (a learned
  per-square distribution over the side-to-move's piece-type
  inventory dotted with the standard piece-value vector),
- `v_d(s)` the symmetric defender value,
- `p_a(s) = sigmoid(attacker_logit(s))` and
  `p_d(s) = sigmoid(defender_logit(s))` the attacker / defender
  intensities (probabilities the next capture or recapture happens).

Define the bounded-depth differentiable static exchange:

```
see_K(s) = v_top_K
see_{k}(s) = v_top_k - p_resp_k * max(0, see_{k+1}(s))   for k < K
```

with the alternating sequence

```
v_top_0 = v_target(s)        (we capture the opponent piece)
v_top_1 = v_a(s)             (defender recaptures our attacker)
v_top_2 = v_d(s)             (we recapture their defender)
v_top_3 = v_a(s)             (defender re-recaptures)
...
```

and `p_resp_k` alternating `p_d, p_a, p_d, p_a, ...`. The
`max(0, .)` operator is the SEE "stop here" rule: the side considering
the next capture only takes if continuing is non-negative for them.

`see(s) = see_0(s)` is the per-square SEE on the learned attack/defense
graph. `exchange_soundness(s) = sigmoid(see(s) / T)` gates a feature
pool over the opponent-piece *target squares* of the side-to-move's
attack graph.

## What This Buys

A position with a real puzzle threat has at least one target square
`s*` with high `see(s*) > 0` -- the threat is sound after exchanges.
A near-puzzle position whose tactic dissolves after exchanges has
`see` close to zero (or negative) on every target square. The same
intermediate quantities give the head explicit graph-network scalars:

- `max_see_target`, `mean_see_target`, `frac_unsound_targets`,
- `graph_pressure`, `reply_pressure`, `defense_gap`,
- `transport_imbalance`, `sheaf_tension`,

which the puzzle classifier consumes alongside the bottleneck pool of
trunk features over the most decisive squares.
