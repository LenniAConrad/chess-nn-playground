# Math Thesis

Replicator Payoff Piece Dynamics

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.

Batch candidate rank: `1`.

Working thesis: Puzzle-like positions often feel like unstable games among
pieces: one attacker increases pressure, one defender is overloaded, one target
becomes strategically dominant. A differentiable payoff game over occupied
pieces models this as a small dynamical system whose equilibrium and
instability statistics are read out as a binary puzzle-likeness signal.

Concretely, the architecture identifies up to `Pmax` occupied piece tokens
from the `simple_18` board, learns an asymmetric pairwise payoff
`P^h_ij = f_theta(token_i, token_j, geometry_ij)` split across `H` role heads,
and runs `T` replicator-dynamics steps for each head:

```text
fitness^h_i = (P^h p^h)_i
avg^h       = sum_i p^h_i fitness^h_i
p^h_i       <- p^h_i * exp(eta_h * (fitness^h_i - avg^h)) / Z
```

with the occupancy mask preserved so empty slots stay at zero mass. Equilibrium
diagnostics (entropy, top mass, KL from the initial population, average
payoff, fitness variance, mass on kings / minors / majors / own / opponent
pieces) and the payoff-matrix asymmetry norm are fused with a small CNN board
summary to predict the `puzzle_binary` logit.
