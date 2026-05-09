# Math Thesis

Defender-Exhaustion Cascade Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.

Batch candidate rank: `2`.

Working thesis: many puzzles exist because one side cannot satisfy
all defensive obligations at once — `king escape`, `piece defense`,
`mate threat`, `queen or rook attack`, `promotion stop`, `back-rank
weakness`. Near-puzzles can have heavy raw threats but a still
*satisfiable* defence graph. Modelling defence as a small recurrent
allocation cascade — `T` steps where typed obligations bid for
typed resources whose capacity is consumed at every step — produces
an *exhaustion curve* whose growth distinguishes "defence runs out"
from "defence is satisfiable but tense".

Concretely, with obligation tokens `o_i` and resource tokens `r_j`
each of dimension `D`, capacities `c_j > 0`, and threat context `z`,
the cascade

```
h^{(0)}_i      = o_i
h^{(t)}_i      = GRU(z, h^{(t-1)}_i)
demand^{(t)}_i = softplus(MLP(h^{(t)}_i))
mod^{(t)}_{ij} = Linear(h^{(t)}_i)_j
A^{(t)}_{ij}   = softmax_j(<o_i, r_j> / sqrt(D) - lambda * (P^{(t)}_{ij} + mod^{(t)}_{ij}))
allocated^{(t)}_i = sum_j A^{(t)}_{ij} c_j
residual^{(t)}_i  = demand^{(t)}_i - allocated^{(t)}_i
P^{(t+1)}      = P^{(t)} + A^{(t)} * c
```

is a board-only readout of a discrete allocation game. The
exhaustion-curve diagnostics

```
phi^{(t)} = (sum_i softplus(residual^{(t)}_i),
             mean_i H(A^{(t)}_{i,:}),
             max_i softplus(residual^{(t)}_i))
```

and the final residual / allocation marginals are fed to a small MLP
puzzle head along with pooled trunk features. Setting
`cascade_steps = 1` collapses this to a static one-shot allocation
ablation.
