# Math Thesis

Traced Threat Motif Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-28_0857_tuesday_new_york_trace_motif.md`.

Working thesis: Select **Traced Threat Motif Network**.

## Setting

Encode the current board as a finite category whose objects are squares
`V = {1, ..., 64}` and whose typed morphisms `R_g: V -> V` describe the ten
piece-relation channels enumerated below. The model treats the side-to-move
as `u` (us) and the opponent as `t` (them). Each piece geometry contributes
three role-typed relations (`ctrl`, `hit`, `quiet`), giving `K = 36` raw
relation channels per board. These are mixed into ten interpretable groups:

```text
GROUPS = {u_ctrl, u_hit, u_quiet, u_ray, u_jump,
          t_ctrl, t_hit, t_quiet, t_ray, t_jump}
```

Each group `g` is a stochastic operator `A_g in [0, 1]^{64 x 64}` whose row
sums are bounded by one. The interaction between attacker and defender is
captured by composition:

```text
M_{g_1 g_2 ... g_L} = A_{g_1} A_{g_2} ... A_{g_L}.
```

## Threat motifs as words

A *traced threat motif* is a word `w = g_1 ... g_L` over the group alphabet
together with three boundary contractions:

```text
trace(M_w)         = (1 / 64) * tr(M_w)
king(M_w)          = u^T M_w k_t   with u = our piece mass,  k_t = enemy king
value(M_w)         = u^T M_w v_t   with v_t = enemy material value vector
mass(M_w)          = log(1 + sum(M_w))
```

The cardinal motif vocabulary is fixed in the implementation and contains
24 board-relevant words spanning the canonical packet menu (forks, pins,
discovered attacks, sacrifice/decoy patterns, ray-skewers, knight forks,
defended-target strikes, etc.). For a learnable per-square gating
`a_raw[k, i, j]` produced from the board trunk, every raw relation is
filtered by a deterministic geometry mask `M_raw[k, i, j]` that encodes
piece-legal moves, blocking, pawn double-step middle-square clearance, and
between-square ray clearance. The gated raw relations are then mixed
into the ten groups via a softmax over `K = 36` raw channels.

## Contest pullback

For each square `j`, the *contest pressure* is

```text
contest(j) = (sum_i u_ctrl[i, j]) * (sum_i t_ctrl[i, j]),
```

which is the product of inbound friendly and enemy control. Its mean,
top-`k` mean, and entropy form a bounded, board-only diagnostic of which
squares are simultaneously attacked and defended.

## Monoidal closure features

In addition to the 24 word features, four monoidal features expose the
group's loop and parallel structure:

```text
loop2_u   = trace(A_{u_ctrl} A_{u_ctrl})
loop2_t   = trace(A_{t_ctrl} A_{t_ctrl})
parallel  = loop2_u + loop2_t
interact  = trace(A_{u_ctrl} A_{t_ctrl})
```

These quantities are 1-D summary statistics of the matrix algebra spanned
by `(A_{u_ctrl}, A_{t_ctrl})` and capture how often friendly and enemy
control schemes feed back into each other.

## Puzzle decision

The puzzle logit is a learned readout over the concatenation of:

- the convolutional pooled trunk features `(B, 2 d_model)`,
- the 24 motif words evaluated under the four boundary contractions
  (`24 x 4 = 96` motif scalars),
- the four monoidal-closure features,
- the three contest-pressure scalars (mean, top-4 mean, entropy).

This produces one BCE logit per board, faithful to the puzzle_binary
target task. The gated relation tensor, the ten group operators, the
contest heatmap, and the per-motif scores are exposed as diagnostics for
ablation and interpretability runs.
