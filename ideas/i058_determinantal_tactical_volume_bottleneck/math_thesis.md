# Math Thesis

Determinantal Tactical Volume Bottleneck

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2044_friday_shanghai_determinantal_volume.md`.

Working thesis: puzzle-like positions may differ from non-puzzles by how the
occupied pieces collapse into, or span, learned tactical role subspaces. The
test statistic is the log-volume of role-gated PSD Gram matrices over the set
of occupied tokens, which is permutation-invariant and reduces to a single
local CNN filter only in the degenerate diagonal-only case.

For a current-board tensor `x`, let `S(x) = {(t_i, s_i)}_{i=1}^N` be the
occupied set, encoded into token embeddings `phi_i in R^d`. For each role
`r`, a non-negative gate `g_{r,i}` and projector `A_r in R^{d x q}` define

```
K_r(x) = D_r Phi A_r A_r^T Phi^T D_r + eps * I_N,
D_r    = diag(sqrt(g_{r,1}), ..., sqrt(g_{r,N})),
V_r(x) = log det K_r(x).
```

`V_r` is invariant under permutations of the occupied tokens
(`det(P K_r P^T) = det(K_r)`) and depends only on the spectrum of the role-
gated covariance, so it cannot reduce to a single local convolutional filter.

Implementation falsifier: the diagonal-only ablation replaces `log det K_r`
with the gated trace, preserving gates and norms but removing all
off-diagonal interaction; section 9 of the packet predicts that, if puzzle-
likeness is captured by the diagonal-only signal alone, the determinant
bottleneck is unnecessary.
