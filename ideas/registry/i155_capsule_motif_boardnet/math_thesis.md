# Math Thesis

Capsule Motif BoardNet

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.

Batch candidate rank: `1`.

Working thesis: Local chess motifs are not only scalar activations;
they have type, pose, orientation, and part-whole relationships. A
capsule-style model can encode local patterns as small vectors and
route them into higher-level tactical motif capsules by agreement.

Concretely, after a small convolutional trunk with rank/file
coordinate channels, primary capsules are extracted at every square:

```
u[b, i] in R^{D_caps},  i = 1, ..., N_caps,  N_caps = 8 * 8 * C_primary
```

with `u[b, i] = squash(Conv3x3(trunk)[b, i])` so each primary capsule
lives on a bounded manifold. A shared tensor `W in R^{M x D_motif x
D_caps}` defines, for each motif `m = 1, ..., M`, a transformation
matrix `W_m`, and motif predictions are computed by

```
u_hat[b, i, m] = W_m u[b, i].
```

Routing-by-agreement performs `T` iterations starting from `b[b, i, m]
= 0`:

```
c[b, i, m] = softmax_m(b[b, i, m])
s[b, m]    = sum_i c[b, i, m] u_hat[b, i, m]
v[b, m]    = squash(s[b, m])
b[b, i, m] += <u_hat[b, i, m], detach(v[b, m])>   (all but last step)
```

where `squash(s) = ||s||^2 / (1 + ||s||^2) * s / ||s||` keeps motif
activations bounded. Motif capsule lengths `||v_m||` together with
pooled trunk features feed a small MLP that emits one puzzle logit.

The architecture is *not* attention (the routing weights are produced
by iterative inner-product agreement, not by query-key softmax over
values), *not* a prototype dictionary (capsules carry vector pose,
not nearest-prototype scores), and *not* a graph or sheaf
construction. Whether routing-by-agreement improves the
puzzle_binary contract over a parameter-matched plain CNN is the
empirical question this idea tests.
