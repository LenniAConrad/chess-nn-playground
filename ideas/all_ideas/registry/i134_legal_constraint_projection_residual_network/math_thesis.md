# Math Thesis

Legal-Constraint Projection Residual Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.

Batch candidate rank: `4`.

Working thesis: Even when the input board is legal, a learned latent
explanation of "why this is puzzle-like" may produce soft piece/square
beliefs that violate basic legal-board constraints. Projecting those beliefs
back onto a soft legal-board constraint set and reading the residual
``R = Y - Y_proj`` may expose tactical contradiction or ambiguity.

## Soft latent board belief

A small CNN trunk produces a per-square soft belief `Y in (B, 13, 8, 8)`
over the twelve piece classes `(P, N, B, R, Q, K, p, n, b, r, q, k)` plus an
explicit "empty" class. By construction `Y(b, :, r, f)` lies on the 12-
simplex via softmax.

## Soft legal-board constraint set `C`

`C` is the intersection of four convex constraint families, each depending
only on the current board:

```
C_simplex = { Y : Y_{b,c,r,f} >= 0,  sum_c Y_{b,c,r,f} = 1 }
C_count   = { Y : sum_{r,f} Y_{b,k,r,f} <= cap_k for every piece class k }
C_king    = { Y : sum_{r,f} Y_{b,K,r,f} <= 1,   sum_{r,f} Y_{b,k,r,f} <= 1 }
C_pawn    = { Y : Y_{b,P,r,f} = 0 if r in {0,7},  Y_{b,p,r,f} = 0 if r in {0,7} }
```

with `cap_K = cap_k = 1`, `cap_Q = cap_q = 9`, `cap_R = cap_r = 10`,
`cap_B = cap_b = 10`, `cap_N = cap_n = 10`, `cap_P = cap_p = 8` reflecting
the maximum legal occupancy under promotion-aware chess rules. The
constraints `C_count`, `C_king`, and `C_pawn` are linear half-spaces on the
class-count statistics, and `C_simplex` is the per-square simplex.

## Approximate projection

The exact projection
``Y_proj = argmin_Z ||Z - Y||_2^2  s.t.  Z in C``
has no closed form, but each individual constraint family admits a cheap
projection:

```
P_simplex(Y)   :  per-square clamp-to-non-negative + renormalise
P_count(Y)     :  scale piece-class counts down by min(1, cap_k / count_k); excess -> empty
P_king(Y)      :  scale king plane to sum at most 1; excess -> empty
P_pawn(Y)      :  zero pawn classes on ranks 0 and 7; mass -> empty
```

The model approximates the joint projection by a few alternating sweeps
over `(P_count, P_king, P_pawn, P_simplex)`. After each sub-step we record
the per-batch residual energy
``e_*(Y) = || Y - P_*(Y) ||_F^2``
to expose how much each constraint family was violated.

## Residual classifier

The projection residual ``R = Y - Y_proj`` is the central feature. A linear
read-out concatenates:

- per-class residual L2 norms `|| R_{:, c, :, :} ||_2` (size 13);
- per-constraint residual energies `[e_simplex, e_count, e_king, e_pawn]`;
- a small CNN summary of the spatial residual map;
- pooled encoder features (mean + max);
- the total residual norm and a per-square belief entropy scalar.

A two-layer GELU MLP maps this to one ``puzzle_binary`` logit. The
hypothesis is that puzzle-like positions correspond to higher residual mass
and more localised violations, while non-puzzle positions push `Y` close to
`C` so the residual stays small.
