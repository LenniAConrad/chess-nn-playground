# Math Thesis

Zobrist Kernel Feature Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.

Batch candidate rank: `5`.

## Thesis

Classical Zobrist hashing assigns a random bitstring `z(p, s)` to each
`(piece-class, square)` pair and XORs them together for every occupied
`(piece-class, square)` to produce a compact fingerprint of the board.  The
fingerprint is stable under the order in which pieces are added and changes by
a single XOR per piece move, which is why it is the standard board-key in
chess engines.

This idea uses Zobrist hashing as the basis of a kernel feature map.  Replace
XOR with addition so the operation is differentiable, draw `M` independent
random sign matrices `Z_m \in \{-1, +1\}^{12 \times 64 \times D}`, and define
the per-bank fingerprint of a board ``B`` with piece-square occupancy
``O[p, s] \in \{0, 1\}`` as

```
s_m(B) = sum_{p, s} O[p, s] * Z_m[p, s]    in R^D.
```

Two boards that share many of the same `(piece, square)` entries collide on
many of the same code rows, so `s_m(B)` is a low-variance estimator of the
piece-square occupancy of `B` even with very few entries.  In expectation
`<s_m(B), s_m(B')> = D * |occ(B) cap occ(B')|` where `occ` is the set of
occupied piece-square pairs and the codes are iid `\pm 1`, which is exactly
the linear "intersection kernel" over the occupancy set.

To get a richer kernel we lift each bank's fingerprint through a random
Fourier feature map.  Draw `W_m \sim N(0, 1/D)` and `b_m \sim U[0, 2\pi)` and
define

```
phi_m(B) = (1 / sqrt(D)) * [cos(W_m s_m(B) + b_m), sin(W_m s_m(B) + b_m)].
```

The Bochner / Rahimi-Recht random Fourier features theorem then says
`<phi_m(B), phi_m(B')> ≈ k(s_m(B), s_m(B'))` where `k` is the RBF kernel whose
inverse bandwidth is set by the variance of `W_m`.  Concatenating `M`
independent banks reduces the variance of this kernel approximation by a
factor of `M`.

Only a small classifier MLP `h_θ` reads the concatenated kernel features
`Phi(B) = [phi_1(B), ..., phi_M(B)]` plus a few diagnostic norms and emits one
``puzzle_binary`` logit.  The Zobrist banks `Z_m`, the random projections
`W_m`, and the phase biases `b_m` are all fixed buffers; no learned parameter
is allowed to look at the board directly.  Generalisation comes entirely from
how stable the Zobrist kernel is across positions that share occupancy
structure.
