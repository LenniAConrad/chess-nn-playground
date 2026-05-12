# Math Thesis

Near-Puzzle Margin Twin Network.

Source packet:
`ideas/research/packets/classic/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`
(batch candidate rank `3`).

## Setup

The puzzle-binary benchmark draws boards from three latent populations:

- `R` — random non-puzzle boards,
- `N` — near-puzzle hard negatives (positions that look puzzle-like
  but are not actually puzzles),
- `P` — real puzzle boards (positive class).

The trainer's labels collapse `R` and `N` into a single negative
class, but the benchmark's failure mode is exactly the
puzzle-vs-near-puzzle ranking. Any single-latent model that produces
similar features for `N` and `P` cannot be calibrated out at the head.

## Twin objective

Let `b` be a board tensor and `phi(b)` a shared encoder. Two latent
projections share `phi`:

```text
z_ordinary(b) = projector_ordinary(phi(b))   in R^{d_o}
z_tactical(b) = projector_tactical(phi(b))   in R^{d_t}
```

`z_ordinary` is unconstrained and can preserve the surface similarity
between `N` and `P`. `z_tactical` is the *only* representation the
puzzle head reads:

```text
s(b) = head(z_tactical(b))            in R
puzzle_logit(b) = s(b)
```

Training combines BCE on `puzzle_logit` with batch-level pair
margins. With group metadata `g(b)` (`sister_group_id` /
`split_group_id`), for any two boards `b_p in P, b_n in N` with the
same group:

```text
L_margin(b_p, b_n) = relu(m - s(b_p) + s(b_n))
```

This is a hinge ranking loss with margin `m > 0`. It is exactly the
condition `s(P) - s(N) >= m` on the puzzle-evidence latent, which
single-latent BCE never sees because both samples carry label `0` or
`1` rather than a relative order.

For pairs `(b_n in N, b_r in R)` an optional weak ordering or
contrastive term can act in the ordinary latent:

```text
L_neg_order(b_n, b_r) = max(0, m_neg - ||z_ordinary(b_n) - z_ordinary(b_r)||)
```

so `N` and `R` are not forced apart in the ordinary latent, while the
tactical latent still puts `N` below `P`.

## Why two latents

If `phi` is forced to deliver a single readout that is simultaneously

1. close for `N` and `P` on ordinary content (which they share by
   construction), and
2. far for `N` and `P` on puzzle evidence (the ranking we need),

the optimizer compromises and the near-puzzle false-positive rate
saturates near `0.25`, matching the BT4 baseline. Splitting the
readout removes the conflict: `z_ordinary` absorbs the surface
similarity, `z_tactical` carries the ranking signal, and the puzzle
head only sees `z_tactical`.

The model never reads group metadata at inference; the margin terms
are an objective the trainer attaches when group metadata is reliable.
The architecture's job is to expose `z_ordinary`, `z_tactical`, and
`puzzle_margin_signal` so those terms can be wired in.
