# Math Thesis

Oriented Matroid Covector Bottleneck

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`.

Batch candidate rank: `5`.

Working thesis: Puzzle-like positions may be characterized by sign-pattern arrangements of occupied pieces in learned tactical coordinate systems. A covector bottleneck records which side of learned hyperplanes each occupied piece lies on, then pools sign-pattern histograms.

## Formal Setup

Let the position give a set of occupied piece tokens `T = {t_1, ..., t_N}` with `N <= max_pieces`. Each token carries a soft 12-d role distribution `r_n in Delta^{12}`, square coordinates `(rank_n, file_n, color_n) in [-1, 1]^3`, square occupancy, and any auxiliary planes; a small MLP lifts each token to an embedding `e_n in R^{token_dim}`.

Let `H = {(w_p, b_p)}_{p=1}^{P}` be a learned hyperplane arrangement in token space, with `||w_p|| = 1`. The signed projection `s_{n,p} = <w_p, e_n> + b_p` produces the (smooth) oriented-matroid covector entry `sigma_{n,p} = tanh(beta * s_{n,p}) in (-1, 1)`. In the limit `beta -> infty` this recovers the discrete oriented matroid covector `sigma in {-, 0, +}^{N x P}` of the position with respect to the arrangement `H`.

## Bottleneck

The puzzle classifier reads the position only through sign-pattern statistics of `sigma`, never through the tokens or scores directly. Define the masked counts

- `pos_p   = (1/N) sum_n max(sigma_{n,p}, 0)`
- `neg_p   = (1/N) sum_n max(-sigma_{n,p}, 0)`
- `zero_p  = (1/N) sum_n (1 - |sigma_{n,p}|)_+`

the pairwise sign-agreement matrix `A_{p,q} = (1/N) sum_n sigma_{n,p} sigma_{n,q}`, the per-role per-hyperplane sign entropy `H_{r,p} = -sum_{e in {+,-,0}} pi^{(r,p)}_e log pi^{(r,p)}_e / log 3` (with `pi^{(r,p)}_e` re-weighted by `r_n`), the role histogram `rho_r = (1/N) sum_n r_{n,r}`, and per-hyperplane sign / score moments. The covector readout is

```
psi(x) = [pos ; neg ; zero ; vec(A) ; vec(H) ; rho ; sign_mean ; sign_abs_mean ; score_abs_mean ; score_std ; globals]
```

with shape `R^{D}` where `D = 3 P + P^2 + R P + R + 4 P + 8`. The puzzle logit is `f(x) = MLP(LayerNorm(psi(x)))`. The classifier decision flows only through `psi(x)`; raw tokens, embeddings, and per-token scores are never visible to the head, so the sign-pattern bottleneck is architectural.

## Why This Is Not the Shared Probe

The shared `ResearchPacketProbe` reads a CNN trunk over the board planes plus a deterministic profile diagnostic block, which is unrelated to the oriented-matroid covector mathematics. The bespoke architecture replaces the trunk with a piece-token tokenizer plus a learned hyperplane arrangement, then forces the head to decide using only sign-pattern / covector statistics — both the inputs to the head and the structural assumption (sign patterns, not magnitudes or raw embeddings) are different.

## Ablation Modes

The architecture exposes five modes that probe the thesis:

- `covector` — full learned arrangement and full sign-pattern + role-conditioned readout.
- `magnitude_only` — replace `s_{n,p}` with `|s_{n,p}|` and resign `sigma`; removes oriented-matroid sign content while keeping capacity.
- `random_hyperplanes` — swap learned `(w_p, b_p)` for a deterministic random arrangement; tests whether learned tactical orientations matter.
- `material_role_hist_only` — zero out sign / score features, leaving only `rho`; tests whether covector content adds anything beyond material counts.
- `coordinate_shuffle_by_piece` — replace per-token square coordinates with a deterministic role-dependent permutation; tests whether the spatial layout (and therefore any genuinely structural sign-pattern arrangement) is what the covector readout is exploiting.

Each mode is a strict reduction of the bottleneck rather than a separate model, so the empirical comparison cleanly isolates the contribution of *learned*, *signed*, *spatially-structured* covectors.
