# Math Thesis

Convex Feasibility Residual Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`.

Batch candidate rank: `3`.

## Working thesis

Puzzle-like positions may be those that lie near the boundary of several learned safe convex feasibility regions in board-feature space. Quiet, non-puzzle positions are expected to fall inside the intersection of those regions; tactical positions should violate at least one constraint and therefore have a non-zero distance to the feasible set. An unrolled projection layer can test whether the *distance to feasibility* is itself a useful classification signal, without relying on any closed-form nuisance residual.

## Notation

Let `x` be the simple_18 board tensor of shape `(B, 18, 8, 8)`, and let `phi: R^{18 x 8 x 8} -> R^d` be a learned board encoder that produces a latent `z = phi(x)` in `R^d`.

The architecture parameterises a bank of `H` half-space constraints

  `c^H_k(z) = <n_k, z> - b_k <= 0`,            (k = 1, ..., H)

with unit normals `n_k` (after normalisation) and softplus-positive offsets `b_k`. It additionally parameterises `Q` ball constraints expressed in low-rank ambient projections,

  `c^B_q(z) = ||P_q z - mu_q||_2 - r_q <= 0`,  (q = 1, ..., Q)

with `P_q in R^{m x d}` (rows unit-normalised), centres `mu_q in R^m`, and softplus-positive radii `r_q`.

The feasibility region of a single position is the intersection

  `K(z) = { z : c^H_k(z) <= 0 for all k, c^B_q(z) <= 0 for all q }`.

## Soft-projection unroll

Closed-form Euclidean projection onto `K` is intractable for a learned bank, so the network performs a finite, differentiable approximation. With temperature `tau > 0` and gate sharpness `gamma`, we use the smoothed hinge

  `h_tau(u) = tau * softplus(u / tau)`,

so that `h_tau(u) -> max(0, u)` as `tau -> 0`. The per-constraint smoothed violation and gate are

  `v^H_k(z) = h_tau(c^H_k(z))`,   `g^H_k(z) = sigmoid(gamma * c^H_k(z))`,
  `v^B_q(z) = h_tau(c^B_q(z))`,   `g^B_q(z) = sigmoid(gamma * c^B_q(z))`.

The total smoothed violation energy is

  `E(z) = sum_k v^H_k(z) + sum_q v^B_q(z)`,

and a (gated) gradient of `E` w.r.t. `z` is given, up to the smoothing temperature, by

  `g(z) = sum_k g^H_k(z) v^H_k(z) n_k
        + sum_q g^B_q(z) v^B_q(z) P_q^T (P_q z - mu_q) / ||P_q z - mu_q||_2`.

The unrolled projection performs `T` steps of

  `z_{t+1} = z_t - eta * g(z_t)`,                (t = 0, ..., T-1; z_0 = z)

with step size `eta`. The terminal `z_T` is interpreted as a soft projection of `z` onto an approximation of `K`, and the *feasibility residual* is the displacement `r(z) = z - z_T` it required.

## Decision rule

The classifier head reads pooled feasibility evidence rather than the raw position. Concretely, the head input is the concatenation

  `psi(z) = [ z ; z_T ; r(z) ; v(z) ; ||z_{t+1} - z_t|| ; m(z) ]`,

where `v(z)` stacks all per-constraint smoothed violations, `||z_{t+1} - z_t||` is the per-step displacement, and `m(z) = (||r||, ||r||^2/d, sum_t ||z_{t+1} - z_t||, max_t ||z_{t+1} - z_t||, mean(v), max(v), mean(v^2), mean(g))` summarises the projection geometry. A small MLP `f: psi(z) -> R` returns the puzzle logit

  `pi(x) = f(psi(phi(x)))`,

trained with binary cross-entropy against the puzzle_binary target `y in {0, 1}` derived from the fine label.

## Predicted geometry

The model is faithful to the working thesis if and only if non-puzzles cluster *inside* the feasible intersection (small `r(z)` and small `E(z)`) while puzzles tend to live *just outside* the boundary (small but nonzero `r(z)`, with a few large per-constraint violations). The diagnostic outputs `feasibility_residual`, `violations`, `feasible_fraction`, and `path_step_norms` are exposed precisely to test this prediction empirically.

## Ablation lattice

The architecture supports five mode codes that share parameter counts where relevant so that any change in metric is attributable to the mechanism, not capacity:

- `projection`: the full unrolled feasibility projection described above.
- `no_projection`: the head reads `z` directly, skipping the projection block.
- `random_constraints`: the half-space and ball banks are replaced by random fixed constraints, isolating whether *learned* feasibility regions matter.
- `linear_head_same_params`: same parameter budget as `no_projection` so that capacity is matched.
- `material_only_encoder`: replaces the convolutional encoder with a simple per-channel material summary, isolating spatial structure from feasibility geometry.
