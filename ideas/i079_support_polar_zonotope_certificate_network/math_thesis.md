# Math Thesis

Support-Polar Zonotope Certificate Network (SPZC-Net)

Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0718_tuesday_new_york_support_polar_zonotope.md`.

## Working thesis

Fine-label `2` (puzzle) positions should be detectable as **latent
convex protrusions**: a small number of square-pair relations create
extreme support in one or more learned directions, while fine-label
`0` and `1` positions stay inside a learned symmetric polar body.

For each board, the model builds 64 square tokens and an ordered-pair
generator family `g_{ij}(x) = a_{ij}(x) * phi(t_i, t_j, rho_{ij})` and
defines the **board zonotope**

    Z_x = c_x + sum_{i != j} alpha_{ij} g_{ij}(x),  alpha_{ij} in [-1, 1].

Its support function is closed form:

    h_{Z_x}(u) = <u, c_x> + sum_{i != j} | <u, g_{ij}(x)> |.

The model also learns `K` directions `u_k` and positive thresholds
`beta_k`, which together span the symmetric polar body
`Q = { z : | <u_k, z> | <= beta_k }`. SPZC-Net classifies the board by
the largest containment violation

    r(x) = max_{k, sigma in {-1, +1}} ( h_{Z_x}( sigma u_k ) - beta_k ).

The puzzle logit is `scale(x) * r(x) + bias` with
`scale = softplus(raw_scale)`, so the head is a calibrated *monotone*
function of the residual: larger violations cannot decrease the
positive logit. The architecture also exposes the **certificate**
`(u_k, sigma, beta_k)` and the per-pair projections `<u_k, g_{ij}>`,
so each prediction is auditable.

## Falsifiability

The thesis fails if any of the following hold (research packet
section 12):

1. SPZC-Net does not beat the strongest current-board-only baseline by
   at least **1.5 AUROC points** or **2.0 PR-AUC points** averaged
   across three seeds.
2. A generic same-budget token baseline matches SPZC-Net within
   **0.5 AUROC points**, indicating the convex object adds no value.
3. Fine-label diagnostic shows label `2` recall is not improved at
   matched false-positive rate.
4. Learned `beta_k` collapse to nearly identical thresholds and
   directions have high cosine similarity, indicating unused polar
   geometry.
5. Top certificate directions vary wildly under small legal
   board-preserving augmentations.
6. Removing zonotope width has no measurable effect, falsifying the
   sparse-extreme-interaction thesis.
7. Calibration worsens materially relative to the baseline with no
   compensating AUROC / PR-AUC gain.
8. Apparent gain disappears when forbidden non-board fields are
   removed from the data loader and run artifacts.
