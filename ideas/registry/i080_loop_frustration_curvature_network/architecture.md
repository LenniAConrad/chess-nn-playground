# Architecture

`Loop-Frustration Curvature Network` (LFCN) is a board-only
`puzzle_binary` classifier that follows the markdown thesis from
`ideas/research/packets/classic/chess_nn_research_2026-04-28_0729_tuesday_new_york_frustration_curvature.md`.
It tests the packet's claim that true puzzles induce a sharply
concentrated frustrated spin-glass response under temperature
perturbation, while near-puzzles do not, by computing a closed-loop
free-energy curvature observable directly on the current board.

## Mechanism

1. **Board encoder.** A compact `Conv1x1 -> Conv3x3 -> Conv3x3` stack
   with `GELU` activations and a final `LayerNorm` over the 64 sites
   produces site embeddings `g_i(x) ∈ R^{F}` from the `simple_18`
   board tensor (packet section 9). The encoder only parameterises
   spin-glass fields and couplings; no global pooling of `g` is fed
   to the head.
2. **Site spin field.** A `Conv1x1` head followed by `tanh` produces
   the Edwards-Anderson spin field `m_{i,k}(x) ∈ [-1, 1]` of shape
   `(B, K, 8, 8)` for `K` replicas.
3. **Static spin-glass graph.** A non-trainable buffer holds the
   packet's `M = 210` undirected edges (56 horizontal, 56 vertical,
   49 down-right diagonal, 49 down-left diagonal) and a static loop
   bank of `L = 520` cycles built deterministically by
   `build_loop_bank()`: 324 orthogonal rectangle boundaries with
   sides in `{1, 2, 3}` plus 196 unit-square corner triangles.
   Padding length is `Lmax = Vmax = 12`.
4. **Edge couplings `J`.** Per-edge pair features
   `[g_i, g_j, |g_i - g_j|, g_i * g_j, edge_type_emb]` flow through
   a 2-layer MLP that outputs `K` replica logits per edge. The
   couplings are bounded by `J = 2.5 * tanh(J_raw / 2.5)` so that
   `tanh(beta * J)` does not saturate at initialisation.
5. **Inverse temperature.** `beta = 0.20 + softplus(raw_beta)` is
   clamped at `beta_max = 3.0`. Finite-difference points use
   `beta_minus = clamp_min(beta - delta, 0.05)`, `beta_mid = beta`,
   `beta_plus = beta + delta` with `delta = 0.125`.
6. **Loop products.** For each loop `ell` and replica `k`,
   `P_{ell,k}(beta, x) = prod_{e ∈ ell} tanh(beta * J_{e,k})`
   is computed in log-stable form via `sign.prod * exp(log|.|.sum)`
   with `mask`-aware aggregation, then clamped into
   `(-1 + 1e-6, 1 - 1e-6)` to keep `log(1 + eta * P)` finite.
7. **Loop free energy and curvature.**
   `A_{ell,k}(beta, x) = log(1 + eta * P_{ell,k}(beta, x))` with
   `eta = 0.90`. The centered finite-difference curvature is
   `D2A = (A(beta + delta) - 2 A(beta) + A(beta - delta)) / delta^2`.
8. **Physical observable.**
   `Omega_{ell,k}(x) = sigmoid(-nu * P_mid) * |D2A|` with
   `nu = 4.0`. The sigmoid weight emphasises negative loop products
   (frustrated cycles) and `|D2A|` measures thermodynamic response.
9. **Loop-to-site scatter.** Each loop's `Omega` value is divided
   by its vertex count and added to every participating site,
   producing `Omega_site ∈ R^{B x K x 8 x 8}` without using search,
   move generation, or engine evaluations.
10. **Observable head.** Only physics-derived statistics are fed to
    the classifier: `[mean, std, top8 mean, max, frustration_rate,
    top8/(|mean|+eps), EA_order]` per replica, concatenated into
    a `(B, 7K)` vector. A `Linear(7K, 32) -> GELU -> Dropout(0.1)
    -> Linear(32, 1)` MLP returns the puzzle logit.

## Output Contract

Forward returns a dict whose `"logits"` entry is `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. Diagnostic
tensors appended to prediction artefacts:

- `J`: `(B, K, M)` learned edge couplings (after the `2.5 *
  tanh(./2.5)` clamp).
- `loop_product_mid`: `(B, K, L)` `P_{ell,k}(beta, x)` at the centre
  inverse temperature.
- `loop_curvature`: `(B, K, L)` finite-difference `D2A`.
- `loop_omega`: `(B, K, L)` physical observable Omega.
- `omega_site`: `(B, K, 8, 8)` scattered saliency map.
- `site_spin`: `(B, K, 8, 8)` Edwards-Anderson spin field `m`.
- `observables`: `(B, 7K)` board-level statistics fed to the head.
- `beta`: `(B,)` learned inverse temperature broadcast across the
  batch.
- `frustration_rate`: `(B, K)` `mean sigmoid(-nu * P_mid)`.
- `omega_concentration`: `(B, K)` `top8 / (|mean| + eps)`.
- `ea_order`: `(B, K)` `mean(m^2) - mean(m)^2` over squares.
- `ablation_*`: per-batch indicator flags consumed by the packet's
  diagnostic table.

## Ablations

The bespoke builder accepts `model.ablation` in
`{"none", "no_loop_product", "cycle_scramble", "no_curvature",
"no_frustration_weighting", "fixed_beta", "single_replica",
"rectangles_only", "triangles_only"}`, matching the packet's
section-11 falsification controls:

- `no_loop_product` (LFCN-NoLoopProduct): replace each loop product
  with `mean_e |tanh(beta * J_e,k)|`, preserving edge magnitudes
  while destroying signed closed-loop frustration.
- `cycle_scramble` (LFCN-CycleScramble): apply a fixed random
  permutation to the first edge slot of every loop, destroying real
  closed-loop topology while keeping module shapes, parameter count,
  and loop lengths.
- `no_curvature`, `no_frustration_weighting`, `fixed_beta`,
  `single_replica`, `rectangles_only`, `triangles_only`: the packet's
  additional ablations that disable individual mechanism components.

## Implementation Binding

- Registered model name: `loop_frustration_curvature_network`.
- Source implementation file: `src/chess_nn_playground/models/loop_frustration_curvature_network.py`.
- Idea-local wrapper: `ideas/registry/i080_loop_frustration_curvature_network/model.py`.
