# Mathematical Thesis

- Mathematical motivation: This idea is a controlled architecture study, not a
  new primitive. It holds the BT4-style residual tower (stem conv, N residual
  blocks of `mixer -> SqueezeExcite -> +residual -> ReLU`, then value head)
  fixed and swaps only the per-block spatial-mixing operator with the
  `pareto_antichain_frontier` primitive adapted to the
  `(B, C, 8, 8) -> (B, C, 8, 8)` shape-preserving contract. Mathematically,
  this isolates the change in the function class induced by replacing a pair
  of 3x3 convs with a learned partial-order (Pareto-frontier) reducer over a
  candidate utility table compiled from the 64 squares by `K` learnable
  set-query attention tokens. The PAFR per-block operator computes
  `s_{ij}^c = sigmoid((U_{ic} - U_{jc} - eps) / tau_dim)`,
  `p_{ij} = prod_c s_{ij}^c`, `log pi_j = sum_{i!=j} log(1 - p_{ij})`, and
  `alpha = softmax((log pi + beta * mean_c(U)) / tau_set)`, then
  scatters the frontier-weighted candidate value summary back onto the
  64 squares using the same per-square attention weights used to compile
  the candidates.
- Assumptions: (i) the BT4 tower depth and width are sufficient to expose the
  mixer as the dominant source of inductive bias; (ii) the learned utility
  channels `U in R^{B x K x C_u}` are a useful learned surrogate for the
  partial-order utilities envisioned by the p001 primitive (forcing claim,
  king exposure, exchange soundness, ...), even though they are not labelled
  with fixed semantics; (iii) the data is `simple_18` puzzle_binary, so CRTK
  metadata remains reporting-only.
- Claimed advantage: Inside a fixed tower / optimizer / data contract,
  swapping the per-block mixer is the cleanest available test of whether the
  partial-order-preserving PAFR operator is a better token mixer than the
  baseline `conv` (3x3 pair) or `attention` mixers on puzzle_binary,
  particularly on near-puzzle false positives at matched recall where the
  partial-order width / entropy signals are hypothesised to distinguish
  one-clean-best-move puzzles from "wide" non-puzzle frontiers.
- Proof sketch: Because the BT4 block reduces to identity-plus-mixer with
  unit residual, end-to-end gradients on the puzzle BCE loss reach
  `ParetoAntichainFrontierMixer` through the same residual paths as the conv
  baseline. As `tau_dim -> 0`, `p_{ij}` converges to the exact non-dominated
  indicator under the product partial order, so the operator's limit form is
  the discrete Pareto-frontier selector. In the finite-`tau_dim` smooth
  regime used here it is everywhere differentiable, so the optimiser can
  reach intermediate solutions on the spectrum from "fully soft scalar
  reduction" (`tau_dim` large) to "hard partial-order frontier"
  (`tau_dim` small) without architectural changes.
- What is actually proven: (a) the model builds, forward-passes, and
  backward-passes on `(B, 18, 8, 8)` simple_18 inputs and emits
  `(B,)` logits suitable for BCE-with-logits; (b) the model is registered
  under `bt4_pareto_antichain_frontier_mixer` and trainable through the
  shared idea guard. These are checked by the scaffold gate (build + forward
  + backward) and by `tests/test_idea_registry.py`.
- What is only hypothesized: (1) that the partial-order reducer provides a
  measurable slice-level lift on near-puzzle false positives at matched
  recall vs. the conv / attention baselines; (2) that the lift is driven by
  the partial-order structure (and so collapses to baseline under the
  `scalar_max` and `shuffle_channels` ablations) rather than by extra
  parameters or a generic set-query attention pool.
- Failure cases: (a) if the BT4 tower already saturates on the dataset, the
  mixer swap will not produce a measurable lift; (b) if the learned utility
  channels are insufficiently decorrelated, the product partial order
  degenerates to a scalar order and PAFR collapses to a soft top-1 token
  mixer; (c) per-square attention masks `alpha_k` may concentrate on a
  small number of squares, in which case the mixer's spatial coverage is
  weaker than a 3x3 conv pair.
