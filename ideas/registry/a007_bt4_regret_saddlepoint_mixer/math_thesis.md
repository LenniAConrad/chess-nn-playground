# Mathematical Thesis

- Mathematical motivation: This idea is a controlled architecture study, not a
  new primitive. It holds the BT4-style residual tower (stem conv, N residual
  blocks of `mixer -> SqueezeExcite -> +residual -> ReLU`, then value head)
  fixed and swaps only the per-block spatial-mixing operator with the
  `regret_saddlepoint` primitive adapted to the
  `(B, C, 8, 8) -> (B, C, 8, 8)` shape-preserving contract. Mathematically,
  this isolates the change in the function class induced by replacing a pair
  of 3x3 convs with an entropy-regularized zero-sum saddle solver over a
  bilinear payoff table compiled from the 64 squares by `K` attacker-candidate
  and `R` defender-reply set-query attention tokens. The RSP per-block
  operator solves
  `max_p min_q  p^T A q + tau_p H(p) - tau_q H(q)` for `p in Delta_K`,
  `q in Delta_R` by damped fixed-point iteration
  `p_new = softmax(A q / tau_p)`,
  `q_new = softmax(-p_new^T A / tau_q)`,
  `(p, q) <- (1 - damp)(p, q) + damp(p_new, q_new)`, then scatters the
  attacker equilibrium `p` back onto the 64 squares via the candidate-
  compiling attention `alpha_k in R^{B x K x 64}` so the spatial mix is
  driven by *which candidates survive the defender's best response* rather
  than by a scalar score.
- Assumptions: (i) the BT4 tower depth and width are sufficient to expose the
  mixer as the dominant source of inductive bias; (ii) the learned candidate
  tokens `c_k in R^{B x K x D}` and reply tokens `r_j in R^{B x R x D}` are
  useful learned surrogates for the attacker / defender action spaces
  envisioned by the p002 primitive, even though they are not labelled with
  fixed semantics; (iii) the data is `simple_18` puzzle_binary, so CRTK
  metadata remains reporting-only.
- Claimed advantage: Inside a fixed tower / optimizer / data contract,
  swapping the per-block mixer is the cleanest available test of whether the
  entropy-regularized saddle solver is a better token mixer than the
  baseline `conv` (3x3 pair) or `attention` mixers on puzzle_binary,
  particularly on near-puzzle false positives at matched recall where the
  exploitability scalar `attacker_regret + defender_regret` is hypothesised
  to distinguish robust-forcing true puzzles (high `value`, low
  exploitability) from tempting near-puzzles (high attacker claim, low
  saddle value because a single defender column refutes the forcing row).
- Proof sketch: Because the BT4 block reduces to identity-plus-mixer with
  unit residual, end-to-end gradients on the puzzle BCE loss reach
  `RegretSaddlepointMixer` through the same residual paths as the conv
  baseline. The damped softmax fixed-point iteration with `tau_p`, `tau_q > 0`
  is a contraction in the entropy-regularized regime, so the unrolled
  `iters = 24` steps converge to a stationary distribution that is unique
  given `A`; gradients through the unrolled solver are well defined because
  every step is a differentiable softmax. As `tau_p, tau_q -> 0`, the
  equilibrium converges to the pure max-min saddle `max_i min_j A_ij`; in
  the finite-temperature regime used here the operator is everywhere
  differentiable, so the optimiser can reach intermediate solutions on the
  spectrum from "fully soft uniform mixture" (`tau` large) to "hard
  pure-strategy saddle" (`tau` small) without architectural changes.
- What is actually proven: (a) the model builds, forward-passes, and
  backward-passes on `(B, 18, 8, 8)` simple_18 inputs and emits
  `(B,)` logits suitable for BCE-with-logits; (b) the model is registered
  under `bt4_regret_saddlepoint_mixer` and trainable through the shared idea
  guard. These are checked by the scaffold gate (build + forward + backward)
  and by `tests/test_idea_registry.py`.
- What is only hypothesized: (1) that the entropy-regularized saddle solver
  provides a measurable slice-level lift on near-puzzle false positives at
  matched recall vs. the conv / attention baselines; (2) that the lift is
  driven by the game-structure of the payoff table (and so collapses to
  baseline under the `row_shuffle_payoff`, `col_shuffle_payoff`, and
  `uniform_payoff` ablations) rather than by extra parameters or a generic
  bilinear token mixer.
- Failure cases: (a) if the BT4 tower already saturates on the dataset, the
  mixer swap will not produce a measurable lift; (b) if the learned token
  sets `c_k` and `r_j` collapse onto a low-rank subspace, the payoff table
  `A` degenerates and the saddle collapses to a near-uniform mixture
  (high exploitability) so the equilibrium-weighted scatter-back path is
  indistinguishable from an average-pooled mixer; (c) per-square attention
  masks `alpha_k` may concentrate on a small number of squares, in which
  case the mixer's spatial coverage is weaker than a 3x3 conv pair.
