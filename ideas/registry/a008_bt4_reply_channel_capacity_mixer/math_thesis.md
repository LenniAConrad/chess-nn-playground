# Mathematical Thesis

- Mathematical motivation: This idea is a controlled architecture study, not a
  new primitive. It holds the BT4-style residual tower (stem conv, N residual
  blocks of `mixer -> SqueezeExcite -> +residual -> ReLU`, then value head)
  fixed and swaps only the per-block spatial-mixing operator with the
  `reply_channel_capacity` primitive adapted to the
  `(B, C, 8, 8) -> (B, C, 8, 8)` shape-preserving contract. Mathematically,
  this isolates the change in the function class induced by replacing a pair
  of 3x3 convs with a Blahut-Arimoto channel-capacity solver over a learned
  candidate-to-reply transition table compiled from the 64 squares by `K`
  candidate and `R` reply set-query attention tokens. The RCC per-block
  operator builds a reply-logit table `L in R^{B x K x R}`, defines the soft
  conditional reply distribution `P_{kr} = softmax_r(L_{kr} / tau)`, and
  solves for the capacity-achieving candidate prior `q* in Delta_K` that
  maximizes the mutual information
  `I(candidate; reply) = sum_k q_k sum_r P_{kr} log (P_{kr} / sum_i q_i P_{ir})`
  via Blahut-Arimoto-style softmax iteration
  `marginal_r = sum_k q_k P_{kr}`,
  `per_row_k = sum_r P_{kr} (log P_{kr} - log marginal_r)`,
  `q_new = softmax_k(per_row)`, then scatters `q*` back onto the 64 squares
  via the candidate-compiling attention weights `alpha_k in R^{B x K x 64}`
  so the spatial mix is driven by *how much the candidate choice can
  control the reply distribution* rather than by a scalar reply entropy.
- Assumptions: (i) the BT4 tower depth and width are sufficient to expose the
  mixer as the dominant source of inductive bias; (ii) the learned candidate
  tokens `c_k in R^{B x K x D}` and reply tokens `r_j in R^{B x R x D}` are
  useful learned surrogates for the attacker / reply action spaces
  envisioned by the p003 primitive, even though they are not labelled with
  fixed semantics; (iii) the data is `simple_18` puzzle_binary, so CRTK
  metadata remains reporting-only.
- Claimed advantage: Inside a fixed tower / optimizer / data contract,
  swapping the per-block mixer is the cleanest available test of whether
  the channel-capacity solver is a better token mixer than the baseline
  `conv` (3x3 pair) or `attention` mixers on puzzle_binary, particularly on
  near-puzzle false positives at matched recall where the capacity gap
  `H(r) - H(reply | candidate)` is hypothesised to distinguish robust
  forcing tactics (large gap: candidate choice strongly controls the reply
  distribution) from tempting near-puzzles (collapsed gap: the candidate
  choice barely changes the reply landscape, so even a sharp single-row
  entropy can be a decoy).
- Proof sketch: Because the BT4 block reduces to identity-plus-mixer with
  unit residual, end-to-end gradients on the puzzle BCE loss reach
  `ReplyChannelCapacityMixer` through the same residual paths as the conv
  baseline. The Blahut-Arimoto-style softmax iteration with `tau > 0` and
  softmax-normalised `q` is the standard ascent on the concave mutual-
  information functional `I(q; P)` over the probability simplex `Delta_K`;
  for fixed `P`, repeated `q <- softmax(per_row(P, q))` converges to the
  unique capacity-achieving prior under the standard log-sum-exp
  parameterisation,
  and gradients through the unrolled `iters = 24` steps are well defined
  because every step is a differentiable softmax. As `tau -> 0`, the
  conditional `P_{k.}` collapses to one-hot rows and capacity approaches
  `log min(K, |distinct rows|)`; in the finite-temperature regime used
  here the operator is everywhere differentiable, so the optimiser can
  reach intermediate solutions on the spectrum from "low-temperature,
  near-deterministic capacity" (`tau` small) to "high-entropy mixture"
  (`tau` large) without architectural changes.
- What is actually proven: (a) the model builds, forward-passes, and
  backward-passes on `(B, 18, 8, 8)` simple_18 inputs and emits
  `(B,)` logits suitable for BCE-with-logits; (b) the model is registered
  under `bt4_reply_channel_capacity_mixer` and trainable through the
  shared idea guard. These are checked by the scaffold gate (build +
  forward + backward) and by `tests/test_idea_registry.py`.
- What is only hypothesized: (1) that the channel-capacity solver provides
  a measurable slice-level lift on near-puzzle false positives at matched
  recall vs. the conv / attention baselines; (2) that the lift is driven
  by the channel structure of the candidate -> reply transition table (and
  so collapses to baseline under the `row_shuffle_channel`,
  `duplicate_rows`, and `entropy_only` ablations) rather than by extra
  parameters or a generic bilinear token mixer.
- Failure cases: (a) if the BT4 tower already saturates on the dataset, the
  mixer swap will not produce a measurable lift; (b) if the learned token
  sets `c_k` and `r_j` collapse onto a low-rank subspace, the transition
  table `P` degenerates to near-identical rows and the capacity collapses
  to near zero, so the equilibrium-weighted scatter-back path is
  indistinguishable from an average-pooled mixer; (c) per-square
  attention masks `alpha_k` may concentrate on a small number of squares,
  in which case the mixer's spatial coverage is weaker than a 3x3 conv
  pair.
