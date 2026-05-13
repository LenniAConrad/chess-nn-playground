# Mathematical Thesis

- Mathematical motivation: Many chess tactical positions are encoded not in
  the static piece-value sum but in *pairwise* piece-interactions whose sign
  carries chess meaning. A pin or fork is **super-additive** — both pieces
  must be present for the tactic to fire, so the second-order forward
  difference of any sensible scoring function is positive on that pair. A
  defender / blocker relationship is **sub-additive** — removing either piece
  makes the *other* more critical, so the same second-order difference is
  negative. Aggregating signed pair-Hessian mass over a saliency-selected
  piece set therefore produces a class-separating signal that neither
  first-order saliency nor unsigned interaction magnitude can recover.

- Assumptions:
  1. Let `phi_theta: simple_18 -> R` be a learned scalar scorer.
  2. Piece existence is a Boolean indicator `s_i in {0,1}` for each
     (channel, square) entry in the 12-plane piece occupancy block.
  3. Removing piece `i` zeroes its (channel, square) entry; this is a
     differentiable map on the simple_18 tensor.
  4. The 4-vertex hypercube `{full, full\i, full\j, full\{i,j}}` is
     well-defined for any pair `(i, j)` of distinct piece positions.

- Claimed advantage: For a discrete pair Hessian

      H_ij = phi(P) - phi(P\{i}) - phi(P\{j}) + phi(P\{i, j}),

  `sign(H_ij) = +1` iff `phi` has a super-additive interaction on the pair,
  `sign(H_ij) = -1` iff it has a sub-additive interaction, and
  `H_ij = 0` iff `phi` is additive on the pair. The DHPE primitive
  surfaces this signed quantity to the discriminator MLP. Aggregations
  `z_pos = sum relu(+H_ij)` and `z_neg = sum relu(-H_ij)` partition the
  pair-interaction mass into constructive and substitutive components, and
  the ratio `z_pos / (z_pos + z_neg)` summarises the position's
  constructive vs substitutive character on a fixed [0, 1] scale.

- Proof sketch:
  1. By Newton's discrete forward-difference identity applied to the
     two-variable Boolean cube,
     `Delta_i Delta_j phi = phi(s_i = 1, s_j = 1) - phi(s_i = 0, s_j = 1) -
     phi(s_i = 1, s_j = 0) + phi(s_i = 0, s_j = 0)`,
     which is the H_ij definition above evaluated at the current position.
  2. If `phi(s) = sum_k f_k(s_k) + const` (purely additive), then
     `Delta_i Delta_j phi = 0` for all pairs — the prototype neutral case.
  3. If `phi(s) = sum_k f_k(s_k) + g_{ij}(s_i, s_j)`, then
     `Delta_i Delta_j phi = g_{ij}(1, 1) - g_{ij}(0, 1) - g_{ij}(1, 0) +
     g_{ij}(0, 0)`, i.e. exactly the pair-interaction `b_{ij}` from the
     second-order ANOVA decomposition of phi. The prototype scripts
     confirm this numerically on planted-pin and planted-fork scorers
     (both produce uniformly positive H on the planted pair) and on a
     near-puzzle scorer that fires only when the defender is *absent*
     (which produces a uniformly negative H on defender pairs).

- What is actually proven: The signed Hessian on the 4-vertex hypercube
  exactly recovers the pair-interaction term of the ANOVA decomposition of
  any scalar scorer; the prototype scripts demonstrate this on hand-crafted
  pin / fork / neutral / near-puzzle scorers. The unit tests
  `test_dhpe_signed_hessian_recovers_planted_pair_interaction` and
  `test_dhpe_signed_hessian_negative_for_substitutive_pair` verify the same
  identity for the implemented module.

- What is only hypothesized: That a learned PhiScorer trained on
  puzzle_binary signal alone will learn pair-interaction structure that is
  load-bearing on the equal / hard slices, and that the `unsigned` and
  `shuffled_pairs` ablations will lose most of any equal-slice lift. These
  are scout-scale empirical predictions; this folder is the implementation
  that makes them falsifiable, not a proof that they hold.

- Failure cases:
  - Sign collapse: phi learns to ignore the pairwise interaction term, so
    `z_pos approx z_neg approx 0` and the primitive contribution is
    indistinguishable from noise. The `no_dhpe` ablation is the appropriate
    sanity check.
  - Saliency selection picks the wrong pieces: a fixed deterministic
    piece-value prior may skip a critical pawn or knight that the
    interaction term concentrates on. The `shuffle_singles` ablation tests
    sensitivity to the saliency permutation.
  - Substitutive lift is matched by ordinary unsigned magnitude: the
    `unsigned` ablation matches the full architecture, falsifying the
    central claim that the sign is load-bearing.
