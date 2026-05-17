# Mathematical Thesis

- Mathematical motivation: This idea is a controlled architecture study, not a
  new primitive. It holds the BT4-style residual tower (stem conv, N residual
  blocks of `mixer -> SqueezeExcite -> +residual -> ReLU`, then value head)
  fixed and swaps only the per-block spatial-mixing operator with the
  `signed_edit_bilinear_memory` (SEBM) primitive adapted to the
  `(B, C, 8, 8) -> (B, C, 8, 8)` shape-preserving contract. Mathematically,
  this isolates the change in the function class induced by replacing a pair
  of 3x3 convs with the SEBM state-triple `(s, u, p)` where
  `s = sum_j a_j`, `u = sum_j b_j`, and
  `p = s (.) u - sum_j a_j (.) b_j` is the factorisation-machine pair
  identity over the 64 board squares treated as the active feature set, with
  per-square `a_j = A x_j` and `b_j = B x_j` learned projections.
- Assumptions: (i) the BT4 tower depth and width are sufficient to expose the
  mixer as the dominant source of inductive bias; (ii) the 64 squares are an
  adequate static-position surrogate for the source primitive's active piece-
  square feature set (a square's vector replaces the source primitive's
  per-feature embedding); (iii) broadcasting the global SEBM memory
  `(s, u, p)` back to every square via a FiLM-style modulation is a faithful
  per-token readout of an operator that originally produced a single
  `(B, 3r)` vector with no per-square output; (iv) the data is `simple_18`
  puzzle_binary so CRTK metadata remains reporting-only.
- Claimed advantage: Inside a fixed tower / optimizer / data contract,
  swapping the per-block mixer is the cleanest available test of whether the
  SEBM-style bilinear pair-state spatial mixer is a better token mixer than
  the baseline `conv` (3x3 pair) or `attention` mixers on puzzle_binary, and
  particularly on slices where attacker/defender, blocker/slider, or
  king-ring/intruder pair interactions are load-bearing (the source primitive
  was designed to summarise those pair interactions in a single rank-r
  bilinear summary).
- Proof sketch: Because the BT4 block reduces to identity-plus-mixer with
  unit residual, end-to-end gradients on the puzzle BCE loss reach
  `SignedEditBilinearMemoryMixer` through the same residual paths as the conv
  baseline. The pair-state identity
  `p = s (.) u - sum_j a_j (.) b_j` is the standard factorisation-machine
  cross-term over the 64 tokens and so couples every pair of squares without
  paying the all-pairs cost; the FiLM broadcast then redistributes the global
  pair memory to per-square outputs while preserving the source primitive's
  load-bearing algebra. Setting the FiLM `gamma`/`beta` outputs to 0 makes
  the mixer reduce to a per-square linear readout of the global memory, so
  the operator's function class strictly extends that of a learnable
  global-pool mixer.
- What is actually proven: (a) the model builds, forward-passes, and
  backward-passes on `(B, 18, 8, 8)` simple_18 inputs and emits `(B,)`
  logits suitable for BCE-with-logits; (b) the model is registered under
  `bt4_signed_edit_bilinear_memory_mixer` and trainable through the shared
  idea guard. These are checked by the scaffold gate (build + forward +
  backward) and by `tests/test_idea_registry.py`.
- What is only hypothesized: (1) that the SEBM pair-state identity provides
  a measurable slice-level lift over the conv / attention baselines on slices
  where pair interactions dominate; (2) that the lift is mechanistic rather
  than parameter-count driven, surviving the `shuffle_pair_state` ablation
  only partially; (3) that the FiLM broadcast is a faithful per-token readout
  of the source primitive's set-level summary rather than a discarded
  adaptation layer.
- Failure cases: (a) if the BT4 tower already saturates on the dataset, the
  mixer swap will not produce a measurable lift; (b) on near-puzzle positions
  that share the same first- and second-order moment structure as true
  puzzles, the bilinear summary may collapse the discriminative signal and
  inflate the false-positive rate; (c) the static 64-token board surrogate is
  strictly weaker than the source primitive's signed-edit O(|Delta|) update
  path, so engine-wrapped make/unmake gains (if any) cannot be measured in
  this static-position training harness.
