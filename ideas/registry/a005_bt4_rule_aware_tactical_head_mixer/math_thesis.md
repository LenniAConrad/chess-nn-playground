# Mathematical Thesis

- Mathematical motivation: This idea is a controlled architecture study, not a
  new primitive. It holds the BT4-style residual tower (stem conv, N residual
  blocks of `mixer -> SqueezeExcite -> +residual -> ReLU`, then value head)
  fixed and swaps only the per-block spatial-mixing operator with the
  `rule_aware_tactical_head` primitive adapted to the
  `(B, C, 8, 8) -> (B, C, 8, 8)` shape-preserving contract. Mathematically,
  this isolates the change in the function class induced by replacing a pair
  of 3x3 convs with a directional ray-scan + forcing-feature + gated-fusion
  operator. Under the gated additive fusion `y = base_mix(x) + sigmoid(gate(x)) * delta(forcing(x))`,
  the mixer adds at most the rank of the forcing field to the per-block
  spatial response and reduces to `base_mix(x)` when the gate saturates to 0.
- Assumptions: (i) the BT4 tower depth and width are sufficient to expose the
  mixer as the dominant source of inductive bias; (ii) ray-geometry features
  in `(B, C, 8, 8)` activations are an adequate learned surrogate for the
  TSDP primitive's rule-exact (check, capture, promotion, ...) indicators,
  which are not differentiably computable from an unlabelled activation
  tensor; (iii) the data is `simple_18` puzzle_binary, so CRTK metadata
  remains reporting-only.
- Claimed advantage: Inside a fixed tower / optimizer / data contract,
  swapping the per-block mixer is the cleanest available test of whether the
  TSDP-style forcing-aware spatial mixer is a better token mixer than the
  baseline `conv` (3x3 pair) or `attention` mixers on puzzle_binary, and
  particularly on tactical motifs where the forcing-feature gate is expected
  to fire (`mate_in_1`, `forcing_density` high).
- Proof sketch: Because the BT4 block reduces to identity-plus-mixer with
  unit residual, end-to-end gradients on the puzzle BCE loss reach
  `RuleAwareTacticalMixer` through the same residual paths as the conv
  baseline. The directional depthwise convs are initialised to read the
  single neighbour cell in each of the 8 (rook + bishop) directions, giving
  the mixer the ability to express any local single-step propagation pattern
  used by the rook/bishop attack rays. The gated additive fusion guarantees
  that the mixer can always recover `base_mix(x)` by driving the gate to 0,
  so the mixer's function class is a strict superset of a pure 3x3 conv
  mixer up to optimisation noise.
- What is actually proven: (a) the model builds, forward-passes, and
  backward-passes on `(B, 18, 8, 8)` simple_18 inputs and emits
  `(B,)` logits suitable for BCE-with-logits; (b) the model is registered
  under `bt4_rule_aware_tactical_head_mixer` and trainable through the
  shared idea guard. These are checked by the scaffold gate (build + forward
  + backward) and by `tests/test_idea_registry.py`.
- What is only hypothesized: (1) that the forcing-aware gate provides a
  measurable slice-level lift on tactical motifs vs. the conv / attention
  baselines; (2) that the lift survives the `shuffle_tsdp` ablation only
  partially, exposing genuine reliance on the ray-geometry channels.
- Failure cases: (a) if the BT4 tower already saturates on the dataset, the
  mixer swap will not produce a measurable lift; (b) on near-puzzle
  positions that share forcing motifs with true puzzles, the mixer may
  inflate the false-positive rate; (c) the differentiable ray surrogate is
  strictly weaker than the rule-exact TSDP oracle, so cases that depend on
  legal-move enumeration (e.g. stalemate-vs-checkmate discrimination) are
  expected to fail.
