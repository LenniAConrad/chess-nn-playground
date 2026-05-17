# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `delta_pair_accumulator` (DPA) primitive from
  `p014_delta_pair_accumulator`. Source primitive math:
  `ideas/registry/p014_delta_pair_accumulator/math_thesis.md`.

- Assumptions:
  1. The DPA primitive is well-defined as a shape-preserving operator
     `(B, C, 8, 8) -> (B, C, 8, 8)` under the
     `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract.
  2. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and across
     the `conv` and `attention` baselines.
  3. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is the
     mixer.

- Claimed advantage: If the DPA primitive carries a spatial mixing
  signal that conv and attention do not, dropping it into the BT4 block
  must lift held-out PR AUC (aggregate or on a target slice) versus the
  two baselines under the same tower, optimizer, and data. This is a
  controlled architecture-level test of "is DPA a better spatial mixer
  than conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the BT4
  block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for DPA itself (a first-order accumulator term
  `sum_i u_i` plus a pair term over the rule-derived alignment edge set
  `E(S) subset S x S` defined by same-rank, same-file, or same-diagonal
  square pairs, with a low-rank bilinear per-edge message conditioned
  on the (rank_diff, file_diff) delta and degree-normalised by aligned
  in-degree) is established in the source primitive's math thesis and
  falsified by its own ablation grid. The static-position adapter here
  evaluates the analytical fixed point and exposes it as a token mixer
  that couples every aligned pair of squares.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. The alignment mask `E(S)` is a fixed
  precomputed buffer over the 8x8 geometry; the only cross-square
  interaction the mixer performs is the first-order accumulator
  broadcast plus the alignment-masked pair message, both of which
  factor through `O(|S| + |E(S)| * k)` rather than `O(|S|^2 * k)`. A
  forward + backward smoke test guards the mixer at registration time.

- What is only hypothesized: That replacing the conv mixer with the
  DPA mixer lifts PR AUC on at least one CRTK slice (most likely
  slices where rook/bishop-style line geometry is load-bearing, such
  as middlegame open-file motifs and `crtk_phase = middlegame` /
  endgame slices) without regressing aggregate PR AUC by more than the
  matched-baseline tolerance.

- Failure cases:
  - The conv baseline already learns enough alignment structure
    through stacked 3x3 receptive fields, and the rule-derived edge
    mask provides no additional signal once the stem conv has
    compressed the simple_18 board into `C` channels.
  - The alignment mask is too uniform: every square pair on a rank,
    file, or diagonal carries equal weight before the bilinear
    bottleneck, so the all-aligned-pairs message collapses to a
    direction-marginalised summary that an `attention` baseline
    recovers more efficiently with input-dependent weights.
  - The factorisation-machine compromise (no `W_{type(i),type(j)}`
    piece-type pair table because `C` has no piece semantics) strips
    the source primitive of its most discriminative feature, leaving
    the mixer with strictly less inductive bias than the source
    primitive's head-form.
  - The make/unmake delta-stream variant of DPA that gives the source
    primitive its inference-time cost advantage has no static-batch
    analogue and is not attempted here; a null at training time is
    not a falsifier for that variant.
