# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `reversible_delta_kernel_memory` primitive from
  `p019_reversible_delta_kernel_memory`. Source primitive math:
  `ideas/registry/p019_reversible_delta_kernel_memory/math_thesis.md`.
  The primitive forms a linear-attention-style unordered-set memory
  `M = sum_i phi(k_i) nu(v_i)^T`, `z = sum_i phi(k_i)` with
  `phi(.) = elu(.) + 1`, and each query reads
  `y_q = phi(q)^T M / (phi(q)^T z + eps)`. Inside the BT4 block this
  reduces to an all-pairs spatial mixer factorised through a global
  kernel-memory pair rather than an explicit 64x64 attention map.

- Assumptions:
  1. The `reversible_delta_kernel_memory` primitive is well-defined as a
     shape-preserving operator
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
  4. All 64 squares are active tokens in the spatial-mixer adaptation
     (the channel tensor has no occupancy mask); the unordered-set
     semantics of the primitive are preserved because `M` and `z` are
     still plain sums over the token set.

- Claimed advantage: If the `reversible_delta_kernel_memory` primitive
  carries a spatial mixing signal that conv and attention do not,
  dropping it into the BT4 block must lift held-out PR AUC (aggregate or
  on a target slice) versus the two baselines under the same tower,
  optimizer, and data. This is a controlled architecture-level test of
  "is reversible_delta_kernel_memory a better spatial mixer than conv or
  attention inside a fixed BT4 tower shell?", not a new primitive claim.
  The kernel-memory factorisation is `O(64 * h * v)` per block rather
  than the `O(64^2 * h)` cost of an explicit attention map, so a slice
  lift at matched throughput would be a useful empirical win.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the BT4
  block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for `reversible_delta_kernel_memory` itself (the
  unordered-set kernel memory with exact signed updates) is proven in
  the source primitive's math thesis and falsified by its own ablation
  grid. This folder inherits that math and tests whether the resulting
  operator, used as a token mixer rather than as an additive head,
  transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The all-pairs kernel-memory reduction is
  mathematically equivalent to a linear-attention factorisation of the
  64-token square-to-square mix.

- What is only hypothesized: That replacing the conv mixer with the
  `reversible_delta_kernel_memory` mixer lifts PR AUC on at least one
  CRTK slice (most likely slices where global piece-piece interaction is
  load-bearing, e.g. king-piece distance or pinned-piece-plus-pinner
  patterns identified by the source primitive) without regressing
  aggregate PR AUC by more than the matched-baseline tolerance.

- Failure cases:
  - The `reversible_delta_kernel_memory` mixer collapses inside the BT4
    shell because the residual + SqueezeExcite path dominates the mixer
    output; the `conv` baseline matches the variant within noise.
  - All 64 squares being active tokens (no occupancy masking on raw
    channels) dilutes the kernel memory, so the unordered-set advantage
    of the primitive is lost; an explicit `attention` baseline matches
    or beats this idea.
  - The mixer's per-sample cost inflates wall-clock enough that the
    matched-budget comparison is unfair; the baselines train for more
    effective optimizer steps inside the same wall-clock budget.
  - The kernel-memory factorisation underfits relative to a true 64x64
    attention map on this task; `attention` strictly dominates.
