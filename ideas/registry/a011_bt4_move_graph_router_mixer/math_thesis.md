# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `move_graph_router` (MGR) primitive from
  `p006_move_graph_router`. Source primitive math:
  `ideas/registry/p006_move_graph_router/math_thesis.md`. Operationally,
  for per-square tokens `X in R^{B x 64 x C}` and a sparse, content-
  derived, stop-grad adjacency `E_b subseteq {(i, j)}`, the mixer computes
  `y_i = (1 / |N(i)|) * sum_{j : (i, j) in E_b} phi_theta([x_i, x_j])`
  with a shared two-layer GELU MLP `phi_theta` and `|N(i)|`-degree
  normalisation, then reshapes back to `(B, C, 8, 8)`.

- Assumptions:
  1. The MGR primitive is well-defined as a shape-preserving operator
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
  4. Because the BT4 mixer only receives `(B, C, 8, 8)` with arbitrary,
     semantically-opaque `C`, the rule-derived legal-move edge set from
     `p006` is not reconstructible at this point in the network. The
     mixer therefore substitutes a **content-derived** thresholded
     adjacency (computed from `src_score(tokens)` x `dst_score(tokens)`
     and detached) for the rule-derived `E_b`. The CORE of the
     primitive -- gather-scatter, concat-MLP edge function, stop-grad
     discrete mask, degree-normalised aggregation -- is preserved
     exactly. This is the operator-faithful adaptation, not a rebrand
     of dense attention.

- Claimed advantage: If the MGR primitive carries a spatial mixing
  signal that conv and attention do not, dropping it into the BT4 block
  must lift held-out PR AUC (aggregate or on a target slice) versus the
  two baselines under the same tower, optimizer, and data. This is a
  controlled architecture-level test of "is MGR a better spatial mixer
  than conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim. The MGR operator is hypothesised to be particularly
  strong on slices where ray-style information (pins, x-rays, battery
  threats) must propagate along sparse edges that a single 3x3 conv
  cannot reach, because MGR delivers that propagation in one mixer call
  rather than across multiple conv depth steps.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the BT4
  block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for MGR itself (gather-scatter over a stop-grad
  sparse adjacency, concat-MLP `phi_theta`, degree-normalised pooling)
  is proven in the source primitive's math thesis and falsified by its
  own ablation grid. This folder inherits that math and tests whether
  the resulting operator, used as a token mixer rather than as an
  additive head over the i193 trunk, transfers its signal through the
  BT4 tower. The stop-gradient on the adjacency makes the mask a
  non-differentiable branch by construction, matching the source
  primitive's "topology is rules, not learned scores" spec.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The degree-normalised mean form is
  well-defined for empty rows because of the `degree.clamp_min(1.0)`
  floor. The adjacency carries zero gradient because the discretisation
  uses `>=` against a per-source quantile threshold under
  `@torch.no_grad()`.

- What is only hypothesized: That replacing the conv mixer with the
  MGR mixer lifts PR AUC on at least one CRTK slice (most likely the
  high-`crtk_difficulty` and ray-tactical motif slices that motivate
  the MGR primitive's legal-move-graph framing) without regressing
  aggregate PR AUC by more than the matched-baseline tolerance. Also
  hypothesised: that the content-derived adjacency, while not chess-
  rule-derived, still produces an inductive bias materially different
  from a generic attention mixer because the mask is sparse,
  discretised, and detached.

- Failure cases:
  - The MGR mixer reduces to a noisy attention head inside the BT4
    shell because the residual + SqueezeExcite path dominates the
    mixer output; the `attention` baseline matches the MGR variant
    within noise.
  - The content-derived adjacency at the BT4 token grid drifts away
    from anything resembling the rule-derived legal-move adjacency
    that the source primitive was validated against; the
    `dense_edges` / `random_edges`-style ablations show the mask is
    not load-bearing inside the tower.
  - The fixed edge-density target (`edge_density=0.25`) over-covers
    or under-covers the relevant per-square edges at the BT4 token
    grid, so the mixer collapses to either a full dense mixer
    (large density) or a near-no-op (very small density).
  - The per-edge MLP `phi_theta` has cost `O(B * 64^2 * 2C)` per
    block because the implementation materialises the dense
    `[x_i, x_j]` tensor before masking. At larger `C` or `num_blocks`
    this dominates the matched-budget cost and the comparison
    becomes wall-clock-unfair to the conv baseline.
