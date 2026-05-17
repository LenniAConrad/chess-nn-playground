# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `attack_ray_sparse_attention` (ARSA) primitive from
  `p007_attack_ray_sparse_attention`. Source primitive math:
  `ideas/registry/p007_attack_ray_sparse_attention/math_thesis.md`.
  Operationally, for per-square tokens `X in R^{B x 64 x C}` and a
  per-square 9-slot key index tensor `K(b, s)` derived from an
  occupancy-driven first-blocker scan along the 8 sliding-piece ray
  directions (plus a self-edge), the mixer computes
  `y_{b,s} = sum_{k in K(b,s)} softmax_k(q_{b,s}^T k_{b,s,k} / sqrt(d_q)
  + bias_dir(k)) * v_{b,s,k}` with learned `q, k, v` linear projections,
  a learned per-slot direction bias, and a softmax mask over slots whose
  ray has no blocker; the result is then reshaped back to
  `(B, C, 8, 8)`.

- Assumptions:
  1. The ARSA primitive is well-defined as a shape-preserving operator
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
     semantically-opaque `C`, the rule-derived `simple_18` occupancy
     plane used by the source `p007` primitive to drive the first-blocker
     scan is not available at this point in the network. The mixer
     therefore substitutes a **content-derived** soft occupancy
     (a thresholded learned scalar per square under `@torch.no_grad()`)
     for the rule-derived occupancy. The CORE of the primitive --
     ray-cast first-blocker key selection, the 9-slot sparse gather,
     `q/k/v` projections with a per-slot direction bias, masked softmax
     over slots whose ray has no blocker, and stop-grad on the index
     tensor -- is preserved exactly. This is the operator-faithful
     adaptation, not a rebrand of dense attention.

- Claimed advantage: If the ARSA primitive carries a spatial mixing
  signal that conv and attention do not, dropping it into the BT4 block
  must lift held-out PR AUC (aggregate or on a target slice) versus the
  two baselines under the same tower, optimizer, and data. This is a
  controlled architecture-level test of "is ARSA a better spatial mixer
  than conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim. ARSA is hypothesised to be particularly strong on
  slices where the relevant key set per square is exactly the first
  blocker on each ray (pins, x-rays, skewers, discovered attacks,
  battery threats) because the operator delivers that adjacency at one
  mixer call rather than across multiple conv depth steps.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the BT4
  block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for ARSA itself (first-blocker key set, 9-slot
  sparse softmax attention with masked empty rays, stop-grad on the
  index tensor) is proven in the source primitive's math thesis and
  falsified by its own ablation grid. This folder inherits that math
  and tests whether the resulting operator, used as a token mixer
  rather than as an additive head over the i193 trunk, transfers its
  signal through the BT4 tower. The stop-gradient on the key index
  tensor makes the sparsity pattern a non-differentiable branch by
  construction, matching the source primitive's "topology is rules and
  occupancy, not learned scores" spec.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The 9-slot key set has a fixed cardinality
  per square (8 ray neighbours plus self-edge); the softmax mask is
  well-defined for source squares with no blocker on a ray because
  those slots are masked to `-inf` before softmax, and the self-edge
  always exists so the per-square softmax denominator is non-zero.
  The key index tensor carries zero gradient because the occupancy
  discretisation and the first-blocker scan are computed under
  `@torch.no_grad()`.

- What is only hypothesized: That replacing the conv mixer with the
  ARSA mixer lifts PR AUC on at least one CRTK slice (most likely the
  high-`crtk_difficulty` and sliding-piece tactical motif slices --
  pins, skewers, x-rays, discovered attacks -- that motivate the ARSA
  primitive's first-blocker framing) without regressing aggregate PR
  AUC by more than the matched-baseline tolerance. Also hypothesised:
  that the content-derived occupancy, while not rule-derived from the
  `simple_18` piece planes, still produces an inductive bias
  materially different from a generic attention mixer because the
  attention is structurally restricted to 9 slots per query along
  fixed ray geometry with a stop-grad mask.

- Failure cases:
  - The ARSA mixer reduces to a noisy attention head inside the BT4
    shell because the residual + SqueezeExcite path dominates the
    mixer output; the `attention` baseline matches the ARSA variant
    within noise.
  - The content-derived occupancy at the BT4 token grid drifts away
    from anything resembling the rule-derived occupancy that the
    source primitive was validated against; the
    `random_keys`-style ablation shows the rule-derived first-blocker
    geometry is not load-bearing inside the tower.
  - The `arsa_self_weight` distribution concentrates near 1, meaning
    each token attends mostly to its own square and the mixer reduces
    to a per-square linear plus residual; the BT4 block then degenerates
    to a SqueezeExcite-only operator.
  - The 9-slot cardinality is small enough that the per-query softmax
    saturates quickly on a single direction bias, and gradients to
    `q_proj` / `k_proj` collapse; the operator degrades to a
    direction-biased mean pool.
