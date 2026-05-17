# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `occlusion_semiring_ray_scan` primitive from
  `p021_occlusion_semiring_ray_scan`. Source primitive math:
  `ideas/registry/p021_occlusion_semiring_ray_scan/math_thesis.md`.
  For each source square `s`, direction `r in 0..7`, ordered ray cell at
  step `l` with index `c_{s,r,l}`, and per-square soft occupancy `O`,
  the primitive computes the *exclusive prefix transmittance* along the
  ray:
  `T_{s,r,l} = prod_{q < l} (1 - O_{c_{s,r,q}})`,
  and reduces along the ray via
  `y_{s,r} = sum_{l=1..L} T_{s,r,l} * A_r * x_{c_{s,r,l}}`,
  where `A_r` is one of 8 distinct per-direction linear projections.
  The defining property is the *non-recurrent* exclusive prefix product:
  cell `l` is reachable from `s` only if every earlier cell on that ray
  is unoccupied, and each cell's own projection `A_r * x_{c_{s,r,l}}` is
  weighted by its transmittance rather than being folded into a recurrent
  hidden state. `T` is computed in log-domain via a shifted `cumsum` of
  `log(1 - O)` (clamped at `log_eps = 1e-4` to avoid `log(0)` when a
  blocker sets `O = 1`, in which case the resulting `T = 0` zeroes the
  cell as required). Inside the BT4 block the 8 directional outputs at
  each source square are concatenated and projected back to the channel
  dimension via `Linear(8 * C -> C)`.

- Assumptions:
  1. The `occlusion_semiring_ray_scan` primitive is well-defined as a
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
  4. The mixer cannot read the piece planes directly (the BT4 block
     hands it a generic `(B, C, 8, 8)` channel tensor), so the
     occupancy indicator is derived inside the operator as
     `O_s = sigmoid(w . x_s + b)` from the per-square channel vector.
     The occupancy mask is therefore *learned* from features, but it is
     still generated *inside* the operator and never supplied
     externally, which preserves the source thesis's defining property
     (the prefix-product transmittance is not driven by an external
     mask).

- Claimed advantage: If the `occlusion_semiring_ray_scan` primitive
  carries a sliding-piece transmittance signal that conv and attention
  do not, dropping it into the BT4 block must lift held-out PR AUC
  (aggregate or on a sliding-piece-dependent slice such as pin / skewer
  / discovered attack / x-ray attack / rook-on-open-file /
  queen-line-into-king-zone) versus the two baselines under the same
  tower, optimizer, and data. This is a controlled architecture-level
  test of "is occlusion_semiring_ray_scan a better spatial mixer than
  conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim. The exclusive-prefix-product reduction is
  `O(64 * 8 * L * h)` per block where `L <= 7` and `h` is the mixer
  hidden width; the prefix product is implemented via a vectorised
  `cumsum` rather than a Python loop, so it is comparable to a 3x3 conv
  in FLOPs and asymptotically cheaper than a full 64x64 attention map.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the BT4
  block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for `occlusion_semiring_ray_scan` itself (the
  exclusive prefix-product transmittance with log-domain stabilisation)
  is proven in the source primitive's math thesis and falsified by its
  own ablation grid (`zero_occupancy`, `uniform_occupancy`,
  `isotropic_A`). This folder inherits that math and tests whether the
  resulting operator, used as a token mixer rather than as an additive
  head, transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The ray-step index and step-mask tables
  are pure rule-derived constants (queen-style 8 directions, up to 7
  steps, off-board steps masked); the exclusive prefix product
  `T_{s,r,l} = exp(cumsum(log(clamp(1 - O, log_eps, 1)))[..., :-1])`
  with a leading zero pad and a step-mask zero-out is implemented
  exactly.

- What is only hypothesized: That replacing the conv mixer with the
  `occlusion_semiring_ray_scan` mixer lifts PR AUC on at least one
  CRTK slice (most likely slices where sliding-piece transmittance is
  load-bearing -- pin / skewer / discovered attack / x-ray attack /
  rook-on-open-file / queen-line-into-king-zone -- and the upper
  `crtk_difficulty` tail where long-range geometry dominates over local
  conv windows) without regressing aggregate PR AUC by more than the
  matched-baseline tolerance.

- Failure cases:
  - The learned soft occupancy `O_s = sigmoid(w . x_s + b)` fails to
    recover the piece-plane occupancy that the source primitive uses,
    so the prefix-product gate is uninformative; the `zero_occupancy`
    ablation matches this idea on its declared target slice.
  - The `occlusion_semiring_ray_scan` mixer collapses inside the BT4
    shell because the residual + SqueezeExcite path dominates the
    mixer output; the `conv` baseline matches the variant within noise.
  - The 8-way directional fuse `out = Linear(8 * C -> C)` underfits and
    bottlenecks the per-direction signal; the `attention` baseline
    matches or beats this idea.
  - The per-direction projection collapses across directions (the
    `isotropic_A` ablation closes the gap), meaning the 8 distinct
    `A_r` maps are not load-bearing and a single shared projection
    would suffice; the directional structure is then decorative.
