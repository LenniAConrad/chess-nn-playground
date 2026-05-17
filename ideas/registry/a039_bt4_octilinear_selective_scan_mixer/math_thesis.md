# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `octilinear_selective_scan` primitive (OSS) from
  `p034_octilinear_selective_scan`. Source primitive math:
  `ideas/registry/p034_octilinear_selective_scan/math_thesis.md`.
  The OSS operator is a *Mamba-style selective state-space scan*
  along each of the 8 chess ray directions
  `k in {E, W, N, S, NE, NW, SE, SW}`. For each direction `k` the
  board is decomposed into a set of maximal scan paths (cardinal
  directions: 8 paths of length 8; diagonal directions: 15 paths of
  variable length 1..8) that traverse all 64 squares exactly once.
  Along each path the state evolves as

  ```
  h_t = sigmoid(A_k(x_t)) * h_{t-1} + B_k(x_t) * x_t
  ```

  where `A_k, B_k : R^C -> R^C` are channelwise linear maps from the
  per-square feature vector to a per-channel gain. The `sigmoid`
  wrapper keeps the multiplicative transition in `(0, 1)` per channel,
  so the iterated state is a contraction. The 8 per-direction per-
  square outputs are stacked to `(B, 64, 8 * C)` and fused back to
  `C` channels through `LayerNorm + Linear(8*C -> C) + GELU` before
  being reshaped to `(B, C, 8, 8)`. The load-bearing idea is that
  the selectivity gate `sigmoid(A_k(x))` lets the scan attenuate or
  *block* at piece-occupancy points: chess sliding-piece blocking
  behaviour emerges from the gate's data dependence on the per-square
  feature (which carries piece-existence after the trunk stem). The
  8-direction decomposition is rule-aware -- a bishop on c1 looks
  along its a3-f6 diagonal, a rook on h1 along the h-file, etc.

- Assumptions:
  1. The `octilinear_selective_scan` primitive is well-defined as a
     shape-preserving operator `(B, C, 8, 8) -> (B, C, 8, 8)` under
     the `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract.
  2. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and
     across the `conv` and `attention` baselines.
  3. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is
     the mixer.
  4. The source primitive (p034) was a *pooled head* over the i193
     trunk that ran the per-direction selective scan over a
     `Linear(13) -> d` projection of the piece planes, gathered the
     per-direction final-state-per-square outputs, concatenated and
     fused them, then own-piece-weighted-mean + global-mean pooled to
     a scalar gated delta logit over the i193 base logit. The mixer
     adaptation keeps the operator shape-preserving: it returns the
     fused per-square feature directly after `LayerNorm + Linear +
     GELU`, without the terminal own-piece / global mean pool and
     without the trunk-fusion gate / delta MLPs. The load-bearing
     selective-scan structure (per-(square, direction, channel)
     selectivity from a `sigmoid` linear projection of the per-square
     feature, eight rule-aware scan-path orderings, 8 * C -> C
     concat-and-fuse) is preserved exactly.
  5. The mixer reads the BT4 block's generic `(B, C, 8, 8)` channel
     tensor rather than a `Linear(13)` projection of the simple_18
     piece planes. The selectivity gate must therefore rediscover
     piece occupancy from whatever the trunk has encoded into the
     channel features; the chess-blocker behaviour is no longer
     a direct read of piece existence but an emergent property of
     the learned `A_k` projection.

- Claimed advantage: If the `octilinear_selective_scan` primitive
  carries a load-bearing rule-aware long-range ray signal beyond what
  conv and dense attention provide, dropping it into the BT4 block
  must lift held-out PR AUC (aggregate or on a slice that depends on
  long-range piece interactions along files / ranks / diagonals --
  pin / skewer / discovered-attack motifs, batteries on files or
  diagonals, X-ray attacks on the king, rook-on-open-file, queen-on-
  open-line, bishop-pair-on-long-diagonal, and long-line `mate_in_*`
  patterns where a single sliding-piece blocker decides the tactic)
  versus the two baselines under the same tower, optimizer, and data.
  This is a controlled architecture-level test of "is
  `octilinear_selective_scan` a better spatial mixer than conv or
  attention inside a fixed BT4 tower shell?", not a new primitive
  claim. The per-block cost is `O(8 * C^2 * 64)` for the eight pairs
  of `A_proj` / `B_proj` `Linear(C -> C)` projections (one per
  direction), `O(8 * 64 * C)` for the iterated scan body (per-square
  multiply-accumulate over the per-direction paths -- an 8 *
  64-element pass per direction with the scan body running across all
  paths in parallel within the direction), and `O(8 * C^2 * 64)` for
  the `LayerNorm + Linear(8*C -> C)` fuse. The dominant cost is the
  per-direction A / B projections at `O(NUM_DIRECTIONS * C^2)` per
  token, which is asymptotically *cheaper per block* than the dense
  `attention` baseline (`O(64 * 64 * C)` token-pair matmul) but
  somewhat more expensive than the conv baseline's `O(9 * C^2)`
  3x3 pair. The scan itself is a *sequential* Python loop over scan-
  path depth (max 8 steps); the per-step body is batched across
  (batch, channel) but not across squares within a path.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for `octilinear_selective_scan` itself
  (the scan is bounded and stable because the multiplicative
  transition `sigmoid(A_k) in (0, 1)` is a per-channel contraction;
  with `A_k = 0` everywhere the operator reduces to a per-square
  `B_k(x) * x` injection -- a per-direction local read; with `A_k
  = 1` everywhere the operator reduces to a plain prefix sum of
  `B_k(x_t) * x_t` along each scan path, a non-selective
  geometric prefix sum) is proven in the source primitive's math
  thesis and falsified by its own ablation grid (`single_direction`,
  `fixed_transition`, `shuffle_features`). This folder inherits that
  math and tests whether the resulting operator, used as a token
  mixer rather than as a pooled additive head, transfers its signal
  through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The per-direction scan
  `h_t = sigmoid(A_k(x_t)) * h_{t-1} + B_k(x_t) * x_t` is bounded
  because `sigmoid(A_k) in (0, 1)`, so for `|sigmoid(A_k)|_inf < 1`
  the iterated state satisfies
  `||h_t|| <= ||B_k(x) * x||_inf * sum_{j=0}^{t-1} a^j <= ||B_k(x)
  * x||_inf / (1 - a)` with `a = ||sigmoid(A_k)||_inf`. The eight
  scan-path tables (cardinal directions: 8 paths of length 8;
  diagonal directions: 15 paths of variable length 1..8 padded with
  -1) are computed at construction time and registered as
  non-persistent buffers; the off-path mask is applied per-step by
  the `valid = path[path >= 0]` filter inside `_scan_direction`. The
  8-direction concatenate + LayerNorm + Linear(8*C -> C) + GELU
  fusion preserves the channel dimension exactly. With `A_k` clamped
  to a constant (no input dependence) the operator reduces to a per-
  direction non-selective geometric prefix sum, matching the
  source primitive's `fixed_transition` ablation. With only the `E`
  direction active and the other seven zeroed the operator reduces
  to a single rightward scan, matching `single_direction`.

- What is only hypothesized: That replacing the conv mixer with the
  `octilinear_selective_scan` mixer lifts PR AUC on at least one
  CRTK slice (most likely long-line tactical slices: pins, skewers,
  discovered attacks, batteries on files / diagonals, X-ray attacks
  on the king, rook-on-open-file, bishop-pair-on-long-diagonal, and
  long-line `mate_in_*` patterns where the blocker on the ray
  decides the tactic) without regressing aggregate PR AUC by more
  than the matched-baseline tolerance. The hypothesis also covers
  the higher `crtk_difficulty` band where the trunk's local-
  receptive-field stack and the dense `attention` baseline are both
  likely to be insufficient: the conv stack misses long rays in a
  single block; dense attention has no chess-ray prior, so it must
  rediscover the 8 directions and the selective per-channel mix
  from data.

- Failure cases:
  - The trunk's stem conv plus the surrounding residual + SE blocks
    already encode long-range ray context densely enough (after `N`
    blocks the effective receptive field already covers the 8x8
    board) that adding a single selective-scan layer per block buys
    no marginal signal; the `conv` baseline matches the variant
    within noise.
  - The dense `attention` baseline matches or beats the OSS mixer;
    all-pairs attention can in principle express any ray pattern and
    the explicit 8-direction selective-scan prior is decorative at
    the BT4 tower's capacity.
  - The selectivity gate `sigmoid(A_k(x))` saturates near 1
    everywhere, so the scan becomes a plain non-selective geometric
    prefix sum of `B_k(x_t) * x_t`. The source primitive's
    `fixed_transition` ablation (A becomes data-independent) then
    matches this idea on its declared target slice.
  - The selectivity gate saturates near 0 everywhere, so the
    iterated state degenerates to a single-step `B_k(x_t) * x_t`
    injection (no long-range accumulation); the per-direction long-
    range scan claim collapses and the operator becomes a per-
    direction gated local read.
  - The 8 direction outputs collapse to redundant features (`single_
    direction` matches the full mixer) because the trunk's channels
    are not separable enough into per-direction signal; the OSS
    decomposition is decorative at the BT4 trunk's capacity.
  - The per-square feature passed to `A_k` / `B_k` does not contain
    enough piece-existence information to drive a blocker-like gate
    (the source primitive read piece planes directly through a
    `Linear(13) -> d` projection; the mixer reads a generic `C`-
    channel feature). The `shuffle_features` ablation (batch-permute
    the seed feature across positions) -- run in the BT4 tower by
    shuffling the channel input to the mixer only -- should then
    match the unablated mixer, indicating that the gate is not
    learning content-conditioned blocking.
  - SqueezeExcite + residual + ReLU absorbs most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.
  - The Python loop over scan-path depth (max 8 steps) is sequential
    and not parallelised across squares within a path. If throughput
    on matched hardware falls below ~40% of the conv baseline, the
    matched comparison must drop `model.num_blocks` or
    `model.channels` to compensate (see `trainer_notes.md`).
