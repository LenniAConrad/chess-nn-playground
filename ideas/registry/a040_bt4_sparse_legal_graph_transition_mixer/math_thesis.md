# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `sparse_legal_graph_transition` primitive (SLMGT)
  from `p035_sparse_legal_graph_transition`. Source primitive math:
  `ideas/registry/p035_sparse_legal_graph_transition/math_thesis.md`.
  The SLMGT operator is a *joint, non-separable edge function over the
  chess move graph* with hard-binary chess-rule masking and
  degree-normalised mean aggregation:

  ```
  phi(X_i, X_j) = LayerNorm(ReLU(
      W_self X_i + W_neighbor X_j + W_interact (X_i (.) X_j)
  ))
  Y[i] = (1 / max(deg(i), 1)) * sum_{j : A[i, j] = 1} phi(X_i, X_j)
  ```

  where `W_self, W_neighbor, W_interact : R^C -> R^{d_edge}` are
  channelwise linear maps over the per-square feature, the Hadamard
  product `X_i (.) X_j` is the joint non-separable interaction term,
  and `A in {0, 1}^{64 x 64}` is the *static* union of knight, king,
  and sliding-piece reach (zero diagonal). The aggregated `(B, 64,
  d_edge)` per-square features are projected back to `C` via
  `Linear(d_edge -> C)` and reshaped to `(B, C, 8, 8)`. The
  load-bearing idea is that the Hadamard interaction
  `W_interact (X_i (.) X_j)` lets the operator learn
  "attacker-defender pair" features: the term is non-trivially active
  only when both squares carry compatible feature signals, which is
  the right inductive bias for hanging-piece / pin / fork detection.
  Standard GAT applies a *separable* score with softmax-normalised
  attention; SLMGT applies a *joint* edge function with a *hard
  binary* chess-rule mask, and mean aggregation prevents high-degree
  squares from saturating.

- Assumptions:
  1. The `sparse_legal_graph_transition` primitive is well-defined as
     a shape-preserving operator `(B, C, 8, 8) -> (B, C, 8, 8)` under
     the `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract.
  2. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and
     across the `conv` and `attention` baselines.
  3. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is
     the mixer.
  4. The source primitive (p035) was a *pooled head* over the i193
     trunk that built the binary adjacency `A(x)` directly from the
     `simple_18` board with piece-specific blocker resolution (the
     blocker-resolved legal-move graph), ran the joint edge function
     `phi` over that adjacency, mean-aggregated per source square,
     pooled, and projected to a gated scalar delta logit. The mixer
     adaptation keeps the operator shape-preserving: it returns the
     fused per-square edge feature directly via the `Linear(d_edge
     -> C)` projection, without the terminal pooling and gated-delta
     fusion. The load-bearing structure (the joint non-separable
     edge function with the Hadamard interaction term, the hard
     binary chess-rule mask, the degree-normalised mean aggregation,
     and the LayerNorm) is preserved exactly.
  5. The mixer reads the BT4 block's generic `(B, C, 8, 8)` channel
     tensor rather than the `simple_18` piece planes, and applies the
     *static* union-of-moves adjacency rather than the
     blocker-resolved per-board legal-move adjacency. The mask is
     therefore position-independent: each square sees its full
     chess-rule reach without any occupancy or blocker filtering.

- Claimed advantage: If the `sparse_legal_graph_transition` primitive
  carries a load-bearing rule-aware joint-edge signal beyond what
  conv and dense attention provide, dropping it into the BT4 block
  must lift held-out PR AUC (aggregate or on a slice that depends on
  joint attacker-defender pair structure -- hanging-piece motifs,
  pin / skewer / fork motifs along the union-of-moves graph, X-ray
  attacks where two-piece compatibility matters, and the mid-to-upper
  `crtk_difficulty` band where attack-defend pair detection is the
  decisive signal) versus the two baselines under the same tower,
  optimizer, and data. This is a controlled architecture-level test
  of "is `sparse_legal_graph_transition` a better spatial mixer than
  conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim. The per-block cost is dominated by the explicit
  pair tensor `pair = feats.unsqueeze(2) * feats.unsqueeze(1) in
  R^{B x 64 x 64 x C}` and the subsequent `W_interact` projection
  `R^{B x 64 x 64 x d_edge}`. With `d_edge = C` the pair tensor
  costs `O(64 * 64 * C) = O(4096 C)` per sample and the `W_interact`
  projection costs `O(64 * 64 * C^2) = O(4096 C^2)` per sample,
  asymptotically the same scale as dense attention's `O(64 * 64 * C)`
  pair-matmul plus `O(64 * C^2)` projections but with a heavier
  per-edge MLP body. The hard binary mask makes the einsum
  `bij,bijd->bid` skip masked pairs at the FLOP level under sparse
  multiplication but is dense in memory because the pair tensor is
  materialised for all 64x64 pairs before masking.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for `sparse_legal_graph_transition`
  itself (the aggregation is well-defined for any board; the
  Hadamard interaction makes `phi` non-separable through any single
  `Linear`; removing it strictly reduces the operator's capacity to
  the separable additive form `ReLU(W_self X_i + W_neighbor X_j)`)
  is proven in the source primitive's math thesis and falsified by
  its own `separable_phi`, `uniform_adjacency`, and
  `shuffle_adjacency` ablations. This folder inherits that math and
  tests whether the resulting operator, used as a token mixer rather
  than as a pooled additive head, transfers its signal through the
  BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The aggregation is well-defined for
  any board: `inv_degree = 1 / max(degree, 1)` is bounded away from
  zero by construction, the static `adjacency` buffer has zero
  diagonal (no self-edge), the union-of-knight-king-sliding reach
  covers a non-empty neighbourhood for every square (every square
  has at least one reachable neighbour under the union, so the
  degree is strictly positive and the `clamp(min=1.0)` floor is
  defensive), and the `einsum` `bij,bijd->bid` is well-defined for
  all `(i, j)` pairs. The Hadamard interaction term `W_interact (X_i
  (.) X_j)` is genuinely non-separable through any single `Linear`
  on either `X_i` or `X_j` alone (removing it reduces the operator
  to the separable additive form `ReLU(W_self X_i + W_neighbor
  X_j)`, the `separable_phi` ablation). With `W_interact = 0` and
  no `LayerNorm` after `ReLU` the operator reduces to a standard
  separable GAT-style aggregation `Y[i] = (1 / deg(i)) sum_j A[i,j]
  ReLU(W_self X_i + W_neighbor X_j)`. With `A` replaced by the
  all-ones matrix (minus identity) the operator reduces to a dense
  joint-edge aggregation over all 63 other squares per source square
  (the `uniform_adjacency` ablation), removing the chess-rule prior.

- What is only hypothesized: That replacing the conv mixer with the
  `sparse_legal_graph_transition` mixer lifts PR AUC on at least one
  CRTK slice (most likely joint attacker-defender pair slices:
  hanging pieces, pins, skewers, forks, X-ray attacks where two-
  piece compatibility matters, and rook-on-open-file / bishop-pair /
  battery patterns where the union-of-moves graph carries the
  decisive geometry) without regressing aggregate PR AUC by more
  than the matched-baseline tolerance. The hypothesis also covers
  the higher `crtk_difficulty` band where the trunk's local-
  receptive-field stack and the dense `attention` baseline are both
  likely to be insufficient: the conv stack misses long-range
  attacker-defender pairs in a single block; dense attention has no
  chess-rule mask, so it must rediscover the union-of-moves
  geometry and the per-square degree normalisation from data.

- Failure cases:
  - The trunk's stem conv plus the surrounding residual + SE blocks
    already encode joint-pair structure densely enough (after `N`
    blocks the effective receptive field already covers the 8x8
    board and the per-channel co-activation already encodes
    attacker-defender pairs) that adding a single SLMGT layer per
    block buys no marginal signal; the `conv` baseline matches the
    variant within noise.
  - The dense `attention` baseline matches or beats the SLMGT mixer;
    all-pairs attention with softmax can in principle express any
    masked joint-edge pattern at sufficient capacity and the
    explicit chess-rule mask is decorative at the BT4 tower's
    capacity.
  - The `separable_phi` ablation (zero out `W_interact`) matches the
    full mixer, so the Hadamard interaction term is decorative and
    the operator is a separable GAT-style mean aggregator.
  - The `uniform_adjacency` ablation (replace `A` with all-ones
    minus identity) matches the full mixer, so the chess-rule mask
    is decorative and the operator is a dense joint-edge aggregator
    over all 63 other squares per source.
  - The `shuffle_adjacency` ablation (batch-permute `A`) matches
    the full mixer, so the rule indicators are decoupled from
    position and the chess-rule mask is not load-bearing.
  - The static union-of-moves mask is too permissive (every square
    sees its full chess-rule reach without occupancy filtering)
    and the per-edge MLP cannot recover the missing blocker
    geometry from the generic `(B, C, 8, 8)` channel features; the
    operator degenerates to a position-independent mean-aggregation
    over a fixed dense neighbourhood (~16-27 neighbours per square
    under the union-of-moves graph).
  - SqueezeExcite + residual + ReLU absorbs most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.
  - The pair tensor `(B, 64, 64, C)` is `O(B * 64 * 64 * C)` memory.
    At default sizes (`B = 256`, `C = 64`) this is `~256 MiB` per
    block before masking, which can OOM on small-VRAM hardware and
    forces a batch-size reduction that breaks the matched-baseline
    contract.
