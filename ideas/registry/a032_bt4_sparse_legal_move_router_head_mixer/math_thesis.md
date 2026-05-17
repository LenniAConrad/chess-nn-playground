# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `sparse_legal_move_router_head` primitive from
  `p027_sparse_legal_move_router_head`. Source primitive math:
  `ideas/registry/p027_sparse_legal_move_router_head/math_thesis.md`.
  The Sparse Legal-Move Router (SLMR) operator, for a per-square
  embedding stack `X in R^{B x 64 x C}` and a per-(source, target)
  adjacency `M in {0, 1}^{64 x 64}` over a chess-structured support,
  computes one round of *masked* attention:

  ```
  attn_{i, j} = (Q_i . K_j) / sqrt(d_attn)   if M_{i, j} = 1
              = -inf                          otherwise
  weights_i   = softmax(attn_i)
  Y_i         = sum_j weights_{i, j} * V_j
  ```

  where `Q = X W_Q`, `K = X W_K`, `V = X W_V` are linear projections
  to attention width `d_attn` (`Q, K`) and channel width `C` (`V`).
  Connectivity is restricted to the union of standard chess move
  shapes (slider rays, knight L-jumps, king steps, pawn forward and
  diagonal moves) instead of the dense 64x64 all-pairs graph.

- Assumptions:
  1. The `sparse_legal_move_router_head` primitive is well-defined as
     a shape-preserving operator `(B, C, 8, 8) -> (B, C, 8, 8)` under
     the
     `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract.
  2. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and
     across the `conv` and `attention` baselines.
  3. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is
     the mixer.
  4. The source primitive (p027) builds the rule-exact legal-move
     adjacency `M` from the `simple_18` piece planes and the
     side-to-move plane via the i193 geometry tables, honouring
     blocker termination for sliders. The mixer cannot read piece
     planes (it sees only a generic `(B, C, 8, 8)` channel tensor)
     so the adjacency is replaced by a fixed chess-geometry support
     `S in {0, 1}^{64 x 64}` (union of slider rays, knight jumps,
     king steps, pawn moves -- unobstructed) plus a *learned*
     per-edge gate `g_{i, j} = sigmoid(theta_{i, j})` applied as a
     log-bias inside the masked softmax. The sparse-routing
     structure -- attend only along chess-structured edges, with a
     masked-softmax aggregator -- is preserved exactly; the
     "is this edge legal given the content" decision becomes a
     learned soft gate rather than a rule-exact mask.

- Claimed advantage: If the `sparse_legal_move_router_head` primitive
  carries a load-bearing chess-structured sparse-routing signal
  beyond what conv and dense attention provide, dropping it into the
  BT4 block must lift held-out PR AUC (aggregate or on a slice that
  depends on legal-move structure -- e.g. fork-style multi-target
  motifs, sliding-piece long-range threats, knight-jump tactics,
  and pin / skewer / discovered-attack structures) versus the two
  baselines under the same tower, optimizer, and data. This is a
  controlled architecture-level test of "is
  sparse_legal_move_router_head a better spatial mixer than conv
  or attention inside a fixed BT4 tower shell?", not a new
  primitive claim. The per-block cost is `O(64 * 64 * d_attn)` for
  the masked attention (the masking does not reduce FLOPs since
  the matmul is still dense before the mask is applied) plus an
  `O(C^2)` output projection back to `C` channels; this is at the
  same big-O as the `attention` baseline but with a *fixed* (not
  learned) sparsity pattern.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for `sparse_legal_move_router_head` itself
  (masked-softmax aggregator is a standard PyTorch construction with
  the usual numerical-stability guarantees; sources with no on-support
  target fall back to a self-loop so softmax does not NaN) is proven
  in the source primitive's math thesis and falsified by its own
  ablation grid (`full_64x64_mask`, `self_loop_only`,
  `shuffle_adjacency`, `zero_router_features`). This folder inherits
  that math and tests whether the resulting operator, used as a
  token mixer rather than as a pooled additive head, transfers its
  signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards
  the mixer at registration time. The attention is computed exactly
  as `softmax(Q K^T / sqrt(d) + log(g . S) - inf . (1 - S))` with
  `S` the fixed chess-geometry support and `g` the learned per-edge
  gate clamped above `1e-9` for numerical stability. Off-support
  edges receive `-inf` logits and contribute zero to the softmax
  exactly; on-support edges receive a soft additive log-gate bias
  that can suppress an edge below numerical resolution but never
  attends off-support. The self-loop term `M_{i, i} = 1` guarantees
  every row of `M` has at least one nonzero entry, so the softmax
  is finite-valued.

- What is only hypothesized: That replacing the conv mixer with the
  `sparse_legal_move_router_head` mixer lifts PR AUC on at least
  one CRTK slice (most likely fork-style multi-target tactics,
  sliding-piece long-range threats, knight-jump motifs, and
  pin / skewer / discovered-attack structures where information
  routing along legal-move edges is more relevant than dense
  all-pairs mixing) without regressing aggregate PR AUC by more
  than the matched-baseline tolerance. The hypothesis also covers
  the higher `crtk_difficulty` band where the trunk's local-
  receptive-field stack and the dense `attention` baseline are
  both likely to be insufficient.

- Failure cases:
  - The trunk's stem conv plus the surrounding residual + SE blocks
    already encode chess move structure densely enough that
    restricting attention to a sparse legal-move support adds no
    marginal signal; the `conv` baseline matches the variant within
    noise.
  - The dense `attention` baseline matches or beats the sparse
    router; the sparsity constraint is decorative and any
    all-pairs token mixer captures the same signal.
  - The learned per-edge gate `g = sigmoid(theta)` collapses to
    near-uniform values on the support, so the gate plays no role
    and the mixer degenerates to a uniform-weighted sparse
    aggregator over the fixed chess-geometry support. The
    `full_64x64_mask` style ablation (drop the support entirely)
    should close the gap if the chess-structured support is not
    load-bearing.
  - The fixed chess-geometry support (slider rays unobstructed,
    knight jumps, king and pawn steps) under-approximates the
    rule-exact legal-move adjacency by ignoring blockers, so the
    soft gate must learn to suppress edges that are physically
    blocked by intervening pieces. If the gate cannot recover the
    blocker structure from the channel features alone, the mixer
    routes signal through impossible moves.
  - SqueezeExcite + residual + ReLU absorbs most of the mixer's
    contribution if the masked-attention output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.
  - The `(B, 64, 64)` dense matmul-then-mask adds wall-clock cost
    that is not amortised by signal. If
    `train_samples_per_second` falls well below the matched conv
    baseline without a slice-level lift, the mixer fails its cost-
    matched comparison.
