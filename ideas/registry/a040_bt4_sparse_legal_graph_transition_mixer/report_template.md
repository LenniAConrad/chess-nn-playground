# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Joint-edge diagnostics: per-block mean and max of the routed-
    token output norm `||Y||`, per-block per-source-square mean of
    the aggregated edge feature norm `||agg[i]||` before the
    `Linear(d_edge -> C)` back-projection, and per-block ratio of
    the interaction term contribution `||W_interact (X_i (.) X_j)||`
    to the additive contributions `||W_self X_i + W_neighbor X_j||`
    inside `pre` before the `ReLU + LayerNorm`. Flag mixer collapse
    (all `||Y|| ~ 0`) or interaction-term degeneracy (the Hadamard
    contribution ratio falls below ~0.05, indicating `W_interact`
    has learned to zero out and the operator is effectively
    `separable_phi`).
  - Adjacency-mask diagnostics: at construction time, confirm that
    the static union-of-knight-king-sliding adjacency has zero
    diagonal, that every square has at least one reachable
    neighbour (degree > 0 for all 64 squares -- the
    `inv_degree.clamp(min=1.0)` floor should be a defensive guard
    that is never active), and that the per-square degree
    distribution matches the expected union-of-moves distribution
    (corner squares ~17, edge squares ~21, central squares ~27,
    with the exact counts pinned by `_static_move_graph`). Report
    the per-square degree histogram once at training start so
    regression test parity is auditable.
  - Edge-correlation diagnostics: per-block correlation between
    per-edge `phi` activation magnitude `||phi[i, j]||` (averaged
    over the held-out test set) and rule-exact attacker-defender
    pair indicators from `simple_18` (e.g. correlate the per-edge
    activation with the indicator that `i` holds an attacker and
    `j` holds a defender of compatible piece type). If the mixer
    is learning the attacker-defender pair geometry, edges with
    matched piece compatibility should carry larger `||phi||`
    than uncorrelated edges. If they do not, the joint edge
    function is decorative and the Hadamard interaction term is
    not aligned with chess-rule pair structure.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`. Also
    report per-block activation memory for the pair tensor `(B,
    64, 64, C)` to flag OOM risk at the matched batch size.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control -- the same all-pairs
    long-range mixing role without the hard-binary chess-rule mask
    and without the joint non-separable edge term; the most
    informative head-to-head for this idea).
  - `p035_sparse_legal_graph_transition` (A3 head-form control
    with the original blocker-resolved per-board legal-move
    adjacency `A(x)` and the pooled scalar trunk-fusion path that
    the mixer cannot use).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer reads a generic `(B, C, 8, 8)` channel tensor rather
    than the `simple_18` piece planes. The adjacency is the
    *static* union-of-moves graph rather than the blocker-resolved
    per-board legal-move graph; the joint edge function has no
    direct access to "which squares are actually connected by a
    legal move in *this* position". The hard mask is the chess-
    rule superset of the legal-move graph, so it carries every
    edge the source primitive had plus the blocked edges; the
    operator must learn to attenuate the blocked edges through
    the per-edge MLP body alone.
  - The explicit `(B, 64, 64, C)` pair tensor is `O(B * 64 * 64
    * C)` activation memory; at default sizes (`B = 256`, `C =
    64`) this is `~256 MiB` per block (FP32) before the masked
    einsum. On small-VRAM hardware the matched comparison must
    either drop batch size for all siblings or drop
    `model.channels` across siblings; do not silently break the
    matched-baseline contract.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.
  - The interaction term `W_interact (X_i (.) X_j)` may degenerate
    if `W_interact` learns to zero out, collapsing the operator
    to `separable_phi` (the A5 ablation matches). Flag this if
    the per-block ratio `||W_interact (X_i (.) X_j)|| / ||W_self
    X_i + W_neighbor X_j||` falls below ~0.05 on the held-out
    test set.

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Do not stop at an
aggregate confusion matrix. Every promoted idea must require:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`,
  `crtk_eval_bucket`, `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives
  for fine label `2`;
- confidence/calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and
  motifs;
- a short conclusion describing what the model appears able and
  unable to learn.

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  joint attacker-defender pair slices where the Hadamard
  interaction term `W_interact (X_i (.) X_j)` aligned with the
  hard-binary chess-rule mask is load-bearing (hanging-piece
  motifs, pin motifs, skewer motifs, fork motifs along the
  union-of-moves graph, X-ray attacks where two-piece
  compatibility matters); slices where the chess-rule
  neighbourhood structure dominates (rook-on-open-file with a
  reachable target, battery motifs on files / diagonals, bishop-
  pair on long diagonals); and the mid-to-upper `crtk_difficulty`
  band where attack-defend pair detection is the decisive signal
  and the trunk's local-receptive-field stack and the dense
  `attention` baseline are both likely to be insufficient (the
  conv stack misses long-range attacker-defender pairs in a
  single block; dense softmax attention has no chess-rule mask
  and no joint Hadamard term, so it must rediscover the
  union-of-moves geometry and the per-square pair-compatibility
  structure from data).
- Slices where this idea is expected to fail or merely match:
  local-tactical slices (one- or two-square exchanges with no
  attacker-defender pair across distance) where the conv mixer's
  local 3x3 window already saturates; positions dominated by
  pure single-piece evaluation with no joint-pair component
  (lowest `crtk_difficulty` band where the trunk's exchange
  features already saturate); positions where the
  blocker-resolved legal-move graph diverges sharply from the
  static union-of-moves graph (heavy mid-game with many blocked
  sliding pieces) -- the source primitive at A3 should beat this
  variant on those slices. These should be measured for
  non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the in-mixer A5 `separable_phi`, A6
  `uniform_adjacency`, and A7 `shuffle_adjacency` ablations. If
  any of these matches this idea on the target slice, the joint
  edge function, the chess-rule mask, or the rule-alignment claim
  is not load-bearing inside the BT4 tower. A2
  (`bt4_attention_mixer` -- dense all-pairs softmax mixing
  without the chess-rule mask and without the joint Hadamard
  term) is the canonical falsifier: if dense attention matches,
  the joint edge function on the chess-rule graph is just a
  more expensive variant of attention. A6 (`uniform_adjacency`)
  closing the slice also kills the chess-rule prior claim.
- Minimum useful slice-level improvement: target-slice PR AUC
  delta `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC
  delta in `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
