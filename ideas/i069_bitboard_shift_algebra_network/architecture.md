# Architecture

`Bitboard Shift-Algebra Network` is a board-only `puzzle_binary` classifier
that exercises the math thesis: chess displacement patterns can be
propagated cheaply through fixed sparse rule-shaped square-shift operators
combined with low-degree learned operator polynomials, replacing dense
attention or attack-graph message passing with bitboard-style shifts and
masks.

## Mechanism

1. `BitboardStem` reads the 18-plane `simple_18` tensor, appends rank,
   file, centerness, and side-relative forward-rank planes, and produces a
   per-square feature map `u0` of width `D`.
2. A fixed bank of 16 sparse single-step shift operators is constructed
   geometrically: 8 king/orthogonal/diagonal one-step shifts and 8 knight
   L-shaped shifts. Each shift is a 64-square gather with masked
   wraparound (no learned parameters in the shift bank).
3. A fixed path basis of 12 short shift-composition path families is
   evaluated against `u0`: identity, one-step orthogonal, one-step
   diagonal, two- and three-step rook and bishop slides, knight jump,
   king ring, side-relative pawn-capture left and right, and a
   knight-then-king-ring composition. Path depth is capped by
   `path_depth_max` (`<= 3`) so the polynomial degree stays small.
4. `CoefficientEmitter` produces board-conditioned head coefficients
   `alpha in R^{B x H x P}` from a compact pooled board summary
   (mean+max+material). Coefficients are normalized either with
   `tanh / sqrt(P)` (default) or `softmax` over `P`, matching the packet's
   thesis; a learned per-head fixed-alpha tensor is held for the
   `fixed_alpha` ablation.
5. Per-head shift fields are formed as `v_h = sum_p alpha[h,p] * path_output[p]`
   and gated against the stem with a sigmoid `Conv1x1` gate so the network
   can suppress shift contributions where the current board has blockers
   or noise: `head_field = gate * v_h + (1 - gate) * u0`. Disabling the
   gate falls back to a simple `u0 + v_h` sum for the `no_gate` ablation.
6. Per-head shift diagnostics — mean/max/topk pooled head fields,
   occupied-square pooled fields, king-zone pooled fields, mean-square
   shift residual `||head_field - u0||`, king-zone absolute residual, and
   occupied-square absolute energy — are stacked, flattened across heads,
   and concatenated with a CNN mean+max summary, a coarse material
   summary, and per-head coefficient diagnostics (entropy, mean and max
   absolute coefficient) before a `LayerNorm + 3-layer MLP` classifier
   produces one puzzle logit.
7. Forward returns a dict with the puzzle logit plus finite per-batch
   diagnostics that the trainer can record (`coefficient_entropy`,
   `coefficient_abs_mean`, `top_path_strength`, `shift_residual`,
   `king_zone_shift_residual`, `occupied_shift_energy`,
   `path_output_energy`, `head_field_energy`, `cnn_energy`,
   `material_balance`, `piece_count`).

A set of ablations (`cnn_only`, `random_shift_bank`, `orthogonal_only`,
`one_step_only`, `fixed_alpha`, `no_gate`, `dense_conv_matched`) is
supported by the bespoke builder so the chess-shift mechanism can be
falsified against random-permutation, orthogonal-only, single-step-only,
fixed-coefficient, ungated, and depth-matched depthwise-convolution
controls.

## Output Contract

Forward returns a `dict` whose `"logits"` entry has shape `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. All diagnostic
tensors are finite per batch and are appended to prediction artifacts.

## Implementation Binding

- Registered model name: `bitboard_shift_algebra_network`
- Source implementation file: `src/chess_nn_playground/models/bitboard_shift_algebra.py`
- Idea-local wrapper: `ideas/i069_bitboard_shift_algebra_network/model.py`
