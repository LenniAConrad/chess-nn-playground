# Architecture

`Schur-Ray Line Algebra Network` is a board-only `puzzle_binary` classifier
that exercises the math thesis: every square is described by an
unconstrained per-square data field, every rank, file, and diagonal carries
a low-rank line operator, and a closed-form Woodbury/Schur line-equilibrium
solve regularizes the square field through those 46 rays. The hypothesis
is that the line-coupled solution carries puzzle-vs-non-puzzle signal that
generic CNN pooling does not extract.

## Mechanism

1. `CoordinateBoardStem` reads the 18-plane `simple_18` tensor and
   appends rank, file, centerness, and side-relative forward-rank planes
   before a small convolutional stem produces a per-square feature map.
2. `field_head` projects each square feature into per-head data
   `(b, d, g)` triplets: `b = tanh(...)` is the unconstrained square
   target, `d = softplus(...) + eps` is the per-square positive data
   weight, and `g = sigmoid(...)` is a blocker gate that softens the
   square's incidence on its rays.
3. `BoardConditionedLineModes` builds compressed line modes
   `M(x) in R^{H x 46 x r}` from the fixed rank/file/diagonal incidence:
   square features are scattered into the 46 lines (8 ranks, 8 files,
   15 diagonals, 15 anti-diagonals), concatenated with a learned line-type
   embedding and a normalized line length, and an MLP produces `r` modes
   per head per line. The modes are L2-normalized along the rank axis.
4. The 64x46 incidence is realized by gathering each square's four ray
   memberships and summing their `r`-dimensional modes, then gating the
   result by `g` to form `U_h in R^{64 x r}`.
5. The Schur form of the line system is solved per head:
   `S = U^T D^{-1} U + diag(c)^{-1} + jitter * I`, with `c = softplus(c_raw) + eps`
   the diagonal positive line coupling. A Cholesky solve of
   `S a = U^T b` yields the line coefficients; the equilibrium square
   field is `z = b - D^{-1} U a`. This is exactly the Woodbury formula
   applied to `(D + U C U^T) z = D b`, replacing a 64x64 solve with an
   `r x r` Cholesky in line space.
6. Per-head Schur diagnostics — mean/max/std/topk of `z`, mean
   absolute correction `|z - b|`, line-coefficient norm `||a||`, log-det
   and trace of `S`, data energy `(z-b)^T D (z-b)`, line energy
   `(U^T z)^T C (U^T z)`, plus king-zone and slider-line masked energies —
   are stacked, flattened across heads, and concatenated with a CNN
   mean+max summary and a coarse material summary before a `LayerNorm +
   3-layer MLP` classifier produces one puzzle logit.
7. Forward returns a dict with the puzzle logit and finite per-batch
   diagnostics that the trainer can record (`schur_logdet`,
   `schur_trace`, `line_correction_norm`, `mean_abs_correction`,
   `data_energy`, `line_energy`, `king_zone_energy`,
   `slider_line_energy`, `schur_feature_norm`, `material_balance`,
   `piece_count`).

A set of ablations (`cnn_only`, `dense_attention_matched`,
`direct_64_solve`, `random_line_incidence`, `rank_file_only`, `diag_only`,
`fixed_M`, `no_blocker_gate`, `large_r`) is supported by the bespoke
builder so the line-equilibrium mechanism can be falsified against simpler
or matched-capacity baselines.

## Output Contract

Forward returns a `dict` whose `"logits"` entry has shape `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. All diagnostic
tensors are finite per batch and are appended to prediction artifacts.

## Implementation Binding

- Registered model name: `schur_ray_line_algebra_network`
- Source implementation file: `src/chess_nn_playground/models/schur_ray_line_algebra.py`
- Idea-local wrapper: `ideas/registry/i068_schur_ray_line_algebra_network/model.py`
