# Architecture

`Oriented Matroid Covector Bottleneck` realises the source packet's covector / sign-pattern thesis as a bespoke PyTorch model for the repository's `puzzle_binary` task. The classifier reads the position only through sign-pattern statistics of an arrangement of learned hyperplanes evaluated on occupied-piece tokens, so the head never sees the raw piece tokens directly — the puzzle decision flows only through covector summaries.

## Implementation Binding

- Registered model name: `oriented_matroid_covector_bottleneck`
- Source implementation file: `src/chess_nn_playground/models/oriented_matroid_covector.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i096_oriented_matroid_covector_bottleneck/model.py`

## Modules

`OccupiedPieceTokenizer` extracts up to `max_pieces` occupied square tokens from the simple_18 board planes. For each selected square it emits a 12-d soft role distribution over the piece planes, the square's `(rank, file, square_color)` coordinates, the square occupancy, and any auxiliary planes from `12..C-1`. A deterministic `randperm(64)` per piece role is registered as a buffer so the `coordinate_shuffle_by_piece` ablation can replace square coordinates with role-dependent permutations and test whether the architecture exploits the underlying piece geometry.

`PieceTokenEncoder` is a small `Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm -> GELU` MLP that lifts each token to the `token_dim`-d learned representation. The token mask is applied multiplicatively so padded tokens contribute zero everywhere downstream.

`HyperplaneArrangement` carries the learned arrangement: `hyperplanes` unit-normalised vectors `w_p` and biases `b_p`, plus a deterministic random alternative `(w_p^{rand}, b_p^{rand})` registered as buffers. Given the token embeddings it returns:

- `scores`: the signed projection `<w_p, e_n> + b_p` for every piece token `n` and every hyperplane `p`.
- `signs = tanh(sign_scale * scores)`: a smooth surrogate for the oriented-matroid covector entry `sigma_n^{(p)} in {-, 0, +}`.

The `random_hyperplanes` ablation swaps the learned arrangement for the random one without changing the rest of the pipeline, so the experiment can isolate whether *learned* hyperplane orientations carry the puzzle signal.

`CovectorStats` is the bottleneck. From `(scores, signs, role_probs, token_mask)` it forms only sign-pattern / role-conditioned summaries:

- Per-hyperplane masked counts `pos_p`, `neg_p`, `near_zero_p` averaged over occupied tokens — the sign-pattern histogram of the covector at the position.
- Pairwise hyperplane sign-agreement matrix `A_{pq} = (1/N) sum_n sigma_n^{(p)} sigma_n^{(q)}` flattened to expose covector co-occurrence.
- Per-role per-hyperplane sign entropy `H_{r,p}` over `{+, -, 0}`, computed by re-weighting tokens with the soft role distribution `role_probs`.
- Per-role histogram `(1/N) sum_n role_probs(n)` across the 12 piece roles.
- Per-hyperplane sign mean and absolute mean, score absolute mean and masked standard deviation, normalised covector entropy `H(pos_p, neg_p, zero_p)`.
- A small `globals_` block with piece count, mean positive/negative/near-zero rates, pairwise agreement energy, mean `|sigma|`, mean `|score|`, and mean covector entropy.

The pooled covector feature vector has dimension `3 * hyperplanes + hyperplanes^2 + roles * hyperplanes + roles + 4 * hyperplanes + 8` and is the *only* board-derived input the head sees. The head is a `LayerNorm -> Linear -> GELU -> Dropout -> Linear -> GELU -> Linear(1)` MLP returning a single puzzle logit.

`OrientedMatroidCovectorBottleneck` glues the trunk together: tokenizer -> token encoder -> hyperplane arrangement -> covector stats -> head. The puzzle decision flows only through `psi(x) = covector_stats(arrangement(token_encoder(tokenize(x))))`, so the sign-pattern bottleneck is enforced architecturally.

## Modes

The `mode` argument selects the active variant:

- `covector` (default): learned hyperplane arrangement with full sign-pattern + role-conditioned covector readout. The reference implementation called for in the source packet.
- `magnitude_only`: replaces signed projections with `|scores|` and re-derives signs as `tanh(|scores|)`, removing oriented-matroid sign content while keeping the same head capacity. Tests whether sign patterns (rather than absolute hyperplane evidence) drive the puzzle signal.
- `random_hyperplanes`: bypasses the learned arrangement and reads sign patterns of fixed random hyperplanes. Tests whether *learned* tactical hyperplanes matter.
- `material_role_hist_only`: zeros out the sign and score features, leaving only the role histogram block as a piece-count baseline. Tests whether covector content adds anything beyond material counts.
- `coordinate_shuffle_by_piece`: applies a deterministic per-role square permutation before tokenisation. Tests whether the spatial layout (and therefore any genuinely structural sign-pattern arrangement) is what the covector readout is exploiting.

The `orientation_mode` scalar is exposed in the diagnostics so ablation harnesses can attach the active mode to each prediction.

## Diagnostics

`forward(x, *, return_covectors=False)` returns a dict containing:

- `logits`: shape `(B,)`, BCE-compatible for the one-logit `puzzle_binary` head.
- `prob`: sigmoid of the puzzle logit.
- `covector_features`: shape `(B, stats.output_dim)`, the flat sign-pattern readout fed to the head.
- `token_mask`: shape `(B, max_pieces)`, the padding mask.
- `piece_count`: shape `(B,)`, number of occupied tokens.
- `soft_signs`: shape `(B, max_pieces, hyperplanes)`, the per-token covector signs masked to the occupied tokens.
- `hyperplane_scores`: shape `(B, max_pieces, hyperplanes)`, the raw signed projections masked to the occupied tokens.
- `positive_counts`, `negative_counts`, `near_zero_counts`: shape `(B, hyperplanes)`, the sign-pattern histograms per hyperplane.
- `sign_agreement`: shape `(B, hyperplanes, hyperplanes)`, the pairwise covector co-occurrence matrix.
- `role_sign_entropy`: shape `(B, roles, hyperplanes)`, per-role per-hyperplane sign entropy.
- `role_histogram`: shape `(B, roles)`, soft per-role count histogram.
- `sign_mean`, `sign_abs_mean`, `score_abs_mean`, `score_std`: shape `(B, hyperplanes)`, per-hyperplane sign / score moments.
- `covector_entropy`: shape `(B, hyperplanes)`, normalised `(+,-,0)` entropy per hyperplane.
- `near_zero_rate`: shape `(B,)`, mean near-zero rate across hyperplanes (a "covector degeneracy" proxy).
- `pairwise_agreement_energy`: shape `(B,)`, the squared sign-agreement Frobenius norm normalised by hyperplane count — the matroid co-occurrence energy.
- `orientation_mode`: integer code identifying the active mode.
- `mechanism_energy`: alias for `pairwise_agreement_energy`, the sign-pattern energy that operationalises the packet's `logic` mechanism family on the covector readout.
- `proposal_profile_strength`: per-board mean of `|sign|` across hyperplanes, a single-scalar proxy for sign-pattern strength.
- `proposal_keyword_count`: integer scalar preserved for compatibility with the project's research-packet diagnostic schema.

When `return_covectors=True` the dict additionally contains `token_features`, `token_embeddings`, `role_probs`, and `square_indices` for ablation harnesses.

## Contract

- Input: `(B, C, 8, 8)` simple_18 board tensor only. CRTK / verification / source / engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- Model shapes: piece tokens `[B, max_pieces, token_feature_dim]`, token embeddings `[B, max_pieces, token_dim]`, hyperplane scores / signs `[B, max_pieces, hyperplanes]`, covector features `[B, 3 * hyperplanes + hyperplanes^2 + roles * hyperplanes + roles + 4 * hyperplanes + 8]`.
- The puzzle decision flows only through `psi(x) = covector_stats(...)` — the head never sees raw piece tokens or per-token scores directly, so the sign-pattern bottleneck is enforced architecturally.
