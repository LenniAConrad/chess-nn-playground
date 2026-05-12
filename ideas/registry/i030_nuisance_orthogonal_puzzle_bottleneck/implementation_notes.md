# Implementation Notes

## Source Code

- Bespoke implementation: `src/chess_nn_playground/models/trunk/nuisance_orthogonal_puzzle_bottleneck.py`
- Idea-local wrapper: `ideas/registry/i030_nuisance_orthogonal_puzzle_bottleneck/model.py`
- Registry key: `nuisance_orthogonal_puzzle_bottleneck`
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0508_tuesday_local_nuisance_orthogonal.md`

## Encoding Contract

The first implementation supports the `simple_18` encoding only. The
`Simple18Adapter` rejects any other encoding (`fail_closed_semantics=true`),
matching the math-thesis requirement that deterministic nuisance extraction
only consume channels with declared semantics. CRTK/source/engine metadata is
never used as model input.

## Deterministic Nuisance Vector (m = 64)

Component layout, in order:

| Slice | Size | Description |
|---|---|---|
| `0..11` | 12 | Normalized piece counts: white `P,N,B,R,Q,K` then black `p,n,b,r,q,k`, divided by canonical maxima `(8,2,2,2,1,1)`. |
| `12..17` | 6 | Material/phase summaries: white material / 39, black material / 39, balance, |imbalance|, non-pawn material total / 14, total piece count / 32. |
| `18` | 1 | Side-to-move sign: `+1` for white-to-move, `-1` for black-to-move (read from the side-to-move plane mean). |
| `19..22` | 4 | Castling-right flags `WK, WQ, BK, BQ` (each plane is constant 0/1 in `simple_18`). |
| `23..31` | 9 | En-passant: presence scalar plus 8 file one-hot. |
| `32..39` | 8 | King coordinates: white file/rank in `[-1,1]`, black file/rank in `[-1,1]`, then white/black edge-distance summaries `1 - |coord|`. |
| `40..55` | 16 | Pawn-file counts: white pawns per file (8) followed by black pawns per file (8), divided by 8. |
| `56..63` | 8 | Coarse occupancy marginals: 4 rank-group sums + 4 file-group sums, each divided by 16. |

The extractor is a `nn.Module` with no trainable parameters. It uses fixed
buffers for piece values and the non-pawn-mask only.

## Fixed Nuisance Feature Map

`FixedNuisanceFeatureMap` concatenates the raw nuisance vector with a
deterministic Gaussian random projection of all unique pairwise products
(including squares) of nuisance entries. The projection matrix is generated
once with `seed=42` and registered as a non-trainable buffer. The expanded
matrix is sliced to the configured `nuisance_rank` columns and passed through
a non-affine LayerNorm so columns share scale before residualisation.

Defaults: `nuisance_rank=64`, `nuisance_expansion_dim=64`.

## Projection Math And Numerical Stability

Inside `BatchRidgeOrthogonalProjector` we compute:

```python
H_c = H - H.mean(0)
Q_c = Q - Q.mean(0)
G   = Q_c.T @ Q_c + ridge_lambda * I_k    # [k, k]
A   = solve(G, Q_c.T @ H_c)               # [k, d]
Z   = H_c - gamma * (Q_c @ A)
```

We never instantiate the `[b, b]` projection matrix. The solve runs in
`float32` for stability; `torch.linalg.solve` is used by default with a
`torch.linalg.lstsq` fallback if the solve fails. Defaults match the math
thesis: `ridge_lambda=1e-3`, `gamma=1.0`.

When `b <= k + 2` the Gram matrix is rank-deficient before regularisation but
the ridge term keeps it invertible; the projection is still the closed-form
ridge residualiser. The recommended training batch size is `>= 256` so this
edge case appears only in tests.

## Diagnostic Outputs

The model returns a dict with at minimum `logits`, the raw nuisance vector
`nuisance_vector`, the fixed feature matrix `nuisance_features`, the trunk
latent `trunk_latent`, the projected latent `projected_latent`, the residual
covariance norm `residual_cov_norm`, the post-projection latent variance
`latent_variance`, the empirical nuisance Gram rank `nuisance_rank_estimate`,
several scalar nuisance summaries, and the projection hyperparameters echoed
per-row so they round-trip into trainer artifacts.

## Fail-Closed Behaviour

- Encoding other than `simple_18`: `Simple18Adapter` raises `ValueError`.
- Wrong number of input channels: `require_board_tensor` raises `ValueError`.
- Unknown nuisance rank larger than the available feature dimension: the
  feature map raises `ValueError` at construction time.

## Configuration Surface

The builder reads from a flat model dict:

```yaml
model:
  name: nuisance_orthogonal_puzzle_bottleneck
  input_channels: 18
  num_classes: 1
  channels: 64
  hidden_dim: 96
  depth: 2
  dropout: 0.1
  use_batchnorm: true
  latent_dim: 256
  nuisance_rank: 64
  nuisance_expansion_dim: 64
  ridge_lambda: 0.001
  projection_gamma: 1.0
  encoding: simple_18
  fail_closed_semantics: true
  nuisance_seed: 42
```

The trainer fills `num_classes=1` for `puzzle_binary` mode.
