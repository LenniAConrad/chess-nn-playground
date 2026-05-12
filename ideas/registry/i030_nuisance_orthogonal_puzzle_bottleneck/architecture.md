# Architecture

`Nuisance-Orthogonal Puzzle Bottleneck` (idea i030) implements a board-only
classifier that residualises a learned CNN latent against a deterministic
chess nuisance design matrix using a closed-form batchwise ridge projection.

## Implementation Binding

- Registered model name: `nuisance_orthogonal_puzzle_bottleneck`
- Source implementation file: `src/chess_nn_playground/models/trunk/nuisance_orthogonal_puzzle_bottleneck.py`
- Idea-local wrapper: `ideas/registry/i030_nuisance_orthogonal_puzzle_bottleneck/model.py`

## Components

- `Simple18Adapter`: validates the `simple_18` channel layout (12 piece planes,
  side-to-move, 4 castling planes, en-passant) and exposes the planes the
  deterministic nuisance extractor needs. Other encodings raise immediately so
  the adapter fails closed.
- `DeterministicNuisanceExtractor`: returns a 64-d nuisance vector
  ``n(B)`` composed of normalized piece counts (12), material/phase summaries
  (6: white material, black material, balance, |imbalance|, non-pawn phase,
  total piece count), the side-to-move sign (1), four castling flags, the
  en-passant presence + 8 file one-hot (9), eight king-coordinate features
  (white/black file/rank in [-1,1] plus four edge-distance summaries), 16
  pawn-file counts (white and black per file), and eight coarse occupancy
  marginals (4 rank groups + 4 file groups). The vector has no trainable
  parameters.
- `FixedNuisanceFeatureMap`: expands ``n`` with all unique pairwise products
  (and squares) projected through a deterministic ``seed=42`` Gaussian
  random projection of dimension ``nuisance_expansion_dim`` (registered as a
  non-trainable buffer), concatenates the raw nuisance vector with the
  expansion, slices to the configured ``nuisance_rank`` columns, and applies a
  non-affine LayerNorm. Output ``Q in R^{B x k}`` is centred per mini-batch.
- `ConvResidualTrunk`: ``Conv3x3 -> Norm -> GELU`` stem followed by ``depth``
  residual blocks at ``channels`` width, a projection conv to ``proj_channels``
  (kept comparable to the latent dimension), global average pool, and a linear
  head into ``latent_dim`` followed by GELU + LayerNorm. The trunk consumes the
  full ``simple_18`` tensor so the model still sees board structure beyond the
  deterministic nuisance vector.
- `BatchRidgeOrthogonalProjector`: closed-form residualiser. Centres ``H`` and
  ``Q`` across the mini-batch, builds the gram ``G = Q^T Q + lambda I_k`` in
  float32, solves ``A = G^{-1} Q^T H`` with ``torch.linalg.solve`` (LSTSQ
  fallback), and returns ``Z = H_c - gamma * Q_c A``. ``gamma`` and
  ``ridge_lambda`` are config knobs; ``gamma=0`` is the central falsification
  ablation. The projection is never materialised as a ``[b, b]`` matrix.
- `ClassifierHead`: ``LayerNorm -> Dropout -> Linear -> GELU -> Linear``
  producing ``num_classes`` logits. For the puzzle_binary contract,
  ``num_classes=1`` and the logit is squeezed so the trainer receives a
  ``[B]``-shaped tensor.

## Forward Contract

```text
output = model(x)
x.shape == (batch, input_channels=18, 8, 8)
output["logits"].shape == (batch,)        # because num_classes == 1
```

`output` additionally exposes the deterministic nuisance vector ``n`` and the
fixed feature matrix ``Q`` (for diagnostics and ablations such as shuffled
``Q``), the trunk and projected latents, the residual covariance norm
``||Q^T Z / b||`` (used to verify empirical orthogonality), the post-projection
latent variance, the empirical rank of ``Q^T Q``, scalar nuisance summaries
(material balance, |imbalance|, side-to-move sign, castling total, en-passant
presence, king separation), and the projection hyperparameters ``gamma`` and
``ridge_lambda`` echoed per-row for trainer artifacts.

## Mathematical Operator

For a mini-batch of size ``b`` with ``H in R^{b x d}`` and ``Q in R^{b x k}``
centred across the batch and ridge ``lambda > 0``:

```text
Z = H - gamma * Q (Q^T Q + lambda I_k)^{-1} Q^T H
```

When ``lambda = 0`` and ``rank(Q) = k``, ``Q^T Z = 0`` exactly: every column of
``Z`` has zero empirical linear covariance with every column of ``Q`` on that
mini-batch. Setting ``gamma = 0`` recovers the central falsification ablation
because ``Z`` collapses to the centred ``H`` and the trunk/head retain their
full nuisance dependence.

## Encoding Support And Fail-Closed Behaviour

The first implementation supports the `simple_18` encoding only because the
deterministic nuisance extractor needs explicit piece/side/castling/en-passant
channel semantics. Passing any other encoding name to the adapter raises a
``ValueError`` immediately, preserving the ``fail_closed_semantics`` contract
demanded by the math thesis. LC0 encodings can be added later by registering
explicit current-board channel indices; the trunk may consume all input
channels even then, but the nuisance extractor must only read channels with
declared semantics.
