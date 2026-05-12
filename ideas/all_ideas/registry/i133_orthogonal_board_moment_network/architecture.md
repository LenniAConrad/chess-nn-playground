# Architecture

`Orthogonal Board Moment Network` projects the simple_18 board tensor into a
small bank of learned scalar fields and reads out their fixed Legendre /
Chebyshev polynomial moments as a compact global shape descriptor. The moment
tensor is fused with a CNN summary and classified into one ``puzzle_binary``
logit (fine labels `0` and `1` map to non-puzzle, fine label `2` maps to
puzzle).

## Input And Field Bank

- Input is the repo `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- A `BoardConvStem` (depth `2` by default) produces a `(batch, channels, 8, 8)`
  feature map. A `1x1` convolution `field_head` projects this into
  `num_fields` (default `24`) learned scalar fields `F_c(u, v)` over the 8x8
  grid.
- CRTK / source / engine metadata is ignored — only the board tensor is
  consumed by the model.

## Orthogonal Polynomial Moments

- Board coordinates are normalised to centred grid points
  `u_k, v_k = (k + 0.5) / 4 - 1` for `k in 0..7`, so `u, v in [-1, 1]`.
- Two fixed polynomial families are evaluated on the same grid up to degree
  `max_degree - 1` (default `4`, giving degrees `0..3`):

  ```text
  Legendre  : P_0, P_1, P_2, P_3
  Chebyshev : T_0, T_1, T_2, T_3
  ```

  Both are computed by their standard recurrences and stored as fixed
  non-trainable buffers.
- For every learned field `F_c` the model computes the mixed tensor-product
  moments

  ```text
  m_{family, c, i, j} = sum_{u, v} F_c(u, v) * basis_i(u) * basis_j(v)
  ```

  via two `einsum("bcrf, fi, rj -> bcij", fields, basis, basis)` calls,
  yielding a moment tensor of shape `(batch, 2, num_fields, K, K)`.

## Degree Families And Degree Dropout

- Moments are split into total-degree groups `i + j`:

  ```text
  low    : i + j in {0, 1}   (material / centre balance)
  middle : i + j in {2, 3}   (side / wing skew)
  high   : i + j in {4, ...} (local concentration / high-order shape)
  ```

- During training a Bernoulli per-sample group mask drops middle and high
  groups with probability `degree_dropout` (default `0.1`); the low group is
  always kept so the gradient path through `field_head` is preserved.
  Inverted-dropout scaling is applied so the expected moment magnitude is
  preserved.

## Moment Head, CNN Summary, And Classifier

- The (post-dropout) moment tensor is flattened and passed through a
  `Linear -> GELU -> Dropout` `moment_mlp` of width `hidden_dim`.
- A compact CNN summary is built from the same backbone by concatenating
  `mean` and `max` pools to a `(batch, 2 * channels)` vector.
- Diagnostic energies (per-degree-group, per-family, normalised) and the
  current degree-dropout keep mask are concatenated to the moment features
  and the CNN summary. A two-layer GELU MLP with dropout reads the fusion
  vector and emits one ``puzzle_binary`` logit.
- The forward pass returns a dict whose `logits` tensor has shape `(batch,)`
  alongside per-degree, per-family, and per-summary diagnostic tensors of
  shape `(batch,)` for ablation analysis.

## Implementation Binding

- Registered model name: `orthogonal_board_moment_network`.
- Source implementation: `src/chess_nn_playground/models/orthogonal_board_moment_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i133_orthogonal_board_moment_network/model.py`.
