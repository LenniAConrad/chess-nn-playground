# Architecture

`Cross-Stitch CNN-Token Fusion Net` realises the source packet's
"cross-stitch coupling between a CNN board branch and an occupied-piece
token branch" thesis as a bespoke architecture for the repo's
`puzzle_binary` task. Two branches read the same simple_18 board and
exchange information at every intermediate stage through learned
``2x2`` (or per-group ``G x 2 x 2``) cross-stitch matrices initialised
to the identity.

## Implementation Binding

- Registered model name: `cross_stitch_cnn_token_fusion_net`
- Source implementation file: `src/chess_nn_playground/models/trunk/cross_stitch_cnn_token_fusion_net.py`
- Idea-local wrapper: `ideas/registry/i157_cross_stitch_cnn_token_fusion_net/model.py`

## Modules

`CrossStitchCNNTokenFusionNet` accepts the project's `(B, 18, 8, 8)`
board tensor only. CRTK / source / engine / verification metadata is
reporting-only and is not consumed.

1. **Stem.** A single ``3x3 Conv2d -> [BatchNorm2d ->] GELU`` block
   lifts the input planes to the shared trunk width
   ``C = board_width = token_width``.
   `Simple18PieceTokenExtractor` reads up to `max_piece_tokens`
   occupied-piece tokens (rank/file coordinates, side-to-move, castling
   rights, en-passant flag, occupancy score, etc.). A small
   ``Linear(TOKEN_FEATURE_DIM, C) -> GELU -> Linear(C, C)`` head lifts
   each token feature vector to the same width ``C`` so the
   cross-stitch unit can mix board and token summaries channel-by-channel.
2. **Cross-stitch stages.** For each of ``num_stages`` stages:
   - One ``3x3 Conv2d -> Norm -> GELU`` block runs on the board branch.
   - One residual token-mixer block (LayerNorm + token-wise MLP +
     masked summary gate) runs on the token branch.
   - The branch summaries ``b = mean_pool(h_board)`` and
     ``p = masked_mean(h_token)`` are computed.
   - The two summaries are mixed by a learned per-group ``2x2``
     ``CrossStitchUnit``:

     ```
     [b'_g]   [a_g  c_g] [b_g]
     [p'_g] = [d_g  e_g] [p_g]
     ```

     The matrix is stored as a ``(num_groups, 2, 2)`` parameter
     initialised to the identity so the unit starts as a no-op and
     learns its mixing coefficients.
   - The mixed summaries are injected back into each branch:
     ``h_board += board_adapter(b_new)`` (broadcast over ``8x8``);
     ``h_token += token_adapter(p_new)`` (broadcast over the token
     axis, masked by the token mask).
3. **Final head.** The final pooled summaries ``b_final`` and
   ``p_final``, the material summary, and the per-stage off-diagonal
   energy are concatenated and run through a small
   ``LayerNorm -> Linear -> GELU -> Linear -> GELU -> Linear`` head
   that produces the puzzle logit.

## Cross-Stitch Mathematics

`CrossStitchUnit` splits the channel-width-``C`` summaries into
``G = cross_stitch_groups`` groups and applies a per-group ``2x2``
matrix. ``A = I`` at initialisation gives the parent late-fusion model
exactly. As training progresses the off-diagonal entries learn the
amount of board-to-token and token-to-board transfer and the diagonal
entries learn within-branch rescaling. The ``diagonal_stitch``
ablation forces the off-diagonals to zero (no cross-branch exchange)
and the ``late_fusion_only`` ablation disables intermediate
cross-stitch entirely (only the final concat fuses branches).

## Loss

The default trainer wires standard BCE-with-logits on
``output["logits"]``. There is no auxiliary loss; gradient flows
through the trunk, the per-stage conv blocks, the per-stage token
mixer blocks, the cross-stitch matrices, the per-stage adapters, and
the readout head.

## Diagnostics

`forward` returns a dict containing:

- `logits`: shape `(B,)`. BCE-compatible log-odds.
- `logit`, `prob`: aliases of the log-odds and the sigmoid probability.
- `board_latent`: shape `(B, channels, 8, 8)`, the post-final-stage
  board feature map.
- `token_latent`: shape `(B, max_piece_tokens, channels)`, the
  post-final-stage token sequence.
- `board_pool_final`, `token_pool_final`: shape `(B, channels)`, the
  pooled branch summaries that feed the head.
- `material_summary`: shape `(B, MATERIAL_SUMMARY_DIM)`, the extractor
  material summary.
- `token_count`: shape `(B,)`, the number of occupied tokens.
- `cross_stitch_matrices`: shape `(num_stages, num_groups, 2, 2)`, the
  detached learned mixing matrices per stage.
- `offdiag_energy_per_stage`: shape `(num_stages,)`, the mean of
  ``a_01^2 + a_10^2`` across groups for each stage.
- `board_to_token_transfer`, `token_to_board_transfer`: shape `(B, num_stages)`,
  the per-stage L2 magnitude of the cross-branch contribution to the
  mixed summary (detached).
- `board_norms_pre`, `board_norms_post`, `token_norms_pre`,
  `token_norms_post`: shape `(B, num_stages)`, the L2 norm of each
  branch summary before and after stitching at each stage (detached).

## Contract

- Input: `(B, 18, 8, 8)` simple_18 board tensor only. Engine,
  verification, source, CRTK, principal-variation, mate-score, and
  best-move metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  puzzle_binary BCE-with-logits trainer, plus the diagnostics listed
  above.
- Target mapping: fine labels `0` and `1` map to binary target `0`;
  fine label `2` maps to binary target `1`.
