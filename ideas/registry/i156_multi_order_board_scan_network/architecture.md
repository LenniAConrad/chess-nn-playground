# Architecture

`Multi-Order Board Scan Network` realises the source packet's
"different scan orders expose different dependencies" thesis as a
bespoke architecture for the repo's `puzzle_binary` task. A shared
convolutional trunk produces per-square features. Those features are
re-ordered by five fixed scan orders and consumed by a single shared
bidirectional GRU; the order-pooled summaries drive the puzzle logit.

## Implementation Binding

- Registered model name: `multi_order_board_scan_network`
- Source implementation file: `src/chess_nn_playground/models/multi_order_board_scan_network.py`
- Idea-local wrapper: `ideas/registry/i156_multi_order_board_scan_network/model.py`

## Modules

`MultiOrderBoardScanNetwork` accepts the project's `(B, 18, 8, 8)`
board tensor only. CRTK / source / engine / verification metadata is
reporting-only and is not consumed.

1. **Stem.** Two normalised rank/file coordinate planes are
   concatenated to the input. A `3x3 Conv2d -> [BatchNorm2d ->] ReLU`
   stack of `depth` blocks lifts the `(input_channels + 2)` planes to
   the trunk channel dimension while preserving the `8x8` layout.
2. **Square tokens.** The trunk is reshaped to `(B, 64, channels)`,
   one feature vector per square in rank-major order. This is the
   common token sequence that the scan permutations re-order.
3. **Scan permutations.** Five permutations of the 64 squares are
   produced, one per scan order:
   - `rank_major`: identity, `(rank, file) -> rank * 8 + file`.
   - `file_major`: `(rank, file) -> file * 8 + rank`.
   - `diagonal`: ascending by `rank + file`, tiebreak by rank
     (south-west to north-east anti-diagonal sweep).
   - `spiral_from_king`: ascending by Chebyshev distance from the
     side-to-move king square, tiebreak by rank-major square index.
     The king square is read from the side-to-move plane and the
     friendly king plane (`K` for white-to-move, `k` for black-to-move).
   - `center_out`: ascending by Chebyshev distance from the board
     centre, tiebreak by rank-major square index.

   The first four orders are deterministic and stored as a static
   `(num_static_scans, 64)` buffer. `spiral_from_king` is per-sample
   and is gathered from a precomputed `(64, 64)` lookup table indexed
   by the friendly king square.
4. **Shared bidirectional GRU.** A learned per-scan, per-position
   embedding of shape `(num_scans, 64, channels)` is added to each
   re-ordered sequence so the shared GRU can tell the orders apart
   while reusing the same weights across them. All five sequences are
   stacked into a `(num_scans * B, 64, channels)` batch and processed
   by a single `nn.GRU(channels, gru_hidden_dim, bidirectional=True,
   num_layers=num_gru_layers)`. The output has shape
   `(num_scans, B, 64, 2 * gru_hidden_dim)` after reshaping.
5. **Order pooling.** For each scan order, the GRU output is
   mean-pooled along the sequence axis, yielding one summary vector
   per order of dimension `2 * gru_hidden_dim`.
6. **Readout.** The five order summaries are concatenated into a
   `(B, num_scans * 2 * gru_hidden_dim)` feature vector and run
   through a `Linear -> ReLU -> Dropout -> Linear` head to produce
   the single puzzle logit.

## Loss

The default trainer wires standard BCE-with-logits on
`output["logits"]`. The scan stack has no auxiliary loss term; all
gradient signal flows through the trunk, the per-scan position
embeddings, the shared GRU, and the readout head.

## Diagnostics

`forward` returns a dict containing:

- `logits`: shape `(B,)`. BCE-compatible log-odds for the one-logit
  puzzle_binary head.
- `logit`, `prob`: aliases of the log-odds and the sigmoid probability.
- `latent`: shape `(B, channels, 8, 8)`, the post-trunk feature map.
- `square_tokens`: shape `(B, 64, channels)`, the rank-major token
  sequence the scan permutations re-order.
- `scan_perms`: shape `(num_scans, B, 64)`, the per-sample
  permutations applied by each scan order (detached).
- `scan_summaries`: shape `(B, num_scans, 2 * gru_hidden_dim)`, the
  mean-pooled GRU output for each scan order.
- `scan_summary_norms`: shape `(B, num_scans)`, the L2 norm of each
  scan summary (detached).
- `friendly_king_square`: shape `(B,)`, the side-to-move king square
  index used to look up the spiral order (detached).

The `scan_perms`, `scan_summary_norms`, and `friendly_king_square`
diagnostics are detached so they are reportable without biasing the
training loss.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. Engine, verification,
  source, CRTK, principal-variation, mate-score, and best-move
  metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  puzzle_binary BCE-with-logits trainer, plus the diagnostics listed
  above.
- Target mapping: fine labels `0` and `1` map to binary target `0`;
  fine label `2` maps to binary target `1`.
