# Architecture

`Occupancy Run-Length Segment Encoder` is a board-only puzzle-binary classifier for sliding-tactic line structure. It compresses each rank, file, diagonal, and anti-diagonal into deterministic run-length segment rows, embeds those rows with a shared MLP, pools by line type, and fuses the segment branch with a compact CNN board summary.

## Input And Line Extraction

- Input is the repo `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- Piece occupancy is computed from the first 12 planes and collapsed into six color-agnostic piece-type channels.
- The model extracts 46 fixed line scans: 8 ranks, 8 files, 15 diagonals, and 15 anti-diagonals.
- Direction buckets are encoded relative to side to move from the `white_to_move` plane, so the same absolute file or diagonal can be interpreted as forward/backward for the active side.

## Deterministic Segment Rows

For every line, the model enumerates all contiguous intervals of length 1 through 8 and keeps the top `Smax` deterministic segment rows. Two segment kinds are represented:

- Empty runs: maximal contiguous empty intervals bounded by occupied blockers or board edges.
- Occupied runs: maximal contiguous occupied intervals bounded by empty squares or board edges.

Each segment feature row contains:

- normalized empty-run length
- normalized occupied-run length/count
- normalized start, end, and center positions along the line
- first and last endpoint piece-type one-hots
- king-slider gap signal for empty segments bounded by a king and a compatible slider candidate
- king-zone contact flag
- open-to-edge flag
- empty/occupied segment kind bits
- line-type one-hot
- side-relative direction bucket
- side-to-move scalar

This is intentionally not a full ray grammar or automaton. It summarizes structural intervals and their endpoint facts rather than processing every square token in sequence.

## Segment And Board Branches

A shared segment MLP embeds each selected segment row. Segment embeddings are pooled into line vectors, then pooled globally and by line type: rank, file, diagonal, and anti-diagonal. A small convolutional board stem supplies mean and max board summaries in parallel.

The classifier receives:

- board CNN mean and max features
- global segment mean and max features
- rank/file/diagonal/anti-diagonal segment summaries
- scalar run-length diagnostics

It returns one puzzle logit for the `puzzle_binary` task. Fine labels 0 and 1 are non-puzzle; fine label 2 is puzzle.

## Diagnostics

The output dictionary includes `logits` plus segment diagnostics such as `empty_run_mean`, `occupied_run_mean`, `open_segment_fraction`, `king_zone_segment_fraction`, `king_slider_gap_mean`, `segment_count_mean`, `endpoint_type_entropy`, `segment_branch_energy`, and line-type contribution/energy values.

## Implementation Binding

- Registered model name: `occupancy_run_length_segment_encoder`.
- Source implementation: `src/chess_nn_playground/models/occupancy_run_length_segment.py`.
- Idea-local wrapper: `ideas/i128_occupancy_run_length_segment_encoder/model.py`.
