# Implementation Notes

- Central code: `src/chess_nn_playground/models/occupancy_run_length_segment.py`.
- Registry key: `occupancy_run_length_segment_encoder`.
- Idea wrapper: `ideas/i128_occupancy_run_length_segment_encoder/model.py`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.
- Batch candidate: `Occupancy Run-Length Segment Encoder`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.

The implementation enumerates all fixed board lines and all contiguous intervals within each line. Empty intervals become segment rows only when they are maximal empty runs bounded by an occupied square or the board edge. Occupied intervals become segment rows only when they are maximal occupied runs. The model keeps the top segment rows per line by deterministic structural score and pads absent rows with zeros.

Endpoint piece types, king-zone contact, open-edge flags, side-relative direction buckets, and king-slider gap signals are computed directly from the `simple_18` current-board planes. A shared segment MLP embeds these rows, line/type pooling summarizes them, and a small CNN branch supplies square-local context before the final one-logit puzzle head.

The packet's ablations (`histogram_only`, `no_endpoint_types`, `random_line_assignment`, `cnn_only`, and `run_lengths_only`) are experiment variants rather than runtime branches in the production model.
