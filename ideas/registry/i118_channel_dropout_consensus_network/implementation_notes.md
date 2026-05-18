# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/channel_dropout_consensus.py`.
- Registry key: `channel_dropout_consensus_network`.
- Builder: `build_channel_dropout_consensus_network_from_config`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.
- Batch candidate: `Channel Dropout Consensus Network`.
- Input contract: simple_18 only (18 planes, 8x8); the model raises on other encodings or input-channel counts.
- Output contract: puzzle_binary one-logit (`num_classes = 1`); the model raises if requested otherwise. The forward dict includes the puzzle logit, probability, per-view pooled latents, consensus/disagreement summaries, and bookkeeping diagnostics consumed by `report_template.md`.
- Trainable parameters: the shared encoder (`Phi`) and the MLP head only. Drop-channel masks and the full-view index are registered buffers (non-persistent) and therefore not part of the optimizer state.
- View construction: deterministic, computed at module init from `DETERMINISTIC_VIEW_DROP_CHANNELS`. The full view is index `0`; the remaining views drop pawns, minors, majors, white pieces, and black pieces respectively. Non-piece planes (side-to-move, castling, en-passant) are always preserved.
- Multi-view shared pass: views are stacked along the batch axis (`reshape` to `(B*V, 18, 8, 8)`) so the encoder, its BatchNorm/Dropout, and gradient flow all see every view in a single forward call. This is what makes the model a *shared*-encoder ensemble rather than `V` independent encoders.
- This idea is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Ablation map (see `ChannelDropoutConsensusNetwork.ABLATIONS`): `none`, `full_view_only`, `mean_only`, `random_channel_masks`, `train_dropout_only`. `train_dropout_only` swaps the multi-view trunk for ordinary `nn.Dropout2d` channel dropout on the full board; this collapses to a single-view classifier at inference.
- Tests live at `tests/test_channel_dropout_consensus_network.py` and cover the registry contract, configuration parsing, forward keys, gradient flow through the shared encoder and head, the buffer-vs-parameter discipline of the mask table, every ablation's documented behavior, and the idea-folder conformance audits.
