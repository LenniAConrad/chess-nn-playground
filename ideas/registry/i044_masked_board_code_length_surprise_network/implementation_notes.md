# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/masked_surprise_codec.py`.
- Idea-local builder: `ideas/registry/i044_masked_board_code_length_surprise_network/model.py`
  is a thin `build_model_from_config` wrapper around
  `build_masked_board_code_length_surprise_network_from_config`.
- Registry key: `masked_board_code_length_surprise_network`.
- Source packet:
  `ideas/research/packets/classic/chess_nn_research_2026-04-21_0739_tuesday_los_angeles_masked_surprise_codec.md`.
- Input contract: board-only `(B, 18, 8, 8)` `simple_18` tensors. CRTK,
  source, engine, and verification metadata remain reporting-only and
  are never consumed by the codec or the classifier.
- LC0 fail-closed: `MaskedBoardCodeLengthSurpriseNet` raises
  `ValueError` for any encoding other than `simple_18` until an
  explicit current-board piece-channel schema is wired in for
  `lc0_static_112` / `lc0_bt4_112`.
- Mask bank: fixed `2x2_residue` (four masks of 16 squares each), each
  square is masked exactly once. The bank is stored as a non-persistent
  buffer so the experiment is deterministic across processes.
- Mask chunking: `mask_chunk_size` (default 2) controls peak memory.
  The forward path processes chunks of the mask bank and accumulates
  spatial sums, so memory scales as
  `O(B * mask_chunk_size * (C + codec_width + 13) * 8 * 8)`.
- Surprise scaling: per-square code length is clipped to
  `surprise_clip_nats=8.0` and rescaled with `log1p(.)` before being
  concatenated to the classifier input.
- Codec freezing and detachment: `freeze_codec=True` excludes codec
  parameters from gradient updates and runs the codec under
  `torch.no_grad`. `detach_surprise=True` additionally detaches the
  spatial fields before the classifier so backprop cannot push
  gradients into the codec. Either flag can be flipped to `False` for
  the joint-fine-tune ablation, but the markdown protocol treats
  joint training as a separate report.
- Codec checkpoint loading: `MaskedBoardCodec.load_state_dict` is
  available for downstream scripts that pretrain the codec with the
  label-free `codec_nll` loss before running the classifier
  trainer.
- Tokenizer strictness: `strict_tokenizer=True` raises if a square has
  more than one active piece plane (a data-quality guard); the default
  config uses non-strict argmax for robustness on noisy splits.
