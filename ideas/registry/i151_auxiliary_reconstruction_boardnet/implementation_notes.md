# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/auxiliary_reconstruction_boardnet.py`.
- Idea-local wrapper: `ideas/registry/i151_auxiliary_reconstruction_boardnet/model.py`.
- Registry key: `auxiliary_reconstruction_boardnet`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.
- The model is intentionally board-only and does not consume engine,
  verification, source, principal-variation, mate-score, best-move, or
  CRTK metadata as input.
- Architecture hyperparameters available in `config.yaml`:
  - `encoder_width` (alias `channels`): encoder trunk width (default 64).
  - `encoder_depth` (alias `depth`): residual blocks in the encoder
    (default 4 in the packet, the local config keeps it at 2).
  - `decoder_width`: hidden width of the auxiliary decoder (default 32).
  - `hidden_dim`: classifier-head width (default 96).
  - `dropout`: dropout used inside residual blocks, the classifier head,
    and the decoder.
  - `use_batchnorm`: toggles `BatchNorm2d` in the trunk and decoder.
  - `lambda_recon`: weight on the reconstruction BCE term (default
    0.05). Only consumed when `auxiliary_reconstruction_loss` is wired.
  - `reconstruction_targets`: optional list of input plane indices to
    reconstruct. Defaults to all simple_18 planes (`0..17`). Set to
    `[0..11]` to reconstruct piece occupancies only,
    `[0..12]` to add side-to-move, etc.
- `auxiliary_reconstruction_loss` exposes the combined puzzle +
  reconstruction objective for ablations. The default trainer runs the
  standard BCE-with-logits on `output["logits"]`, which already trains
  the encoder + classifier path; the reconstruction term is only added
  when ablation configs request it.
