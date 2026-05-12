# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/trunk/agreement_variance_head_net.py`.
- Idea-local wrapper: `ideas/registry/i153_agreement_variance_head_net/model.py`.
- Registry key: `agreement_variance_head_net`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.
- The model is intentionally board-only and does not consume engine,
  verification, source, principal-variation, mate-score, best-move, or
  CRTK metadata as input.
- Architecture hyperparameters available in `config.yaml`:
  - `channels`: width of the convolutional trunk (default 64).
  - `depth`: number of residual blocks in the trunk (default 2).
  - `hidden_dim`: hidden width of each cheap classifier head
    (default 96).
  - `num_heads`: number of cheap heads sharing the trunk (default 5).
    Must be `>= 2` so the variance diagnostic is well defined.
  - `dropout`: dropout used in residual blocks and head MLPs.
  - `use_batchnorm`: toggles `BatchNorm2d` in the trunk.
- The mean of per-head logits is the BCE-with-logits training target.
  Per-head variance / disagreement is computed under `torch.no_grad()`
  and reported as a diagnostic; it is not added to the loss.
- Each head is initialised with an independent Kaiming-uniform draw so
  the heads do not start identical and the variance diagnostic stays
  informative early in training.
