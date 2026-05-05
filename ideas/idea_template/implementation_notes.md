# Implementation Notes

- Start from the pre-wired chunks in `model.py`: `BoardConvStem`, `GlobalPoolClassifier`, and `require_board_tensor`.
- Keep the model builder named `build_model_from_config(config)` until registering it in `src/chess_nn_playground/models/registry.py`.
- Keep `config.yaml` fields aligned with `idea.yaml`: `idea_id`, `slug`/`model.name`, and `device: nvidia`.
- Replace only the idea-specific feature body; do not rewrite the shared trainer, dataloader contract, checkpointing, or report pipeline.
- Add a forward-shape test before training: input `(batch, input_channels, 8, 8)`, output `(batch, num_classes)`.
