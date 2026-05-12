# Implementation Notes

- Central code: `src/chess_nn_playground/models/patch_mixer_boardnet.py`.
- Registry key: `patch_mixer_boardnet`.
- Idea wrapper: `ideas/registry/i146_patch_mixer_boardnet/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.
- Batch candidate: `Patch Mixer BoardNet`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Patch extraction uses `torch.nn.Unfold`, so no extra dependency is needed.
