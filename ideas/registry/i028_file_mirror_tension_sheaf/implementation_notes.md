# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/file_mirror_tension_sheaf.py` (`FileMirrorTensionSheafNet`).
- Idea-local wrapper: `ideas/registry/i028_file_mirror_tension_sheaf/model.py` (`build_model_from_config`).
- Registry key: `file_mirror_tension_sheaf`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0437_tuesday_los_angeles_mirror_tension_sheaf.md`.
- Board-only by construction: only the `simple_18` board tensor enters the model. Engine, verification, source, proposal, unresolved, and CRTK fields are never consumed as inputs.
- The file-mirror operator is `Simple18Mirror` (spatial file flip plus kingside <-> queenside castling-plane swap). The mirror is involutive on the supported encoding.
