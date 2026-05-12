# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/harmonic_board_potential_network.py`.
- Idea-local wrapper: `ideas/registry/i059_harmonic_board_potential_network/model.py`
  exposes `build_model_from_config(config)` and delegates to
  `build_harmonic_board_potential_network_from_config`.
- Registry key: `harmonic_board_potential_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2045_friday_shanghai_harmonic_potential.md`.
- Input is the simple_18 board tensor only; CRTK/source/engine metadata
  are never consumed by the model.
- The Green matrices are precomputed dense `(64, 64)` buffers; they are
  not trainable parameters and only the 1x1 charge encoder plus the head
  MLP carry learnable weights.
- `ablation` field selects between `none`, `random_orthogonal_solver`,
  `local_gaussian_solver`, and `charge_only_stats`. The first three keep
  the same matrix contract so the head input dimensionality is unchanged;
  `charge_only_stats` zeros the potentials so only charge moments reach
  the head.
