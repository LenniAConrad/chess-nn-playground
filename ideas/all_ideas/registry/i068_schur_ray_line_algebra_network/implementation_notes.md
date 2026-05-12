# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/schur_ray_line_algebra.py`.
- Registry key: `schur_ray_line_algebra_network`.
- Idea-local wrapper: `ideas/all_ideas/registry/i068_schur_ray_line_algebra_network/model.py` (thin `build_model_from_config`).
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2127_friday_shanghai_schur_ray_line_algebra.md`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The 46 ray incidence is fixed (8 ranks, 8 files, 15 diagonals, 15 anti-diagonals); only the line mode coefficients are learned.
- The Schur system is solved per head with a Cholesky factorization of an `r x r` matrix (`r = line_rank`), which is cheaper than a 64x64 solve for the `direct_64_solve` ablation.
