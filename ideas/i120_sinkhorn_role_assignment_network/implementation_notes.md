# Implementation Notes

- Central code: `src/chess_nn_playground/models/sinkhorn_role_assignment_network.py`.
- Idea-local wrapper: `ideas/i120_sinkhorn_role_assignment_network/model.py`.
- Registry key: `sinkhorn_role_assignment_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.
- Batch candidate: `Sinkhorn Role Assignment Network`.
- Input is the simple_18 board tensor only; CRTK / engine / source metadata is reporting-only and never consumed by the model.
- Sinkhorn iterations run in log domain on a masked kernel so padded piece slots and inactive role columns transport zero mass.
- The dustbin role (extra column) absorbs pieces that do not match any prototype, so the row mass constraint ``sum_j A[i, j] = mask[i]`` is exactly satisfied.
