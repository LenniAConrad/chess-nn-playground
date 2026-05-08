# Implementation Notes

- Central code: `src/chess_nn_playground/models/support_polar_zonotope.py`.
- Idea-local wrapper: `ideas/i079_support_polar_zonotope_certificate_network/model.py`.
- Registry key: `support_polar_zonotope_certificate_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0718_tuesday_new_york_support_polar_zonotope.md`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The forward pass exposes the certificate `(u_k, sigma, beta_k)` and per-pair projections so the packet's `forward_with_details` requirement is realised by the standard `forward`.
