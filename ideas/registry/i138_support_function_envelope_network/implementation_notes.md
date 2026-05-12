# Implementation Notes

- Bespoke model: `src/chess_nn_playground/models/trunk/support_function_envelope_network.py`.
- Idea-local wrapper: `ideas/registry/i138_support_function_envelope_network/model.py`.
- Registry key: `support_function_envelope_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.
- Batch candidate: `Support-Function Envelope Network`.
- Board-only: the model consumes the `simple_18` board tensor and never reads
  engine, verification, source, or CRTK metadata.
- Side-to-move handling: own/opponent piece planes are computed from the
  side-to-move flip via the shared helper; no FEN string is needed.
- Differentiable support function: implemented as
  `tau * logsumexp_s ((<u, coord_s> + log(eps + rho_c(s))) / tau)` with a
  small `epsilon` floor inside the `log`. The `hard_max_support` ablation
  swaps the `logsumexp` for a per-square `max`.
- Direction set: 16 fixed unit vectors with antipodal pairs at indices
  `(2k, 2k + 1)`, so widths and centers are obtained by a single
  `index_select` over the antipode buffer.
