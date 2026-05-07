# Implementation Notes

- Central code: `src/chess_nn_playground/models/soft_king_cage_path.py`.
- Registry key: `soft_king_cage_path_bottleneck_network`.
- Idea wrapper: `ideas/i052_soft_king_cage_path_bottleneck_network/model.py`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0812_tuesday_los_angeles_king_cage_dp.md`.

The implementation is board-only. It does not consume engine scores, verification metadata, source labels, legal move counts, checkmate/stalemate oracles, candidate move sets, or game-tree consequences.

## Geometry Adapter

The rule branch supports `simple_18` only and fails closed for unknown encodings. The learned trunk still receives the full input tensor.

## Attack And Barrier Fields

Pseudo-legal attack pressure is computed from current-board pieces and blockers. `MonotoneBarrierField` uses positive softplus coefficients for opponent attacks, own occupancy, and opponent occupancy, plus a small nonnegative local adapter for king/edge/side context.

## Dynamic Program

`SoftKingEscapeDP` uses absorbing Chebyshev shells and the true 8-neighbor king-step grid for the main model. It returns final distance fields and cage scalars for all configured radii and temperatures. The central random-grid ablation uses a deterministic degree-matched outgoing neighbor table, and shell shuffling uses deterministic cyclic shifts inside king-centered rings.

The repo config keeps `num_classes: 1`; the model computes internal two-class scores and returns the puzzle margin as `output["logits"]` for the BCE trainer.
