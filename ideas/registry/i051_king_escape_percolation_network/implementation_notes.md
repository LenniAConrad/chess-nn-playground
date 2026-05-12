# Implementation Notes

- Central code: `src/chess_nn_playground/models/king_escape_percolation.py`.
- Registry key: `king_escape_percolation_network`.
- Idea wrapper: `ideas/registry/i051_king_escape_percolation_network/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0811_tuesday_pacific_king_escape_percolation.md`.

The implementation is board-only. It does not consume engine output, verification metadata, source tags, row provenance, candidate labels, legal move counts, or move-tree consequences.

## Encoding Adapter

The rule geometry is implemented for `simple_18` only. The adapter decodes the current-board piece planes and side-to-move channel directly. Other encodings raise `ValueError` until their current-board channel semantics are explicitly mapped.

## Attack Maps

`PseudoLegalAttackMaps` computes frozen-board attacks for both colors:

- pawns use color-specific diagonals;
- knights and kings use fixed offsets;
- bishops, rooks, and queens scan rays through empty squares and stop after the first occupied square.

The attack maps are deterministic current-board geometry and are not legal move generation.

## Escape Operator

`EscapeCostField` builds side-relative geometric cost features and applies a shared nonnegative `1x1` MLP. `SoftMinEscapeDP` runs the multi-temperature king-neighborhood recurrence, saves configured snapshots, and emits both escape maps and a compact vector of edge, ring, mass, and side-to-move aligned summaries.

The default config still uses the project BCE trainer with `num_classes: 1`; internally the model has two scores and returns the puzzle margin as `output["logits"]`.

## Controls

The code supports the main packet controls through `model.ablation_mode`: `ring_bin_cost_shuffle`, `no_attack_cost`, and `no_occupancy_barrier`. The ring/bin control uses a deterministic cyclic permutation inside each preserved bin, so repeated runs are stable without an extra random seed.
