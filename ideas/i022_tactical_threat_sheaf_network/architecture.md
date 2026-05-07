# Architecture

`Tactical Threat-Sheaf Network` implements the packet's board-only attack-defense sheaf for the repo's `puzzle_binary` task.

## Implementation Binding

- Registered model name: `tactical_threat_sheaf_network`
- Source implementation file: `src/chess_nn_playground/models/tactical_threat_sheaf.py`
- Idea-local wrapper: `ideas/i022_tactical_threat_sheaf_network/model.py`

## Modules

`EncodingPieceAdapter` decodes current piece planes, side-to-move, colors, piece types, and side-relative roles from `(B, C, 8, 8)` board tensors. It supports `simple_18` and LC0-style 112-plane tensors with the current piece slice in the first twelve planes. It does not consume engine output, fine labels, source tags, candidate metadata, or verifier fields.

`PseudoLegalAttackBuilder` constructs a padded dynamic complex over the 64 board squares. Pawns create diagonal attack/control edges; knights and kings create jump/step edges; bishops, rooks, and queens trace slider rays through empty squares and stop at the first occupied blocker. Blocker edges are typed as own defense, enemy attack, king contact, or pin-line when an enemy blocker shields its own king.

Each edge carries source and target square, relation type, target role, source side-relative role, geometry family, edge group, padding mask, degree-normalized edge weight, and pin flag. The learned relation type compactly ties source role, source piece, target tactical bucket, and direction bucket; target-role and geometry embeddings are also supplied to the gate.

`SquareStem` maps raw square planes plus decoded piece/color/role features, side-to-move, coordinates, and optional square embeddings into vertex stalks `z in R^d`.

`SheafRestrictionBank` stores learned source and target restrictions for every relation type. It supports `diagonal_lowrank`, `full`, and `identity_ablation` forms. The default coboundary is:

```text
delta_e = A_src[type_e] z_src(e) - A_dst[type_e] z_dst(e)
```

`ThreatSheafLayer` computes gated edge tension, scatters the sheaf-gradient terms back to source and target squares, adds contest-cell messages, and updates node stalks with a learned step size, residual MLP, and layer norm.

`ContestCellPool` aggregates incoming target-square tension by side-to-move and not-side-to-move energy, maximum incoming energy, side counts, and signed imbalance. These summaries are used as node messages and as readout features.

`SheafReadout` pools final node states, per-layer sheaf energy statistics, tactical group energies, contest pressure, overload pressure, and board-level counts. The classifier returns the repo trainer's one BCE puzzle logit in `output["logits"]`; diagnostics such as `sheaf_tension`, `attack_energy`, `defense_energy`, `pin_energy`, `contest_pressure`, `overload_pressure`, `gate_mean`, and `edge_density` are reporting outputs only.
