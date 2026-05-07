# Architecture

`King-Shelter Microkernel Network` is a board-only puzzle-binary classifier that gives the king zone a dedicated high-resolution branch. A compact global CNN reads the full board, while side-relative microkernels inspect both kings through fixed 5x5 and 7x7 crops.

## Input And King Views

- Input is the repo `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- White and black king locations are read from the king planes in the current board tensor.
- Each color receives a side-relative board view: own piece planes, opponent piece planes, occupancy, empty squares, own/opponent king masks, relative side-to-move, and coordinate planes.
- The black view is rotated into the same forward direction as the white view so front shield, side escape, diagonal entry, and rank backdoor filters share one orientation.
- The side-to-move plane selects the own-king view and opponent-king view before fusion.

## Microkernel Branch

For each selected king view, the model extracts padded 5x5 and 7x7 crops around the king. A shared bank of asymmetric convolutional microkernels processes each crop:

- front-shield filters use 3x5 kernels over the pawn shield and adjacent front files
- side-escape filters use 5x3 kernels over lateral exits
- diagonal-entry filters use 3x3 kernels for diagonal access into the king zone
- rank-backdoor filters use 1x5 kernels along the king rank

The 5x5 and 7x7 crop embeddings are projected into a king-zone vector for the own king and the opponent king. The classifier receives the own vector, opponent vector, signed residual, and absolute residual so it can compare shelter and pressure asymmetrically.

## Deterministic Shelter Features

In parallel with learned microkernels, the crop branch computes deterministic king-zone scalars:

- front shield pawn density
- side escape emptiness
- diagonal entry pressure from opponent pieces
- rank backdoor pressure from opponent rooks and queens
- near-slider pressure from opponent bishops, rooks, and queens
- local blocker density
- escape-ring emptiness
- local king-zone density
- combined local pressure
- shield/escape minus pressure gap

These scalars are computed for own and opponent king views and fused with their signed residuals.

## Global Fusion And Output

A compact CNN board stem supplies full-board mean and max pooled features. The classifier fuses those global features with the microkernel king-zone summary and returns one puzzle logit for the `puzzle_binary` task. Fine labels 0 and 1 are non-puzzle; fine label 2 is puzzle.

The output dictionary includes `logits` plus diagnostics such as `king_crop_branch_logit`, `own_microkernel_energy`, `opponent_microkernel_energy`, `king_zone_residual`, `front_shield_score`, `side_escape_score`, `diagonal_entry_pressure`, `rank_backdoor_pressure`, `near_slider_pressure`, `local_blocker_density`, `shelter_escape_gap`, and crop activation energies.

## Implementation Binding

- Registered model name: `king_shelter_microkernel_network`.
- Source implementation: `src/chess_nn_playground/models/king_shelter_microkernel.py`.
- Idea-local wrapper: `ideas/i129_king_shelter_microkernel_network/model.py`.
