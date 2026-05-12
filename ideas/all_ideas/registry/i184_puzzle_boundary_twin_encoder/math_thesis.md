# Math Thesis

Puzzle Boundary Twin Encoder

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.

Batch candidate rank: `8`.

Working thesis: The hardest part of the `puzzle_binary` task is the
decision boundary between verified puzzles and verified near-puzzles.
A single-readout classifier collapses them because they share material,
king geometry, and threat structure. Learn that boundary directly with
a siamese (twin) encoder shared across in-batch pairs (puzzle, near,
random) and a margin objective on a linear boundary surface in a
unit-norm embedding space.

Geometry. Let `z = encoder(board)` be a shared encoder. Project to a
margin embedding `e = projector(z)` and L2-normalise to the unit
sphere `e_unit = e / ||e||`. The decision boundary is the hyperplane
`{e_unit : <e_unit, w_unit> = -b / s}` for a learned unit direction
`w_unit`, learned scale `s > 0`, and learned bias `b`. The puzzle
logit is the signed cosine margin to that hyperplane,

    boundary_score(board) = <e_unit, w_unit> * s + b
    logit                  = boundary_score.

Training contract. The trainer applies BCE-with-logits on `logit` and,
when reliable in-batch pair groups are available, the packet's pair-
margin terms,

    boundary_score(puzzle) >= boundary_score(near)   + m_near,
    boundary_score(near)   >= boundary_score(random) + m_random_surface.

These are exactly the inequalities the markdown idea writes down; the
forward pass exposes `boundary_score` and the unit-norm embedding so
the trainer can compute them without rerunning the encoder.
