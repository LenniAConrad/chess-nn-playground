# Math Thesis

File-Mirror Tension Sheaf

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0437_tuesday_los_angeles_mirror_tension_sheaf.md`.

Working thesis: puzzle-like chess positions can be detected from board-only inputs by learning a small typed signed directed sheaf over pseudo-legal attack, defense, x-ray, pawn-control, and king-zone relations, then measuring how much the resulting local tactical tension changes under a learnable file-mirror partial-equivariance gate.

For a board tensor `X in R^{C x 8 x 8}` we build a directed multigraph `G_X = (V, E_X, tau, sigma)` over the 64 squares with twelve edge types and signed coboundary `(delta_F h)_e = A^{dst}_{tau(e)} h_{t(e)} - sigma(e) A^{src}_{tau(e)} h_{s(e)}`. The sheaf energy `E_F(h; X) = sum_e w_e || (delta_F h)_e ||_2^2` is reduced through stable diffusion steps `h^{k+1} = LN(h^k - eta_k delta_F^T W delta_F h^k + phi_k(h^k))`. Energy statistics `s_F(X)` are compared to `s_F(M X)` for the file-mirror operator `M`; the partial-equivariance gate `rho(X) = sigmoid(gate_mlp([s, s_M, delta_s]))` decides how much of the mirror discrepancy enters the classifier.

Proposition 1 (architectural equivariance). If file mirror is realized as a permutation of squares plus the kingside <-> queenside castling-plane swap, edge construction is `M`-equivariant, and restriction maps are shared across mirrored edge types, then `s_F(M X) = Pi_M s_F(X)` for the type-stat permutation `Pi_M`. Because file mirror does not flip own/enemy color, the type permutation reduces to the identity in this implementation, so the statistics are invariant under the mirror at initialization and the gate `rho` is what introduces partial asymmetry from data.

Proposition 2 (no engine input). All edges are computed from board occupancy and pseudo-legal piece geometry; no Stockfish scores, principal variations, node counts, verification metadata, source labels, proposed labels, or split identity enter the model.

Hypothesis: localized incompatibilities in attack-defense fibers (overloaded defenders, x-ray pressure, king-adjacent imbalance) carry the discriminative signal between non-puzzles and puzzles in this benchmark.
