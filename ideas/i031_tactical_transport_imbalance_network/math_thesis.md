# Math Thesis

Tactical Transport Imbalance Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0512_tuesday_local_transport_imbalance.md`.

Working thesis: a chess puzzle-like position should often exhibit an asymmetric low-cost, low-entropy optimal-transport plan from the side-to-move's pieces onto opponent target squares relative to the reverse direction. We model this with an entropically-regularised Sinkhorn transport on a fixed chess-geometry cost basis, contrast forward vs. reverse plans, and combine the contrast signal with a compact local CNN trunk for puzzle-binary classification.

Key quantities, per transport head `h` and direction `d ∈ {fwd, rev}`:

- Source/target measures `mu^d_h, nu^d_h ∈ Δ^{64}` from softmax over piece-value-weighted occupancy with per-square bias and king-ring attraction.
- Cost matrix `C_h ∈ R^{64×64}` from a softplus combination of eight chess-geometry channels (Manhattan, Chebyshev, off-rank/file, off-diagonal, knight-graph distance, colour parity, off-queen-line, backward-rank push).
- Sinkhorn plan `P^d_h = argmin_{P ∈ Π(mu^d_h, nu^d_h)} ⟨P, C_h⟩ - ε H(P)`.

The diagnostic signal is the forward-vs-reverse contrast in transport cost, entropy, peak mass, mass concentration, and rank/file moment shift, summarised in `transport_imbalance` and supporting per-head features that feed the classifier alongside the local CNN feature vector.

The architecture is board-only: CRTK, engine, and source metadata are reporting-only and never enter the network as input.
