# Architecture

`Tactical Transport Imbalance Network` measures whether a position exhibits an asymmetric, low-cost, low-entropy entropic-transport plan from the side-to-move's pieces onto opponent target squares. The architecture follows the markdown thesis directly:

- **Side-to-move canonicalisation.** A simple_18 board tensor is split into white and black piece planes; the side-to-move metadata channel selects the active side and rank-flips the board so the side to move is always "own". This makes the transport signal symmetry-equivariant under colour swap.
- **Local board trunk.** A compact residual CNN over the 18-plane current board produces a globally pooled feature vector `z_cnn` shared with the classifier head.
- **Transport mass head.** Per-head softmax distributions assign mass to source/target squares based on learned piece-value priors (initialised at canonical 1/3/3/5/9 for sources and 1/3/3/5/9/12 for targets, with king-ring attraction) plus per-square biases. This produces forward source `mu_own`, forward target `nu_opp`, reverse source `mu_opp`, and reverse target `nu_own` distributions over 64 squares per head.
- **Chess-geometry cost basis.** Square-to-square transport costs are a head-wise softplus combination of eight deterministic geometry channels: Manhattan, Chebyshev, off-rank/file, off-diagonal, knight-graph distance, square-colour parity, off-queen-line, and backward-rank push. No learned position embedding is used; the cost geometry is fixed chess geometry.
- **Entropic Sinkhorn transport.** A small fixed-iteration Sinkhorn block solves the regularised optimal-transport problem twice per head: side-to-move attacking the opponent (forward) and the opponent attacking side-to-move (reverse). Both plans share the same head-specific cost matrix.
- **Transport feature pool.** For each head we read off transport cost, normalised entropy, peak mass, L2 mass concentration, and rank/file moment shifts on each plan, then form forward/reverse contrasts. These per-head signals are flattened and concatenated with `z_cnn`.
- **Head.** A two-layer MLP over `[z_cnn, transport features]` produces one puzzle logit. The forward returns the logit and a small set of named diagnostics (`transport_imbalance`, `forward_transport_cost`, `reverse_transport_cost`, `transport_entropy_gap`, `transport_concentration_gap`, `transport_rank_moment_gap`) for the prediction artifact.

The model is board-only: CRTK, engine, and source metadata are reporting-only and never enter the network as input.

## Implementation Binding

- Registered model name: `tactical_transport_imbalance_network`
- Source implementation file: `src/chess_nn_playground/models/tactical_transport_imbalance.py`
- Idea-local wrapper: `ideas/i031_tactical_transport_imbalance_network/model.py`

The wrapper calls `build_tactical_transport_imbalance_network_from_config` with the idea's `model:` config block. The registry key `tactical_transport_imbalance_network` resolves to the same builder, so `build_model(name, model_cfg)` and the idea wrapper produce equivalent modules.
