# Math Thesis

Sinkhorn Role Assignment Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.

Batch candidate rank: `2`.

Working thesis: Puzzle-like positions often contain latent tactical roles
(target king, forcing piece, blocker, loose defender, escape square,
overloaded piece, promotion candidate). Instead of asking attention to
discover these roles implicitly, this network assigns occupied piece tokens
to a fixed set of ``M`` learned role prototypes through a differentiable
optimal-transport layer. The assignment ``A in R^{P x (M + 1)}`` is the
unique matrix consistent with a Sinkhorn-Knopp factorisation
``A = diag(u) * exp(-C / tau) * diag(v)`` of the cosine-cost kernel between
projected piece tokens and learned prototypes, computed by ``T`` log-domain
Sinkhorn iterations subject to row mass ``r_i = mask_i`` (1 for occupied
pieces, 0 for padding) and column mass ``c_j = pi_j * sum_i r_i`` where
``pi`` is a learned softmax over the ``M + 1`` role slots and the extra
slot acts as a dustbin for irrelevant pieces. Role vectors
``role_j = sum_i A_{i, j} * token_i`` therefore obey explicit transport
constraints (no role can absorb more mass than the learned prior allots,
and no piece can transport more mass than its mask), so the puzzle logit is
a function of structurally normalised role aggregates rather than an
unconstrained attention pool.

This is the bespoke implementation of the markdown architecture and is not
backed by the shared ``ResearchPacketProbe`` scaffold.
