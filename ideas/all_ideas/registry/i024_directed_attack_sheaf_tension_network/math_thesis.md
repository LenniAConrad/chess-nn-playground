# Math Thesis

Directed Attack-Sheaf Tension Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0427_tuesday_local_attack_sheaf_tension.md`.

The thesis is that puzzle-likeness can be exposed by localized, asymmetric inconsistencies in static directed attack geometry. For a board tensor \(x\), the model decodes board-only piece occupancy and builds a directed attack graph with edges \(e=(u,v,\kappa)\) for pawn, knight, king, rook-ray, bishop-ray, and queen-ray relations. Ray edges include path-clear and blocked-path summaries so pins, batteries, and x-rays remain visible rather than being reduced to only currently legal attacks.

Each square state \(z_v\in\mathbb{R}^d\) is restricted into an edge stalk by learned source and target maps:

\[
\delta_e z = \sqrt{g_e(x,z)}\left(A_{\kappa(e)}z_u - B_{\kappa(e)}z_v\right).
\]

Here \(g_e\in[0,1]\) is a learned gate conditioned on endpoint states, relation type, direction, distance, path features, reciprocal-edge status, x-ray status, king-zone status, and side-relative roles. The edge tension energy is

\[
\mathcal{E}(z)=\sum_e g_e\lVert A_{\kappa(e)}z_u - B_{\kappa(e)}z_v\rVert_2^2.
\]

For fixed gates this is nonnegative because it is a squared norm of the sheaf coboundary. The layer applies a degree-normalized Laplacian-style update, but keeps outgoing and incoming gradients separate before combining them through a directed residual block. This preserves the packet's key asymmetry claim: a threat from \(u\) to \(v\) is not treated as interchangeable with pressure from \(v\) to \(u\).

The readout pools final square states and sheaf energy statistics, including one-way versus reciprocal tension, outgoing versus incoming tension, attack and defense energy, x-ray tension, and king-zone tension. The folder's benchmark contract uses a single puzzle logit: fine labels `0` and `1` map to non-puzzle, and fine label `2` maps to puzzle.
