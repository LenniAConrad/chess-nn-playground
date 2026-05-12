# Math Thesis

King-Anchored Material-Null Transport Bottleneck (KAMN-OTB)

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0657_tuesday_local_material_ot_bottleneck.md`.

Working thesis: a puzzle-like position should often exhibit unusually efficient side-to-move transport of forcing material toward opponent king and value targets, after subtracting a deterministic material-preserving null geometry. The residual transport descriptors should classify near-puzzle and puzzle positions better than raw material or local texture alone.

Let `S = {0,...,7}^2` be the squares, `R = {K,Q,R,B,N,P}` the piece roles, and `A = {white, black}` the colours. A `simple_18` adapter `A_e` extracts deterministic piece occupancy `P(x) ∈ {0,1}^{2×6×8×8}` and a side-to-move flag `s(x) ∈ A`. For each direction `a -> b ∈ {stm->opp, opp->stm}` we form padded source candidates `U_a(x)` (up to 16 own pieces) and target candidates `V_b(x)` (up to 16 opponent pieces plus 9 opponent king-zone pseudo-targets clipped to the board). Mass weights `mu_u, nu_v` are softplus-normalised role/type priors over the masked candidates.

For each transport head `h ∈ {1,..,H}` (default `H=4`) the cost matrix `C_h(u,v;x) = softplus(theta_h^T phi(u,v;x) + b_h) ∈ [c_floor, 20]` uses only rule-independent current-board geometry (Manhattan, Chebyshev, file/rank/diagonal alignment, queen-line, knight-graph distance, side-relative pawn-forward, role-aware distance, king-zone indicator, high-value indicator). The entropic OT plan

`Pi_h^ε(x) = argmin_{Pi ∈ Π(mu, nu)} ⟨Pi, C_h(x)⟩ + ε * sum_{u,v} Pi_{uv}(log Pi_{uv} - 1)`

is solved by masked log-domain Sinkhorn with default `ε=0.08`, `iterations=12`. The transport descriptor `T_real(x, a->b)` pools `(Pi_h^ε, C_h)` into 15 scalars per head: expected cost, normalised plan entropy, max and top-4 pair mass, king-zone target mass, value-bucket masses (pawn/minor/rook/queen/king), distance-bucket masses (≤1, =2, =3, ≥4), and forward-mass (toward opponent). A deterministic king-anchored material-null sampler permutes non-king source/target squares while preserving side-to-move, both king squares, piece identities, source counts, target counts, and target-role histogram; we approximate `E_null T_null(x, a->b)` by averaging `K=4` such draws under the same cost head.

The bottleneck residual

`Z(x) = [T_real(x, stm->opp) - mean_K T_null(x, stm->opp), T_real(x, opp->stm) - mean_K T_null(x, opp->stm), signed difference between the two directions]`

is fed to a small LayerNorm-MLP classifier head producing one puzzle logit. By Sinkhorn uniqueness on the convex/strictly-convex entropic OT subproblem with positive smoothed marginals, `Pi_h^ε` is unique and permutation-invariant under simultaneous candidate relabelling, so all pooled descriptors are independent of padding order. Null shuffles are used only as unsupervised centering controls and never enter the supervised label set.

The model is board-only: CRTK, engine, verification, and source metadata are reporting-only and never consumed as input.
