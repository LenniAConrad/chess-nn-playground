# Math Thesis

Oriented Tactical Sheaf Laplacian

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0254_tuesday_local_oriented_tactical_sheaf.md`.

Working thesis: puzzle-likeness in the `puzzle_binary` task is detectable as gluing-defect energy on a side-to-move-oriented chess attack/defense incidence sheaf. The bespoke implementation realizes that thesis as a learned cellular sheaf Laplacian over typed tactical relations.

## Object

For each board `x`, build a finite cell complex `K(x)`:

- 0-cells `V`: the 64 board squares, each carrying a stalk `F(v) = R^s` with default `s = 8`.
- 1-cells: typed tactical relations `e = (u, v, r)` with `r` ranging over us-/them- attacks of pieces, us-/them- defenses, us-/them- attacks of empty squares near the opposing king, visible bishop / rook / queen rays, knight attacks, oriented pawn attacks, and king-ray pin candidates (12 relations in the implementation).
- Optional 2-cells: attacker-target-defender tactical triples, realized in the implementation as `TriadDefectPool` mean / peak statistics.

Edge weights `w_e(x) in [0, 1]` are deterministic, board-only, and computed exactly from the mover-oriented piece planes plus precomputed knight, king, ray, and between-square blocker masks.

## Sheaf Laplacian

A learned cellular sheaf assigns relation-typed restriction maps `rho_{e,u}^{(r)}, rho_{e,v}^{(r)}: R^s -> R^s` and a fixed sign `sigma_r in {-1, +1}` per relation. The coboundary is

```text
(delta_rho h)_e = sqrt(w_e) * (rho_{e,v}^{(r)} h_v - sigma_r * rho_{e,u}^{(r)} h_u),
```

so the sheaf Laplacian `L_rho(x) = delta_rho(x)^T delta_rho(x)` is symmetric positive semidefinite. The bounded heat step `h_{t+1} = (I - eta L_rho(x)) h_t` is stable in Euclidean norm for `0 <= eta <= 2 / lambda_max(L_rho)`; the implementation enforces a learned-but-clipped `eta` and bounded relation gates `g_r in (0, 2)`.

## Hypothesis

Tactical positions concentrate gluing defect on the typed tactical incidence complex (overloads, pinned defenders, king-ray geometry), so a learned sheaf-energy readout is a more sample-efficient puzzle-likeness statistic than a generic spatial CNN. The model returns one BCE puzzle logit plus per-relation sheaf-energy diagnostics, mover/opponent pooled stalks, triad-defect statistics, and board summaries used as reporting fields by the `puzzle_binary` trainer.

## Falsifiers and counterexamples

- Quiet endgame studies and zugzwang positions may be puzzle-like with weak attack/defense incidence; static one-ply tactics will miss them.
- Non-puzzle blunder-rich positions can exhibit high tactical tension without verified puzzle status.
- The central falsifier (per the source packet) is replacing real relation masks with degree-preserving random masks; if performance is unchanged the typed-relation thesis is rejected and the family must not be re-scaled.
