# Architecture

`Oriented Tactical Sheaf Laplacian` implements the packet's side-to-move-oriented tactical incidence sheaf for the repo's `puzzle_binary` task.

## Implementation Binding

- Registered model name: `oriented_tactical_sheaf_laplacian`
- Source implementation file: `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf.py`
- Idea-local wrapper: `ideas/registry/i018_oriented_tactical_sheaf_laplacian/model.py`

## Modules

`BoardStateAdapter` consumes a `(B, C, 8, 8)` board tensor (currently `simple_18`, with hooks for `lc0_static_112` and `lc0_bt4_112`), reads the side-to-move flag, and rotates / color-swaps the board so that the mover always sees their own pieces in the bottom ranks. It returns the canonical raw planes, a 13-channel mover-oriented `piece_state` (empty plus six us-/them- piece classes), an occupancy vector, and a `side_info` flag. Unsupported encodings fall back to a learned 1x1 piece-state probe; no labels, engine, or verification fields are read.

`TacticalIncidenceBuilder` constructs the typed tactical incidence complex over the 64 squares from `piece_state` and `occupancy`. It uses precomputed knight, king, king-zone, oriented pawn, rook-ray, bishop-ray, and between-square blocker masks, so visible rook/bishop/queen edges are gated by occupancy along the segment. The builder emits a dense `(B, R, 64, 64)` relation tensor with `R = 12` typed edges:

- `us_attacks_them_piece`, `them_attacks_us_piece`
- `us_defends_us_piece`, `them_defends_them_piece`
- `us_attacks_empty_near_king`, `them_attacks_empty_near_king`
- `bishop_ray_visible`, `rook_ray_visible`, `queen_ray_visible`
- `knight_attack`, `pawn_attack_forward_oriented`
- `king_ray_pin_candidate` (king-blocker-slider triples on a clear ray)

`SquareTokenEncoder` mixes the canonical raw square slice, the mover-oriented piece state, and fixed 6-D coordinate features (rank, file, centered rank/file, edge distance, mover promotion distance) through small projections and a fused MLP into 64-dim node states `h0`.

`SheafDiffusionBlock` learns the cellular sheaf. For each of the `R = 12` relations it stores `8x8` source/target restriction maps `rho_src[r], rho_dst[r]`, a bounded scalar gate `g_r in (0, 2)`, and a fixed sign `sigma_r in {-1, +1}` separating attacker from target context. Each block projects nodes to stalks `z = node_to_stalk(h)`, computes the per-relation coboundary

```text
delta_e = sqrt(w_e) * (rho_dst[r] z_v - sigma_r * rho_src[r] z_u)
```

so that `L_rho = delta^T delta` is symmetric positive semidefinite (Hansen-Ghrist sheaf Laplacian), runs a bounded heat step `z <- z - eta * delta^T delta` with a learned `eta` clamped to a stable spectral range, and adds the result back through `stalk_to_node` plus a residual MLP and `LayerNorm`. The block returns updated node states, per-relation sheaf energies, and the relation gates.

`TriadDefectPool` realizes the optional 2-cell tactical-triple defect pool from the math thesis. For both us- and them- centric triads it forms attacker-mean and defender-mean stalks weighted by the corresponding incidence rows, computes a learned coboundary on (attacker, target, defender), and pools the squared defect by attacker x defender x target-piece weight to produce mean and peak triad energies.

`TacticalReadout` concatenates mean / max node embeddings, mover-piece and opponent-piece weighted node pools, per-relation sheaf-energy mean and max across blocks, relation density, mean relation gates, the triad statistics, and a small board-statistics vector (occupancy mean, side material counts, attack densities, pin density, rank/file occupancy spread). It feeds them through a `LayerNorm -> Linear -> GELU -> Dropout -> Linear` head and returns one BCE-compatible puzzle logit in `output["logits"]`.

Diagnostic outputs (`mechanism_energy`, `sheaf_tension`, `transport_imbalance`, `symmetry_residual`, `topology_pressure`, `ray_language_energy`, `information_surprisal`, `sparse_certificate_energy`, `rank_file_imbalance`, `king_ring_pressure`, `reply_pressure`, `defense_gap`, `triad_defect_energy`, `pin_pressure`, `proposal_profile_strength`, `proposal_keyword_count`) accompany the logit so the puzzle_binary trainer's prediction artifacts retain the packet-profile reporting fields. They are reporting-only and never used in the loss.

## Contract

- Input: `(B, C, 8, 8)` board tensor only; `simple_18` is the primary encoding. CRTK / source / verification metadata is reporting-only and never enters the model.
- Output: `dict` with `logits` of shape `(B,)` for the one-logit puzzle_binary BCE-with-logits trainer, plus the diagnostic tensors listed above.
- Symmetry: only the side-to-move canonicalization (color swap + 180-degree rotation) is applied; full `D4` board symmetry is intentionally not assumed.
