# Architecture

`File-Mirror Tension Sheaf` implements a board-only typed signed directed sheaf over pseudo-legal attack/defense/x-ray relations and combines its energy statistics with the same statistics computed on the file-mirrored input through a learned partial-equivariance gate.

## Implementation Binding

- Registered model name: `file_mirror_tension_sheaf`
- Source implementation file: `src/chess_nn_playground/models/file_mirror_tension_sheaf.py`
- Idea-local wrapper: `ideas/i028_file_mirror_tension_sheaf/model.py`

## Components

- `EncodingAdapter`: decodes piece type, piece color, side-to-move, and side-relative role (own/enemy) from the `simple_18` board planes. It does not consume engine, verification, source, or CRTK metadata.
- `AttackDefenseGraphBuilder`: builds the directed multigraph `G_X = (V, E_X, tau, sigma)` from the decoded board. Edge types are: `own_attack_enemy`, `enemy_attack_own`, `own_defense`, `enemy_defense`, `own_xray_enemy`, `enemy_xray_own`, `own_pawn_control`, `enemy_pawn_control`, `own_king_zone_pressure`, `enemy_king_zone_pressure`, `own_line_blocker_pressure`, `enemy_line_blocker_pressure`. Defenders carry sign `-1`; attackers, x-rays, blockers, pawn controls, and king-zone edges carry sign `+1`. Pseudo-legal pawn captures, knight jumps, king moves, and sliding rays (with single-blocker x-rays) are emitted; king-zone edges are added for own/enemy pieces directly adjacent to the targeted king. Edge counts are capped at `e_max` with a deterministic priority order (king-zone, attacks, x-rays, defenses, pawn controls, blocker pressure).
- `NodeInitializer`: concatenates raw `simple_18` square planes, decoded one-hot piece/color/role features, square coordinates, and side-to-move into a learned stalk feature `h^0 in R^d`.
- `TypedSheafDiffusionLayer`: holds learnable type-conditioned restriction maps `A^src_tau, A^dst_tau` and applies the diffusion step
  `h^{k+1} = LayerNorm(h^k - eta_k * delta_F^T W delta_F h^k + phi_k(h^k))`
  with `eta_k = sigmoid(eta_logit) * eta_max`, a sigmoid edge gate `w_e`, and a small node MLP `phi_k`. The coboundary `(delta_F h)_e = A^dst_tau h_{t(e)} - sigma(e) A^src_tau h_{s(e)}` matches the math thesis. Restriction maps are initialized near the identity so the operator starts close to a clean directed Laplacian.
- `SheafEnergyReadout`: computes the energy statistic vector `s` containing per-edge-type means and maxes (twelve types each), top-`k` global pooled energies, own/enemy king-zone energy, the top-`k`/total concentration ratio, divergence-norm mean and max, and node mean/max pooling.
- `FileMirrorPartialGate`: computes `s_F(MX)` with the SAME encoder (shared weights) on the file-mirrored input `M(X)`, computes `delta_s = |s_F(X) - Pi_M s_F(MX)|` (the type-stat permutation `Pi_M` is the identity because file mirror does not flip own/enemy color), and learns `rho = sigmoid(gate_mlp([s, s_M, delta_s]))` with `gate_mode in {"scalar", "vector"}`. The mirror is implemented as `Simple18Mirror`: spatial file flip plus the kingside <-> queenside castling-plane swap. Logits are NOT forced to match under mirror.
- `Classifier`: `LayerNorm -> Linear -> GELU -> Dropout -> Linear` over `[s, rho * delta_s, rho, pooled_node_features]`, returning one puzzle logit (config `num_classes: 1`).

## Forward Contract

```text
output = model(x)
x.shape == (batch, input_channels=18, 8, 8)
output["logits"].shape == (batch,)        # because num_classes == 1
```

The diagnostic dictionary additionally exposes `sheaf_tension`, `transport_imbalance`, `symmetry_residual`, `topology_pressure`, `ray_language_energy`, `information_surprisal`, `sparse_certificate_energy`, `rank_file_imbalance`, `king_ring_pressure`, `reply_pressure`, `defense_gap`, `mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count`, and `mirror_gate_rho` for downstream artifact reporting.

## Symmetry

Full chessboard rotation/reflection invariance is rejected. Only the file-mirror direction is treated as an approximate symmetry, and the gate `rho` lets data decide how much the mirror discrepancy matters. Because the file mirror is an involution and the type-stat permutation is identity, the energy statistics are equivariant under `M` when the gate is disabled and restriction maps are shared across mirrored edge types (Proposition 1 of the math thesis).
