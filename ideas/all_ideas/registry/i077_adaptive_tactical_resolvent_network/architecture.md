# Architecture

`Adaptive Tactical Resolvent Network` is a board-only `puzzle_binary`
classifier that follows the markdown thesis from
`ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_2002_saturday_shanghai_adaptive_tactical_resolvent.md`.
It tests the packet's claim that a stable shifted-inverse / Green's-function
operator over learned chess geometry can directly read attacker-to-target
transfer and defender cancellation.

## Mechanism

1. **Board encoder.** A compact convolutional stem (`BoardConvStem`)
   consumes the `simple_18` board tensor and produces a
   `(B, channels, 8, 8)` per-square feature map. The 64 squares give
   `X ∈ R^{B x 64 x channels}`.
2. **Operator builder `A(X)`.** A 64x64 batched operator
   `A = sum_g gate_g(X) * mask_g + U(X) V(X)^T`
   combines five fixed deterministic chess-geometry masks (rook+bishop
   ray, knight, pawn-attack, king, rook-line defense) gated by softplus
   weights from a pooled-board MLP, plus a low-rank context update
   `U V^T` whose factors are linear projections of the per-square
   features.
3. **Spectral normalization.** `A_hat = A / max(1, sigma(A))` where
   `sigma(A)` is estimated by `spectral_norm_iters` power iterations of
   `A^T A`. This implements the packet's recommended
   `A_hat = A / max(1, spectral_norm_estimate(A))` step so the
   resolvents `(I - alpha A_hat)^(-1)` stay well-defined.
4. **Resolvents.** For each `alpha_k ∈ {0.25, 0.5, 0.75}` (each gated
   through `sigmoid(alpha_logits)` so they stay in `(0, 1)` while
   remaining trainable), the model assembles
   `R_k = (I - alpha_k * A_hat)^(-1)` *implicitly* via batched
   `torch.linalg.solve`. The packet explicitly authorises direct 64x64
   solves for v1.
5. **Role seeds `v_r(X)`.** A linear head reads six role scalars per
   square (`attack`, `defense`, `king_target`, `material_target`,
   `blocker`, `tempo`) and L2-normalises each role vector.
6. **Forward and transposed propagation.** For each `alpha_k`:
   - `y_attack_k  = R_k @ s_attack`,
   - `y_defense_k = R_k @ s_defense`,
   - `y_target_k  = R_k^T @ s_target` (transposed solve).
7. **Transfer / cancellation readout.** Per `alpha_k`:
   - `attack_to_target_k  = <y_attack_k, s_target>`,
   - `defense_to_target_k = <y_defense_k, s_target>`,
   - `net_pressure_k = attack_to_target_k - defense_to_target_k`,
   - `transfer_ratio_k = attack_to_target_k / (|attack| + |defense| + ε)`,
   - `resolvent_sensitivity_k = || y_attack_k - y_defense_k ||`,
   - king-zone and material-target resolvent energies of the attacker,
     defender and (transposed) target propagated fields, computed from
     simple_18 piece planes (side-to-move king, opposing high-value
     pieces).
8. **Puzzle head.** A `LayerNorm + MLP` consumes
   `[pool(X), per-alpha (attack, defense, net, ratio, sensitivity,
   king/material energies, alpha), operator gate weights,
   operator-norm proxy, low-rank energy]` and emits one puzzle logit.
   The single-alpha ablation pads remaining alpha slots with zeros so
   the head shape is fixed.

## Output Contract

Forward returns a dict whose `"logits"` entry is `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. Diagnostic tensors
appended to prediction artefacts:

- `operator_norm`: estimated spectral-norm proxy of `A(X)`.
- `operator_gate_weights`: `(B, 5)` softplus gates over the chess-
  geometry masks.
- `operator_low_rank_energy`: Frobenius mass of `U V^T`.
- `attack_to_target`, `defense_to_target`, `net_pressure`,
  `transfer_ratio`, `resolvent_sensitivity`: `(B, num_alpha)` packet
  diagnostics.
- `king_zone_resolvent_energy`, `material_target_resolvent_energy`:
  `(B, num_alpha, 3)` (attacker, defender, target).
- `alpha_values`: `(num_alpha,)` propagation scales actually used.
- `ablation_*`: per-batch indicator flags consumed by the packet's
  diagnostic table.

## Ablations

The bespoke builder accepts `model.ablation` in
`{"none", "no_resolvent_direct_pool", "neumann_1_step",
"single_alpha", "fixed_operator_no_gates", "no_low_rank_update",
"random_geometry_operator", "attack_only_no_defense",
"cnn_same_params"}`, matching the packet's required ablation table.
The `cnn_same_params` ablation is enforced at trainer-level; the model
itself only marks the `ablation_cnn_same_params` output flag.

## Implementation Binding

- Registered model name: `adaptive_tactical_resolvent_network`.
- Source implementation file: `src/chess_nn_playground/models/adaptive_tactical_resolvent_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i077_adaptive_tactical_resolvent_network/model.py`.
