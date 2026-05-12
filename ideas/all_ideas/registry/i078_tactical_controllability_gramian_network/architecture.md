# Architecture

`Tactical Controllability Gramian Network` is a board-only `puzzle_binary`
classifier that follows the markdown thesis from
`ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_2004_saturday_shanghai_tactical_controllability_gramian.md`.
It tests the packet's claim that controllability/observability Gramians
on a learned 64x64 chess linear system separate true puzzles from
near-puzzles by directly comparing attacker control, defender control,
and target observability in the same linear-algebra picture.

## Mechanism

1. **Board encoder.** A compact convolutional stem (`BoardConvStem`)
   consumes the `simple_18` board tensor and produces a
   `(B, channels, 8, 8)` per-square feature map. The 64 squares give
   `X ∈ R^{B x 64 x channels}`.
2. **Stable operator `A(X)`.** A 64x64 batched operator
   `A = sum_g gate_g(X) * mask_g + U(X) V(X)^T`
   combines five fixed deterministic chess-geometry masks (rook+bishop
   ray, knight, pawn-attack, king, rook-line defense) gated by softplus
   weights from a pooled-board MLP, plus a low-rank context update
   `U V^T` whose factors are linear projections of the per-square
   features. `A_hat = A / max(1, sigma(A))` with
   `spectral_norm_iters` power iterations keeps the unrolled Gramian
   sums bounded.
3. **Attacker / defender input columns.** Per-square `input_rank`
   vectors `B_a(X)` and `B_d(X)` are produced by linear heads from the
   trunk features and then *gated* by side-to-move and opposing piece
   occupancy planes built from the simple_18 channels — only the
   side-to-move pieces contribute to attacker influence, only the
   opposing pieces to defender influence, exactly as the packet
   specifies.
4. **Target output matrix `C`.** A linear head emits `target_rank` rows
   over the 64 squares. Half the rows are gated by a soft side-to-move
   king zone (3x3 dilation of the king plane) and half by opposing
   high-value piece occupancy (queen, rook, bishop, knight). Rows are
   L2-normalised so `trace(C C^T)` does not explode at initialisation.
5. **Unrolled Gramians.** With `K = gramian_steps` the model forms
   `W_a = sum_{k=0..K} A_hat^k B_a B_a^T (A_hat^T)^k`,
   `W_d = sum_{k=0..K} A_hat^k B_d B_d^T (A_hat^T)^k`, and
   `W_o = sum_{k=0..K} (A_hat^T)^k C^T C A_hat^k`,
   via the recursions `W <- B B^T + A_hat W A_hat^T` and
   `W <- C^T C + A_hat^T W A_hat` that the packet's solver-choice
   section explicitly authorises for v1.
6. **Hankel-like modal readout.** A symmetric eigendecomposition of
   `W_o` (with tiny diagonal jitter for stability) produces
   `W_o^{1/2}` from which the model takes the top-`readout_modes`
   singular values of `W_o^{1/2} W_a W_o^{1/2}` (attacker observable
   Hankel modes) and `W_o^{1/2} W_d W_o^{1/2}` (defender cancellation
   modes). Principal angles between the leading eigenspaces of
   `W_a` and `W_d` measure attacker-vs-defender subspace mismatch.
7. **Tactical readouts.** Per board:
   - `T_a   = trace(C W_a C^T)`, `T_d = trace(C W_d C^T)`,
     `T_net = T_a - T_d`,
   - `target_diag_attacker = diag(C W_a C^T)`,
     `target_diag_defender = diag(C W_d C^T)`,
   - `observability_trace = trace(W_o)`,
   - `attacker_hankel_modes`, `defender_hankel_modes`,
   - `mode_ratio = attacker / (|attacker| + |defender| + ε)`,
   - `subspace_principal_angles`.
8. **Puzzle head.** A `LayerNorm + MLP` consumes
   `[pool(X), T_a, T_d, T_net, observability_trace, attacker modes,
   defender modes, mode ratio, principal angles, target diagonals,
   operator gate weights, operator-norm proxy, low-rank energy]` and
   emits one puzzle logit.

## Output Contract

Forward returns a dict whose `"logits"` entry is `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. Diagnostic tensors
appended to prediction artefacts:

- `T_a`, `T_d`, `T_net`: attacker / defender / net target reach `(B,)`.
- `observability_trace`: `trace(W_o)` `(B,)`.
- `attacker_hankel_modes`, `defender_hankel_modes`,
  `mode_ratio`, `subspace_principal_angles`: `(B, readout_modes)`.
- `target_diag_attacker`, `target_diag_defender`:
  `(B, target_rank)` per-target Gramian energies.
- `operator_norm`: estimated spectral-norm proxy of `A(X)`.
- `operator_gate_weights`: `(B, 5)` softplus gates over the chess-
  geometry masks.
- `operator_low_rank_energy`: Frobenius mass of `U V^T`.
- `ablation_*`: per-batch indicator flags consumed by the packet's
  diagnostic table.

## Ablations

The bespoke builder accepts `model.ablation` in
`{"none", "attacker_only", "defender_only", "no_observability",
"one_step_gramian", "random_target_C", "random_geometry_A",
"fixed_A_no_gates", "diag_only_gramian", "cnn_same_params"}`,
matching the packet's required ablation table. The
`cnn_same_params` ablation is enforced at trainer-level; the model
itself only marks the `ablation_cnn_same_params` output flag.

## Implementation Binding

- Registered model name: `tactical_controllability_gramian_network`.
- Source implementation file: `src/chess_nn_playground/models/tactical_controllability_gramian_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i078_tactical_controllability_gramian_network/model.py`.
