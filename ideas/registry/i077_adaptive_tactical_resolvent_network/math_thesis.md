# Math Thesis

Adaptive Tactical Resolvent Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_2002_saturday_shanghai_adaptive_tactical_resolvent.md`.

## Working thesis

A learned chess-structured resolvent operator
`R_k(A) = (I - alpha_k A)^{-1} = sum_{t>=0} alpha_k^t A^t` propagates
attacker and defender influence through every path length at once.
True puzzles should have high attacker-to-target transfer that the
defender's resolvent transfer cannot cancel; near-puzzles should look
locally similar but get cancelled by global defender propagation.

## Operator construction

The model encodes a position as `X ∈ R^{64 x d}` (compact CNN trunk
over simple_18 planes) and assembles

  A(X) = sum_g gate_g(X) * mask_g + U(X) V(X)^T

over five chess-geometry masks `g ∈ {ray, knight, pawn, king, defense}`
plus a low-rank board-conditioned update `U V^T`. The operator is
spectrally normalised via a small power-iteration estimate of
`||A||_2`:

  A_hat = A / max(1, sigma_hat(A))

so the resolvents `(I - alpha A_hat)^{-1}` are well-defined for the
trainable `alpha_k = sigmoid(alpha_logits_k) ∈ (0, 1)`.

## Resolvent transfer

For each `alpha_k` we solve directly

  y_attack_k  = (I - alpha_k A_hat)^{-1} s_attack
  y_defense_k = (I - alpha_k A_hat)^{-1} s_defense
  y_target_k  = (I - alpha_k A_hat)^{-T} s_target

with `torch.linalg.solve` (cheap on 64x64) and read

  attack_to_target_k  = <y_attack_k, s_target>
  defense_to_target_k = <y_defense_k, s_target>
  net_pressure_k      = attack_to_target_k - defense_to_target_k
  sensitivity_k       = || y_attack_k - y_defense_k ||

plus king-zone and material-target energies of `y_attack_k`,
`y_defense_k`, and `y_target_k`. The puzzle logit is a LayerNorm+MLP
over the concatenation of these per-alpha features, the pooled board
context, and operator diagnostics.

## Falsification

The packet's required ablations all live in the bespoke model and run
end-to-end on the same head shape: `no_resolvent_direct_pool`,
`neumann_1_step`, `single_alpha`, `fixed_operator_no_gates`,
`no_low_rank_update`, `random_geometry_operator`,
`attack_only_no_defense`, and the trainer-side `cnn_same_params`
baseline.
