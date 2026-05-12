# Architecture

`Safe-Reply Certificate Verifier` implements the packet's disproof-first classifier for puzzle-binary detection.

## Board Contract

- Input: current-board `simple_18` tensor with shape `(batch, 18, 8, 8)`.
- Output: one puzzle logit per board for the repository puzzle-binary trainer.
- Metadata: CRTK/source fields remain reporting-only and are never consumed by the model.

## Certificate Proposal Program

The model builds a deterministic set of board-local safe-reply candidates before applying neural scoring. The proposal program derives attack maps, slider ray clearance, side-to-move attacker/defender roles, king zones, tactical material values, and line blockers from the board tensor.

It creates candidates for the six packet certificate families:

- `move_away_king`: defender-king adjacent safe squares not occupied by defender pieces and not currently attacked.
- `capture_attacker`: attacking pieces that defender pieces already attack.
- `block_line`: empty squares between an attacking slider and a high-value defender target with at most one current blocker.
- `defend_target`: high-value defender targets under attack that have defender coverage.
- `counter_threat`: high-value attacker targets or attacker king-zone squares that the defender can attack.
- `trade_down`: attacked attacker pieces where the local material exchange is favorable or cheap for the defender.

Each certificate token contains the certificate family one-hot, deterministic base score, square geometry, local piece/resource features, attack/defense indicators, king-zone indicators, and line-blocking evidence. The configured `max_certificates` allocates slots across all six families, preserving at least one slot per family.

## Verifier

A compact convolutional board trunk with coordinate planes produces square tokens and a pooled board context. The pooled context is fused with aggregate certificate statistics, then projected into a global verifier token.

For each candidate certificate:

```text
c_i = certificate_encoder(candidate_features_i, local_square_token_i, global_board_token)
validity_i = sigmoid(validity_head(c_i))
strength_i = softplus(strength_head(c_i))
disproof_i = validity_i * strength_i
best_disproof = max_i disproof_i
```

The final classifier separately computes `positive_puzzle_logit` from the board context and subtracts the learned-scale disproof witness:

```text
puzzle_logit = positive_puzzle_logit - softplus(alpha) * best_disproof
```

This implements the thesis that a cheap safe-reply certificate is evidence against puzzlehood rather than merely an auxiliary diagnostic.

## Diagnostics and Ablations

The forward output includes `logits`, `positive_puzzle_logit`, `best_disproof`, per-certificate validity/strength/score tensors, certificate masks and kinds, aggregate certificate counts, and per-family max scores. The configured ablation mode supports:

- `mean_disproof_instead_of_max`: replace the strongest witness with a mean over valid witnesses.
- `no_validity_gate`: use only deterministic candidate validity as the gate.
- `certificate_count_only`: hide board-local certificate content and expose only candidate family/count structure.

## Implementation Binding

- Registered model name: `safe_reply_certificate_verifier`.
- Source implementation file: `src/chess_nn_playground/models/safe_reply_certificate.py`.
- Idea-local wrapper: `ideas/registry/i191_safe_reply_certificate_verifier/model.py`.
