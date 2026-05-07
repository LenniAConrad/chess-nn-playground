# Implementation Notes

- Central code: `src/chess_nn_playground/models/safe_reply_certificate.py`.
- Registry key: `safe_reply_certificate_verifier`.
- Idea-local wrapper: `ideas/i191_safe_reply_certificate_verifier/model.py`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Safe-Reply Certificate Verifier`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.

The implementation has two stages:

1. `SafeReplyCertificateBuilder` deterministically proposes safe-reply candidates from `simple_18` board planes. It computes attack relations, line-clearance masks, king zones, material values, and side-to-move attacker/defender roles, then emits candidate tokens for move-away-king, capture-attacker, block-line, defend-target, counter-threat, and trade-down replies.
2. `SafeReplyCertificateVerifier` encodes those candidate tokens with local square features and a global board context. It estimates validity and strength separately, multiplies them into certificate disproof scores, and subtracts the strongest disproof witness from a positive puzzle logit.

The model exposes the packet's required ablations through `model.ablation` in config:

- `mean_disproof_instead_of_max`
- `no_validity_gate`
- `certificate_count_only`
