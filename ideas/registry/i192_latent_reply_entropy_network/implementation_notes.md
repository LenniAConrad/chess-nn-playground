# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/latent_reply_entropy.py`.
- Registry key: `latent_reply_entropy_network`.
- Idea-local wrapper: `ideas/registry/i192_latent_reply_entropy_network/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Latent Reply Entropy Network`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.

The implementation has two stages:

1. `LatentReplyCandidateBuilder` deterministically proposes reply/resource tokens from `simple_18` board planes. It computes attack relations, movement relations, line-block pressure, king zones, material values, and side-to-move attacker/defender roles, then emits tokens for king escapes, captures, blocks, defenses, counter-threats, and quiet resources.
2. `LatentReplyEntropyNetwork` encodes those tokens with source-square, target-square, and global board context. It scores replies with a learned safe-reply scorer, applies a temperature softmax, extracts entropy/concentration features, and feeds them to the final puzzle logit head.

The model exposes the packet's required ablations through `model.ablation` in config:

- `reply_count_only`
- `fixed_uniform_scores`
- `no_entropy_features`
