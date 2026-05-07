# Architecture

`Latent Reply Entropy Network` implements the packet's reply-set compression idea for puzzle-binary classification.

## Board Contract

- Input: current-board `simple_18` tensor with shape `(batch, 18, 8, 8)`.
- Output: one puzzle logit per board for the repository puzzle-binary trainer.
- Metadata: CRTK/source fields remain reporting-only and are never consumed by the model.

## Reply Candidate Program

The model first builds deterministic board-local reply/resource tokens for the side not to move. It derives pseudo-legal attack and movement relations, side-to-move attacker/defender roles, slider line blocks, king zones, tactical material values, and attacked-square maps from the board tensor.

The reply set is split into six families:

- `king_escape`: defender king moves to empty squares not currently attacked.
- `capture_attacker`: defender resources that can capture attacking pieces.
- `block_line`: defender moves to empty squares between attacking sliders and high-value defender targets.
- `defend_target`: defender resources already covering high-value defender targets under attack.
- `counter_threat`: defender resources attacking high-value attacker targets or attacker king-zone squares.
- `quiet_resource`: safe quiet mobility that keeps a broad reply set available.

Each reply token stores its family, deterministic base score, source and target square geometry, local piece values, attack/control flags, king-zone flags, and line-block pressure.

## Latent Reply Distribution

A compact convolutional board trunk with coordinate planes produces square tokens and a pooled board context. The pooled context is fused with aggregate reply statistics. Each reply token is encoded with its source token, target token, and global board token, then scored by a learned safe-reply scorer:

```text
s_i = safe_reply_score(r_i, board_context)
p_i = softmax(s_i / temperature)
```

The entropy readout computes:

```text
H = -sum_i p_i log p_i
top1 = max_i p_i
top2_gap = top1 - second_largest_i p_i
effective_reply_count = exp(H)
```

The final head follows the packet formula:

```text
puzzle_logit = MLP([board_context, H, top1, top2_gap, effective_reply_count])
```

## Diagnostics and Ablations

The forward output includes `logits`, reply entropy, normalized entropy, top-1 probability, top-2 gap, effective reply count, valid reply count, safe-reply mass, per-reply scores and probabilities, reply masks/kinds/source/target squares, and per-family counts/probability masses.

The configured ablation mode supports:

- `reply_count_only`: hide semantic token content and expose only reply-family/count structure.
- `fixed_uniform_scores`: replace learned reply scores with uniform scores over valid replies.
- `no_entropy_features`: remove entropy and concentration features from the classifier input while still reporting them.

## Implementation Binding

- Registered model name: `latent_reply_entropy_network`.
- Source implementation file: `src/chess_nn_playground/models/latent_reply_entropy.py`.
- Idea-local wrapper: `ideas/i192_latent_reply_entropy_network/model.py`.
