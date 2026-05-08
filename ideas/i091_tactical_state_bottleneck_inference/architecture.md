# Architecture

`Tactical State Bottleneck Inference` (`TSBI`) realises the source packet's
discrete-latent tactical-state bottleneck as a bespoke PyTorch model for the
repository's `puzzle_binary` task.

## Implementation Binding

- Registered model name: `tactical_state_bottleneck_inference`
- Source implementation file: `src/chess_nn_playground/models/tactical_state_bottleneck.py`
- Idea-local wrapper: `ideas/i091_tactical_state_bottleneck_inference/model.py`

## Modules

`ChessBoardTrunk` is a residual board encoder with two appended coordinate
planes. It maps `(B, C, 8, 8)` board tensors through a `3x3` conv lift to
`hidden_dim` channels, runs a stack of `ResidualBlock` units (GroupNorm +
GELU + 3x3 convs with optional 2D dropout), and finishes with a GroupNorm /
GELU stage. It returns both the spatial map `h` of shape
`(B, hidden_dim, 8, 8)` and the global-mean pooled vector
`pooled` of shape `(B, hidden_dim)`.

`PriorTacticalHead` produces the inference-time latent logits
`p_psi(z | x)`. It contains six categorical heads matching the source
packet's typed latent tuple:

```text
motif         -> Linear(hidden_dim -> 10)
anchor        -> SquareLogitHead -> 65
target        -> SquareLogitHead -> 65
relation      -> Linear(hidden_dim -> 8)
vulnerability -> Linear(hidden_dim -> 8)
tempo         -> Linear(hidden_dim -> 4)
```

`SquareLogitHead` follows the packet recipe `[B, 64] = conv1x1(h).flatten()`
plus a learnable `null_logit` that completes the 65-way categorical with the
`null` square slot.

`PosteriorTacticalHead` mirrors the prior head but additionally consumes a
binary label embedding of the puzzle target and a posterior spatial fuse
(`1x1` conv over `[h | y_map]`). It is training-only; the prior path is the
sole path used at inference.

`CategoricalLatent` modules own the per-group embedding tables `E_motif`,
`E_square`, `E_relation`, `E_vuln`, `E_tempo`. They expose two projection
modes:

- `expected_embedding(logits)`: returns `(probs, probs @ E)` for inference.
- `sample_embedding(logits, tau, hard)`: returns the Gumbel-softmax sample
  and `sample @ E` for training-time posterior or prior latents.

`TacticalStateBottleneckModel` glues the trunk, prior head, posterior head,
six categorical latents, the latent-conditioned puzzle head, and the capped
direct head together. The forward path computes:

1. `h, pooled = trunk(board)`
2. `prior_logits = prior_head(h, pooled)`
3. `prior_probs, prior_emb = project_latents(prior_logits, sample=False)`
   where `prior_emb = concat(probs_g @ E_g)` over groups
   `(motif, anchor, target, relation, vulnerability, tempo)`.
4. `latent_logit = latent_head([pooled, prior_emb])`,
   `direct_logit = direct_head(pooled)`.
5. `logit = latent_logit + direct_alpha * direct_logit`.

`forward_train(board, fine_label)` additionally evaluates the posterior head,
samples Gumbel-softmax categorical latents from both prior and posterior,
and returns `loss_pred`, `loss_prior_pred`, `loss_kl` (with free bits per
group), `loss_usage`, and `loss_entropy` from
`tactical_state_loss_components`. It is exposed for ablation harnesses; the
shared puzzle_binary trainer in this repository uses `forward_eval`, which
trains the model through the prior-path BCE term that the packet labels
`L_prior_pred` and that closes the train/inference gap.

`NoLatentMatchedBaseline` shares the trunk and replaces the
latent-and-direct head with a wider MLP head so ablations can compare the
TSBI puzzle logit against a no-latent matched control as required by the
packet.

## Diagnostics

`forward` returns a dict containing:

- `logits`: shape `(B,)`, BCE-compatible for the one-logit puzzle_binary head.
- `logit`, `prob`: alias and sigmoid of the puzzle logit.
- `latent_probs`: dict of categorical posteriors over the six latent groups.
- `prior_entropy_by_group`, `prior_usage_by_group`: per-group entropies and
  batch-mean usage distributions of `p_psi(z | x)`.
- `motif_entropy`, `anchor_entropy`, `target_entropy`, `relation_entropy`,
  `vulnerability_entropy`, `tempo_entropy`: scalar entropy diagnostics
  written into prediction artifacts.
- `motif_usage`, `relation_usage`, `vulnerability_usage`, `tempo_usage`,
  `anchor_null_rate`, `target_null_rate`: usage and null-slot rates that
  monitor the packet's collapse-control objectives.
- `pooled_energy`, `direct_alpha`: structural sanity diagnostics.
- `mechanism_energy`, `proposal_profile_strength`,
  `proposal_keyword_count`: scalar reporting fields preserved for
  compatibility with the project's research-packet diagnostic schema.

`forward_train` additionally returns `logit_q`, `logit_p`, the prior and
posterior categorical logits, sampled probabilities, the `losses` bundle
(`loss_pred`, `loss_prior_pred`, `loss_kl`, `loss_usage`, `loss_entropy`),
the prior/posterior agreement rate, and `diag_3x2`, the mandatory `3 x 2`
fine-label-by-prediction count table.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. CRTK / verification / source /
  engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  puzzle_binary BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine
  label `2` maps to binary target `1`.
- Latent cardinalities: `motif=10`, `anchor=65`, `target=65`,
  `relation=8`, `vulnerability=8`, `tempo=4`. The `anchor` and `target`
  groups each include the `null` 65th slot.
- Inference uses only the prior latents; the posterior head and the
  Gumbel-softmax sampler are reachable through `forward_train` for
  ablation runs that wire the full multi-loss objective.
