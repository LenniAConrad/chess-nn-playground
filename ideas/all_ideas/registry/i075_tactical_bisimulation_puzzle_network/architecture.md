# Architecture

`Tactical Bisimulation Puzzle Network` is a board-only `puzzle_binary`
classifier that learns a latent representation in which two positions
are close only if their one-step legal continuations are behaviourally
similar. It follows the markdown thesis from
`ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0113_saturday_shanghai_tactical_bisimulation.md`.

## Mechanism

1. **Board encoder `E`.** A compact convolutional stem
   (`BoardConvStem`) consumes the `simple_18` board tensor and produces
   a `(B, channels, 8, 8)` feature map. Mean and max pools are
   concatenated and projected through a `LayerNorm + MLP` to the latent
   `z = E(board)` of shape `(B, latent_dim)`.
2. **Learned move proposer `pi(a | x)`.** `max_moves` learnable query
   vectors attend over the 64 board squares using the trunk feature
   map. Each query produces (i) a score whose softmax across the K
   queries gives `pi(a | x)`, and (ii) a value-pooled `(B, K, move_dim)`
   move token. The proposer is the board-only stand-in for the
   deterministic legal-move sampler described in the thesis: it never
   reads engine, source, or CRTK metadata, and the random-sampler
   ablation replaces `pi` with a uniform distribution.
3. **Latent transition `T(z, a)`.** A small MLP applied to
   `[z; move_token]` produces a successor latent `z_next^k` for every
   candidate move; the `(B, K, latent_dim)` cloud is the successor
   signature `mu_x` from the thesis.
4. **Prototype bank `P`.** A learnable `(prototype_count, latent_dim)`
   bank, partitioned into puzzle / disproof / random behavioural bands.
   The bank is queried with a learned diagonal-metric distance
   `d(z, p) = || (z - p) odot scale ||`. The euclidean-only ablation
   freezes the metric to the identity.
5. **Bisimulation residual.** The Bellman-style consistency residual
   `|| z - sum_k pi_k * T(z, a_k) ||` measures whether the latent state
   is a fixed point of the policy-mixed transition. The
   `no_transition_consistency` and `no_bisim_loss` ablations zero or
   detach this term.
6. **Successor signature stats.** Per-batch entropy of `pi(a | x)`,
   pi-weighted spread `sum_k pi_k * || z_next^k - centroid ||`, the
   diameter `max_k || z_next^k - centroid ||`, and a transition-norm
   summary. The `no_successor_signature` ablation zeroes these stats.
7. **Final puzzle head.** A `LayerNorm + MLP` head consumes
   `[base_logit g(z), prototype distances, per-band min distance,
   successor stats, bisim residual]` and emits one puzzle logit. The
   `binary_margin_only` ablation collapses the head to `g(z)` so a
   contrastive-only baseline can be measured.

## Output Contract

Forward returns a dict whose `"logits"` entry is `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. All diagnostic
tensors are finite per batch and are appended to prediction artifacts:

- `base_logit`: `g(z)` evidence before the head.
- `prototype_distances`: `(B, prototype_count)` distances to the bank.
- `min_prototype_distance`, `soft_min_prototype_distance`,
  `mean_prototype_distance`: pooled bank summaries.
- `puzzle_prototype_distance`, `disproof_prototype_distance`,
  `random_prototype_distance`: per-band minima used to verify the
  diagnostic ordering required by the packet
  (`puzzle < near < random` for the puzzle band, `near < puzzle` for
  the disproof band).
- `successor_signature_entropy`, `successor_spread`,
  `successor_diameter`, `transition_norm`: successor-cloud stats.
- `bisim_residual`: Bellman-style consistency residual.
- `move_proposal_entropy`, `move_attention_entropy`, `latent_norm`:
  proposer / encoder diagnostics.
- `gamma`, `fine_label_pair_mining_active`, and `ablation_*` flags:
  per-batch indicators consumed by the packet's diagnostic table.

## Ablations

The bespoke builder accepts `model.ablation in {"none",
"no_bisim_loss", "no_successor_signature",
"no_transition_consistency", "euclidean_metric_only",
"random_move_sampler", "no_prototypes", "binary_margin_only",
"fine_label_pair_mining_off"}` matching the packet's required ablation
table. The `fine_label_pair_mining_off` ablation is honoured at
trainer level (the model itself never reads the fine source label) and
only flips the `fine_label_pair_mining_active` flag in the output dict.

## Implementation Binding

- Registered model name: `tactical_bisimulation_puzzle_network`.
- Source implementation file: `src/chess_nn_playground/models/tactical_bisimulation_puzzle_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i075_tactical_bisimulation_puzzle_network/model.py`.
