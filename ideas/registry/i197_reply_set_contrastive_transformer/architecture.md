# Architecture

`Reply-Set Contrastive Transformer` is a bespoke board-only puzzle_binary
architecture that turns the contrastive thesis from `math_thesis.md` into a
concrete network: a *real* puzzle position embeds differently from its
plausible reply positions, while a near-puzzle stays close to one or more
"safe" replies.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit for the BCE-with-logits
`puzzle_binary` trainer. CRTK / source / engine metadata is reporting-only
and is never used as model input.

## Mechanism

1. **Conv trunk.** A compact `BoardFeatureTrunk` (Conv → BN/GroupNorm → GELU
   → optional Dropout2d, repeated `depth` times) produces a per-square
   feature map `(B, channels, 8, 8)` from the current board.

2. **Pseudo-reply set.** A deterministic `PseudoReplyGenerator` produces
   `K = 12` pseudo-reply boards. Each pseudo-reply translates the
   *side-to-move's own* piece planes by one of twelve chess-relevant
   offsets (eight rook/bishop ray steps and four canonical knight jumps),
   flips the side-to-move plane, and zeroes the en-passant plane. Castling
   planes are preserved. This is a coarse, fully differentiable reply
   lattice; it does not require any move-legality oracle.

3. **Shared trunk over replies.** The same conv trunk encodes every
   pseudo-reply (weights shared with the current-board encode). Each reply
   collapses to a `channels`-vector via mean pooling.

4. **Token-attention block.** A small self-attention layer over the
   per-square tokens of the current feature map produces a `(B, 64,
   channels)` token sequence. This realises the `token_attention` proposal
   profile: per-square tokens attend to each other before pooling.

5. **Defender-reply pool.** An attention pool over the post-attention tokens
   weighted by a 3x3 dilation around the *enemy* king square (the side that
   must defend the move). When the king plane is empty the pool falls back
   to uniform attention. This realises the `defender_reply` proposal profile.

6. **Contrastive aggregator.** Cosine similarities between the current
   embedding and each pseudo-reply embedding are aggregated into a
   six-dimensional contrastive feature: `[min, mean, std, top-1, top-2,
   sum-of-positive]`. A real puzzle should drive the minimum down and reduce
   the positive-similarity sum; a near-puzzle keeps the minimum (and at
   least one similarity) high, matching the `graph` proposal profile.

7. **Head.** A LayerNorm → Linear → GELU → (Dropout) → Linear head reads the
   concatenation of the conv-pooled current code (mean+max), the
   token-attention summary (mean over tokens), the defender-reply summary,
   and the contrastive features. It returns one puzzle logit.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for the
`puzzle_binary` BCE-with-logits trainer (`num_classes == 1`), plus
diagnostics, all of shape `(B,)`:

- `logits`, `current_embedding_norm`, `token_summary_norm`,
  `defender_summary_norm`, `reply_similarity_mean`, `reply_similarity_min`,
  `reply_similarity_std`, `reply_pressure`, `defense_gap`,
  `mechanism_energy`, `graph_pressure`, `ray_language_energy`,
  `proposal_profile_strength`, `proposal_keyword_count`, `num_replies`,
  `reply_embedding_mean_norm`, `king_ring_pressure`.

`mechanism_energy` is `1 - reply_similarity_mean`; `graph_pressure` and
`reply_pressure` alias the per-reply similarity standard deviation;
`defense_gap` is the contrast between the mean and the minimum reply
similarity (large = at least one safe reply exists, small = the puzzle is
far from every reply). These keep the puzzle_binary diagnostic packet
contract consistent with the `mechanism_family: graph` family expected by
the source packet.

## Ablations

The constructor accepts the following ablations:

- `none` — the full reply-set contrastive network described above.
- `no_replies` — drop the pseudo-reply generator and the contrastive
  features. Tests whether the contrastive signal is load-bearing.
- `no_token_attention` — drop the token-attention block; the head consumes
  only the conv-pooled current code, the defender pool, and the contrastive
  features.
- `no_defender_reply` — drop the king-ring defender pool; the head consumes
  only the conv-pooled current code, the token-attention summary, and the
  contrastive features.

## Implementation Binding

- Registered model name: `reply_set_contrastive_transformer`
- Source implementation file: `src/chess_nn_playground/models/reply_set_contrastive_transformer.py`
- Idea-local wrapper: `ideas/registry/i197_reply_set_contrastive_transformer/model.py`
