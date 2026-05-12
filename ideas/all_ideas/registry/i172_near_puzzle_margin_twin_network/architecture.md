# Architecture

`Near-Puzzle Margin Twin Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and returns one puzzle
logit per position, plus the two latent projections that make the
"twin" structure explicit and let a margin-aware trainer apply
pairwise ranking losses.

## Mechanism

The benchmark's failure mode is ranking: real puzzles must score
*above* near-puzzle hard negatives. A single latent representation
encourages near-puzzle and puzzle to collapse together, because they
look almost identical on an ordinary-board readout. The architecture
splits the readout into two branches sharing one encoder.

1. **Shared board encoder.** `BoardConvStem(input_channels=18,
   channels, depth, use_batchnorm)` produces an `(B, channels, 8, 8)`
   feature map; mean and max over the 8x8 grid are concatenated into a
   `(B, 2 * channels)` descriptor and projected through `LayerNorm ->
   Linear -> GELU (-> Dropout)` into the shared latent
   `z_shared = encoder(board)` of shape `(B, shared_dim)`. The
   encoder consumes `simple_18` only; CRTK / source / engine metadata
   stays reporting-only.
2. **Two latent projections (the twin).** Two independent
   `LayerNorm -> Linear -> GELU (-> Dropout) -> Linear` projectors
   read the shared latent into two heads:

   ```text
   z_ordinary = projector_ordinary(z_shared)   # generic position descriptor
   z_tactical = projector_tactical(z_shared)   # puzzle-evidence descriptor
   ```

   `z_ordinary` is the latent in which near-puzzle hard negatives are
   by construction *close* to puzzles (same material, same king
   geometry, same threats); `z_tactical` is the latent in which the
   puzzle/near-puzzle separation must live so that the head can use it.
3. **Puzzle head.** A small `LayerNorm -> Linear -> GELU
   (-> Dropout) -> Linear` head reads `z_tactical` only:

   ```text
   logit = head(z_tactical)
   ```

   Routing the head through `z_tactical` is what forces the network to
   put the ranking-relevant signal into a representation that the
   classifier can consume.

## Pairwise margin contract

The forward pass exposes the two latents and a raw
`puzzle_margin_signal` so a trainer with reliable group metadata
(`sister_group_id` / `split_group_id`) can attach the packet's
batch-level pair losses on top of the BCE term:

- For pairs `(puzzle, near)` from the same group:
  `loss_margin = relu(margin - logit_puzzle + logit_near)`.
- For pairs `(near, random)`: an optional weak ordering or
  contrastive separation in `z_ordinary`.
- Optional contrast: `z_tactical` should *separate* puzzle and near-
  puzzle even when `z_ordinary` does not, so a representational
  contrast term can be layered on the two latents.

The model itself never reads group metadata; trainers attach the
margin terms by reading `puzzle_margin_signal`, `z_ordinary`, and
`z_tactical` from the forward output.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. All tensors are
finite per batch:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` for
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `z_shared`: `(B, shared_dim)` shared post-pool descriptor.
- `z_ordinary`: `(B, ordinary_dim)` ordinary-board latent.
- `z_tactical`: `(B, tactical_dim)` puzzle-evidence latent.
- `ordinary_norm`, `tactical_norm`: `(B,)` L2 norms of each latent
  (representational-collapse monitors).
- `ordinary_tactical_alignment`: `(B,)` cosine similarity between the
  two latents on their common dimension prefix; high values flag a
  collapsed twin.
- `trunk_energy`: `(B,)` mean-square trunk activation.
- `puzzle_margin_signal`: `(B,)` value the puzzle head consumes
  (equals `logits` when `num_classes == 1`); the trainer reads this
  for pair-margin losses.

## Implementation Binding

- Registered model name: `near_puzzle_margin_twin_network`
- Source implementation file: `src/chess_nn_playground/models/near_puzzle_margin_twin_network.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i172_near_puzzle_margin_twin_network/model.py`
