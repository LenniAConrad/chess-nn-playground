# Architecture

`Puzzle Boundary Twin Encoder` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and returns one puzzle
logit per position, exposing the raw boundary score so a margin-aware
trainer can attach the packet's in-batch pair losses.

## Mechanism

The benchmark's failure mode is the boundary between verified puzzles
and verified near-puzzles: they share material, king geometry, and
threat structure, so a single global readout collapses them. The
packet proposes learning that boundary directly. The architecture is a
*siamese* twin encoder: the same encoder is applied identically to
every batch item (puzzle, near-puzzle, random) and a single linear
boundary surface produces the puzzle logit. The "twin" lives in the
in-batch pair structure, not in two parallel heads.

1. **Shared board encoder.** `BoardConvStem(input_channels=18,
   channels, depth, use_batchnorm)` produces an `(B, channels, 8, 8)`
   feature map. Mean, max, and standard-deviation pooling are
   concatenated into a `(B, 3 * channels)` descriptor and projected
   through `LayerNorm -> Linear -> GELU (-> Dropout)` into a shared
   latent `z_shared = encoder(board)` of shape `(B, shared_dim)`. The
   third (std) pooling channel is the cheap second-moment cue that
   separates puzzles (peaky activations) from near-puzzles (similar
   means but flatter).
2. **Boundary embedding projector.** A `LayerNorm -> Linear -> GELU
   (-> Dropout) -> Linear` projector maps `z_shared` to an
   `embedding_dim` boundary embedding `e` and then L2-normalises it to
   the unit sphere. Norming makes the margin objective scale-free and
   keeps the boundary surface from drifting during training.
3. **Boundary surface (linear margin head).** A learned unit-norm
   direction `w` is the explicit decision boundary in embedding space.
   The puzzle logit is its signed cosine distance times a learned
   scale, plus a learned bias:

   ```text
   e_unit = normalize(projector(z_shared))
   w_unit = normalize(boundary_direction)
   boundary_score = <e_unit, w_unit> * boundary_scale + boundary_bias
   logit          = boundary_score
   ```

   The decision surface `<e_unit, w_unit> = -bias / scale` is the
   margin surface the packet talks about; `boundary_score` is the
   signed margin to it.

## Pairwise margin contract

The packet's training objective is pair-margin on top of BCE,

```text
boundary_score(puzzle) >= boundary_score(near)   + margin_near
boundary_score(near)   >= boundary_score(random) + margin_random_surface
```

The forward pass exposes `boundary_score`, `boundary_distance`, the
unit-norm `boundary_embedding`, and the pre-normalisation
`embedding_norm` so a trainer with reliable group metadata
(`sister_group_id` / `split_group_id`) can attach pair-margin losses
directly without rerunning the encoder. The model itself never reads
group metadata; pair mining lives in the trainer.

At inference the model is single-board single-logit: the head is just
`logit = head(z)` and the BCE-with-logits trainer treats it as a
standard puzzle classifier.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. All tensors are
finite per batch:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` for
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `boundary_score`: `(B,)` raw signed margin to the decision surface;
  equals `logits` when `num_classes == 1`.
- `boundary_distance`: `(B,)` `|boundary_score|`.
- `boundary_embedding`: `(B, embedding_dim)` unit-norm embedding the
  boundary surface reads.
- `z_shared`: `(B, shared_dim)` post-pool descriptor before the margin
  projector.
- `embedding_norm`: `(B,)` pre-normalisation L2 norm of the boundary
  embedding (representational-collapse monitor).
- `trunk_energy`: `(B,)` mean-square trunk activation.

## Implementation Binding

- Registered model name: `puzzle_boundary_twin_encoder`
- Source implementation file: `src/chess_nn_playground/models/puzzle_boundary_twin_encoder.py`
- Idea-local wrapper: `ideas/i184_puzzle_boundary_twin_encoder/model.py`
