# Math Thesis

Symmetric Difference Twin Encoder

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `4`.

Working thesis: safe deterministic board transforms should preserve
some evidence and change other evidence. Instead of enforcing
invariance, compare the original and transformed board latents by
symmetric-difference features.

## Model

Let `x \in R^{18 \times 8 \times 8}` be the simple_18 board tensor and
`T` a deterministic safe board transform (in this implementation,
file mirror with the appropriate kingside/queenside castling-channel
swap). Let `Phi` be a shared convolutional encoder. The model
computes

```
z          = Phi(x)
z'         = Phi(T(x))
z_aligned  = T^{-1}(z')                  # T^{-1} = file flip in latent space
preserved  = (z + z_aligned) / 2          # intersection in real-vector sense
changed    = |z - z_aligned|              # symmetric difference
y_hat      = head( pool(preserved) || pool(changed) || pool(fuse(preserved, changed)) )
```

with a single shared `Phi` applied to both branches (twin pass).

## Set-Theoretic Reading

For binary sets `A, B`, the symmetric difference is
`A \triangle B = (A \cup B) \setminus (A \cap B)`. Lifting to real
vectors, with element-wise `min` for intersection and element-wise
`max` for union, the symmetric difference becomes
`max(z, z_aligned) - min(z, z_aligned) = |z - z_aligned|`. The
`preserved` stream measures evidence that the safe transform leaves
intact (intersection-like), and the `changed` stream measures evidence
that the safe transform breaks (symmetric-difference-like). Both are
informative: `preserved` carries the rule-equivalent skeleton of the
position, `changed` carries the residual that depends on the
particular file/king-side a piece is sitting on.

## Why Not Invariance

A symmetry-invariant network would force `z = z_aligned`. That
collapses the `changed` signal to zero and erases exactly the
information this classifier wants to read. By feeding both
`preserved` and `changed` to the head as parallel streams, the model
can learn how much of the puzzle signal is symmetric and how much is
broken by the safe transform.

## Frame Alignment

Because `Phi` is fully convolutional, the inverse `T^{-1}` in latent
space is the same spatial file flip used on the input. Applying the
flip to `Phi(T(x))` therefore returns the transformed latent to the
original frame, so `z` and `z_aligned` can be compared cell-by-cell.
This alignment is what makes the element-wise comparison meaningful;
without it, `(z - z_aligned)` would mostly measure spatial offset
rather than symmetry residual.
