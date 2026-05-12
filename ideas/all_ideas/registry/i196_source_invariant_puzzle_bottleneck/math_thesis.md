# Math Thesis

Source-Invariant Puzzle Bottleneck

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `11`.

## Working thesis

The puzzle_binary dataset is drawn from three source groups, so a board-only
classifier can accidentally learn source artifacts (encoding offsets, ply
patterns, rare-piece priors, framing biases) instead of the actual puzzle
structure. We want a representation that *preserves the puzzle signal but
removes any axis aligned with source identity from the main prediction path*.

We do not have source labels at inference time, but we observe that source
artifacts are not, in general, invariant to natural board symmetries: a clean
puzzle remains a puzzle under file-flip, rank-flip, and 180-rotation, while
source-specific encoding artifacts typically *do* shift under those
transformations. We therefore use the symmetry orbit as a proxy for source
variation and force the main puzzle representation to live in the
symmetry-invariant subspace.

## Construction

Let `f_θ(x)` be a shared trunk producing a pooled feature vector for a board
`x`. For a fixed group of `K` symmetry views `T_1, …, T_K` (identity, file
flip, rank flip, 180-rotation), define the per-view code

```text
c_k = g_φ(f_θ(T_k(x)))
```

through a shared bottleneck MLP `g_φ`. The orbit-invariant code and per-view
residuals are

```text
c̄ = (1/K) Σ_k c_k
r_k = c_k − c̄
```

and the residual energy `E_resid = (1/K) Σ_k ||r_k||² / D` measures how much
of the representation lives outside the invariant subspace.

The `c̄` vector is the only quantity allowed to enter the puzzle head. To
suppress any leftover residual axis that survived the averaging (e.g. a
direction common to most views), we additionally subtract the projection of
`c̄` onto the L2-normalised residual sum direction `u = Σ_k r_k / ||Σ_k r_k||`,
gated by `α = sigmoid(orthogonalize_logit)`:

```text
c_main = c̄ − α · (c̄·u) · u
puzzle_logit = h_ψ(c_main)
```

This is a smooth, differentiable, Gram–Schmidt-style suppression of the
dominant residual direction. The residual codes themselves are routed to a
separate auxiliary residual head that produces an `aux_residual_logit` for
diagnostic / regularisation purposes; that aux logit does not enter the main
puzzle prediction.

## Why this is expected to help

If the network attempted to encode a source-specific feature, that feature
would (i) break under at least one of the four symmetry views, and (ii)
appear as a non-zero `r_k`. The orbit-mean averaging plus the residual-axis
subtraction together guarantee that any feature whose energy is concentrated
in the residual subspace is removed from the main prediction path. The puzzle
signal — invariant under the same symmetries — survives.
