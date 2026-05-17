# Architecture

`Oriented Tactical Sheaf Laplacian (Fast)` is a pure execution rewrite of i018
`oriented_tactical_sheaf_laplacian`. It keeps the same board adapter,
tactical-incidence builder, square encoder, sheaf parameters, triad pool,
readout head, and diagnostics. The intended model family and prediction
function are unchanged; only the sheaf diffusion execution plan changes.

## Implementation Binding

- Registered model name: `oriented_tactical_sheaf_fast`
- Source implementation: `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf_fast.py`
- Idea-local wrapper: `ideas/registry/i249_oriented_tactical_sheaf_fast/model.py`

## Core Rewrite

i018's `SheafDiffusionBlock` loops over 12 relation types. For every relation
it materializes an edge residual tensor of shape `(B, 64, 64, stalk_dim)`,
weights it, projects it back through the source/target restriction maps, and
reduces over endpoints. On GPUs this is dominated by memory traffic and launch
overhead, not by useful arithmetic.

i249 expands the same sums algebraically. For each relation with edge weights
`W_ij`, projected source stalks `src_i`, target stalks `dst_j`, and sign
`sigma`, the original edge residual is:

```text
r_ij = dst_j - sigma * src_i
```

The update only needs two reductions:

```text
sum_j W_ij r_ij = (W @ dst)_i - sigma * out_degree_i * src_i
sum_i W_ij r_ij = in_degree_j * dst_j - sigma * (W.T @ src)_j
```

So the block computes `W @ dst`, `W.T @ src`, in-degrees, and out-degrees
directly, then applies the same `rho_src.T` and `rho_dst.T` restriction-map
backsweeps. It also computes the relation energy from the expanded quadratic
form:

```text
sum_ij W_ij ||dst_j - sigma * src_i||^2
```

This removes the dense edge residual intermediate while preserving the same
parameters and the same mathematical update.

## Acceleration Layers

1. **Algebraic `FastSheafDiffusionBlock`** replaces the residual-materializing
   block. Its parameter names and shapes match i018 exactly, so i018 and i249
   checkpoints load into each other with `strict=True`.
2. **Optional `torch.compile(mode="reduce-overhead")`** wraps the bound forward
   method. This keeps `state_dict` keys clean and lets PyTorch fuse the static
   `(B, 18, 8, 8)` board path.
3. **Optional eval/inference autocast** via
   `model.inference_autocast_dtype: float16` runs no-grad CUDA inference in
   FP16. This is disabled during training/gradient checks and is the fastest
   serving path measured so far.
4. **Optional logits-only output** via `model.return_diagnostics: false` skips
   diagnostic columns for serving. The default remains `true` so paper-grade
   reporting keeps the i018 diagnostic contract.

## Numerical Equivalence

With i018 weights loaded into i249:

- eval-mode `logits` matched to about `3e-8` to `1e-7` max absolute error;
- `sheaf_tension`, `mechanism_energy`, `ray_language_energy`, and `pin_pressure`
  matched within normal floating-point reduction noise;
- BCE loss matched exactly in the checked batch;
- the largest checked parameter-gradient difference was about `4.5e-8`.

The rewrite therefore should not move accuracy beyond normal floating-point and
training-order noise. If a future edit increases shared-weight logit drift above
`1e-5` or gradient drift above `1e-7`, i249 should be treated as a changed model
rather than a speed rewrite.

## Measured Inference

On the local RTX 4070 Laptop GPU, base scale `(channels=64, depth=2,
stalk_dim=8)`, synthetic `(B, 18, 8, 8)` inputs:

| model | mode | batch 1 | batch 32 | batch 256 |
|---|---|---:|---:|---:|
| i018 | eager | ~6.2 ms | ~6.4 ms | ~71.7 ms |
| i249 algebraic | eager | ~2.9 ms | ~3.4 ms | ~13.5 ms |
| i249 algebraic | compiled + TF32/high | ~0.4-0.5 ms | ~0.8 ms | ~5.9 ms |
| i249 algebraic | compiled + eval FP16 autocast | ~0.46 ms | ~0.69 ms | ~4.23 ms |

The compiled path has a one-time per-shape compilation cost. For fixed inference
batch shapes or trainer-scale batches, that cost is amortized.
FP16 autocast moved logits by about `2e-4` max on random batches in the local
check; keep `inference_autocast_dtype: none` when exact i018-equivalent logits
are required.

## Config Variants

- `config.yaml`: exact-logit algebraic i249. This is the audit baseline for
  i018-equivalence checks.
- `config_eval_fp16.yaml`: lower-precision eval/serving i249. Same model and
  weights, but no-grad CUDA inference uses FP16 autocast for maximum throughput.

## Contract

Input is the same current-board tensor as i018. Default output is a
`dict[str, Tensor]` with `logits` plus the inherited i018 scalar diagnostics.
Set `return_diagnostics: false` only for serving paths that consume logits and
do not need diagnostic prediction columns.
