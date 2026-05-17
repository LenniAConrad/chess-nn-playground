# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf_fast.py`
  (`OrientedTacticalSheafFastNet`, `FastSheafDiffusionBlock`,
  `build_oriented_tactical_sheaf_fast_from_config`).
- Idea-local wrapper: `ideas/registry/i249_oriented_tactical_sheaf_fast/model.py`
  (`build_model_from_config`).
- Registry key: `oriented_tactical_sheaf_fast`.
- Parent idea: `i018 oriented_tactical_sheaf_laplacian`.

## What Changed

`FastSheafDiffusionBlock` is a drop-in replacement for i018's
`SheafDiffusionBlock`. It has the same parameter names and shapes:

- `node_to_stalk`, `stalk_to_node`
- `rho_src`, `rho_dst`
- `relation_gate_logits`, `eta_logit`
- `relation_signs`
- `node_mlp`, `norm`

The original block materializes per-edge residuals:

```text
residual[i, j] = dst[j] - sign * src[i]
weighted[i, j] = gate * W[i, j] * residual[i, j]
```

i249 computes the same update from reduced quantities:

```text
W_dst = W @ dst
WT_src = W.T @ src
out_degree = W.sum(dst)
in_degree = W.sum(src)
```

Then it applies the same source/target restriction-map backsweeps. Relation
energy is computed from the expanded quadratic form instead of from the
materialized residual tensor.

## Optional Compilation

`model.compile_model: true` wraps the bound `_raw_forward` method with
`torch.compile`. This avoids `_orig_mod.` state-dict prefixes and keeps i018
checkpoint compatibility. `compile_mode: reduce-overhead` was fastest overall
on the local RTX 4070 Laptop GPU; `max-autotune` had slightly faster batch-1
latency but worse compile cost and no batch-256 gain.

The repo trainer config already sets:

- `training.allow_tf32: true`
- `training.matmul_precision: high`

Those settings matter for the compiled path. Standalone inference benchmarks
should set the same precision flags before the first compile.

## Optional Eval Autocast

`model.inference_autocast_dtype: float16` wraps eval/no-grad CUDA forward passes
in FP16 autocast. Training and gradient-enabled calls remain on the surrounding
precision path, so the exact shared-weight gradient guard still uses
`inference_autocast_dtype: none`.

Local random-batch checks showed max logit drift around `2e-4` versus the
FP32/TF32 path, while compiled batch-256 latency improved from about `5.9 ms`
to about `4.23 ms`. Use `none` for exact-logit audits; use `float16` for the
fastest serving/eval path.

Two configs are intentionally kept side by side:

- `config.yaml`: exact-logit algebraic i249 (`inference_autocast_dtype: none`).
- `config_eval_fp16.yaml`: lower-precision eval/serving i249
  (`inference_autocast_dtype: float16`).

## Diagnostics

Default output is the full i018 diagnostic dictionary. Serving-only callers can
set `model.return_diagnostics: false`; logits are unchanged, but scalar
diagnostic columns are omitted. This is intentionally not the paper-grade
default because the reporting pipeline benefits from those diagnostics.

## Numerical Guard

The current rewrite was checked with shared i018 weights:

- strict state-dict load: passed;
- max logit difference: about `3e-8` to `1e-7`;
- max checked diagnostic difference: normal reduction noise;
- BCE loss difference: `0.0` on the checked batch;
- max checked gradient difference: about `4.5e-8`.

Re-run this check after any edit to `FastSheafDiffusionBlock`. If shared-weight
logit drift exceeds `1e-5` or gradient drift exceeds `1e-7`, the change is no
longer an execution-only rewrite.
