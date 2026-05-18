# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The default config
is the recommended first XXL cell from the research markdown:
width=160, hidden_dim=320, depth=4, stalk_dim=8, restriction_mode=full,
compile_model=false, fuse_incidence=false.

Differences vs `ideas/registry/i018_oriented_tactical_sheaf_laplacian/config.yaml`:

- `model.name = efficient_i018_scale_xxl` (i254 builder).
- `model.channels` is 160 (vs 64 at i018 base, 128 at i018 scale_xl).
- `model.hidden_dim` is 320 (vs 96 at i018 base, 192 at i018 scale_xl).
- `model.depth` is 4 (vs 2 at i018 base; matches i018 scale_xl).
- `model.stalk_dim` is 8 (unchanged across the entire i018 family).
- `model.restriction_mode` is `full` by default. Switch to
  `grouped_lowrank` only for the stalk-scaling experiment row.
- `model.compile_model` and `model.fuse_incidence` are both `false`.
  These are reserved for the separate execution branch; they must
  pass the train-mode mixed-precision parity ladder before being
  benchmarked for speed.
- Training header: `batch_size = 128` (per the research markdown's
  memory geometry argument), `epochs = 20`, `min_epochs = 10`,
  `early_stopping_patience = 5`, `monitor = pr_auc`. Same LR
  (`0.0007`) and weight decay (`0.0001`) as i018, same
  `reduce_on_plateau` schedule.

If you change a trunk hyperparameter (`channels`, `hidden_dim`,
`depth`, `stalk_dim`, `dropout`, `use_batchnorm`, `restriction_mode`),
change it on i018 too and re-run both - the scale comparison is only
honest when both nets share the rest of trunk geometry.

## Loss

`bce_with_logits` on the puzzle logit. No i254-specific auxiliary
loss. All i018 mechanism-energy diagnostics are emitted unchanged.

## Decision rule

Following the research markdown:

- **Scale falsifier**: the 3-seed i254 capacity run must beat i018
  scale_xl's 0.8901 mean PR-AUC by at least `+0.003` to count as
  scale headroom; otherwise the family has plateaued.
- **Efficiency falsifier**: if the first XXL candidate reduces
  measured train throughput by more than 25% versus current scale_xl
  while failing the scale falsifier, stop and reconsider.
- **Stalk falsifier**: any `s > 8` variant must beat the best
  `s = 8` width/head-scaled variant or stalk scaling is unsupported.
- **Grouped-map falsifier**: grouped low-rank restrictions must beat
  same-budget full maps by more than seed noise to be kept.
- **Systems falsifier**: do not write a fused incidence kernel
  unless profiling shows the incidence builder is a top hotspot.
  Do not attribute speed changes to compile unless compile-only A/B
  on the unmodified baseline shows them first.
- **Parity falsifier**: any execution-only rewrite must pass the
  train-mode mixed-precision parity ladder before being benchmarked
  for speed.

## Cost expectation

- Trunk shape: ~1.66x the i018 scale_xl parameter budget
  (785,217 params vs 474,437 at scale_xl).
- Rough forward arithmetic: ~1.26x scale_xl, because the extra
  capacity goes into regular dense work (BLAS-friendly), not into the
  `64 * 64 * s` relation work.
- The relation tensor `(B, 12, 64, 64)` is unchanged; at batch 128
  it is ~12 MiB in bf16.
- Wall-clock should not be a problem at `batch_size = 128` on a
  single GPU; the actual measurement must use the profile-first
  protocol described in `math_thesis.md`.

## Benchmark plan

Following the research markdown's capacity-branch first, execution-
branch second split:

| Tranche | Cells | Seeds | Runs |
|---|---|---:|---:|
| Capacity scout | first XXL (width=160, head=320, depth=4, stalk=8, full) | 1 | 1 |
| Capacity confirm | first XXL | 3 | 3 |
| Stalk diagnostic | width=160, head=320, depth=4, stalk=12, full | 3 | 3 |
| Stalk + grouped | width=160, head=320, depth=4, stalk=12, grouped (G=4, k=4) | 3 | 3 |
| Depth diagnostic | width=160, head=320, depth=6, stalk=8, full | 3 | 3 |
| Capacity falsifier | width=128, head=192, depth=4, stalk=8 (i018 scale_xl) | 3 | 3 |
| **Total** | | | **16** |

Promote the capacity scout to confirm only if it is stable and the
validation curve is at least consistent with current scale_xl by the
midpoint.

## Profile-first protocol (execution branch)

Per the research markdown, the recommended profile is:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        with torch.autocast("cuda", dtype=resolved_amp_dtype):
            loss = training_step(batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if step == 19:
            break
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
prof.export_chrome_trace("i254_train_trace.json")
```

The new block emits `record_function` scopes named
`i254/efficient_sheaf_block` and `i254/per_relation_loop`, so they
show up directly in the profiler table. Sort by
`self_cuda_time_total`, `cuda_time_total`, `cpu_time_total`, and
memory. Export a Chrome trace.

## Reports

Standard idea report (see `report_template.md`). The slice analysis
is inherited from i018's reporting contract and must include
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. The new columns are
the scale ladder axis (i018 base / i018 scale_xl / i254 first XXL),
the restriction mode (`full` / `grouped_lowrank`), and any
execution-branch annotations (`compile_only`, `fuse_incidence`).
