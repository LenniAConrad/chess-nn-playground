# Mathematical Thesis — a015 BT4 Primitive Mixer (ray_occlusion_semiring_scan)

This is a controlled architecture study, not a new primitive. The
mathematical claims about the spatial mixer itself live in
`ideas/registry/p010_ray_occlusion_semiring_scan/math_thesis.md`. The
thesis here is about the *swap*: holding a BT4-style residual tower,
optimizer protocol, and data contract fixed, and replacing only the
per-block spatial mixer with the `ray_occlusion_semiring_scan`
primitive adapted to a shape-preserving mixer contract.

## Operator signature

The BT4 residual block factors as

```
y = ReLU(SqueezeExcite(M(x)) + x)
```

where `x, y ∈ R^{B × C × 8 × 8}`. The shared `bt4_primitive_mixer`
tower fixes the stem, residual depth `N`, SqueezeExcite reduction, and
value head; the only operator that varies between sibling
`a###_bt4_*_mixer` ideas is the mixer `M`.

In `a015` we instantiate

```
M(x) = ray_occlusion_semiring_scan_mixer(x)
```

adapted from the primitive's `(B, 64, d)` token-tensor form to the
`(B, C, 8, 8) -> (B, C, 8, 8)` mixer contract. For each of the 8 chess
ray directions `d`, the mixer gathers along ray steps and weights each
step by the prefix product of `(1 - occupancy)` (the semiring
transmittance) and a learned per-direction step-decay scalar `λ_d`,
recovering Mamba-style selective decay over a chess-rule-derived
topology. Per-direction outputs are concatenated and projected back to
the spatial mixer output. The transmittance and ray-step indices are
computed inside `torch.no_grad()` from the same `simple_18` board the
trainer already consumes.

## Claimed advantage of the swap

Because the only changed term in the residual recurrence is `M`, any
difference in puzzle_binary metrics between `a015` and the sibling
`bt4_conv_mixer` / `bt4_attention_mixer` baselines is attributable to
the mixer choice. A measurable lift would mean the occlusion-gated
ray-semiring operator beats both conv and dense attention as a
per-block spatial mixer at this tower depth and width on sliding-
piece tactics that depend on first-blocker geometry (pins, skewers,
x-rays, discovered attacks).

## Assumptions

- The BT4 tower (`bt4_primitive_mixer`) is held byte-for-byte fixed
  across the `a###_bt4_*_mixer` sweep.
- Optimizer, loss, batch size, augmentation, scheduler, and seed are
  the same across the sweep (see `config.yaml`).
- The mixer obeys the shape-preserving contract — input
  `(B, C, 8, 8)`, output `(B, C, 8, 8)`, no per-block parameter sharing
  outside the mixer itself.
- The transmittance prefix product is computed from `simple_18`
  occupancy and is not differentiated through.

## What is actually proven

- The model builds, runs a forward and backward pass under the shared
  `bt4_primitive_mixer` registration (smoke-tested before this folder
  was created — see `idea.yaml notes`).
- The wrapper enforces the `mixer = ray_occlusion_semiring_scan`
  contract via `build_model_from_config`.

## What is only hypothesised

- Whether the swap actually improves puzzle_binary aggregate or
  CRTK-sliced metrics versus the conv / attention baselines.
- Whether any lift survives the `uniform_transmittance` /
  `constant_direction` / `no_step_decay` falsifiers documented in the
  source primitive's `ablations.md` (lift must come from the blocker-
  aware prefix product, not from a generic depthwise ray-conv).

## Failure cases

- The transmittance construction dominates wall-clock at this tower
  width and the run cannot complete inside the shared training budget.
- The per-direction linears under-utilise their parameter budget at
  small channel widths, producing noisy gradients that the residual
  sum cannot dampen.
- The mixer's lift, if any, fails to survive the
  `uniform_transmittance` ablation, in which case the operator is no
  more than an 8-direction depthwise conv with extra steps.
- Occlusion structure is washed out after one or two BT4 blocks
  because the residual + SqueezeExcite path averages the
  per-direction signals back into a conv-like operator.
