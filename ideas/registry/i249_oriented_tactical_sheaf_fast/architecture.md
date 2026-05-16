# Architecture

`Oriented Tactical Sheaf Laplacian (Fast)` is a pure execution optimization of
i018 `oriented_tactical_sheaf_laplacian`. **Same math, same parameters, same
numerics** — only the GPU execution pattern changes.

## Implementation Binding

- Registered model name: `oriented_tactical_sheaf_fast`
- Source implementation: `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf_fast.py`
- Idea-local wrapper: `ideas/registry/i249_oriented_tactical_sheaf_fast/model.py`

## What changed vs i018

i018 is FLOP-light but wall-clock-slow: its `SheafDiffusionBlock` runs a
12-iteration Python loop over typed relations (~72 small kernel launches per
block, x depth) and rematerializes large `(B, 64, 64, stalk)` intermediates
every iteration. On GPUs this is launch-overhead- and bandwidth-bound.

This variant changes two things and nothing else:

1. **`FastSheafDiffusionBlock`** — replaces the per-relation Python loop with:
   - one batched einsum projecting *all* relation source/target stalks at once
     (`einsum('bns,rst->brnt', z, rho_src)`), instead of 12 separate `z @ rho[r]`;
   - a chunked batched coboundary: relations are processed in groups of
     `sheaf_chunk_size` (default 3) so the peak `(B, chunk, 64, 64, stalk)`
     intermediate stays within an 8 GB GPU.
   The parameter set is byte-for-byte the same as i018's block (`rho_src`,
   `rho_dst`, `relation_gate_logits`, `eta_logit`, `node_to_stalk`,
   `stalk_to_node`, `node_mlp`, `norm`), and the arithmetic is the same — only
   reordered. A model trained with either block is the same architecture.

2. **Optional `torch.compile`** — i018 has fully static shapes (board
   `(B, 18, 8, 8)`, relations `(B, 12, 64, 64)`), the ideal case for kernel
   fusion + CUDA graphs. `compile_model: true` wraps a bound method (not the
   module) so `state_dict` keys stay clean.

`BoardStateAdapter`, `TacticalIncidenceBuilder`, `SquareTokenEncoder`,
`TriadDefectPool`, and the readout head are imported unchanged from the i018
source module, so this variant cannot silently drift from i018.

## Numerical equivalence (verified)

With i018 weights loaded into the fast net:

- eval-mode forward `logits` match to **~6e-8** max abs diff;
- `sheaf_tension` / `pin_pressure` diagnostics match to **0** / `~1e-7`;
- loss matches to **0.0**, and per-parameter gradients (`rho_src`, `rho_dst`,
  `relation_gate_logits`, linear weights, head weights) match to **~1e-10**.

So the expected test PR-AUC is identical to i018; the only intended difference
is GPU wall-clock.

## Benchmark plan

3 seeds (42, 43, 44) x 3 scales (base, scale_up:1.5, scale_xl:2), identical
training hyperparameters to the i018 paper-grade runs. Reports measured
`samples_per_second` and `fit_elapsed_seconds` next to test PR-AUC, compared
against the i018 baseline already in `results/paper_grade_top3/`.

## Contract

Identical to i018: input `(B, C, 8, 8)` board tensor only; output `dict` with
`logits` of shape `(B,)` plus the i018 diagnostic tensors.
