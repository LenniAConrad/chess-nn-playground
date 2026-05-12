# Codex Research Packet: Stable-Rank Multiscale Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1710_tuesday_local_stable_rank_multiscale.md`
- Generated at: 2026-05-05 17:10
- Author: Claude (Opus 4.7, 1M context)
- Status: bespoke implementation already in `src/chess_nn_playground/models/stable_rank_multiscale.py`

## Thesis

Compute the **stable rank**

```text
sr(M) = ||M||_F^2 / ||M||_2^2   (continuous, differentiable, sr(M) <= rank(M))
```

of a learned 64x64 chess interaction `M` at three block scales:

- full: 1 value over (64 x 64)
- quadrants: 4 values over (32 x 32) sub-blocks
- sub-blocks: 16 values over (16 x 16) sub-blocks

Stable rank is unitarily invariant, scale-equivariant, and quantifies "effective
degrees of freedom". Multiscale stable rank expresses how interactions
concentrate vs disperse across the board — a notion of "effective tactical
spread" that no spectrum-or-norm packet directly exposes.

## Distinct From

- Determinantal volume (i058): det of Gram matrix; not unitarily invariant up to scale.
- Krylov (i076): subspace-based; not bulk rank.
- Tucker certificate (i090): tensor decomposition rank, not stable rank.
- Low-rank cuts (i136): cut-style; not multiscale Frobenius/spectral ratio.

## Architecture

`StableRankMultiscaleNetwork` in `src/chess_nn_playground/models/stable_rank_multiscale.py`:

```text
input (B, 18, 8, 8)
  -> BoardConvStem -> (B, C, 8, 8)
  -> Conv 1x1 left/right projections
  -> bilinear interaction M[s, t] = sum_c L[c, s] * R[c, t]   (B, 64, 64)
  -> stable_rank(M_full)                          (B, 1)
  -> stable_rank on 4 quadrants                   (B, 4)
  -> stable_rank on 16 sub-blocks                 (B, 16)
  -> concat with pooled CNN features
  -> MLP -> (B, num_classes)
```

`||M||_2` via `torch.linalg.svdvals(M)[..., 0]` (differentiable).

## Ablations

| Ablation | Target |
|---|---|
| `single_scale_full` | only full-board sr | tests multiscale |
| `single_scale_subblocks` | only 16 sub-block sr | tests sub-block scale |
| `random_M` | scramble 1x1 projections | tests learned interaction |
| `nuclear_norm_swap` | replace sr with `||M||_*` | tests sr vs nuclear |
| `cnn_same_params` | matched baseline | |

## Falsifier

`single_scale_full` should drop PR AUC ≥ 0.005 (multiscale matters). `nuclear_norm_swap` should drop PR AUC ≥ 0.01 (sr is not just a norm).

## Targets

PR AUC ≥ 0.82, F1 ≥ 0.76, near-puzzle FPR ≤ 0.20.
