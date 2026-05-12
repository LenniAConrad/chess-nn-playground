# Codex Research Packet: Permanent Ryser Coupling Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1715_tuesday_local_permanent_ryser.md`
- Generated at: 2026-05-05 17:15
- Author: Claude (Opus 4.7, 1M context)
- Status: bespoke implementation already in `src/chess_nn_playground/models/permanent_ryser.py`

## Thesis

For top-k attacker squares `(s_1, ..., s_k)` and top-k defender squares
`(t_1, ..., t_k)` with `k = 6`, build a learned bilinear interaction matrix
`M[i, j] = sigmoid(bilinear(att_feat_i, def_feat_j))` and compute its
**permanent** via Ryser's formula:

```text
perm(M) = (-1)^k sum_{S subset {1..k}} (-1)^|S| prod_{i=1..k} sum_{j in S} M[i, j]
```

`perm(M)` counts unsigned perfect matchings of attackers to defenders — the
Hall-style "can every attacker engage some distinct defender" question made
quantitative. Distinct from i058 DPP (signed determinant), i226 Pfaffian
(signed matchings of skew-graphs), i120 Sinkhorn (doubly-stochastic).

`k = 6` keeps Ryser tractable: 2^6 = 64 subsets per board.

## Distinct From

- DPP (i058): det, not perm; signed.
- Pfaffian (i226): signed perfect matchings of skew adjacency; perm of bipartite is unsigned.
- Sinkhorn (i120): solves the OT polytope; perm is exact count.

## Architecture

`PermanentRyserNetwork` in `src/chess_nn_playground/models/permanent_ryser.py`:

```text
input (B, 18, 8, 8)
  -> BoardConvStem -> (B, C, 8, 8)
  -> attacker_score, defender_score (1x1 conv) -> (B, 64), (B, 64)
  -> top-6 indices for attackers, defenders
  -> gather feature vectors (B, 6, C) for each side
  -> bilinear pairwise -> (B, 6, 6)
  -> sigmoid -> M in [0, 1]^{6x6}
  -> Ryser permanent (64 subset iterations)
  -> features: log|perm|, smooth-sign(perm), ||M||_F, mean(M)
  -> concat pooled CNN
  -> MLP -> (B, num_classes)
```

## Ablations

| Ablation | Target |
|---|---|
| `det_swap` | replace perm with det | tests unsigned vs signed |
| `random_attacker_topk` | random 6 attackers | tests selection |
| `k_eq_4` | smaller bipartite | tests motif size |
| `magnitude_only` | drop sign(perm) feature | (sigmoid M makes perm always >= 0; use w/ unbounded M ablation) |
| `cnn_same_params` | matched baseline | |

## Falsifier

`det_swap` should drop PR AUC ≥ 0.01 (sign-cancellation in det loses signal that permanent preserves).

## Targets

PR AUC ≥ 0.82, F1 ≥ 0.76, near-puzzle FPR ≤ 0.20.
