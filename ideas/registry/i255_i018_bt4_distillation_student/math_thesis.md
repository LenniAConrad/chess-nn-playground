# Math Thesis

i255 takes the BT4 conv backbone as given and asks one controlled
question: can the gap between BT4 (~0.86 PR-AUC at base) and i018
`scale_xl` (~0.89 PR-AUC) be closed by training the BT4 student with
richer-than-logits supervision distilled from i018, while keeping BT4's
dense-convolutional deployment shape (and BT4-class CPU latency).

The full research-markdown loss is:

```
L = lambda_sup  * L_sup
  + lambda_kd   * L_kd
  + lambda_diag * L_diag
  + lambda_plane* L_plane
  + lambda_read * L_read
  + lambda_brier* L_brier
  + lambda_rank * L_rank
```

with the per-term definitions:

```
L_sup    = BCEWithLogits(z_s, y)
L_kd     = T_t^2 * KL( Bern(sigma(z_t / T_t)) || Bern(sigma(z_s / T_t)) )
L_diag   = (1/M) sum_m w_m * Huber( hat_d_{s,m} - hat_d_{t,m} )
L_plane  = (1/K) sum_k SmoothL1( P_{s,k}, P_{t,k}; omega_k )
L_read   = || W_s r_s - LayerNorm(r_t) ||_1
L_brier  = (sigma(z_s) - y)^2
L_rank   = (1/|P|) sum_{(i,j) in P} log(1 + exp(-s_ij * (z_s^i - z_s^j))),
           s_ij = sign(z_t^i - z_t^j)
```

with the research-markdown default weights:

| term            | weight | note |
|-----------------|------:|------|
| `lambda_sup`    | 1.00  | supervised anchor (always on) |
| `lambda_kd`     | 0.80  | main soft-target transfer |
| `lambda_diag`   | 0.25  | scalar tactical supervision |
| `lambda_plane`  | 0.15  | spatial tactical supervision |
| `lambda_read`   | 0.10  | compact feature transfer |
| `lambda_brier`  | 0.05  | calibration-aware regulariser |
| `lambda_rank`   | 0.05  | optional, only on hard-negative batches |

The student emits all the output heads that those losses need
(`logits`, `diagnostic_logits`, `summary_plane_logits`,
`pooled_features`, and `readout_features` when `readout_dim > 0`); the
loss machinery itself is intentionally not bundled in this scaffold
(see `implementation_notes.md`). The model's contract is therefore
*loss-ready*, not *loss-coupled*.

## Teacher

The teacher of record is
`i018 oriented_tactical_sheaf_laplacian` at `scale_xl`. The teacher is
board-only (CRTK metadata is reporting-only by the repo's contract).
Teacher quantities consumed by the distillation loss:

- `z_t`: teacher logit on each position.
- `p_t`: calibrated teacher probability after a single validation-fit
  temperature `T_t`.
- `d_t in R^6`: six mandatory scalar diagnostics (`sheaf_tension`,
  `king_ring_pressure`, `defense_gap`, `triad_defect_energy`,
  `pin_pressure`, `transport_imbalance`) which i018 already exposes in
  its output dict.
- `relation_density_t in R^12`: the typed relation density vector
  (already computed by `TacticalIncidenceBuilder`, but currently
  internal - needs a small `return_teacher_targets=True` export hook
  on the i018 model to surface it).
- `P_t in R^{8 x 8 x 8}`: eight 8x8 summary planes derived from the
  i018 relation tensor (see `SUMMARY_PLANE_NAMES` in
  `bt4_distill_student.py`). These are cheap projections of relation
  masks - the research markdown is explicit that the full
  `12 x 64 x 64` relation tensor must NOT be cached (~65.9 GB at
  360k samples).
- `r_t in R^{readout_dim}`: optional teacher readout vector for
  compact feature distillation.

## Student

The student forward is:

```
x' = canonical(x)                          # MoverCanonicalize
h0 = ReLU(BN(Conv3x3(x')))                 # stem
hk = BT4_block(h_{k-1})  for k = 1..N      # residual trunk
v  = value_neck(hN)                        # pooled summary
z_s = head(v)                              # main puzzle logit
hat_d_s = diagnostic_head(v)               # 18-d diagnostic head (6 + 12)
P_s = plane_head(hN)                       # (8, 8, 8) summary planes
r_s = readout_head(v)                      # optional readout projector
```

`hN` has shape `(B, channels, 8, 8)`, so `plane_head` is a single 1x1
`Conv2d(channels -> 8)`. `v` has dimension `value_hidden`, so
`diagnostic_head` and `readout_head` are tiny MLPs / linear maps.

## Calibration Discipline

The research markdown is explicit about three calibration knobs:

1. Fit a single teacher temperature `T_t` on validation **before**
   caching teacher targets.
2. Include a small Brier term `L_brier` during student training.
3. Temperature-scale the student post-hoc on validation before final
   reporting.

None of these change the student architecture; they live in the loss
and in the post-hoc reporting code.

## Hard-Negative Emphasis

The trainer already supports `fine_label`-aware losses without making
`fine_label` an input feature. The research markdown's hard-negative
recipe is:

- Up-weight BCE on `source_class == 1` (near-puzzle) positions by
  a factor `~1.5-2.0` after a short warm start.
- Optional `L_rank` term on a hard set `H` (puzzles + near-puzzles +
  teacher-ambiguous negatives), using teacher logit orderings within
  the batch.

Both are loss-time signals; the student model contract is unchanged.

## Expected Effect Size (Promotion Gate)

| metric                              | base target | scale_up target |
|-------------------------------------|------------:|----------------:|
| PR-AUC                              | >= 0.875    | >= 0.880        |
| near-puzzle FP @ recall 0.80        | <= 0.16     | <= 0.155        |
| puzzle recall                       | >= 0.80     | >= 0.80         |
| batch-1 CPU latency                 | <= 1.2 ms   | <= 1.6 ms       |

The CPU latency targets follow the repo's measurement methodology:
CPU-only, eager mode, no `torch.compile`, warmup + timed forward
passes, batch sizes 1 / 8 / 32, batch-1 per-position latency is the
decision metric.

## Failure Modes That Drop i255

- **Diagnostic mimicry without decision gain.** Student matches teacher
  diagnostics nicely but `logits` barely improves on the supervised
  task. Response: keep `lambda_sup` dominant, gate promotion on the
  slice metrics, not on diagnostic fit.
- **Teacher blind-spot inheritance.** i018 has known stress slices
  (`equal`, `hard`, `very_hard`, `mate_in_1`, `promotion`); the student
  inherits them. Response: slice-aware hard-negative weighting via the
  trainer's `fine_label`-aware loss hooks, *not* a more exotic student.
- **Over-smoothing.** Too much KD weight or temperature improves
  calibration while hurting puzzle recall. Response: report calibration
  + recall together, do not promote on calibration alone.
- **Latency erosion.** Training-only heads (`summary_plane_head`,
  `readout_head`) leaking compute into inference. Mitigation: heads are
  tiny by construction (1x1 conv and a single Linear), and the
  deployment path can strip them entirely if needed.
