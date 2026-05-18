# Math Thesis

Source: `ideas/research/packets/classic/i259_i018_bt4_ensemble_compression.md`.

## Working thesis

For a board state `x`, let `z_s(x)` be the i018 sheaf-trunk teacher
logit and `z_b(x)` be the BT4 conv teacher logit. Per-model
temperature parameters `T_s, T_b > 0` produce calibrated logits

```
tilde z_s = z_s / T_s,        tilde z_b = z_b / T_b
p_s = sigmoid(tilde z_s),     p_b = sigmoid(tilde z_b)
```

The teacher ensemble logit is one of:

```
equal_weight:        z_ens(x) = 0.5 * tilde z_s + 0.5 * tilde z_b
tuned_alpha:         z_ens(x) = alpha * tilde z_s + (1-alpha) * tilde z_b
uncertainty_gated:   w(x)     = sigmoid(gate^T phi(x))
                     z_ens(x) = w(x) * tilde z_s + (1 - w(x)) * tilde z_b
```

with `phi(x) = [|p_s - p_b|, H((p_s + p_b)/2), d_1, ..., d_k]` (the
binary-entropy + diagnostic-aware fusion gate input). The repo's
deployment artifact is *not* the ensemble — it is the distilled
student `z_stu(x)` produced by the student conv tower defined in
`src/chess_nn_playground/models/architecture/i018_bt4_ensemble_compression.py`.

The distillation loss the packet defines is

```
L = lambda_hard * BCE(y, sigmoid(z_stu))
  + lambda_KD   * T_d^2 *
        KL( sigmoid(z_teach / T_d) || sigmoid(z_stu / T_d) )
  + lambda_diag * sum_j || h_j(x) - hat d_j(x) ||^2
```

where `z_teach = z_ens` (chosen fusion variant), `hat d_j` are the
normalized teacher diagnostics (`diagnostic_hint_keys`), and `h_j`
are the student's `diagnostic_hint_<name>` heads. In this repo the
hard-label BCE is wired through the shared trainer because `logits`
is bound to `z_stu`; the KD and `diag` terms run as offline scripts
over the surfaced `teacher_*` and `diagnostic_hint_*` columns (this
keeps the trainer surface minimal and lets us reuse the existing
puzzle_binary BCE-with-logits loss).

## Stop-gradient contract

Teacher parameters are frozen at construction
(`requires_grad_(False)`) and the teacher forward runs under
`torch.no_grad()`. The student gradient flows only through the
student conv tower and its diagnostic-hint heads. The fusion gate
runs over teacher signals only; in `uncertainty_gated` mode it
participates in the teacher output but is itself trained from offline
calibration sweeps, not from the BCE loss (the gate parameters are
exercised in the forward but their gradient term in the BCE loss is
identically zero because BCE only sees `z_stu`).

## Calibration

Per-teacher temperatures `T_s, T_b` and the distillation temperature
`T_d` are post-hoc parameters fit on the validation split as
described in the source packet's Phase A. They are not learned end
to end; the registered config exposes `teacher_temperature` and
`teacher_alpha` so different temperature/weight choices can be swept
without rebuilding the architecture.

## Why this matters

The repo summary documented in the source packet flags i018 as the
accuracy-per-parameter champion and BT4 conv as the fast, robust
baseline. The packet's claim is that a clean distillation from a
calibrated i018 + BT4 teacher into a BT4-shaped student should keep
most of the near-puzzle rejection lift at BT4 latency, which matches
the repo's documented deployment shape ("one encoding, one student,
one calibration layer"). The success criterion is matched-recall
near-puzzle FP at recall 0.80 / 0.85, not raw PR-AUC — the
distillation network surfaces exactly the tensors needed to compute
that metric on `predictions_<split>.parquet` columns.
