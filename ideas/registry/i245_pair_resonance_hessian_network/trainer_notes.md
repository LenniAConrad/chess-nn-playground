# Trainer Notes

- Use `train.py` as-is — it calls `idea_train_cli(__file__)`, which checks
  idea/config/model identity through `validate_idea_for_training` before
  starting the shared `train_from_config` trainer.

- The shared `puzzle_binary` BCE-with-logits loss is sufficient. The model
  emits the puzzle logit at `out["logits"]`, which the trainer's
  `_primary_logits` helper picks up automatically. No custom loss or
  scheduler is required.

- Batch size: the DHPE side path evaluates `1 + top_k + C(top_k, 2)`
  variants per position through the compact PhiScorer. At `top_k=4` that
  is 11 variants per board. The default config uses `batch_size: 192`
  (down from i193's 256) to keep peak GPU memory comparable; you can
  raise it back to 256 on a 12 GB GPU.

- Cost: roughly **2.5x i193 wall-clock per epoch**. For the scout protocol
  (12 epochs, single seed, single 3070) this stays well under the
  per-primitive overnight budget.

- Mixed precision: enabled (`mixed_precision: true`), matches i193. The
  PhiScorer uses standard PyTorch ops that are AMP-safe.

- Reliability tier: `paper_grade` — same as i193 and the rest of the
  scout-comparable primitives.

- Diagnostics surfaced via the standard trainer pathway:
  - `base_logit`, `primitive_delta`, `primitive_delta_raw`,
    `primitive_gate`, `primitive_gate_applied`, `primitive_gate_logit`,
    `primitive_gate_entropy`, `primitive_contribution`.
  - `dhpe_base_phi`, `dhpe_z_pos`, `dhpe_z_neg`, `dhpe_z_total`,
    `dhpe_z_ratio`, `dhpe_z_top1`, `dhpe_valid_count`.
  - All `trunk_*` keys re-export the i193 trunk diagnostics for
    side-by-side comparison.

- Ablation runs: add `model.ablation: <mode>` to the config to swap the
  ablation. The allowed strings are listed in `architecture.md`. Each
  ablation needs its own run directory to keep the slice reports
  comparable to the i193 baseline.

- Resume: standard `training.resume_from` / `training.resume_run_dir`
  paths work because the model state dict is plain PyTorch state.
