# Ablations

- Ablation switches (string `model.ablation` in config.yaml):
  | id | label | description |
  |---|---|---|
  | A0 | `none` | Full DHPE primitive — control. |
  | A1 | `unsigned` | Replace signed Hessian with `|H|` before aggregating. |
  | A2 | `no_dhpe` | Zero the DHPE fingerprint and force the gate to 0. |
  | A3 | `shuffled_pairs` | Random permutation of the pair-Hessian entries. |
  | A4 | `shuffle_singles` | Random permutation of the per-piece singles inside the Hessian formation. |
  | A5 | `zero_gate` | Sigmoid gate forced to 0 (primitive still trains, no contribution). |
  | A6 | `trunk_only` | Force `logits = base_logit` (i193 verbatim). |

- What each ablation tests:
  - **A1 `unsigned`**: whether the *sign* of the pair-Hessian is the
    discriminating quantity. The DHPE spec calls this out as the most
    important falsifier — if `|H|` matches the full architecture, the
    primitive is dropped.
  - **A2 `no_dhpe`**: whether the primitive contributes anything at all
    beyond the i193 base.
  - **A3 `shuffled_pairs`**: whether the *pair-identity* matters or only
    the aggregate statistics; matched performance under this ablation
    would imply the primitive is acting as a noisy regulariser, not as a
    pair-interaction probe.
  - **A4 `shuffle_singles`**: whether the saliency-selected piece order
    matters for the Hessian.
  - **A5 `zero_gate`**: whether the gate is doing the work, not the
    primitive delta itself.
  - **A6 `trunk_only`**: sanity baseline equal to i193.

- Falsification criteria (from the DHPE spec, applied to the scout run):
  - `crtk_eval_bucket = equal` PR AUC: i245 >= **0.835** (i193 0.817 + 0.018).
  - Aggregate test PR AUC: i245 >= **0.871** (i193 - 0.005 tolerance).
  - No slice regresses by more than `0.01` PR AUC against i193.
  - A1 ablation must lose >= 50% of the equal-slice lift; otherwise the
    primitive is dropped.
  - A2 ablation must match the i193 baseline (sanity check on the gate).

- Promotion threshold (matches the operator-level rule in
  `PRIMITIVE_TRAINING_TODO.md`): keep DHPE only if it improves its
  declared target slice **and** the matched ablation (A1) loses most of
  that lift. Aggregate PR AUC alone is not enough.
