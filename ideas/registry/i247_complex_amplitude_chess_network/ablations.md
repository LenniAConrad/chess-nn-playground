# Ablations

- Ablation switches (string `model.ablation` in config.yaml):
  | id | label | description |
  |---|---|---|
  | A0 | `none` | Full CAIO primitive — control. |
  | A1 | `real_only` | Force `theta = 0`; amplitudes become real, complex layer reduces to a real relation head. |
  | A2 | `random_phase` | Replace learned phase with uniformly random phase. |
  | A3 | `free_phase` | Drop the rule-phase contribution; keep learned phase. |
  | A4 | `shuffle_relation_masks` | Random permutation of relation mask entries. |
  | A5 | `no_conjugacy` | Force `conj_error = 0`; skip the colour-flip pass. |
  | A6 | `constructive_only` | Zero destructive / curl features. |
  | A7 | `no_caio` | Zero CAIO fingerprint and gate; equivalent to "primitive off" sanity check. |
  | A8 | `zero_gate` | Sigmoid gate forced to 0 (primitive trains but contributes nothing). |
  | A9 | `trunk_only` | Force `logits = base_logit` (i193 verbatim). |

- What each ablation tests:
  - **A1 `real_only`**: whether complex phase is load-bearing. If the
    real-only ablation matches, CAIO reduces to a real bilinear relation
    head and the primitive should be dropped.
  - **A2 `random_phase`**: whether the *learned* phase carries signal.
  - **A3 `free_phase`**: whether the chess-rule phase tying matters
    (piece colour, side-to-move, square colour). If `free_phase` matches,
    the model is just learning arbitrary phase from data.
  - **A4 `shuffle_relation_masks`**: whether the chess-relation
    structure (king-zone / ray / square-colour / file-rank) drives the
    signal, or whether any (64, 64) mask of similar density would do.
  - **A5 `no_conjugacy`**: whether the Z2 colour-flip equivariance
    diagnostic contributes.
  - **A6 `constructive_only`**: whether destructive interference is
    load-bearing or whether constructive mass alone suffices.
  - **A7 `no_caio`**: hard sanity check that the primitive contributes
    anything.
  - **A8 `zero_gate`**: whether the gate is doing the work, not the
    primitive delta itself.
  - **A9 `trunk_only`**: sanity baseline equal to i193.

- Falsification criteria (from the CAIO spec, applied to the scout run):
  - Matched-recall near-puzzle false-positive rate at recall 0.80
    improves by `>= 0.01` absolute over i193.
  - Aggregate PR AUC does not regress by more than `0.005`.
  - At least two hard slices (`equal`, `hard`, `mate_in_1`, promotion,
    underpromotion) improve.
  - A1 `real_only` and A2 `random_phase` ablations lose `>= 50%` of the
    measured near-puzzle FP improvement; otherwise complex phase /
    chess-rule phase is not load-bearing and the primitive is dropped.
  - A4 `shuffle_relation_masks` ablation loses `>= 50%` of the slice
    lift; otherwise the relation structure is not load-bearing.

- Promotion threshold: keep CAIO only if it improves at least two hard
  slices **and** at least one phase/mask ablation loses `>= 50%` of the
  measured lift. Aggregate PR AUC alone is not enough.

- Notes on "gains via parameters": A free-phase complex MLP control
  with matched parameter count (A3 + matched real-only baseline) must
  *not* match the full CAIO architecture. The discriminator MLP at the
  end is intentionally small (2-layer LayerNorm + Linear + GELU + Linear)
  so any lift can be attributed to the interference fingerprint, not
  to MLP capacity.
