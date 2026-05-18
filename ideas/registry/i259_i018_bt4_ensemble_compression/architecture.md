# Architecture

`i018 + BT4 Ensemble Compression` (i259) promotes
`ideas/research/packets/classic/i259_i018_bt4_ensemble_compression.md`
into a single bespoke network whose deployment forward is a fast,
BT4-shaped convolutional student over the repo's `simple_18`
current-board tensor. The packet's research-time teacher ensemble
(`i018` + `lc0_bt4_classifier`) is implemented as a pair of *frozen*
teacher modules co-evaluated inside the same network so a single
forward can emit both the student logit and the cached teacher logits
needed to fit the distillation loss described in
`math_thesis.md`. The teachers are never trainable here; they only
serve as offline distillation oracles.

The network consumes `(B, 18, 8, 8)` and returns a `dict[str, Tensor]`
with at least:

| key | shape | meaning |
|---|---|---|
| `logits` | `(B,)` | student puzzle logit (BCE-with-logits target) |
| `student_logit` | `(B,)` | detached alias of `logits` |
| `student_probability` | `(B,)` | `sigmoid(student_logit)` |
| `diagnostic_hint_<name>` | `(B,)` | scalar hint head per `diagnostic_hint_keys` entry |

When `model.teacher_mode != 'off'` the network additionally returns
`teacher_i018_logit`, `teacher_bt4_logit`, `teacher_ensemble_logit`,
`teacher_disagreement`, `teacher_entropy`, `teacher_alpha`, and the
selected `teacher_i018_<diagnostic>` keys. With `teacher_mode='off'`
(the default config) the teacher-prefixed tensors are returned as
zero tensors so the trainer and audits can rely on a stable output
dict regardless of mode.

## Mechanism

1. **Student conv tower.** `_StudentConvTower` wraps
   `LC0BT4Classifier` (BT4-shaped stem + N residual SE blocks + value
   head). The student value head emits the puzzle logit. Auxiliary
   diagnostic-hint heads (one per `diagnostic_hint_keys` entry) read
   from the trunk's pooled feature map and emit one scalar per
   sample. These are the KD compression targets defined by the
   packet's
   `diag` term, so they let the offline trainer regress the student's
   internal representation onto a normalized subset of i018's
   diagnostics without changing the i018 forward.

2. **i018 teacher (`teacher_mode != 'off'`).** A bespoke
   `OrientedTacticalSheafNet` (the registered `i018` trunk) runs on
   the same `simple_18` input under `torch.no_grad()`. Its dict
   output supplies `teacher_i018_logit` and the
   `teacher_i018_<diagnostic>` tensors used as KD hint targets in
   distillation runs.

3. **BT4 teacher (`teacher_mode != 'off'`).** A frozen
   `LC0BT4Classifier` configured to consume 18 channels runs on the
   same input under `torch.no_grad()` and supplies
   `teacher_bt4_logit`.

4. **Teacher fusion.** Calibrated logits
   `tilde z_s = z_s / T`, `tilde z_b = z_b / T` are combined by one
   of three fusion modes (`equal_weight`, `tuned_alpha`,
   `uncertainty_gated`), producing the `teacher_ensemble_logit`
   surfaced to the trainer. `teacher_disagreement` and
   `teacher_entropy` are exported for the audit packet.

5. **Loss flow.** Gradient flows only through the student
   (`requires_grad_(False)` on both teachers + `torch.no_grad()` on
   their forward). The shared trainer's BCE-with-logits term uses
   `logits` directly. Distillation cache generation and the KD /
   diagnostic-hint terms in `math_thesis.md` are computed by the
   downstream offline scripts using the surfaced teacher outputs.

6. **Ablations.** Five supported modes, controlled by `model.ablation`:

   - `none`: full architecture (default).
   - `student_only`: short-circuits teacher evaluation; equivalent to
     `teacher_mode='off'`.
   - `zero_hint_heads`: zeros the diagnostic_hint outputs, leaving
     the student logit intact. Tests whether the hint heads regress
     to non-trivial regression targets at training time.
   - `teacher_logits_only`: rebinds `logits` to the ensemble
     `teacher_ensemble_logit`. Used to evaluate the teacher boundary
     directly without the student; do not train with this ablation
     because the teachers are frozen.
   - `shuffle_teacher_logits`: in-batch permutation of teacher
     logits (falsifier — if KD lift survives this, the teacher
     boundary is not load-bearing).

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, motif tags, and any other
report-only audit columns are *not* consumed by the model. Both
teachers run on the same `simple_18` tensor the student consumes, so
the i018 teacher inherits its existing `simple_18` contract verbatim.

## Cost

| Stage | Cost |
|---|---|
| Student forward | One BT4-shaped conv stack |
| Teacher i018 (research mode only) | One i018 sheaf forward, no grad |
| Teacher BT4 (research mode only) | One BT4 conv forward, no grad |
| Fusion | One sigmoid + linear gate (negligible) |

At inference the deployment build sets `teacher_mode='off'` so the
network is effectively a single BT4-shaped student — that is the
"one encoding, one student, one calibration layer" target documented
in the source packet.

## Implementation Binding

- Registered model name: `i018_bt4_ensemble_compression`.
- Source implementation:
  `src/chess_nn_playground/models/architecture/i018_bt4_ensemble_compression.py`
  (defines `I018Bt4EnsembleCompressionNet` and
  `build_i018_bt4_ensemble_compression_from_config`).
- Builder entry in
  `src/chess_nn_playground/models/_registry_manifest.py`:
  `'i018_bt4_ensemble_compression': ('chess_nn_playground.models.architecture.i018_bt4_ensemble_compression',
   'build_i018_bt4_ensemble_compression_from_config')`.
- Idea-local wrapper:
  `ideas/registry/i259_i018_bt4_ensemble_compression/model.py`
  (re-exports `build_model_from_config(config)` for the shared
  trainer guard).
- Training config:
  `ideas/registry/i259_i018_bt4_ensemble_compression/config.yaml`.
- Source research packet (preserved in place, referenced from this
  idea folder):
  `ideas/research/packets/classic/i259_i018_bt4_ensemble_compression.md`.
