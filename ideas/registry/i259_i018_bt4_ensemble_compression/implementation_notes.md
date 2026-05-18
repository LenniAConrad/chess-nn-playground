# Implementation Notes

- Central model code:
  `src/chess_nn_playground/models/architecture/i018_bt4_ensemble_compression.py`.
- Idea-local wrapper:
  `ideas/registry/i259_i018_bt4_ensemble_compression/model.py`.
- Registry key: `i018_bt4_ensemble_compression`.
- Source research packet (preserved in place):
  `ideas/research/packets/classic/i259_i018_bt4_ensemble_compression.md`.
- Teacher implementations re-used unchanged:
  - i018: `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf.py`
    (`OrientedTacticalSheafNet`).
  - BT4 conv: `src/chess_nn_playground/models/trunk/lc0_bt4.py`
    (`LC0BT4Classifier`).
- Distillation hint targets are the i018 diagnostic keys already
  exported by the registered `oriented_tactical_sheaf_laplacian`
  model: `sheaf_tension`, `triad_defect_energy`,
  `king_ring_pressure`, `reply_pressure`, `defense_gap`,
  `pin_pressure`.

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. The BT4 teacher is configured to take 18 input channels so the
i018 student/teacher and BT4 teacher all share the same input — no
`lc0_bt4_112` exporter is required for this architecture, which keeps
the controlled comparison clean and the trainer config minimal. CRTK
metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and motif tags are *not* used
as model inputs.

## Teacher mode

The registered config sets `model.teacher_mode='off'` so the smoke
tests and the default training run see a clean student-only network.
Two other modes are supported:

- `research`: both teachers are built and run under `torch.no_grad()`
  every forward. Used to generate teacher caches and to evaluate
  fusion variants.
- `frozen`: identical to `research` in this implementation; the name
  exists so a later phase can swap in pretrained teacher checkpoints
  without changing the trainer wiring.

Both modes require teacher checkpoints to be loaded externally (the
idea-local `train.py` is a thin wrapper around `idea_train_cli` and
does not currently load teacher weights). Phase B / C of the source
packet's plan describes how the teacher checkpoints flow into the
distillation cache; that pipeline is documented in `trainer_notes.md`
and `ablations.md` and is not run automatically by this idea folder
under `CLAUDE_ALLOW_TRAINING=0`.

## Stop-gradient contract

Both teacher trunks are constructed with `requires_grad_(False)` and
their forward calls are wrapped in `torch.no_grad()`. The fusion gate
contains trainable parameters but its output only affects
`teacher_alpha` and `teacher_ensemble_logit`; the student's
BCE-with-logits loss is computed against `logits = student_logit`, so
the fusion gate gradient through that loss is identically zero.

## Output dict contract

The model output is a `dict[str, Tensor]` with the following keys
emitted on every forward (with zeros when `teacher_mode='off'` so
downstream consumers see a stable schema):

- `logits`, `student_logit`, `student_probability`
- `teacher_i018_logit`, `teacher_bt4_logit`,
  `teacher_ensemble_logit`, `teacher_disagreement`,
  `teacher_entropy`, `teacher_alpha`
- `diagnostic_hint_<name>` for each entry in `diagnostic_hint_keys`
- `teacher_i018_<name>` for each i018 diagnostic in
  `diagnostic_hint_keys` (research mode only)

All per-sample scalar tensors have shape `(B,)` so the shared trainer
copies them into `predictions_<split>.parquet` without further
reshaping.

## Ablation modes

See `I018Bt4EnsembleCompressionNet.ALLOWED_ABLATIONS`. The
distillation falsifier is `shuffle_teacher_logits`; if KD lift
survives in-batch permutation of the teacher logits the teacher
boundary is not load-bearing and the compression scheme should be
dropped.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` that composes the bespoke i018
trunk and a freshly built BT4 conv tower with new student / fusion
heads. It does not import or call
`build_research_packet_probe_from_config`, does not delegate to any
shared baseline builder from the idea-local `model.py`, and has its
own forward pass. The implementation_kind audit therefore detects
this folder as `bespoke_model`, which matches the `idea.yaml`
declaration.
