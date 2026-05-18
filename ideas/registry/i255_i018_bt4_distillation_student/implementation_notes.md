# Implementation Notes

- Central code:
  `src/chess_nn_playground/models/trunk/bt4_distill_student.py`
  (`BT4DistillationStudent`, `MoverCanonicalize`, `BT4StudentBlock`,
  `SqueezeExcite`, `build_bt4_distill_student_from_config`).
- Idea-local wrapper:
  `ideas/registry/i255_i018_bt4_distillation_student/model.py`
  (`build_model_from_config`).
- Registry key: `i018_bt4_distillation_student`.
- Teacher of record (NOT modified by this packet):
  `i018 oriented_tactical_sheaf_laplacian`
  (`src/chess_nn_playground/models/trunk/oriented_tactical_sheaf.py`).

## What is implemented

- `MoverCanonicalize` reuses i018's `BoardStateAdapter` canonicalization
  for `simple_18` and `lc0_static_112` (rotate + color-swap when the
  side to move is black) and is a pass-through for `lc0_bt4_112` (the
  exporter already canonicalises). No new learned parameters.
- `BT4StudentBlock` is structurally identical to `LC0BT4Block`
  (two 3x3 `Conv2d` + BatchNorm + Squeeze-Excite + residual + ReLU).
  The student backbone is intentionally the same dense-conv shape that
  the repo measured at ~0.83 ms per position on CPU.
- `BT4DistillationStudent` exposes four / five outputs in `forward`:
    * `logits` (always; main puzzle logit).
    * `pooled_features` (always; pre-head feature vector for optional
      feature distillation).
    * `diagnostic_logits` (`diagnostic_dim`-d, default 18).
    * `summary_plane_logits` (`(B, summary_plane_dim, 8, 8)`,
      default 8 planes).
    * `readout_features` (only when `readout_dim > 0`).
- Disabling an auxiliary head is a config edit:
  `model.diagnostic_dim=0` removes the diagnostic MLP entirely;
  `model.summary_plane_dim=0` removes the 1x1 plane head;
  `model.readout_dim=0` keeps the readout projector off.
- The model is trainable on plain BCE today (the trainer's existing
  `bce_with_logits` loss path consumes `logits` and ignores the other
  output keys), so `tests/test_idea_registry.py::test_fully_implemented_idea_is_smoke_testable`
  passes against the standard smoke contract.

## What is intentionally NOT implemented in this packet

The research markdown describes a six-piece pipeline. Three pieces are
in this packet; three pieces are scaffold-only and are explicitly
flagged in the trainer notes:

| piece                                   | status            |
|-----------------------------------------|-------------------|
| 1. New idea registry folder             | implemented       |
| 2. New student model                    | implemented       |
| 3. Teacher export hook (`return_teacher_targets`) | scaffold-only |
| 4. Teacher-cache script                 | scaffold-only     |
| 5. New loss path (`I018BT4DistillationLoss`) | scaffold-only |
| 6. Benchmark configs                    | implemented (one config; matrix as inline notes) |

The scaffold-only pieces are deliberate. The student already emits the
right tensor shapes for the distillation loss; bolting the loss onto
the trainer is a separate change that touches the loss registry and
the teacher I/O path, and that change should land in a focused PR
rather than as a side-effect of this promotion.

## Parameter Budget Validation

The base-scale parameter count is 453,159 (verified by
`chess_nn_playground.models.registry.build_model`):

| Variant   | channels | num_blocks | input_channels | Params      |
|-----------|---------:|-----------:|---------------:|------------:|
| base      |       64 |          4 |             18 |   453,159   |
| scale_up  |       96 |          6 |             18 | 1,179,719   |
| base_bt4  |       64 |          4 |            112 |   507,303   |

(`base_bt4` adds the wider stem `Conv2d(112 -> 64)`; the rest is
identical.)

## Inputs not used

The model does not consume CRTK metadata, source labels, verification
flags, engine evaluations, Stockfish scores, principal variations, or
any report-only metadata. It only sees the board tensor for the
configured encoding.

## How to walk the research-markdown matrix

The default `config.yaml` is the `base / simple_18 / canonicalize=on`
row. Other rows are single-line edits:

```
model.channels: 96
model.num_blocks: 6
# -> scale_up student

data.encoding: lc0_bt4_112
model.input_channels: 112
model.encoding: lc0_bt4_112
# -> lc0_bt4_112 student (benchmark comparability)

model.canonicalize: false
# -> canonicalization ablation

model.diagnostic_dim: 0
# -> drop the scalar diagnostic head

model.summary_plane_dim: 0
# -> drop the spatial plane head

model.readout_dim: 64
# -> turn on the readout projector
```

## Why the scaffold-only loss is not a blocker

The student is trainable end-to-end on plain BCE today. The
distillation loss is *additive* over the supervised anchor:
`L_sup` is always on, and the supervised baseline is a real,
informative student. The model contract is *loss-ready* (every output
the distillation loss needs is already emitted by `forward`), so when
the loss lands it can plug into the trainer without modifying this
folder. That is exactly the "minimal production implementation that
plugs into the shared trainer" rule from the task brief.
