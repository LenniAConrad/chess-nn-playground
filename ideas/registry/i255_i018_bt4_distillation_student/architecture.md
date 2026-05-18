# Architecture

`i018 BT4 Distillation Student` (i255) is a BT4-shaped residual convolutional
student designed to be trained with rich supervision distilled from the
i018 `oriented_tactical_sheaf_laplacian` teacher. The student keeps the
BT4 deployment shape (3x3 conv stem, residual 3x3 conv blocks with
Squeeze-Excite, global value head) and adds three lightweight
training-time auxiliary heads that the distillation loss consumes.

The source research markdown is
`ideas/research/packets/classic/i255_i018_bt4_distillation_student.md`;
this folder is the implementation promotion of that packet.

## Mechanism

1. **Mover-oriented canonicalization (`MoverCanonicalize`)**. A fixed
   preprocessing layer that reuses i018's `BoardStateAdapter`
   canonicalization for `simple_18` and `lc0_static_112` inputs so the
   conv tower sees the board from the side to move. For `lc0_bt4_112`
   the input is already canonicalised by the exporter, so the layer is
   a pass-through. No learned parameters - this is a cheap symmetry the
   student inherits without having to relearn.

2. **BT4-style residual conv trunk**. One 3x3 `Conv2d` stem mapping
   `input_channels` to `channels`, followed by `num_blocks` residual
   blocks. Each block is two `3x3 Conv2d` layers with BatchNorm and
   Squeeze-Excite, a residual addition, and ReLU. The trunk shape is
   intentionally identical to `LC0BT4Classifier` so the deployment
   pattern (dense `Conv2d` on a tiny spatial map) is preserved.

3. **Value neck**. The same 1x1 conv -> flatten -> Linear stack as
   `LC0BT4Classifier`. The output `pooled_features` (`value_hidden`-d
   vector) feeds every head, so distillation losses share a single
   summary representation.

4. **Heads** (all emitted in `forward(...)`):

   * `logits` (puzzle binary logit). The only head needed at inference
     time. Built from `value_head: Linear(value_hidden -> num_classes)`.
   * `pooled_features` (`value_hidden`-d). The pre-head feature vector,
     exposed so the trainer can attach a feature-distillation loss
     without bolting a separate hook into the model.
   * `diagnostic_logits` (`diagnostic_dim`-d, default 18 = 6 scalar
     i018 diagnostics + 12-d typed relation density). Small MLP over
     `pooled_features`.
   * `summary_plane_logits` (`(B, summary_plane_dim, 8, 8)`,
     default 8 planes). 1x1 conv over the final 8x8 feature map.
   * `readout_features` (`readout_dim`-d, optional - only present when
     `readout_dim > 0`). Linear projector over `pooled_features` for
     compact teacher-readout matching.

   The diagnostic and summary-plane heads are training-time signals.
   They are emitted unconditionally so the trainer never has to swap
   models between train and infer; their inference cost is small
   compared to the BT4 trunk and irrelevant if the deployment path
   strips them.

## Distillation Targets (consumed by an external loss)

The student is trainable on plain BCE today. When teacher targets are
wired up via an `I018BT4DistillationLoss` term in the trainer, the
auxiliary heads emit predictions for:

| head                     | teacher source                                                                              | role                                                       |
|--------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------|
| `logits`                 | calibrated `sigma(z_t / T_t)`                                                               | Hinton-style soft target + supervised BCE anchor          |
| `diagnostic_logits[0:6]` | i018 `sheaf_tension`, `king_ring_pressure`, `defense_gap`, `triad_defect_energy`, `pin_pressure`, `transport_imbalance` | scalar tactical diagnostic distillation                    |
| `diagnostic_logits[6:18]`| i018 `incidence.relation_density` (12-d typed)                                              | tactical mass distillation                                 |
| `summary_plane_logits`   | 8 compact projections of i018 relation masks (see `SUMMARY_PLANE_NAMES`)                    | spatial relation summary distillation                      |
| `pooled_features` / `readout_features` | i018 readout pre-head vector                                                  | optional compact feature distillation                      |

The 8 summary planes are deliberately compact: the full teacher
relation tensor is `12 * 64 * 64 = 49,152` floats per sample
(~192 KB), and the research markdown explicitly argues against caching
it (~65.9 GB at 360k samples). The 8 summary planes are only
`8 * 64 = 512` floats per sample (~2 KB), so the cache stays
practical (~0.69 GB).

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed by the model. The contract is identical to i018
and BT4.

## Parameter Budget

At the default `base` scale (`channels=64`, `num_blocks=4`,
`value_channels=16`, `value_hidden=128`, `simple_18`):

| Block                              | Params    |
|------------------------------------|----------:|
| Stem `Conv2d(18 -> 64)` + BN       |     ~10k  |
| 4 x BT4 residual block             |    ~298k  |
| Value neck (1x1 conv + Linear)     |    ~135k  |
| Logit head                         |       129 |
| Diagnostic head (MLP)              |     ~5.3k |
| Summary plane head (1x1 conv)      |       520 |
| **Total (base)**                   | **453,159**|

At `scale_up` (`channels=96`, `num_blocks=6`): ~1.18M params. Still well
inside the BT4-class footprint and well under i018 `scale_xl`.

## Implementation Binding

- Registered model name: `i018_bt4_distillation_student`.
- Source implementation:
  `src/chess_nn_playground/models/trunk/bt4_distill_student.py`
  (`BT4DistillationStudent`, `MoverCanonicalize`, `BT4StudentBlock`,
  `SqueezeExcite`, `build_bt4_distill_student_from_config`).
- Idea-local wrapper:
  `ideas/registry/i255_i018_bt4_distillation_student/model.py`
  (`build_model_from_config`).
- Training config: `ideas/registry/i255_i018_bt4_distillation_student/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  `'i018_bt4_distillation_student': ('chess_nn_playground.models.trunk.bt4_distill_student', 'build_bt4_distill_student_from_config')`.
- Mover canonicalization reuses
  `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf.py`
  (`BoardStateAdapter`) without modifying the i018 trunk.
