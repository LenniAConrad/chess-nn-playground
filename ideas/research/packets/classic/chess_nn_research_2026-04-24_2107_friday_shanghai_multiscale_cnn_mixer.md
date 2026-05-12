# Codex Handoff Packet: Multi-Scale Dilated Board Mixer CNN

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-24_2107_friday_shanghai_multiscale_cnn_mixer.md`
- Generated at: 2026-04-24 21:07
- Weekday: Friday
- Timezone: Asia/Shanghai
- Idea slug: `multiscale_cnn_mixer`
- Intended next consumer: Codex
- Status: regular practical architecture idea, not implemented

## 2. Executive Selection

- Idea name: Multi-Scale Dilated Board Mixer CNN
- One-sentence thesis: A practical chess-board CNN should mix local, knight-distance, diagonal/ray-like, and board-wide context early, so a compact parallel-dilation mixer may be a stronger regular benchmark than the current simple CNN without introducing exotic research machinery.
- Idea fingerprint: `simple_18` input + parallel 3x3 convolutions with dilation 1/2/3 + channel mixer + coordinate planes + global context token + binary classifier.
- Why this is a regular idea: it is a straightforward neural architecture baseline, not a new mathematical family.
- Why it is still worth adding: many exotic packets should be compared against a stronger conventional CNN that has multi-scale receptive fields but remains easy to train and ablate.
- Current-data minimal experiment: train on `data/splits/crtk_sample_3class` for 3 epochs and compare against the existing small/medium/deep simple CNN and residual CNN configs.
- Smallest central ablation: replace all parallel dilated branches with a single ordinary 3x3 branch at matched parameter count.
- Expected information gain if it fails: if it does not improve over existing CNNs, the current baseline suite is probably sufficient as a conventional CNN reference.

## 3. Problem Restatement And Data Contract

Task: binary chess puzzle-likeness classification from current board tensors:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` stay evaluation diagnostics only. The model must return logits shaped `(batch, 2)` and use the shared trainer/reporting pipeline.

Allowed neural inputs:

- Current-board `simple_18` tensor.
- Side-to-move, castling, and en-passant planes already included.
- Optional deterministic coordinate planes generated inside the model from board indices.

Forbidden neural inputs:

- Stockfish scores, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Legal move generation, search, or checkmate/stalemate oracles.

Tensor contract:

```text
input:      (B, 18, 8, 8)
features:   (B, C, 8, 8)
global:     (B, G)
logits:     (B, 2)
```

## 4. Architecture Specification

Module names:

- `BoardCoordinatePlanes`
- `MultiScaleDilatedMixerBlock`
- `GlobalContextGate`
- `MultiScaleBoardMixerCNN`

Forward pass:

1. Input `(B, 18, 8, 8)`.
2. Append four fixed coordinate planes:
   - normalized rank,
   - normalized file,
   - center-distance,
   - side-to-move-relative forward direction.
3. Stem convolution:

```text
(B, 22, 8, 8) -> (B, width, 8, 8)
```

4. Run `num_blocks` mixer blocks. Each block has parallel branches:

```text
3x3 dilation 1
3x3 dilation 2
3x3 dilation 3
1x1 channel branch
```

5. Concatenate branches and mix with a `1x1` projection back to `width`.
6. Add a residual connection and normalization.
7. Compute a global context vector using mean/max pooling plus a small MLP.
8. Use the global vector to gate channels with sigmoid scale.
9. Final pooled head returns binary logits.

Default shape sketch:

```text
input:        (B, 18, 8, 8)
with coords:  (B, 22, 8, 8)
stem:         (B, 64, 8, 8)
block output: (B, 64, 8, 8)
pooled:       (B, 128)
logits:       (B, 2)
```

Default config:

```yaml
model:
  name: multiscale_board_mixer_cnn
  input_channels: 18
  num_classes: 2
  width: 64
  num_blocks: 4
  branch_width: 32
  dropout: 0.1
  use_batchnorm: true
  use_coordinate_planes: true
  use_global_context_gate: true
  ablation: none
```

Parameter estimate:

- Around 250k to 600k depending on width and branch width.
- This should be in the practical CNN benchmark range, not a huge model.

Complexity:

- Small: all feature maps are `8x8`.
- Dilation does not increase activation size.

## 5. Why This Is Not The Same As Existing CNNs

The current simple CNN mostly tests stacked local convolution. The residual CNN tests skip-connected local convolution. This model tests whether a compact conventional CNN improves when each block sees several chess-relevant spatial ranges at once:

- dilation 1: adjacent local patterns,
- dilation 2: knight-like and short diagonal spacing,
- dilation 3: longer board relations,
- global context gate: whole-board material/phase/context modulation.

This is not a claim of deep novelty. It is a better conventional comparator for the more unusual research packets.

## 6. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `single_dilation_matched` | Replace parallel branches with a single 3x3 stack at matched parameters | Multi-scale dilation matters | If it matches, parallel dilation is unnecessary. |
| `no_dilation_3` | Remove dilation-3 branch | Longer board range matters | If it matches, far branch is not useful. |
| `no_coordinate_planes` | Remove fixed coordinate planes | Explicit board coordinates help | If it matches, coordinates are unnecessary. |
| `no_global_context_gate` | Remove channel gate from global pooling | Global context modulation helps | If it matches, gate is not useful. |
| `small_width_control` | Match parameter count to small CNN | Gains are not only parameter count | If small-width still helps, architecture matters. |
| `residual_cnn_matched_params` | Compare against residual CNN with similar parameter count | Tests against ordinary residual depth | If residual CNN matches, this is just conventional capacity. |

## 7. Benchmark And Falsification Criteria

Compare against:

- `configs/bench_cnn_small_simple18.yaml`
- `configs/bench_cnn_medium_simple18.yaml`
- `configs/bench_cnn_deep_simple18.yaml`
- `configs/bench_residual_small_simple18.yaml`
- `configs/bench_residual_medium_simple18.yaml`

Metrics:

- AUROC.
- Accuracy and balanced accuracy.
- F1.
- Calibration.
- Fine-label `0/1/2 -> predicted 0/1` confusion matrix.
- Class `1` recall or precision at matched fine-label-`0` false-positive rate if tooling supports it.

Success threshold:

- Beats the best same-budget simple/residual CNN by at least `+0.5` AUROC point, or improves class-`1` recall at matched fine-label-`0` FPR by at least `+1.0` point.
- Beats `single_dilation_matched`.

Failure threshold:

- Does not beat medium/deep simple CNN or matched residual CNN.
- `single_dilation_matched` and `no_global_context_gate` match the full model.

Use as a regular baseline if:

- It is not more novel than the exotic packets, but it is consistently stronger than the current simple CNN suite.

Abandon if:

- It is not better than existing residual CNN configs or only wins by having more parameters.

## 8. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260424_multiscale_cnn_mixer/idea.yaml` | Create | Idea metadata. |
| `ideas/20260424_multiscale_cnn_mixer/architecture.md` | Create | Architecture details from this packet. |
| `ideas/20260424_multiscale_cnn_mixer/ablations.md` | Create | Ablation table. |
| `src/chess_nn_playground/models/multiscale_cnn_mixer.py` | Create | Model, mixer block, builder function. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `multiscale_board_mixer_cnn`. |
| `configs/bench_multiscale_cnn_mixer_simple18.yaml` | Create | Main config. |
| `configs/bench_multiscale_cnn_mixer_single_dilation.yaml` | Create | Central ablation config. |
| `tests/test_multiscale_cnn_mixer_forward.py` | Create | Forward shape, finite logits, ablation smoke tests. |

## 9. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-24_2107_friday_shanghai_multiscale_cnn_mixer.md
  generated_at: 2026-04-24 21:07
  weekday: Friday
  timezone: Asia/Shanghai
  idea_slug: multiscale_cnn_mixer
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260424_multiscale_cnn_mixer
  name: Multi-Scale Dilated Board Mixer CNN
  slug: multiscale_cnn_mixer
  status: draft
  created_at: 2026-04-24
  author: Codex
  short_thesis: A compact conventional CNN with parallel dilated branches and global context gating may be a stronger regular baseline for chess puzzle-likeness classification.
  novelty_claim: Practical architecture baseline, not a high-novelty research mechanism.
  expected_advantage: Better multi-scale context than the current simple CNN while staying easy to train and ablate.
  central_falsification_ablation: single_dilation_matched
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary logits
  compute_notes: Small 8x8 feature maps; width 64 and four blocks should remain practical.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/bench_multiscale_cnn_mixer_simple18.yaml
  model_path: src/chess_nn_playground/models/multiscale_cnn_mixer.py
  latest_result_path: null
  notes: Treat as a regular benchmark candidate, not an exotic idea.
```

```yaml
config_yaml:
  run:
    name: bench_multiscale_cnn_mixer_simple18
    output_dir: results
  seed: 42
  deterministic: true
  mode: coarse_binary
  device: nvidia
  data:
    train_path: data/splits/crtk_sample_3class/split_train.parquet
    val_path: data/splits/crtk_sample_3class/split_val.parquet
    test_path: data/splits/crtk_sample_3class/split_test.parquet
    encoding: simple_18
    cache_features: false
  model:
    name: multiscale_board_mixer_cnn
    input_channels: 18
    num_classes: 2
    width: 64
    num_blocks: 4
    branch_width: 32
    dropout: 0.1
    use_batchnorm: true
    use_coordinate_planes: true
    use_global_context_gate: true
    ablation: none
  training:
    epochs: 3
    batch_size: 512
    num_workers: 0
    learning_rate: 0.001
    weight_decay: 0.0001
    class_weighting: balanced
    early_stopping_patience: 2
    mixed_precision: false
```

```yaml
model_spec:
  model_name: multiscale_board_mixer_cnn
  file_path: src/chess_nn_playground/models/multiscale_cnn_mixer.py
  builder_function: build_multiscale_board_mixer_cnn_from_config
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - BoardCoordinatePlanes
    - MultiScaleDilatedMixerBlock
    - GlobalContextGate
    - MultiScaleBoardMixerCNN
  required_config_fields:
    - input_channels
    - num_classes
    - width
    - num_blocks
    - branch_width
    - use_coordinate_planes
    - use_global_context_gate
    - ablation
  expected_parameter_count: 250000-600000
  expected_memory_notes: Activations are small because all feature maps remain 8x8.
```

## 10. Prompt Maintenance Notes

This does not need to be added to the anti-duplicate research prompt unless it is treated as a research result. If implemented, record it as a stronger conventional baseline so future exotic ideas are compared against it.

Suggested prompt update:

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Multi-Scale Dilated Board Mixer CNN` to the baseline list if it outperforms current CNNs. | Future research packets should compare against the strongest regular CNN, not only the original simple CNN. | `Current baselines already exist` |
| Mark parallel dilated CNN blocks as regular engineering, not high novelty. | Prevents future prompts from treating dilation branches as a new research family. | `Common Approaches Rejected` |

## 11. Final Sanity Check

- Stored as a Markdown file in `ideas/research/packets/classic/`: yes
- Regular practical architecture idea: yes
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Minimal current-data experiment exists: yes
- Uses shared trainer contract: yes
- Includes central ablations: yes
- Gives implementation hooks for future Codex work: yes
