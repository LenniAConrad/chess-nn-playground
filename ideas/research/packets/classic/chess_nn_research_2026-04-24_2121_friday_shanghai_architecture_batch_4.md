# Codex Research Batch: Additional Architecture Candidates 4

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`
- Generated at: 2026-04-24 21:21
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: draft architecture batch, not implemented

## Purpose

This batch adds more implementable architecture candidates. The emphasis is on ideas that could be benchmarked quickly and that fill gaps between pure CNN baselines and the more exotic packets.

These ideas are not implementation commits. They are candidates to promote into full packets or code later.

## Shared Data Contract

All candidates target the current binary puzzle-likeness task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` remain diagnostics only. First implementations should use `simple_18`, the existing `crtk_sample_3class` splits, and the shared trainer.

Forbidden model inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, or future game outcomes.

Allowed model inputs:

- Current board occupancy, side-to-move, castling/en-passant planes, deterministic square coordinates, side-relative coordinates, material/count summaries, and deterministic transforms of current board tensors.

## Ranked Shortlist

| Rank | Candidate | Main object | Why expand it |
|---|---|---|---|
| 1 | Row-File Factor Mixer | Factorized rank/file/piece interactions | Strong practical architecture that respects board axes without full attention. |
| 2 | Piece-Conditioned Hypernetwork CNN | Material/piece inventory conditions lightweight CNN kernels/gates | Tests whether board context should adapt the local feature extractor. |
| 3 | Neural Board Cellular Automaton | Shared local update dynamics and convergence diagnostics | Interesting iterative regular model, distinct from residual-defect latent MLP. |
| 4 | Symmetric Difference Twin Encoder | Safe transform pairs and feature-difference diagnostics | Practical contrastive-style model without enforcing invariance. |
| 5 | Prototype Patch Dictionary Network | Learned local patch dictionary and residual codes | Strong regular baseline for motif-like recognition. |
| 6 | Channel Dropout Consensus Network | Multiple piece-channel views and consensus/disagreement | Cheap robustness model for source-artifact resistance. |

Best next full packet from this batch: `Row-File Factor Mixer`.

## Candidate 1: Row-File Factor Mixer

### Thesis

Chess boards have two privileged axes: ranks and files. A model can exploit this without a full Transformer by factorizing board processing into rank mixers, file mixers, and piece-channel mixers, then recombining them with bilinear interactions.

### Fingerprint

```text
simple_18 board tensor
+ rank mixer
+ file mixer
+ piece/channel mixer
+ bilinear row-file fusion
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not a CNN: no sliding local filter is the central operator.
- Not a vanilla MLP-Mixer: rank and file axes are treated separately with chess-specific side-relative coordinates.
- Not attention: no learned query-token softmax.

### Architecture Sketch

1. Input `(B, 18, 8, 8)`.
2. Add coordinate planes and side-relative rank/file planes.
3. Project channels to width `D=64`.
4. Rank mixer:

```text
rank_summary = mean over files -> (B, D, 8)
rank_mixed = MLP over rank axis
```

5. File mixer:

```text
file_summary = mean over ranks -> (B, D, 8)
file_mixed = MLP over file axis
```

6. Reconstruct board-conditioned factors:

```text
rank_factor: (B, D, 8, 1)
file_factor: (B, D, 1, 8)
interaction = rank_factor * file_factor
```

7. Fuse interaction with original projected board and run a small head.

### Tensor Contract

```text
input:          (B, 18, 8, 8)
projected:      (B, D, 8, 8)
rank_factor:    (B, D, 8, 1)
file_factor:    (B, D, 1, 8)
interaction:    (B, D, 8, 8)
pooled:         (B, H)
logits:         (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `channel_mlp_only` | Remove rank/file factor mixers | Axis factorization matters | If it matches, factor mixer is unnecessary. |
| `rank_only` | Remove file mixer | File structure matters | If it matches, file branch unnecessary. |
| `file_only` | Remove rank mixer | Rank structure matters | If it matches, rank branch unnecessary. |
| `random_axis_permutation` | Randomly permute ranks/files with fixed permutation | Board axis semantics matter | If it matches, axis geometry is not used. |
| `cnn_matched_params` | Matched CNN baseline | Factor mixer beats ordinary CNN capacity | If it matches, use CNN. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/row_file_factor_mixer.py`
- Registry name: `row_file_factor_mixer`
- Main config: `configs/bench_row_file_factor_mixer_simple18.yaml`
- Central ablation config: `configs/bench_row_file_factor_mixer_channel_only.yaml`
- Tests: forward shape, finite logits, ablation modes.

### Decision Rule

Promote if it beats channel-only and matched-CNN controls. Drop if random axis permutation does not hurt.

## Candidate 2: Piece-Conditioned Hypernetwork CNN

### Thesis

The best local filters may depend on material and piece inventory. A lightweight hypernetwork can condition CNN channel gates or depthwise kernels on safe current-board summaries, adapting the feature extractor without using engine metadata.

### Fingerprint

```text
current-board material/context summary
+ hypernetwork-generated CNN gates or depthwise scales
+ compact board CNN
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not just a CNN: filters/gates are conditioned by current board inventory.
- Not nuisance projection: material/context features condition the model instead of being projected out.
- Not ensembling: one shared network with generated gates, not multiple independent models.

### Architecture Sketch

1. Extract safe summary:
   - piece counts by type/color,
   - side-to-move,
   - castling/en-passant scalars,
   - occupancy count,
   - simple phase/material totals.
2. Hypernetwork maps summary to per-block channel gates:

```text
gate_b = sigmoid(MLP(summary)) -> (B, width)
```

3. CNN block:

```text
h = conv(h)
h = h * gate_b[:, :, None, None]
h = norm/relu/dropout
```

4. Pool and classify.

### Tensor Contract

```text
input:     (B, 18, 8, 8)
summary:   (B, S)
gates:     (B, num_blocks, width)
features:  (B, width, 8, 8)
logits:    (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `static_gates` | Replace generated gates with learned constants | Conditioning matters | If it matches, hypernetwork unnecessary. |
| `random_summary` | Shuffle summaries across batch | Sample-specific conditioning matters | If it matches, gates are generic. |
| `summary_only_head` | Classify from summary only | Detects material shortcut | If it matches, model is not using board texture. |
| `cnn_matched_params` | Ordinary CNN with matched params | Hypernetwork beats capacity | If it matches, use simpler CNN. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/piece_conditioned_hyper_cnn.py`
- Registry name: `piece_conditioned_hyper_cnn`
- Main config: `configs/bench_piece_conditioned_hyper_cnn_simple18.yaml`
- Central ablation config: `configs/bench_piece_conditioned_hyper_cnn_static.yaml`
- Tests: forward shape, finite logits, gate shape/range.

### Decision Rule

Promote if generated gates beat static gates and shuffled-summary controls. Drop if summary-only is competitive.

## Candidate 3: Neural Board Cellular Automaton

### Thesis

Some board patterns may be recognized by repeated local relaxation. A neural cellular automaton applies the same local update rule for several steps and classifies from the evolving board state and update energy.

### Fingerprint

```text
simple_18 board tensor
+ shared local update cell
+ multi-step board-state trajectory
+ update-energy diagnostics
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not residual CNN depth: the update rule is shared across time.
- Not fixed-point residual defect network: that model updates a global latent; this updates the spatial board state.
- Not graph/sheaf: no attack graph or relation complex.

### Architecture Sketch

1. Project input to hidden board state `(B, H, 8, 8)`.
2. For `T=6` steps:

```text
delta_t = local_update(h_t)
h_{t+1} = h_t + alpha * delta_t
```

3. Record update energy:
   - mean `||delta_t||`,
   - max `||delta_t||`,
   - contraction ratio,
   - spatial entropy of update magnitude.
4. Classify from final state plus trajectory diagnostics.

### Tensor Contract

```text
input:       (B, 18, 8, 8)
h_path:      (B, T + 1, H, 8, 8)
delta_path:  (B, T, H, 8, 8)
stats:       (B, T * S)
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `untied_steps` | Different conv block per step | Shared automaton dynamics matter | If it matches, ordinary depth is enough. |
| `final_state_only` | Remove trajectory stats | Update diagnostics matter | If it matches, trajectory unnecessary. |
| `single_step` | Use one update | Iteration matters | If it matches, no relaxation signal. |
| `random_update_rule` | Freeze random update rule, train head | Learned update matters | If it matches, trajectory stats are generic. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/neural_board_ca.py`
- Registry name: `neural_board_cellular_automaton`
- Main config: `configs/bench_neural_board_ca_simple18.yaml`
- Central ablation config: `configs/bench_neural_board_ca_final_only.yaml`
- Tests: forward shape, finite logits, path shape.

### Decision Rule

Promote if trajectory diagnostics beat final-state-only and untied-step controls. Drop if it behaves like an ordinary CNN.

## Candidate 4: Symmetric Difference Twin Encoder

### Thesis

Safe deterministic board transforms should preserve some evidence and change other evidence. Instead of enforcing invariance, compare the original and transformed board latents by symmetric difference features.

### Fingerprint

```text
original board and safe transformed board
+ shared encoder
+ latent sum/product/absolute-difference
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not orbit quotient: it does not pool away transform differences.
- Not orbit disagreement residual: simpler twin encoder with explicit difference/product features.
- Regular and easy to test.

### Architecture Sketch

1. Generate one or two safe transforms:
   - file mirror if castling/en-passant remapping is correct,
   - color/side canonical transform only if channel semantics are exact.
2. Shared CNN encodes original and transformed boards.
3. Build features:

```text
z_sum = z1 + z2
z_absdiff = abs(z1 - z2)
z_product = z1 * z2
```

4. Classify from concatenated features.

### Tensor Contract

```text
input:      (B, 18, 8, 8)
views:      (B, V, 18, 8, 8)
latents:    (B, V, D)
features:   (B, 3D)
logits:     (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `original_only` | Use only original board | Transform comparison matters | If it matches, twin unnecessary. |
| `sum_only` | Remove difference/product | Difference carries signal | If it matches, simple augmentation enough. |
| `random_transform` | Use semantics-destroying transform | Safe transform semantics matter | If it matches, transform is generic. |
| `augmentation_only` | Train with transforms but classify one view | Explicit twin features matter | If it matches, no twin needed. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/symmetric_difference_twin.py`
- Registry name: `symmetric_difference_twin`
- Main config: `configs/bench_symmetric_difference_twin_simple18.yaml`
- Central ablation config: `configs/bench_symmetric_difference_twin_original_only.yaml`

### Decision Rule

Promote if abs-difference/product features beat original-only and augmentation-only controls. Drop if random transform works as well as safe transform.

## Candidate 5: Prototype Patch Dictionary Network

### Thesis

Puzzle-like positions may contain local motifs, but a standard CNN may hide them in distributed filters. A learned patch dictionary can expose motif assignments, reconstruction residuals, and prototype activation histograms.

### Fingerprint

```text
3x3 or 5x5 board patches
+ learned patch prototypes
+ soft assignment and residual codes
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Regular motif model, not high math.
- More interpretable than a CNN filter stack.
- Strong local-pattern baseline for puzzle motifs.

### Architecture Sketch

1. Extract all 3x3 patches from `simple_18`, padded safely.
2. Project patches to `D=64`.
3. Compare to `K=64` learned prototypes.
4. Compute:
   - soft assignment weights,
   - nearest-prototype margin,
   - residual norm,
   - prototype activation histogram,
   - spatial activation maps.
5. MLP/CNN head classifies from histograms and residual maps.

### Tensor Contract

```text
input:        (B, 18, 8, 8)
patches:      (B, 64, patch_dim)
assignments:  (B, 64, K)
residuals:    (B, 64)
histogram:    (B, K)
logits:       (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `random_prototypes` | Freeze random dictionary | Learned motifs matter | If it matches, dictionary unnecessary. |
| `histogram_only` | Remove spatial prototype maps | Spatial arrangement matters | If it matches, motif counts suffice. |
| `cnn_matched_params` | Matched CNN | Dictionary beats CNN capacity | If it matches, use CNN. |
| `patch_shuffle` | Shuffle patch positions | Patch spatial arrangement matters | If it matches, location not used. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/prototype_patch_dictionary.py`
- Registry name: `prototype_patch_dictionary`
- Main config: `configs/bench_prototype_patch_dictionary_simple18.yaml`
- Central ablation config: `configs/bench_prototype_patch_dictionary_random.yaml`

### Decision Rule

Promote if learned prototypes beat random prototypes and matched CNN. Drop if histogram-only and patch-shuffle controls match full model.

## Candidate 6: Channel Dropout Consensus Network

### Thesis

The classifier should not depend too heavily on one piece channel or artifact. Train several shared encoders on deterministic channel-dropped views and classify from consensus and disagreement.

### Fingerprint

```text
piece-channel dropped board views
+ shared encoder
+ consensus latent and disagreement stats
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Regular robustness model.
- Not dropout only: disagreement features are part of the output.
- Useful source-artifact check.

### Architecture Sketch

1. Create deterministic views:
   - remove pawns,
   - remove minors,
   - remove majors,
   - remove own pieces,
   - remove opponent pieces,
   - full board.
2. Shared CNN encodes each view.
3. Classify from:
   - mean latent,
   - variance across views,
   - max pairwise distance,
   - full-view latent.

### Tensor Contract

```text
input:       (B, 18, 8, 8)
views:       (B, V, 18, 8, 8)
latents:     (B, V, D)
consensus:   (B, features)
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `full_view_only` | Use only full board | Consensus matters | If it matches, no need for views. |
| `mean_only` | Remove disagreement stats | Disagreement matters | If it matches, averaging enough. |
| `random_channel_masks` | Random masks matched by channel count | Semantic masks matter | If it matches, view semantics weak. |
| `train_dropout_only` | Ordinary channel dropout during training | Explicit consensus features matter | If it matches, simpler regularization enough. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/channel_dropout_consensus.py`
- Registry name: `channel_dropout_consensus`
- Main config: `configs/bench_channel_dropout_consensus_simple18.yaml`
- Central ablation config: `configs/bench_channel_dropout_consensus_full_only.yaml`

### Decision Rule

Promote if consensus/disagreement beats full-view-only and ordinary channel-dropout training. Drop if random masks match semantic masks.

## Recommended Expansion Order From This Batch

1. `Row-File Factor Mixer`
2. `Piece-Conditioned Hypernetwork CNN`
3. `Neural Board Cellular Automaton`
4. `Prototype Patch Dictionary Network`
5. `Symmetric Difference Twin Encoder`
6. `Channel Dropout Consensus Network`

## Shared Benchmark Setup

Each candidate should start from:

```yaml
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
  input_channels: 18
  num_classes: 2
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

Compare against:

- `configs/bench_cnn_small_simple18.yaml`
- `configs/bench_cnn_medium_simple18.yaml`
- `configs/bench_residual_small_simple18.yaml`
- `configs/bench_multiscale_cnn_mixer_simple18.yaml` if implemented
- `configs/bench_piece_token_cnn_hybrid_simple18.yaml` if implemented

Required diagnostics:

- AUROC, balanced accuracy, F1, calibration.
- Fine-label `0/1/2 -> predicted 0/1` confusion matrix.
- Central ablation delta.
- Class `1` recall or precision at matched fine-label `0` false-positive rate where available.

## Prompt Maintenance Notes

If any candidate is implemented and fails, add a focused anti-duplicate rule:

| Candidate | Anti-duplicate rule if it fails |
|---|---|
| Row-File Factor Mixer | Do not repeat rank/file factorized board mixers with only different hidden widths or mixer depths. |
| Piece-Conditioned Hypernetwork CNN | Do not repeat material-conditioned CNN gate hypernetworks with only different gate sizes. |
| Neural Board Cellular Automaton | Do not repeat shared spatial board-update CA models with only different step counts or widths. |
| Symmetric Difference Twin Encoder | Do not repeat safe-transform twin encoders with only different latent widths or one more transform. |
| Prototype Patch Dictionary | Do not repeat learned patch dictionary motif models with only different prototype counts. |
| Channel Dropout Consensus | Do not repeat deterministic channel-drop consensus models with only different channel mask groups. |

## Final Sanity Check

- Stored as a Markdown file in `ideas/research/packets/classic/`: yes
- Adds multiple fresh candidates: yes
- Avoids forbidden engine/search/source features: yes
- Does not fabricate labels: yes
- Keeps current-data minimal experiments possible: yes
- Includes central ablations and stop conditions: yes
- Gives implementation hooks for future Codex work: yes
