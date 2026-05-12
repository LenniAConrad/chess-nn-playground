# Codex Research Batch: Additional Practical Architecture Ideas 9

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`
- Generated at: 2026-04-24 22:10
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: practical architecture batch, not implemented

## Purpose

This batch adds more plain-to-moderately-novel neural architecture ideas after the practical baseline batch. The intent is to keep the ideas implementable while still testing different model shapes.

Avoid direct repeats of the immediately previous batch:

- no ConvNeXt clone
- no FPN clone
- no piece-plane gated CNN clone
- no MLP-Mixer patch clone
- no specialist-head clone
- no shallow-wide residual clone

The new focus:

- axial rank/file convolution
- cheap-to-expensive cascades
- auxiliary current-board reconstruction
- iterative logit refinement
- uncertainty and agreement heads
- lightweight adapter layers

These are not benchmark results.

## Shared Data Contract

Task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- train on binary labels only
- keep fine labels `0`, `1`, and `2` for diagnostics
- always report the fine-label `3 x 2` diagnostic matrix

First implementation:

- input: `simple_18`
- dataset: existing `crtk_sample_3class` splits
- trainer: shared experimental training pipeline

Forbidden model inputs:

- Stockfish scores, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, unresolved candidate status, dataset provenance, or anything derived from them.
- Engine search, legal move search, forced-line search, mate/stalemate oracles, tablebase outcomes, or future game outcomes.

Allowed model inputs:

- Current-board tensor.
- Side-to-move, castling, and en-passant planes already included in `simple_18`.
- Deterministic coordinate planes.
- Safe material/count summaries where explicitly listed.

## Ranked Shortlist

| Rank | Candidate | Main object | Why it is useful |
|---|---|---|---|
| 1 | Axial Rank-File ConvNet | Alternating rank/file 1D convolutions plus local convs | Plain long-range mixing without attention or line solvers. |
| 2 | Early-Exit Cascade BoardNet | Cheap trunk with optional deeper refinement exits | Tests accuracy/latency tradeoffs and hard-position routing. |
| 3 | Auxiliary Reconstruction BoardNet | Classifier trunk regularized by reconstructing safe current-board planes | Tests whether preserving board detail helps classification. |
| 4 | Iterative Logit Refinement CNN | Repeated small correction heads over a shared feature map | Tests whether staged evidence accumulation helps. |
| 5 | Agreement-Variance Head Net | Multiple lightweight heads with mean/variance diagnostics | Tests uncertainty and near-puzzle ambiguity without full ensembles. |
| 6 | Adapter-Sandwich Residual CNN | Frozen-ish simple trunk style with small bottleneck adapters inserted | Tests parameter-efficient improvements over existing CNNs. |

Best next implementation from this batch:

```text
Axial Rank-File ConvNet
```

Reason: it is simple, fast, chess-shaped, and gives long-range rank/file communication without becoming a Schur solve, ray automaton, or attention model.

## Candidate 1: Axial Rank-File ConvNet

### Thesis

Use ordinary convolutions, but factor long-range board mixing into alternating `8`-length rank and file convolutions. This gives every square access to same-rank and same-file context cheaply while preserving an ordinary CNN training path.

### Fingerprint

```text
simple_18
+ local 3x3 conv stem
+ rank-wise 1D conv
+ file-wise 1D conv
+ local residual conv
+ pooled head
```

### Why It Is Distinct

- Not Schur-Ray: no line incidence solve.
- Not ray-language automaton: no ray token strings.
- Not attention: no query-key routing.
- Not Board FPN: no multiresolution pyramid.
- Not TinyChessMicroNet: this is a midweight trainable conv model, not a hard tiny sketch model.

### Architecture Sketch

Input:

```text
x: (B, 18, 8, 8)
```

Stem:

```text
h = Conv3x3(18 + coords -> W)
```

Axial block:

```text
local = Conv3x3(h)
rank = Conv1d_along_files(h)   # each rank processed as length-8 sequence
file = Conv1d_along_ranks(h)   # each file processed as length-8 sequence
h = h + project(concat(local, rank, file))
```

Recommended block details:

```text
rank/file kernel_size = 5 or 7
depthwise separable 1D convs
pointwise 1x1 fusion
BatchNorm/ReLU
```

Head:

```text
mean_pool
max_pool
row_pool_summary
file_pool_summary
MLP
logits
```

Default config:

```yaml
model:
  name: axial_rank_file_convnet
  input_channels: 18
  width: 64
  depth: 5
  axial_kernel: 7
  branch_width: 32
  dropout: 0.1
  use_coordinate_planes: true
```

Expected parameter range:

```text
250k-700k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `local_only_matched` | Replace axial branches with local convs at matched params | Axial long-range mixing matters | If equal, use local CNN. |
| `rank_only` | Remove file branch | File context matters | If equal, rank branch dominates or axial is weak. |
| `file_only` | Remove rank branch | Rank context matters | If equal, file branch dominates or axial is weak. |
| `kernel3_axial` | Use 1D kernel size 3 | Longer line context matters | If equal, shorter kernels are enough. |
| `shuffled_axis_control` | Randomly permute files/ranks before axial conv with inverse unshuffle disabled | Real board axis order matters | If equal, axial semantics are weak. |

### Diagnostics

- Axial branch activation norms.
- Rank-only and file-only validation deltas.
- Class-1 recall changes on positions with heavy rooks/queens if such metadata is derivable from current board counts.
- Latency versus regular CNN.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/axial_rank_file_convnet.py
tests/test_axial_rank_file_convnet.py
configs/bench_axial_rank_file_convnet_simple18.yaml
configs/bench_axial_rank_file_convnet_local_only.yaml
```

Implementation can reshape safely:

```python
# rank branch: B, C, 8, 8 -> B*8, C, 8
# file branch: B, C, 8, 8 -> B*8, C, 8 after transpose
```

Keep it plain: no attention, no dynamic line extraction, no move logic.

## Candidate 2: Early-Exit Cascade BoardNet

### Thesis

Some positions may be easy and should not need a heavy model, while ambiguous near-puzzles need deeper computation. Build a cascade with several classifier exits and train it to produce useful early predictions plus a final refined prediction.

### Fingerprint

```text
shared CNN stages
+ early classifier exits
+ confidence/entropy gate diagnostics
+ final classifier
+ optional inference-time early exit
```

### Architecture Sketch

Stages:

```text
h1 = stage1(x)
logits1 = head1(h1)

h2 = stage2(h1)
logits2 = head2(h2)

h3 = stage3(h2)
logits3 = head3(h3)
```

Training loss:

```text
loss = w1 * CE(logits1, y) + w2 * CE(logits2, y) + w3 * CE(logits3, y)
```

Recommended:

```text
w1 = 0.3
w2 = 0.5
w3 = 1.0
```

Inference modes:

- full: always use `logits3`
- cascade: exit at `head1` or `head2` if entropy is below threshold

Default config:

```yaml
model:
  name: early_exit_cascade_boardnet
  widths: [32, 64, 96]
  blocks_per_stage: [2, 2, 2]
  dropout: 0.1
  exit_weights: [0.3, 0.5, 1.0]
```

Expected parameter range:

```text
300k-900k
```

### Why It Is Useful

This is not a new chess theory. It tests whether the benchmark has a useful easy/hard structure:

```text
Can shallow exits handle obvious non-puzzles while deeper stages improve ambiguous cases?
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `final_loss_only` | Train only final head | Deep supervision matters | If equal, auxiliary exits are unnecessary. |
| `same_depth_single_head` | Same trunk, one final head | Exit heads improve features | If equal, use simpler model. |
| `random_exit_thresholds` | Use random/confidence-agnostic early exit | Entropy confidence is meaningful | If equal, confidence is not useful. |
| `head1_only` | Use first exit only | Shallow model lower bound | If close, deeper stages are unnecessary. |
| `head_disagreement_features` | Add disagreement to final head | Tests whether exit disagreement helps | If no gain, keep exits diagnostic only. |

### Diagnostics

- Accuracy/AUROC per exit.
- Fraction of samples exiting at each threshold.
- Latency versus AUROC curve.
- Fine-label distribution by exit depth.
- Entropy calibration of each exit.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/early_exit_cascade_boardnet.py
tests/test_early_exit_cascade_boardnet.py
configs/bench_early_exit_cascade_boardnet_simple18.yaml
configs/bench_early_exit_cascade_boardnet_final_only.yaml
```

If the shared trainer expects only one logits tensor, return final logits by default and expose auxiliary logits through an optional flag or model method. Do not break trainer contracts.

## Candidate 3: Auxiliary Reconstruction BoardNet

### Thesis

A classifier trunk may discard board detail too early. Add a lightweight decoder that reconstructs safe current-board planes from the latent feature map, using reconstruction only as an auxiliary training loss. The classifier still sees no future or engine information.

### Fingerprint

```text
CNN encoder
+ binary classifier head
+ current-board reconstruction decoder
+ CE + small reconstruction loss
```

### Why It Is Distinct

- Not masked codec surprise: no label-free masked pretraining or code-length readout.
- Not autoencoder anomaly scoring: reconstruction error is not the classifier feature.
- Not multi-task with engine targets: only reconstructs the allowed input board.

### Architecture Sketch

Encoder:

```text
h = CNNEncoder(x)
```

Classifier:

```text
logits = ClassifierHead(pool(h))
```

Decoder:

```text
recon = Decoder(h)  # reconstruct selected input planes
```

Loss:

```text
loss = CE(logits, y) + lambda_recon * BCEWithLogits(recon, x_reconstruct)
```

Recommended reconstruct targets:

- piece occupancy planes
- side-to-move plane
- castling/en-passant planes only if stable

Default config:

```yaml
model:
  name: auxiliary_reconstruction_boardnet
  encoder_width: 64
  encoder_depth: 4
  decoder_width: 32
  lambda_recon: 0.05
  dropout: 0.1
```

Expected parameter range:

```text
350k-900k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `classifier_only` | Remove decoder and recon loss | Reconstruction regularization helps | If equal, auxiliary loss unnecessary. |
| `decoder_no_loss` | Keep decoder params but no recon loss | Gain is not just capacity | If equal, decoder capacity caused gain. |
| `high_lambda_recon` | Increase reconstruction weight | Tests over-regularization | If worse, recon should stay small. |
| `piece_planes_only` | Reconstruct only piece planes | State planes may not matter | If equal, simpler target. |
| `recon_error_head` | Feed reconstruction error to classifier | Tests anomaly-style shortcut | If no gain, keep recon auxiliary only. |

### Diagnostics

- Reconstruction BCE by fine label.
- Classification metrics with and without auxiliary loss.
- Whether reconstruction loss improves calibration.
- Examples where classifier improves but recon loss is not lower.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/auxiliary_reconstruction_boardnet.py
tests/test_auxiliary_reconstruction_boardnet.py
configs/bench_auxiliary_reconstruction_boardnet_simple18.yaml
configs/bench_auxiliary_reconstruction_boardnet_classifier_only.yaml
```

Implement carefully so the default `forward(x)` still returns logits. Auxiliary outputs can be returned with:

```python
forward(x, return_aux=True)
```

## Candidate 4: Iterative Logit Refinement CNN

### Thesis

Instead of producing a single logit vector at the end, let a model make an initial prediction and then apply several learned correction steps from shared board features. The model tests whether puzzle evidence is better accumulated as staged corrections.

### Fingerprint

```text
CNN feature map
+ initial logits
+ repeated correction heads
+ correction trajectory diagnostics
+ final logits
```

### Architecture Sketch

Feature trunk:

```text
h = CNN(x)
z = pool(h)
```

Initial logits:

```text
l0 = Head0(z)
```

Correction steps:

```text
for t in 1..T:
    c_t = CorrectionMLP([z, l_{t-1}, confidence_features(l_{t-1})])
    l_t = l_{t-1} + c_t
```

Return:

```text
l_T
```

Train either final-only:

```text
CE(l_T, y)
```

or with light deep supervision:

```text
sum_t w_t CE(l_t, y)
```

Default config:

```yaml
model:
  name: iterative_logit_refinement_cnn
  trunk_width: 64
  trunk_depth: 4
  refinement_steps: 4
  correction_hidden: 64
  dropout: 0.1
```

Expected parameter range:

```text
250k-700k
```

### Why It Is Distinct

- Not residual CNN: corrections happen in logit/evidence space, not feature maps.
- Not fixed-point residual defect: no convergence operator over latent states.
- Not cascade: every sample runs all correction steps unless ablated.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `single_head_matched` | One MLP head with matched params | Refinement trajectory matters | If equal, corrections unnecessary. |
| `no_logit_feedback` | Correction heads see only `z` | Feedback matters | If equal, staged heads are just ensemble capacity. |
| `one_step_only` | Only one correction | Multiple refinements matter | If equal, use simpler head. |
| `untied_corrections` | Separate correction heads per step | Weight sharing matters | If untied wins, shared refinement too constrained. |
| `trajectory_features_head` | Final head sees all intermediate logits | Tests trajectory diagnostics | If helpful, expose correction path. |

### Diagnostics

- Average correction norm per step.
- Fraction of samples whose predicted class flips after step 1.
- Fine-label distribution of flips.
- Calibration by step.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/iterative_logit_refinement_cnn.py
tests/test_iterative_logit_refinement_cnn.py
configs/bench_iterative_logit_refinement_cnn_simple18.yaml
configs/bench_iterative_logit_refinement_cnn_single_head.yaml
```

Keep correction magnitudes stable:

```text
c_t = 0.25 * tanh(raw_c_t)
```

for the first implementation.

## Candidate 5: Agreement-Variance Head Net

### Thesis

Use one shared trunk and several cheap heads trained on the same label. Classify from the mean logits, and log head variance as an uncertainty diagnostic. This is a lightweight alternative to full ensembles.

### Fingerprint

```text
shared CNN trunk
+ K independent lightweight heads
+ mean logits
+ variance/entropy diagnostics
+ optional variance-aware calibration
```

### Why It Is Distinct

- Not channel-dropout consensus: no deterministic channel masks are the central object.
- Not sparse expert routing: all heads run and share the same trunk.
- Not evidential Dirichlet: uncertainty comes from head disagreement, not evidence parameters.

### Architecture Sketch

Trunk:

```text
z = pool(CNN(x))
```

Heads:

```text
logits_k = Head_k(z), k = 1..K
```

Mean prediction:

```text
logits = mean_k logits_k
```

Agreement features:

```text
var_logits = variance_k logits_k
entropy_mean = entropy(softmax(logits))
mean_head_entropy = mean_k entropy(softmax(logits_k))
```

Optional calibration head:

```text
temperature = softplus(MLP([z, var_logits, entropy_mean])) + 1e-3
calibrated_logits = logits / temperature
```

Default config:

```yaml
model:
  name: agreement_variance_head_net
  trunk_width: 64
  trunk_depth: 4
  num_heads: 5
  head_hidden: 32
  use_temperature_head: false
```

Expected parameter range:

```text
250k-800k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `single_head_matched` | One larger head with matched params | Head diversity helps | If equal, multi-head unnecessary. |
| `shared_head_weights` | Tie all head weights | Disagreement needs independent heads | Should collapse variance. |
| `variance_to_classifier` | Feed variance into final classifier | Disagreement improves prediction | If no gain, keep diagnostic only. |
| `temperature_head` | Add sample-wise temperature | Agreement helps calibration | If no calibration gain, omit. |
| `head_dropout_only` | Use ordinary dropout in one head | Multi-head beats dropout | If equal, use simpler dropout. |

### Diagnostics

- Head variance by fine label.
- ECE with and without temperature head.
- Cases where heads disagree strongly.
- Whether disagreement is higher on fine-label `1` than `0`.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/agreement_variance_head_net.py
tests/test_agreement_variance_head_net.py
configs/bench_agreement_variance_head_net_simple18.yaml
configs/bench_agreement_variance_head_net_single_head.yaml
```

Default forward should return mean logits. Auxiliary diagnostics can be returned with `return_aux=True`.

## Candidate 6: Adapter-Sandwich Residual CNN

### Thesis

Instead of building a much larger new backbone, insert small bottleneck adapters before and after ordinary residual blocks. This tests whether parameter-efficient adapters can improve the existing CNN family while leaving most of the architecture conventional.

### Fingerprint

```text
ordinary residual CNN
+ bottleneck adapters around blocks
+ adapter scaling
+ pooled head
```

### Architecture Sketch

Base block:

```text
h = ResidualBlock(h)
```

Adapter sandwich:

```text
h = h + alpha_pre * AdapterPre(h)
h = ResidualBlock(h)
h = h + alpha_post * AdapterPost(h)
```

Adapter:

```text
Conv1x1(W -> r)
ReLU
Conv1x1(r -> W)
```

Recommended:

```text
r = W / 4
alpha initialized to 0.1
```

Default config:

```yaml
model:
  name: adapter_sandwich_residual_cnn
  width: 64
  depth: 5
  adapter_rank: 16
  dropout: 0.1
  use_coordinate_planes: true
```

Expected parameter range:

```text
250k-800k
```

### Why It Is Useful

This is a practical parameter-efficiency baseline. If adapters improve a normal residual CNN, future ideas should compare against adapter-enhanced baselines, not only the original residual tower.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `no_adapters` | Ordinary residual CNN | Adapters add value | If equal, skip adapters. |
| `post_only` | Only post-block adapters | Placement matters | If equal, simplify. |
| `pre_only` | Only pre-block adapters | Placement matters | If equal, simplify. |
| `full_width_extra_conv` | Replace adapters with same-param convs | Bottleneck adapter shape matters | If equal, adapters are just capacity. |
| `frozen_base_adapters_only` | Train adapters/head while base is frozen after warmup | Parameter-efficient adaptation works | If poor, full training needed. |

### Diagnostics

- Adapter output norm by layer.
- Whether adapters concentrate near early or late blocks.
- Parameter count versus residual baseline.
- Calibration changes from adapters.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/adapter_sandwich_residual_cnn.py
tests/test_adapter_sandwich_residual_cnn.py
configs/bench_adapter_sandwich_residual_cnn_simple18.yaml
configs/bench_adapter_sandwich_residual_cnn_no_adapters.yaml
```

Keep it compatible with existing residual CNN code if possible. This should be an incremental implementation, not a separate framework.

## Implementation Queue

Recommended order:

1. `Axial Rank-File ConvNet`
2. `Iterative Logit Refinement CNN`
3. `Agreement-Variance Head Net`
4. `Early-Exit Cascade BoardNet`
5. `Adapter-Sandwich Residual CNN`
6. `Auxiliary Reconstruction BoardNet`

Reasoning:

- Axial convs are simple and chess-shaped.
- Logit refinement is easy to add on top of a trunk.
- Agreement heads add diagnostics with little model risk.
- Early exits need trainer/reporting care but can be useful.
- Adapters are practical if residual code is clean.
- Reconstruction needs auxiliary-loss plumbing.

## Shared Benchmark Rules

For every candidate:

- same train/val/test splits
- same coarse binary labels
- same fine-label diagnostics
- compare against simple CNN and residual CNN baselines
- compare against `ConvNeXt BoardNet` if implemented
- report parameter count and inference latency

Important metrics:

- AUROC
- balanced accuracy
- F1
- calibration
- class-1 recall at matched fine-label-0 false-positive rate if available
- fine-label `3 x 2` diagnostic matrix

## Anti-Duplicate Rules

Do not repeat these later with only width/depth changes:

| Family | Avoid Near-Duplicate |
|---|---|
| Axial Rank-File ConvNet | Another rank/file axial conv model unless it changes the axis operator or falsifier. |
| Early-Exit Cascade | Another cascade unless exit policy or supervision changes materially. |
| Auxiliary Reconstruction | Another reconstruction auxiliary model unless target or use of reconstruction changes. |
| Iterative Logit Refinement | Another staged-logit correction model with only step count changes. |
| Agreement-Variance Heads | Another multi-head disagreement model unless uncertainty is used differently. |
| Adapter-Sandwich Residual | Another adapter residual baseline with only rank/depth changes. |

## Continuity Note

These ideas are useful because they fill gaps between plain CNN baselines and exotic research architectures. If one works well, it becomes a stronger baseline. If all fail, that helps justify spending implementation time on the more distinctive packets.
