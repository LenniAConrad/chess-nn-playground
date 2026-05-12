# Codex Research Batch: Attention-Inspired Architecture Candidates

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`
- Generated at: 2026-04-24 20:56
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: draft research batch, not implemented

## Purpose

This file explores attention-inspired architectures without proposing a plain Transformer over 64 squares. The intended research object is the attention pattern itself:

- query selectivity,
- attention entropy and margins,
- disagreement between query families,
- cross-scale attention residuals,
- attention sensitivity under deterministic safe perturbations.

These candidates should be tested against mean-pooling, random-query, and ordinary small-CNN controls. If those controls match, the attention mechanism is not adding useful structure.

## Shared Data Contract

All candidates target the current binary puzzle-likeness task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` are diagnostics only. The first experiment for each candidate should use:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Allowed inputs:

- Current `simple_18` board tensor.
- Side-to-move, castling, en-passant planes already present.
- Deterministic square coordinates and safe side-relative coordinates.
- Current-board occupied-piece masks.

Forbidden inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, future game outcomes, or label-informed masks.

## Ranked Shortlist

| Rank | Candidate | Attention object | Why expand it |
|---|---|---|---|
| 1 | Set-Query Attention Bottleneck | Learned query-to-board attention distributions and margins | Small, implementable, attention-specific falsifiers are clean. |
| 2 | Attention Disagreement Residual Network | Disagreement among independent query families | Tests whether ambiguous near-puzzles induce unstable evidence routing. |
| 3 | Cross-Scale Attention Residual Network | Fine-token attention not explained by coarse tokens | Attention analogue of coarse-to-fine residuals. |
| 4 | Slot Attention Role Binding Network | Iterative soft assignment of pieces to latent tactical roles | Attention-like role binding without sparse hard witnesses. |
| 5 | Attention Perturbation Sensitivity Network | Change in logits/features under deterministic attention-guided masks | Tests whether attended regions are causally useful, not just decorative. |

The best first full packet is `Set-Query Attention Bottleneck`. It is attention-inspired, compact, and has direct controls: uniform attention, random queries, value-only pooling, and entropy-only readout.

## Candidate 1: Set-Query Attention Bottleneck

### Thesis

Puzzle-like positions may be recognized by a small number of latent tactical questions, each expressed as an attention distribution over board tokens. The model should classify not from unconstrained token mixing, but from query attention statistics, attended values, entropy, and best-second attention margins.

### Fingerprint

```text
current-board square or occupied-piece tokens
+ small learned query bank
+ query-to-token attention maps
+ attention entropy/margin/value summaries
+ binary puzzle-likeness head
```

### Why It Is Not A Vanilla Transformer

- No token-to-token self-attention stack.
- A fixed small query bank reads from board tokens once or twice.
- The bottleneck exports attention distributions and summary statistics.
- The central ablations replace attention with mean pooling, uniform attention, or random frozen queries.

### Architecture Sketch

1. Convert input `(B, 18, 8, 8)` to 64 square tokens:

```text
token_i = MLP([piece planes at square i, side-to-move, castling/en-passant scalars, coordinates])
```

2. Learn `Q=24` query vectors in `R^d`.
3. Compute multihead query attention:

```text
a_{q,i} = softmax_i(q_q^T k_i / sqrt(d))
v_q = sum_i a_{q,i} value_i
```

4. Compute attention diagnostics:
   - entropy per query,
   - max attention,
   - best-second margin,
   - occupied-mass versus empty-mass,
   - side-to-move piece mass versus opponent piece mass,
   - attended coordinate mean and variance.
5. Classify from attended values plus diagnostics.

### Tensor Contract

```text
input:        (B, 18, 8, 8)
tokens:       (B, 64, D), default D=64
queries:      (Q, D), default Q=24
attention:    (B, Q, 64)
attended:     (B, Q, D)
diagnostics:  (B, Q, S)
logits:       (B, 2)
```

### Default Config Sketch

```yaml
model:
  name: set_query_attention_bottleneck
  input_channels: 18
  num_classes: 2
  token_dim: 64
  query_count: 24
  head_count: 4
  head_hidden: 128
  attention_dropout: 0.0
  include_attention_diagnostics: true
  ablation: none
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `uniform_attention` | Replace learned attention with uniform token averaging per query | Tests whether selective attention matters. |
| `random_frozen_queries` | Freeze random query vectors | Tests whether learned tactical questions matter. |
| `value_only_no_diagnostics` | Use attended values but remove entropy/margin/mass diagnostics | Tests whether attention maps themselves add signal. |
| `diagnostics_only` | Use attention diagnostics without attended values | Tests whether routing shape alone carries signal. |
| `mean_pool_matched_params` | Replace query attention with mean/max token pooling and same-size head | Tests against ordinary set pooling. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/set_query_attention.py`
- Registry name: `set_query_attention_bottleneck`
- Main config: `configs/bench_set_query_attention_simple18.yaml`
- Central ablation config: `configs/bench_set_query_attention_uniform.yaml`
- Tests:
  - forward shape,
  - finite logits,
  - attention rows sum to 1,
  - ablation modes run,
  - deterministic output with fixed seed and eval mode.

### Success/Failure

Success:

- Main model beats `uniform_attention`, `random_frozen_queries`, and `mean_pool_matched_params`.
- Class `1` recall improves at matched fine-label-`0` false-positive rate, or calibration improves without hurting AUROC.

Failure:

- Mean pooling, random queries, or diagnostics-only match the main model.

## Candidate 2: Attention Disagreement Residual Network

### Thesis

Near-puzzle and puzzle-like positions may contain competing interpretations. Independent attention query families should disagree more on ambiguous or tactically dense boards. The classifier uses the residual disagreement among attention maps as evidence.

### Fingerprint

```text
multiple independent query banks
+ attention maps over shared board tokens
+ pairwise disagreement / covariance / JS divergence summaries
+ binary puzzle-likeness head
```

### Relationship To Prior Research

This is adjacent to orbit disagreement and credal ambiguity ideas, but the disagreement is among learned attention routes over the same board, not exact symmetry views or output evidence distributions.

### Architecture Sketch

1. Build square tokens `(B, 64, D)`.
2. Use `F=4` independent query banks, each with `Q=8` queries.
3. Compute attention maps `A_f in (B, Q, 64)`.
4. Pool each family to attended values.
5. Compute disagreement summaries:
   - Jensen-Shannon divergence between family-averaged maps,
   - covariance of attended coordinate means,
   - entropy variance across families,
   - maximum query-map cosine distance.
6. Classify from mean attended value plus disagreement summaries.

### Tensor Contract

```text
input:          (B, 18, 8, 8)
tokens:         (B, 64, D)
attention:      (B, F, Q, 64)
attended:       (B, F, Q, D)
disagreement:   (B, features)
logits:         (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `shared_query_bank` | Force all families to share queries | Tests whether independent attention interpretations matter. |
| `mean_attention_only` | Use average attention map, no disagreement stats | Tests whether disagreement is useful. |
| `random_family_permutation` | Shuffle family attention maps across samples before disagreement | Tests whether disagreement is sample-specific. |
| `single_family_matched_params` | Use one larger query bank with matched parameters | Tests against ordinary attention capacity. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/attention_disagreement.py`
- Registry name: `attention_disagreement_residual`
- Main config: `configs/bench_attention_disagreement_simple18.yaml`
- Central ablation config: `configs/bench_attention_disagreement_mean_only.yaml`

### Success/Failure

Success:

- Beats `mean_attention_only` and `single_family_matched_params`, with better near-puzzle diagnostics or calibration.

Failure:

- A single larger query bank matches it or random family shuffling does not hurt.

## Candidate 3: Cross-Scale Attention Residual Network

### Thesis

Puzzle-like evidence may appear when fine-square attention cannot be predicted from coarse board context. This model computes attention from fine tokens to coarse tokens, reconstructs expected fine attention, and classifies from the residual attention map.

### Fingerprint

```text
fine square tokens and coarse pooled tokens
+ cross-scale attention
+ fine-minus-coarse attention residual maps
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not a standard multiscale CNN: the residual is in attention space.
- Not wavelet scattering: cross-scale matching is learned through attention, not fixed filters.
- Not coarse-to-fine residual pyramid: this residual is about routing/explanation, not raw board reconstruction.

### Architecture Sketch

1. Build fine tokens `(B, 64, D)` from squares.
2. Build coarse tokens by pooling into 4x4 and 2x2 cells.
3. Fine tokens attend to coarse tokens, producing coarse-explained fine values.
4. Fine tokens also attend to learned query bank.
5. Predict expected fine attention from coarse attention.
6. Compute residual:

```text
R = A_fine_query - upsample(project(A_coarse_query))
```

7. Classify from residual attention maps, residual entropy, and attended fine values.

### Tensor Contract

```text
fine_tokens:      (B, 64, D)
coarse_tokens:    (B, 16, D) and (B, 4, D)
fine_attention:   (B, Q, 64)
coarse_attention: (B, Q, 16)
residual_map:     (B, Q, 64)
logits:           (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `fine_attention_only` | Drop coarse prediction and residual map | Tests whether cross-scale residual matters. |
| `coarse_attention_only` | Use only coarse attention | Tests whether fine unexplained detail matters. |
| `random_coarse_assignment` | Randomly assign squares to coarse cells preserving cell sizes | Tests whether board scale geometry matters. |
| `residual_norm_only` | Use only residual magnitude, not signed residual map | Tests whether residual structure matters. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/cross_scale_attention.py`
- Registry name: `cross_scale_attention_residual`
- Main config: `configs/bench_cross_scale_attention_simple18.yaml`
- Central ablation config: `configs/bench_cross_scale_attention_fine_only.yaml`

### Success/Failure

Success:

- Beats fine-only and coarse-only attention controls.

Failure:

- Residual maps do not improve over fine-only attention, or random coarse assignment matches real scale geometry.

## Candidate 4: Slot Attention Role Binding Network

### Thesis

Puzzle-like positions may be characterized by how occupied pieces bind to a small number of latent tactical roles. Slot attention can softly assign pieces to roles and expose role competition without selecting a hard witness subset.

### Fingerprint

```text
occupied piece tokens
+ iterative slot attention role assignment
+ slot residual updates and assignment entropy
+ binary puzzle-likeness head
```

### Relationship To Prior Research

This is adjacent to sparse witness-piece bottlenecks, but it does not censor the board or pick top-k pieces. Every occupied piece contributes softly to role slots, and the diagnostics are assignment entropy, role mass, and update residuals.

### Architecture Sketch

1. Extract up to 32 occupied piece tokens from `simple_18`.
2. Initialize `S=8` learned role slots.
3. Run `T=3` slot-attention iterations:

```text
attention_{slot,piece} = softmax_slot(slot_key^T piece_key)
slot_update = GRU(slot, weighted_piece_values)
```

4. Pool:
   - final slot vectors,
   - assignment entropy by slot and piece,
   - slot mass,
   - slot update residual norm per iteration.
5. Classify from slot vectors and diagnostics.

### Tensor Contract

```text
tokens:       (B, 32, F)
slots:        (B, S, D), default S=8
assignment:   (B, T, S, 32)
slot_updates: (B, T, S, D)
logits:       (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `single_iteration` | Use one slot update only | Tests whether iterative binding matters. |
| `mean_piece_pool` | Replace slots with mean/max occupied-piece pooling | Tests role binding against ordinary set pooling. |
| `random_slots_frozen` | Freeze random slot initializations/projections | Tests learned role slots. |
| `assignment_entropy_only` | Classify only from entropy/mass diagnostics | Tests whether assignment shape itself is signal. |
| `top_material_slots` | Initialize slots from material roles only | Tests whether learned roles beat material grouping. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/slot_attention_roles.py`
- Registry name: `slot_attention_roles`
- Main config: `configs/bench_slot_attention_roles_simple18.yaml`
- Central ablation config: `configs/bench_slot_attention_roles_mean_pool.yaml`
- Tests: forward shape, finite logits, assignment normalization, mask handling for fewer than 32 pieces.

### Success/Failure

Success:

- Beats mean/max occupied-piece pooling and single-iteration controls.

Failure:

- Material grouping or mean pooling matches the full slot model.

## Candidate 5: Attention Perturbation Sensitivity Network

### Thesis

Attention maps are often decorative unless perturbing attended regions changes evidence. This model uses deterministic attention-guided perturbation sensitivity as the bottleneck: how much the latent or logits move when high-attention versus low-attention board regions are safely masked.

### Fingerprint

```text
base attention reader
+ deterministic top-attention and low-attention masks
+ feature/logit sensitivity under safe current-board masking
+ binary puzzle-likeness head
```

### Risk

This is computationally heavier because it runs the encoder multiple times per sample. It is also adjacent to sparse witness masking. The difference is that the central feature is sensitivity delta, not a classifier forced to see only a subset.

### Architecture Sketch

1. Run a small attention reader to get attention maps over 64 squares.
2. Create deterministic masks:
   - top-attention squares,
   - low-attention occupied squares,
   - matched random occupied squares,
   - top-attention neighborhood.
3. Re-run a small shared encoder on masked board variants where selected square piece planes are zeroed and a mask-indicator plane is optionally added only if trainer/model config supports it.
4. Compute sensitivity:

```text
delta_top = ||z(x) - z(mask_top(x))||
delta_low = ||z(x) - z(mask_low(x))||
contrast = delta_top - delta_low
```

5. Classify from base latent, attention diagnostics, and sensitivity contrasts.

### Tensor Contract

```text
input:             (B, 18, 8, 8)
attention:         (B, Q, 64)
masked_variants:   (B, M, 18, 8, 8)
variant_latents:   (B, M, D)
sensitivity:       (B, features)
logits:            (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `random_mask_sensitivity` | Use matched random masks only | Tests whether attention-guided masks matter. |
| `attention_no_perturbation` | Use attention diagnostics without reruns | Tests whether sensitivity adds value. |
| `top_material_mask` | Mask highest-value pieces instead of highest-attention squares | Tests against material shortcut. |
| `stopgrad_attention_masks` | Stop gradients through mask selection | Prevents model from gaming mask choice; compare to differentiable version. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/attention_sensitivity.py`
- Registry name: `attention_perturbation_sensitivity`
- Main config: `configs/bench_attention_sensitivity_simple18.yaml`
- Central ablation config: `configs/bench_attention_sensitivity_random_mask.yaml`

### Success/Failure

Success:

- Attention-guided sensitivity beats random masks and attention-without-perturbation.

Failure:

- Material masks or random masks match the full model.

## Cross-Candidate Benchmark Rules

Use the same default training setup:

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
- A same-parameter mean/max token pooling model.
- Each candidate's central attention-removal ablation.

Required diagnostics:

- AUROC, accuracy, balanced accuracy, F1, calibration.
- Fine-label `0/1/2 -> predicted 0/1` confusion matrices for main model and central ablation.
- Class `1` recall or precision at matched fine-label-`0` false-positive rate where available.
- Attention-specific diagnostics:
  - query entropy,
  - max attention and best-second margins,
  - occupied versus empty attention mass,
  - side-to-move versus opponent mass,
  - disagreement or slot assignment entropy where applicable.

## Prompt Maintenance Notes

If one of these attention-inspired ideas is implemented and fails, add the corresponding anti-duplicate rule to the deep research prompt:

| Candidate | Anti-duplicate rule if it fails |
|---|---|
| Set-Query Attention Bottleneck | Do not repeat small learned query-to-board attention bottlenecks with only different query counts, head counts, or token dimensions. |
| Attention Disagreement Residual Network | Do not repeat independent attention-family disagreement summaries with only different family counts or divergence metrics. |
| Cross-Scale Attention Residual Network | Do not repeat fine-minus-coarse attention residual maps with only different scale grids or query counts. |
| Slot Attention Role Binding Network | Do not repeat slot-attention occupied-piece role binding with only different slot counts or iteration counts. |
| Attention Perturbation Sensitivity Network | Do not repeat attention-guided masking sensitivity unless the perturbation target or falsifier changes materially. |

## Final Sanity Check

- Stored as a Markdown file in `ideas/all_ideas/research/packets/classic/`: yes
- Attention-inspired but not a plain Transformer proposal: yes
- Includes multiple architecture candidates: yes
- Avoids forbidden engine/search/source features: yes
- Does not fabricate labels: yes
- Keeps current-data minimal experiments possible: yes
- Includes central ablations: yes
- Gives implementation hooks for future Codex work: yes
