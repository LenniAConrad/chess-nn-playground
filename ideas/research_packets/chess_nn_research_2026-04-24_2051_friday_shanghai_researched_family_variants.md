# Codex Research Batch: Variants On Already-Researched Families

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2051_friday_shanghai_researched_family_variants.md`
- Generated at: 2026-04-24 20:51
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: deliberate second-generation variants, not implemented

## Purpose

This file intentionally does something different from the novelty-only prompt: it starts from imported research packets and proposes controlled variants. These are not claimed to be wholly new families. They are useful if the research program wants to go deeper on a family that already looks promising or needs a sharper falsifier.

Use these only after deciding that a researched family deserves a second pass. Do not treat them as fresh independent ideas.

## Shared Data Contract

All variants use the current `chess-nn-playground` binary task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` are diagnostics only. The first implementation for each variant should use `simple_18`, the existing `crtk_sample_3class` splits, the shared trainer, and the same 3-epoch budget as the benchmark configs.

Forbidden as model inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, checkmate/stalemate oracles, or future game outcomes.

Allowed:

- Current board occupancy, side-to-move, castling/en-passant planes, deterministic square coordinates, safe current-board transforms, and label-free or class-0-only pretraining where explicitly stated.

## Ranked Shortlist

| Rank | Variant | Parent researched family | Why expand it |
|---|---|---|---|
| 1 | Masked Codec Interaction-Curvature Network | Masked Board Code-Length Surprise | Adds a crisp second-order falsifier instead of just more codec capacity. |
| 2 | Non-Puzzle Score Curl-Divergence Bottleneck | Non-Puzzle Score-Field Bottleneck | Keeps the class-0 prior but tests geometry of the repair field rather than raw repair size. |
| 3 | Ray Grammar Edit-Distance Network | Ray-Language Automaton | Replaces automaton scoring with soft edit distance to learned ray grammars. |
| 4 | Orbit Disagreement Residual Network | Orbit/Reynolds bottlenecks | Uses transform disagreement as evidence rather than forcing invariance. |
| 5 | Hall-Defect Dual-Residual Network | Hall-Defect Obligation Matroid | Uses unrolled dual residuals instead of exact Hall profiles. |
| 6 | Credal Temperature Field Network | Credal Near-Puzzle Evidence | Moves ambiguity handling into sample-wise temperature calibration rather than a Dirichlet evidence head. |

If implementing only one, start with `Masked Codec Interaction-Curvature Network`. It is closest to a real improvement over an imported idea because the central ablation is clean: first-order surprise versus second-order interaction curvature.

## Variant 1: Masked Codec Interaction-Curvature Network

### Parent Packet

- Parent family: `Masked Board Code-Length Surprise Network`
- Parent fingerprint: label-free masked board codec, spatial code-length/entropy fields, classifier over surprise maps.

### Delta From Parent

The parent uses first-order masked reconstruction surprise. This variant measures second-order interaction curvature: how much the surprise from masking two regions differs from the sum of masking each region independently.

```text
curv(A, B) = surprise(A union B) - surprise(A) - surprise(B)
```

The thesis is that tactical motifs may appear as non-additive reconstruction interactions between board regions, not merely as high surprise at one square.

### Architecture Sketch

1. Pretrain a small masked-board codec on all training positions without labels.
2. For each input, generate deterministic safe masks:
   - local king ring
   - center box
   - side-to-move piece squares
   - opponent piece squares
   - rank/file/diagonal stripe masks
3. Compute first-order surprise for each mask.
4. Compute pairwise curvature for a small fixed mask pair set.
5. Pool curvature maps and first-order surprise maps into a classifier head.

Tensor sketch:

```text
input:             (B, 18, 8, 8)
mask_set:          M masks, default M=12
first_surprise:    (B, M, 8, 8)
pair_curvature:    (B, P, 8, 8), P about 24
stats:             (B, pooled curvature and surprise stats)
logits:            (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `first_order_only` | Use only `surprise(A)` fields | Tests whether second-order curvature adds signal beyond the parent idea. |
| `random_mask_pairs` | Preserve mask sizes but randomize pair identities | Tests whether semantic region interactions matter. |
| `unigram_codec` | Replace codec with material/channel unigram predictor | Tests whether learned codec structure matters. |
| `curvature_shuffled` | Shuffle curvature maps across samples inside a batch | Tests whether curvature is real sample evidence. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/masked_codec_curvature.py`
- Registry name: `masked_codec_curvature`
- Main config: `configs/bench_masked_codec_curvature_simple18.yaml`
- Central ablation config: `configs/bench_masked_codec_curvature_first_order.yaml`

### Decision Rule

Pursue if `pair_curvature` beats `first_order_only` on AUROC or class-`1` recall at matched fine-label-`0` false-positive rate. Abandon if first-order surprise, unigram codec, or random mask-pair controls match the main model.

## Variant 2: Non-Puzzle Score Curl-Divergence Bottleneck

### Parent Packet

- Parent family: `Non-Puzzle Score-Field Bottleneck Network`
- Parent fingerprint: class-0-only denoising score prior, classify from non-puzzle repair vector fields.

### Delta From Parent

The parent classifies from repair vectors or score-field magnitudes. This variant treats the score field over board squares as a vector field and classifies from discrete divergence, curl-like rotational components, and Helmholtz-style residual summaries.

The point is not "how abnormal is this board?" but "what shape does the non-puzzle repair flow have?"

### Architecture Sketch

1. Train or reuse a class-0-only denoising score model over `simple_18`.
2. Evaluate the score/repair field at the current board.
3. Convert channel-wise repair maps into spatial vector summaries:
   - horizontal and vertical finite differences
   - divergence-like accumulation
   - curl-like local circulation over 2x2 cells
   - boundary flux
4. Pool these field-shape summaries and classify.

Tensor sketch:

```text
input:          (B, 18, 8, 8)
repair_field:   (B, 18, 8, 8)
divergence:     (B, K, 8, 8)
curl:           (B, K, 7, 7)
flux_stats:     (B, features)
logits:         (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `repair_norm_only` | Use only norm/energy of the score field | Tests whether field geometry matters beyond abnormality magnitude. |
| `all_class_score_prior` | Train score prior on all classes instead of class 0 only | Tests whether class-0 ordinaryness is necessary. |
| `random_field_rotation` | Rotate or permute repair vectors in channel-space | Tests whether channel semantics matter. |
| `divergence_only` | Remove curl and circulation | Tests whether rotational repair structure is useful. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/score_field_curl.py`
- Registry name: `score_field_curl_bottleneck`
- Main config: `configs/bench_score_field_curl_simple18.yaml`
- Central ablation config: `configs/bench_score_field_curl_norm_only.yaml`

### Decision Rule

Pursue if curl/divergence features beat repair norm-only and all-class-prior controls. Abandon if score magnitude explains everything.

## Variant 3: Ray Grammar Edit-Distance Network

### Parent Packet

- Parent family: `Ray-Language Automaton Network`
- Parent fingerprint: side-relative rank/file/diagonal ray token strings scored by differentiable weighted finite automata.

### Delta From Parent

The parent scores rays with automata. This variant learns a small set of prototype ray grammars and computes differentiable edit distances from each board ray to each prototype. It tests whether puzzle-likeness is closer to "near-miss grammar matching" than automaton acceptance.

### Architecture Sketch

1. Extract the same side-relative rank/file/diagonal ray token strings as the parent.
2. Learn `G` prototype token sequences of bounded length.
3. Compute soft Levenshtein/edit distance between each observed ray and each prototype.
4. Pool min, mean, and margin-to-second-prototype distances over ray families.
5. Classify from edit-distance histograms.

Tensor sketch:

```text
rays:           (B, R, T) discrete token ids
prototypes:     (G, T_max, token_logits)
edit_distance:  (B, R, G)
stats:          (B, pooled distance and margin features)
logits:         (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `bag_of_ray_tokens` | Use token counts without order/edit distance | Tests whether sequence grammar matters. |
| `random_prototypes` | Freeze random prototypes | Tests whether learned ray grammars matter. |
| `substitution_only` | Remove insertion/deletion operations | Tests whether edit-distance flexibility matters. |
| `ray_token_shuffle` | Shuffle tokens within each ray | Tests whether ordered ray structure matters. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/ray_edit_distance.py`
- Registry name: `ray_grammar_edit_distance`
- Main config: `configs/bench_ray_edit_distance_simple18.yaml`
- Central ablation config: `configs/bench_ray_edit_distance_bag_tokens.yaml`

### Decision Rule

Pursue only if it beats bag-of-ray tokens and token-shuffle controls. If not, the original automaton family should not be rescued by edit-distance wording.

## Variant 4: Orbit Disagreement Residual Network

### Parent Packet

- Parent families: `Legal Automorphism Quotient Network`, `Rule-Exact Orbit Bottleneck Network`, and `Color-Flip Orbit Evidence Bottleneck`
- Parent fingerprint: exact legal transform views, orbit pooling/Reynolds projection, invariant or evidence-intersection readouts.

### Delta From Parent

The parent orbit models mostly suppress transform disagreement by enforcing or pooling invariance. This variant treats disagreement itself as a signal: if exact safe transforms produce unstable evidence, that instability may identify source artifacts or ambiguous near-puzzle cases.

### Architecture Sketch

1. Generate exact safe transform views supported by the encoding adapter.
2. Run a shared encoder on each view.
3. Compute:
   - invariant mean latent
   - residuals from orbit mean
   - covariance/eigenvalue summaries of orbit latents
   - disagreement between view logits
4. Classify from invariant mean plus disagreement residuals.

Tensor sketch:

```text
views:          (B, G, C, 8, 8)
latents:        (B, G, D)
orbit_mean:     (B, D)
residual_stats: (B, features)
logits:         (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `invariant_mean_only` | Use only Reynolds/orbit mean | Tests whether disagreement adds to the parent model. |
| `augmentation_only` | Train with transforms but classify a single view | Tests whether explicit residual stats matter. |
| `random_pseudo_orbit` | Use semantics-destroying transforms with same count | Tests whether exact chess transforms matter. |
| `disagreement_stopgrad` | Stop gradients through disagreement branch | Tests whether the model learns to manufacture disagreement. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/orbit_disagreement.py`
- Registry name: `orbit_disagreement_residual`
- Main config: `configs/bench_orbit_disagreement_simple18.yaml`
- Central ablation config: `configs/bench_orbit_disagreement_mean_only.yaml`

### Decision Rule

Pursue if disagreement residuals improve fine-label `1` diagnostics or calibration while beating `invariant_mean_only`. Abandon if random pseudo-orbits match exact transforms.

## Variant 5: Hall-Defect Dual-Residual Network

### Parent Packet

- Parent family: `Hall-Defect Obligation Matroid Network`
- Parent fingerprint: exact Hall deficiency/transversal-matroid overload profiles over defender-obligation set systems.

### Delta From Parent

The parent computes exact set-system deficiency summaries. This variant unrolls a small Lagrangian dual relaxation of a defender-obligation covering problem and classifies from primal/dual residual trajectories.

This is still Hall-family-adjacent, but it asks whether the optimization path carries signal beyond final exact deficiency counts.

### Architecture Sketch

1. Build the same safe current-board obligation incidence matrix as the parent packet.
2. Define a relaxed covering objective:

```text
min c^T z subject to A z >= demand, 0 <= z <= 1
```

3. Run `T=5` differentiable projected dual-ascent steps.
4. Pool:
   - primal violation by step
   - dual variable norms
   - complementarity residuals
   - final relaxed objective
5. Classify from the residual trajectory.

Tensor sketch:

```text
incidence:    (B, obligations, defenders)
dual_path:    (B, T, obligations)
primal_path:  (B, T, defenders)
residuals:    (B, T, features)
logits:       (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `final_defect_only` | Use exact or final deficiency only | Tests whether optimization trajectory adds signal. |
| `degree_matched_rewire` | Rewire incidence preserving row/column degrees | Tests whether set semantics matter. |
| `random_dual_steps` | Use random update matrices with matched shapes | Tests whether dual ascent matters. |
| `obligation_count_only` | Use counts/material summaries | Tests whether residuals beat obvious shortcuts. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/hall_dual_residual.py`
- Registry name: `hall_dual_residual`
- Main config: `configs/bench_hall_dual_residual_simple18.yaml`
- Central ablation config: `configs/bench_hall_dual_residual_final_only.yaml`

### Decision Rule

Pursue only if dual residual paths beat final-defect-only and degree-matched rewires. Otherwise the exact Hall packet is already the right representative for this family.

## Variant 6: Credal Temperature Field Network

### Parent Packet

- Parent family: `Credal Near-Puzzle Evidence Network`
- Parent fingerprint: binary Dirichlet evidence treatment of fine-label-1 ambiguity.

### Delta From Parent

The parent changes the output evidence semantics. This variant keeps ordinary binary logits but predicts a sample-wise temperature and smoothing factor from current-board uncertainty fields. It asks whether near-puzzle ambiguity is better handled as calibrated uncertainty on the decision surface rather than Dirichlet evidence.

The model still returns standard logits for compatibility, with optional exported temperature diagnostics.

### Architecture Sketch

1. Shared board encoder produces latent `h`.
2. Binary classifier produces raw logits `z`.
3. A calibration branch predicts:
   - positive temperature `T(x) = softplus(t(x)) + eps`
   - optional smoothing `alpha(x) in [0, max_alpha]`
4. Training uses temperature-scaled logits and optional bounded label smoothing:

```text
loss = CE(z / T(x), y_smooth(alpha(x)))
```

5. Evaluation exports raw logits, calibrated logits, `T(x)`, `alpha(x)`, entropy, and fine-label diagnostics.

Tensor sketch:

```text
input:       (B, 18, 8, 8)
latent:      (B, D)
raw_logits:  (B, 2)
temperature: (B, 1)
smoothing:   (B, 1)
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `fixed_temperature` | Use global learned scalar temperature | Tests whether sample-wise calibration matters. |
| `ordinary_bce` | Plain binary classifier | Tests whether ambiguity branch adds value. |
| `dirichlet_evidence_parent` | Reproduce parent-style evidence head if implemented | Tests whether temperature field is a better second pass. |
| `temperature_stopgrad` | Stop calibration branch from shaping the encoder | Tests whether calibration is only post-hoc. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/credal_temperature.py`
- Registry name: `credal_temperature_field`
- Main config: `configs/bench_credal_temperature_simple18.yaml`
- Central ablation config: `configs/bench_credal_temperature_fixed.yaml`

### Decision Rule

Pursue if sample-wise temperature improves calibration and class-`1` diagnostics without hurting AUROC. Abandon if a global temperature or ordinary BCE matches it.

## Cross-Variant Benchmark Rules

Each variant should keep:

```yaml
seed: 42
deterministic: true
mode: coarse_binary
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

- Existing same-budget `simple_18` CNN and residual CNN.
- The closest parent-family ablation, not only the generic baseline.
- The central falsifier listed above.

Required diagnostics:

- AUROC, balanced accuracy, F1, calibration.
- Fine-label `0/1/2 -> predicted 0/1` confusion matrix for main model and central ablation.
- Class `1` recall or precision at matched fine-label-`0` false-positive rate where available.
- Family-specific diagnostic: curvature maps, curl/divergence energy, edit-distance margins, orbit disagreement, dual residual path, or sample-wise temperature histogram.

## Prompt Maintenance Notes

If a variant fails, update the deep research prompt with a stronger anti-duplicate rule. Do not allow future prompts to repeat the same family with only larger hidden sizes, more masks, more prototypes, more transforms, more dual steps, or different calibration branch widths.

| Variant | Add to anti-duplicate memory if it fails |
|---|---|
| Masked Codec Interaction-Curvature | masked-codec second-order mask interaction curvature and pairwise surprise additivity tests |
| Non-Puzzle Score Curl-Divergence | class-0 score-field curl/divergence/flux shape descriptors |
| Ray Grammar Edit-Distance | soft edit-distance ray grammar prototypes over rank/file/diagonal strings |
| Orbit Disagreement Residual | exact-transform latent disagreement residuals and orbit covariance as evidence |
| Hall-Defect Dual-Residual | unrolled Hall/covering dual residual trajectories |
| Credal Temperature Field | sample-wise temperature/smoothing fields for near-puzzle ambiguity |

## Final Sanity Check

- Stored as a Markdown file in `ideas/research_packets/`: yes
- Explicitly varies already-researched families: yes
- Marks parent family for each variant: yes
- Gives a concrete delta from the parent idea: yes
- Avoids forbidden engine/search/source features: yes
- Does not fabricate labels: yes
- Keeps current-data minimal experiment possible: yes
- Includes central falsification ablations: yes
- Gives implementation hooks for future Codex work: yes
