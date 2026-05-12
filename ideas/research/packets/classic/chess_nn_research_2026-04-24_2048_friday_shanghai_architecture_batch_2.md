# Codex Research Batch: Additional Architecture Candidates 2

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`
- Generated at: 2026-04-24 20:48
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: draft research batch, not implemented

## Shared Data Contract

All candidates below target the current `chess-nn-playground` binary task:

- output `0`: non-puzzle
- output `1`: puzzle-like

The fine labels `0`, `1`, and `2` are diagnostics only. Every proposed model must accept `(batch, C, 8, 8)` and return `(batch, 2)` logits through the shared trainer.

Allowed inputs:

- Current board piece occupancy from `simple_18`.
- Side-to-move, castling, and en-passant planes already present in the encoding.
- Deterministic square coordinates and safe transforms of current board tensors.

Forbidden inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate-pool status, or anything computed from them.
- Engine search, forced-line search, legal checkmate/stalemate oracles, or future game outcomes.

Already-covered families to avoid repeating:

- Tactical sheaf/Hodge/attack graph variants.
- One-ply move-delta set, spectrum, or landscape models.
- Current-board Sinkhorn/optimal transport variants.
- Deterministic nuisance projection.
- Ordinal/evidential heads, sparse witness bottlenecks, ray automata, Mobius/ANOVA constellations, pseudo-likelihood ratios.
- Orbit/tempo/symmetry bottlenecks, masked code-length surprise, topology/Hall/king-path/FCA/score-field/non-backtracking walks.
- New 2026-04-24 local drafts: determinant log-volume, harmonic potential, and tropical min-plus constraint circuits.

## Ranked Shortlist

| Rank | Candidate | Why it is worth expanding |
|---|---|---|
| 1 | Parity-Syndrome Puzzle Bottleneck | Very different algebraic inductive bias, small implementation, crisp ablations. |
| 2 | Wavelet Scattering Board Network | Fixed multiscale operator with low leakage risk and fast baseline comparison. |
| 3 | Convex Feasibility Residual Network | Tests whether puzzle-likeness is distance from learned safe board-feasibility regions. |
| 4 | Rank-Quantile Evidence Field Network | Tests sparse extreme evidence without masking the board like the imported witness model. |
| 5 | Oriented Matroid Covector Bottleneck | High-risk, high-novelty sign-pattern geometry over occupied pieces. |

The best next full packet is probably `Parity-Syndrome Puzzle Bottleneck`: it is compact, not too close to the prior packets, and has a decisive sum-vs-parity falsifier.

## Candidate 1: Parity-Syndrome Puzzle Bottleneck

### Thesis

Puzzle-like positions may produce distinctive parity or syndrome patterns over current-board facts: not just which local facts are present, but whether learned sparse XOR-like constraints are satisfied or violated. This tests a mod-2 algebraic bottleneck rather than averaging, min-plus clauses, determinants, graphs, or move consequences.

### Fingerprint

```text
current-board literal activations
+ learned sparse parity-check matrix
+ differentiable mod-2 syndrome energy
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not tropical: uses parity/XOR-style syndromes, not min-plus OR-of-AND clauses.
- Not Mobius/ANOVA: does not enumerate degree-2/3 interactions; checks sparse parity constraints.
- Not sparse witness: the classifier sees syndrome summaries, not a selected subset of board pieces.
- Not masked codec: no generative reconstruction or code-length model.

### Model Sketch

1. Encode `(B, 18, 8, 8)` into bounded literal probabilities `p in [0, 1]`, default `(B, 32, 8, 8)`.
2. Flatten to `p in (B, L)`, `L=2048`.
3. Learn sparse nonnegative parity-check weights `H in (K, L)` through low-rank factors plus top-k masks, default `K=96`, `topk=16`.
4. Soft parity for check `k`:

```text
s_k = 0.5 * (1 - prod_i (1 - 2 * p_i) ** H_ki)
```

or, for numerical stability:

```text
log_abs = sum_i H_ki * log(clamp(abs(1 - 2 * p_i)))
sign = product sign(1 - 2 * p_i) for active literals, approximated by tanh product
s_k = 0.5 * (1 - sign * exp(log_abs))
```

5. Pool syndrome features: mean, max, entropy, top-k syndrome values, and check-margin histograms.
6. Feed `(B, features)` to an MLP head returning `(B, 2)`.

### Tensor Contract

```text
input:       (B, 18, 8, 8)
literals:    (B, 32, 8, 8)
flat:        (B, 2048)
syndromes:   (B, 96)
stats:       (B, about 256)
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it changes | Interpretation |
|---|---|---|
| `sum_checks` | Replace parity with weighted sums using same `H` | If it matches, mod-2 structure is unnecessary. |
| `random_parity_checks` | Freeze random sparse `H` with same degree | If it matches, learned checks are unnecessary. |
| `material_only_literals` | Only material/type aggregate literals | If it matches, the model uses material shortcuts. |
| `square_shuffle_preserve_channels` | Fixed random square permutation before literals | If it matches, board geometry is not used. |
| `dense_parity_no_sparsity` | Remove sparse top-k pressure | If dense wins but sparse fails, the idea may be generic capacity. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/trunk/parity_syndrome.py`
- Registry name: `parity_syndrome_bottleneck`
- Main config: `configs/bench_parity_syndrome_simple18.yaml`
- Central ablation config: `configs/bench_parity_syndrome_sum_checks.yaml`
- Focused tests: forward shape, finite logits, ablation mode, syndrome range in `[0, 1]`.

### Success/Failure

Success means beating same-budget `simple_18` CNN/residual baselines or improving class-`1` recall at matched fine-label-`0` false-positive rate, while beating `sum_checks`. Failure means sum checks, random checks, or material-only literals match the main model.

## Candidate 2: Wavelet Scattering Board Network

### Thesis

Puzzle-like structure may live in multiscale arrangements of piece planes. A fixed wavelet scattering front end can test whether stable multiscale modulus features help beyond learned CNN filters while avoiding engine-specific priors.

### Fingerprint

```text
current-board planes
+ fixed separable Haar/wavelet filter bank
+ modulus cascades and scale/channel energies
+ small binary head
```

### Why It Is Distinct

- Not CNN scaling: the main filters are fixed, not learned convolutional towers.
- Not harmonic potential: uses localized multiscale wavelets, not inverse-Laplacian Green functions.
- Not topology: no connected components, Betti curves, or Euler transforms.

### Model Sketch

1. Input `(B, 18, 8, 8)`.
2. Apply fixed separable Haar filters at scales `1`, `2`, and `4`, with orientation-like bands horizontal, vertical, and diagonal.
3. Apply modulus nonlinearity.
4. Optionally apply a second scattering layer: wavelet on modulus fields.
5. Pool per channel/scale/orientation: mean, std, max, signed low-pass energy.
6. MLP head returns logits.

### Tensor Contract

```text
input:       (B, 18, 8, 8)
bands_1:     (B, 18, S, O, 8, 8)
bands_2:     optional (B, 18, S2, O2, 8, 8)
stats:       (B, 500-2000)
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it changes | Interpretation |
|---|---|---|
| `random_fixed_filters` | Replace Haar filters with random orthogonal filters | If it matches, wavelet structure is not important. |
| `lowpass_only` | Remove high-frequency bands | If it matches, multiscale edges are unnecessary. |
| `cnn_matched_params` | Small learned CNN with same parameter count | If it matches, fixed scattering is not adding bias. |
| `channel_shuffle` | Shuffle input piece channels with a fixed permutation | Tests whether semantic channels matter. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/wavelet_scattering.py`
- Registry name: `wavelet_scattering_board`
- Main config: `configs/bench_wavelet_scattering_simple18.yaml`
- First implementation should use only fixed Haar filters to avoid new dependencies.

### Success/Failure

Success requires beating a matched-parameter CNN or showing better class-`1` diagnostic behavior. Abandon if random fixed filters or lowpass-only features match the full scattering model.

## Candidate 3: Convex Feasibility Residual Network

### Thesis

Puzzle-like positions may be those that lie near the boundary of several learned safe convex feasibility regions in board-feature space. An unrolled projection layer can test whether distance-to-feasibility is useful without using closed-form nuisance residualization or generative score fields.

### Fingerprint

```text
current-board encoder
+ learned convex halfspace/ball constraints
+ unrolled soft projection residuals
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not nuisance projection: constraints are learned feasibility sets, not known material/phase/king vectors projected out by ridge regression.
- Not score-field: no class-0 denoising prior or repair vector field.
- Not tropical: uses Euclidean/Bregman projection residuals, not min-plus clauses.

### Model Sketch

1. Encode board to compact vector `z in R^d`, default `d=64`, with a shallow shared CNN or MLP over flattened planes.
2. Define `K` learned convex constraints:
   - halfspaces `a_k^T z <= b_k`
   - balls `||P_k z - c_k||_2 <= r_k`
3. Run `T=3` unrolled soft projection steps:

```text
z_{t+1} = z_t - eta * sum_k gate_k(z_t) * violation_k(z_t) * grad violation_k
```

4. Classify from original `z`, final projected `z_T`, residual `z - z_T`, violation vector, and projection path length.

### Tensor Contract

```text
input:       (B, 18, 8, 8)
z:           (B, 64)
violations:  (B, K)
path:        (B, T, 64)
stats:       (B, 64 + 64 + K + T)
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it changes | Interpretation |
|---|---|---|
| `no_projection` | Classify from encoder vector only | If it matches, feasibility residual is unnecessary. |
| `random_constraints` | Freeze random constraints with matched scales | If it matches, learned feasibility geometry is unnecessary. |
| `linear_head_same_params` | Replace projection module with MLP of equal parameters | If it matches, unrolled projection is just capacity. |
| `material_only_encoder` | Use material/count features only | If it matches, constraints are material shortcuts. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/trunk/convex_feasibility.py`
- Registry name: `convex_feasibility_residual`
- Main config: `configs/bench_convex_feasibility_simple18.yaml`
- Central ablation config: `configs/bench_convex_feasibility_no_projection.yaml`

### Success/Failure

Success means the projection residual model beats `no_projection` and a same-parameter MLP while improving or matching CNN baseline diagnostics. Abandon if projection residuals add no measurable delta or collapse to material-only behavior.

## Candidate 4: Rank-Quantile Evidence Field Network

### Thesis

Puzzle-likeness may be driven by extreme sparse evidence fields rather than average board evidence. Differentiable rank and quantile pooling can test this while still allowing the classifier to see the full board, unlike a sparse witness mask.

### Fingerprint

```text
current-board evidence fields
+ differentiable sorting / quantile pooling
+ tail gaps and extreme-value summaries
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not sparse witness: it does not choose a subset of pieces and hide the rest of the board.
- Not ordinary attention: it constrains readout to rank/quantile statistics, not arbitrary weighted sums.
- Not ordinal/evidential: rank pooling is an architecture bottleneck, not a label-head reinterpretation.

### Model Sketch

1. Encode input into `K` evidence fields `(B, K, 8, 8)`.
2. Flatten each field to 64 values.
3. Use differentiable sorting or soft quantile approximation to compute quantiles `[0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]`.
4. Compute tail gaps: `q99-q95`, `q95-q50`, `q50-q05`, `q05-q01`.
5. Concatenate quantile stats with small global material-safe stats and classify.

### Tensor Contract

```text
input:       (B, 18, 8, 8)
fields:      (B, 24, 8, 8)
quantiles:   (B, 24, 7)
tail_gaps:   (B, 24, 4)
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it changes | Interpretation |
|---|---|---|
| `mean_pool_only` | Replace quantiles with mean/std | If it matches, tail evidence is unnecessary. |
| `topk_only` | Use hard top-k means without quantile curve | If it matches, full rank profile is unnecessary. |
| `random_field_encoder` | Freeze random field projections | If it matches, learned evidence fields are unnecessary. |
| `square_shuffle` | Shuffle squares before rank pooling | If it matches, geometry before evidence extraction is weak. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/trunk/rank_quantile.py`
- Registry name: `rank_quantile_evidence`
- Main config: `configs/bench_rank_quantile_simple18.yaml`
- Keep the first implementation simple: exact sort over 64 values is deterministic and cheap.

### Success/Failure

Success requires beating mean-pool-only and showing a useful class-`1` diagnostic. Failure means average pooling or hard top-k summaries match the full quantile profile.

## Candidate 5: Oriented Matroid Covector Bottleneck

### Thesis

Puzzle-like positions may be characterized by sign-pattern arrangements of occupied pieces in learned tactical coordinate systems. A covector bottleneck records which side of learned hyperplanes each occupied piece lies on, then pools sign-pattern histograms.

### Fingerprint

```text
occupied piece tokens
+ learned hyperplane arrangement
+ sign/covector histogram summaries
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not Hall-defect matroid: no matching, defender obligations, or transversal matroid.
- Not determinant volume: uses signs of learned hyperplane evaluations, not logdet/eigen volume.
- Not sparse witness: no top-k chosen subset.
- Not Mobius/ANOVA: no explicit polynomial tuple interactions.

### Model Sketch

1. Extract up to 32 occupied piece tokens from `simple_18`.
2. Encode each token into `h_i in R^d`.
3. Learn `P` hyperplanes `w_p^T h_i + b_p`.
4. Convert to soft signs `sigma_{i,p} = tanh(alpha * (w_p^T h_i + b_p))`.
5. Pool covector statistics:
   - per-hyperplane positive/negative/near-zero counts
   - pairwise sign agreement matrix among hyperplanes
   - entropy of sign histograms by piece role
6. MLP head returns logits.

### Tensor Contract

```text
input:       (B, 18, 8, 8)
tokens:      (B, 32, F)
embeddings:  (B, 32, 48)
signs:       (B, 32, P), default P=24
stats:       (B, about 1000)
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it changes | Interpretation |
|---|---|---|
| `magnitude_only` | Use hyperplane magnitudes without signs | If it matches, covector orientation is unnecessary. |
| `random_hyperplanes` | Freeze random hyperplanes | If it matches, learned arrangement is unnecessary. |
| `material_role_hist_only` | Use piece role histograms only | If it matches, sign geometry adds no signal. |
| `coordinate_shuffle_by_piece` | Shuffle coordinates within piece types | If it matches, geometry is not used. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/trunk/oriented_matroid_covector.py`
- Registry name: `oriented_matroid_covector`
- Main config: `configs/bench_oriented_matroid_covector_simple18.yaml`
- This is higher risk than parity or wavelet; implement after at least one cheaper idea.

### Success/Failure

Success requires beating magnitude-only and material-role-only controls. Failure means sign arrangements are not useful or the model collapses to material histograms.

## Cross-Candidate Benchmark Rules

Each candidate should use:

```yaml
run:
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

Required diagnostics:

- AUROC, balanced accuracy, F1, calibration.
- Fine-label `0/1/2 -> predicted 0/1` confusion matrices for main model and central ablation.
- Class `1` recall or precision at matched fine-label-`0` false-positive rate where tooling supports it.

## Prompt Maintenance Notes

If any candidate is implemented, update `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` and `ideas/research/packets/README.md` with the exact result.

| Candidate | Anti-duplicate rule if it fails |
|---|---|
| Parity-Syndrome Puzzle Bottleneck | Do not repeat sparse parity/XOR syndrome checks over current-board literals with only different check counts, literal channels, or sparsity levels. |
| Wavelet Scattering Board Network | Do not repeat fixed wavelet/scattering front ends with only different wavelet bases or scale lists. |
| Convex Feasibility Residual Network | Do not repeat learned convex projection residuals with only different halfspace counts or projection steps. |
| Rank-Quantile Evidence Field Network | Do not repeat differentiable rank/quantile field pooling with only different quantile grids or field counts. |
| Oriented Matroid Covector Bottleneck | Do not repeat hyperplane sign/covector histograms over occupied pieces with only different hyperplane counts or token dimensions. |

## Final Sanity Check

- Stored as a Markdown file in `ideas/research/packets/classic/`: yes
- Contains multiple new architecture candidates: yes
- Avoids forbidden engine/search/source features: yes
- Does not fabricate labels: yes
- Keeps current-data minimal experiment possible: yes
- Avoids exact duplicates of imported packets: yes
- Identifies central ablations: yes
- Gives implementation hooks for future Codex work: yes
