# Codex Handoff Packet: Finite-Field Character-Sum Board Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-24_2115_friday_shanghai_character_sums.md`
- Generated at: 2026-04-24 21:15
- Weekday: Friday
- Timezone: Asia/Shanghai
- Idea slug: `character_sums`
- Intended next consumer: Codex
- Status: draft research packet, not implemented

## 2. Executive Selection

- Idea name: Finite-Field Character-Sum Board Network
- Heavy math concept: finite-field harmonic analysis, additive and multiplicative characters, Gauss/Jacobi-style character sums, and Weil-style cancellation intuition.
- One-sentence thesis: Puzzle-like positions may create arithmetic structure in sparse board polynomials that appears as non-random finite-field character-sum signatures, giving a label-safe way to test global piece arrangement outside CNNs, attention, graph operators, transport, topology, and matrix decompositions.
- Idea fingerprint: current-board piece planes + deterministic finite-field board polynomials + additive/multiplicative character sums over small primes + cancellation/phase statistics + binary puzzle-likeness head.
- Why this is novel relative to current repo memory: no imported or local packet treats a chess board as a polynomial over finite fields or classifies from character-sum cancellation patterns.
- Why this is not a common CNN/ResNet/Transformer variant: the central operator is fixed arithmetic evaluation of sparse board polynomials over `F_p` followed by character sums, not learned convolution depth, residual stacking, attention, graph propagation, move deltas, OT, topology, or linear-algebra matrix spectra.
- Current-data minimal experiment: train on `simple_18` using existing `crtk_sample_3class` train/val/test splits for 3 epochs, compare with same-budget CNN/residual baselines plus residue-only and phase-shuffled controls.
- Smallest central falsification ablation: keep all finite-field residue histograms and polynomial count summaries, but remove additive/multiplicative character phases and signs.
- Expected information gain if it fails: a clean failure rules out arithmetic character-sum board signatures as useful for this task before anyone tries larger primes, more probes, or neural wrappers.

## 3. Problem Restatement And Data Contract

Task: binary chess puzzle-likeness classification from current board tensors:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- `0`: known non-puzzle
- `1`: verified near-puzzle
- `2`: verified puzzle

Fine labels stay evaluation diagnostics only. The model returns logits shaped `(batch, 2)` and should use the shared trainer.

Allowed neural inputs:

- Current-board `simple_18` tensor.
- Side-to-move, castling, and en-passant planes already present.
- Deterministic square coordinates mapped to finite-field residues.
- Deterministic piece/channel integer codes derived from current board occupancy.

Forbidden neural inputs:

- Stockfish scores, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Legal move generation, search, checkmate/stalemate oracles, or future game outcomes.

Tensor contract:

```text
input:              (B, 18, 8, 8)
board_polynomial:   implicit sparse values over F_p
character_features: (B, P, Q, S)
pooled_features:    (B, H)
logits:             (B, 2)
```

The first implementation should use deterministic feature extraction plus a small MLP head. This keeps the mathematical bottleneck clean.

Leakage checklist:

- Character sums use only current board occupancy and fixed finite-field probes.
- No engine/search/source metadata enters the model.
- No legal move generation or attack graph is needed.
- Fine labels are evaluation-only.

## 4. Research Map

External research anchors are conceptual only. No external citation was verified during generation.

| Source or concept | Borrowed | Not copied |
|---|---|---|
| Finite-field additive characters | Use `psi_p(t) = exp(2*pi*i*t/p)` to turn board polynomial evaluations into phases. | No theorem claiming chess labels obey finite-field equidistribution. |
| Multiplicative characters / Legendre symbols | Use quadratic character `chi_p(t)` as a sign-like arithmetic nonlinearity. | No number-theoretic proof of puzzle structure. |
| Character-sum cancellation | Non-random structure can create biased sums instead of cancellation. | No asymptotic Weil bound is used as a guarantee on 8x8 boards. |
| Polynomial method | Encode a board as sparse polynomial evaluations over small finite fields. | No algebraic geometry solver and no learned symbolic rule system. |

Candidate search trace:

| Candidate mechanism | Why not selected |
|---|---|
| Integer hash features over piece-square IDs | Too ad hoc; character sums give a principled harmonic basis over finite fields. |
| Real Fourier transform only | Already covered by bispectral phase packet; finite-field characters are arithmetic rather than Euclidean spectral. |
| p-adic neural network with learned valuations | Harder to implement and explain; small finite fields are simpler and deterministic. |
| Groebner-basis constraint solver | Too brittle and likely unimplementable for current data without hand-built constraints. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Additive character | `cos(2*pi*P(a,b)/p), sin(2*pi*P(a,b)/p)` for board polynomial `P` | `(B, primes, probes)` | residue-only | Not FFT over real grid; arithmetic finite-field phase. |
| Multiplicative character | Legendre sign `chi_p(P(a,b))` | `(B, primes, probes)` | sign shuffle | Tests quadratic-residue structure. |
| Character-sum cancellation | Mean/vector norm of sums over fixed probe sets | `(B, primes, probe_families)` | random residue remap | Tests arithmetic structure beyond counts. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/cnn.py` | Tests local filters, not arithmetic character sums. |
| Residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | More depth is regular scaling. |
| Vanilla ViT | Common square-token Transformer | Too broad and does not isolate arithmetic structure. |
| Real FFT bispectrum | Local bispectral phase packet | Uses Euclidean Fourier phase; this packet uses finite-field characters and Legendre signs. |
| Ray automata | Imported ray-language packet | Uses ordered ray strings; no finite-field polynomial probes. |
| Mobius/ANOVA constellation | Imported high-order constellation packet | Explicit tuple interactions differ from global arithmetic character sums. |
| Matrix-decomposition packets | Local linear-algebra packets | Matrix spectra differ from finite-field harmonic signatures. |
| Hyperparameter tuning | Existing configs | Not a research architecture. |

## 6. Mathematical Thesis

Represent a board as sparse piece-square data. For each occupied square `s = (r, f)` and piece/channel code `c`, assign fixed integers:

```text
r, f in {0, ..., 7}
piece_code(c) in Z
side_code in Z
```

For a small prime `p > 8`, map coordinates into `F_p`. Define a family of board polynomials:

```text
P_{alpha,beta,gamma}^{(p)}(x)
  = sum_{occupied pieces i} piece_code_i
      * (alpha_0 + alpha_1 r_i + alpha_2 f_i + alpha_3 r_i f_i
         + alpha_4 r_i^2 + alpha_5 f_i^2 + beta * side_i + gamma)
    mod p
```

This is not the only possible polynomial family. It is intentionally low-degree and deterministic so the first experiment is interpretable.

For an additive character:

```text
psi_p(t) = exp(2*pi*i*t/p)
```

define features:

```text
A_p(q, x) = psi_p(P_q^{(p)}(x))
```

For the quadratic multiplicative character:

```text
chi_p(t) =
  0  if t = 0
  1  if t is a nonzero square mod p
 -1  otherwise
```

define:

```text
M_p(q, x) = chi_p(P_q^{(p)}(x))
```

Then aggregate over a deterministic set of probes `q` and primes `p`.

Core hypothesis:

Puzzle-like boards may create non-random arithmetic signatures in low-degree polynomial summaries of piece-square arrangements. Non-puzzles may behave closer to cancellation under many character probes. The classifier tests whether the distribution of character phases, signs, and cancellation norms separates puzzle-like positions from non-puzzles.

What is actually proven:

- Additive character features are periodic finite-field harmonic functions of current-board polynomial summaries.
- Multiplicative character features detect quadratic-residue structure in those summaries.
- The central ablation can preserve residues and counts while deleting character phase/sign information.

What remains hypothesized:

- That chess puzzle-likeness correlates with these arithmetic signatures.
- That low-degree board polynomials are expressive enough.
- That the features do not collapse to material or phase shortcuts.

Counterexamples:

- Labels are driven mostly by material, source artifacts, or local motifs.
- Character sums behave like random hash features and add no semantic signal.
- Prime/probe selection accidentally aliases common material counts.
- Exact tactical information requires move consequences not visible in current-board arithmetic summaries.

Self-critique:

This is deliberately speculative. The biggest risk is that the character features become fancy random hashes. The experiment is still useful because the controls are strong: residue-only, random residue remap, material-only polynomial, and phase-shuffled character features. If those match the main model, abandon this idea.

## 7. Architecture Specification

Module names:

- `Simple18FiniteFieldEncoder`
- `CharacterProbeTable`
- `FiniteFieldCharacterFeatures`
- `CharacterSumHead`

Forward pass:

1. Input `(B, 18, 8, 8)`.
2. Extract deterministic occupied-piece table:

```text
(B, 32, piece_code, color_code, side_relative_code, rank, file, mask)
```

3. For each prime `p` in a fixed list, default:

```text
primes = [11, 13, 17, 19, 23]
```

4. For each probe polynomial `q`, compute `P_q^{(p)}(x) mod p`.
5. Compute features:

```text
cos(2*pi*P/p)
sin(2*pi*P/p)
legendre(P, p)
zero_indicator(P == 0)
residue_bucket(P)
```

6. Aggregate:
   - per-prime mean and std of cos/sin,
   - character-sum norm per probe family,
   - Legendre sign mean/std,
   - zero frequency,
   - optional residue histogram.
7. Concatenate material/count summary so ablations are fair.
8. MLP head returns `(B, 2)`.

Shapes:

```text
input:       (B, 18, 8, 8)
tokens:      (B, 32, F_int)
residues:    (B, P, Q)
char_feats:  (B, P, Q, S)
features:    (B, about 1000-3000)
logits:      (B, 2)
```

Default `Q=128` probes. Keep this small in the first run.

Parameter estimate:

- Character extraction has no trainable parameters.
- MLP head: 100k to 400k depending on feature count and hidden size.

Complexity:

- Pure integer/float feature extraction over at most 32 occupied pieces, 5 primes, and 128 probes.
- Cheap relative to CNN training, though vectorization should avoid Python loops over batch.

Required config fields:

```yaml
model:
  name: finite_field_character_sums
  input_channels: 18
  num_classes: 2
  primes: [11, 13, 17, 19, 23]
  probe_count: 128
  polynomial_degree: 2
  head_hidden: 192
  include_residue_histogram: true
  include_legendre: true
  include_material_summary: true
  ablation: none
```

Implementation notes:

- Precompute probe coefficients as fixed integer buffers.
- Precompute Legendre lookup tables for each prime as fixed buffers.
- Use integer-like tensor arithmetic where possible, then convert residues to float features.
- Since autograd does not need to flow into deterministic character features, feature extraction can be wrapped carefully without trainable operations. The MLP head is trainable.

Encoding adapters:

- `simple_18`: supported first.
- `lc0_static_112` and `lc0_bt4_112`: avoid initially; integer piece-code extraction must fail closed unless current-board channel semantics are explicit.

Pseudocode:

```python
tokens, mask = extract_piece_tokens_simple18(x)
features = []
for p in primes:
    coeff = probe_coeffs[p]  # (Q, num_terms)
    terms = build_terms_mod_p(tokens, p)  # (B, 32, num_terms)
    values = (terms[:, :, None, :] * coeff[None, None, :, :]).sum(dim=(1, 3)) % p
    angle = 2 * math.pi * values.float() / float(p)
    features.append(torch.cos(angle))
    features.append(torch.sin(angle))
    if include_legendre:
        features.append(legendre_lut[p][values])
    if include_residue_histogram:
        features.append(residue_histogram(values, p))
features.append(material_summary(tokens, mask))
z = torch.cat([flatten_feature(f) for f in features], dim=1)
z = apply_ablation(z, values, material_summary)
return head(z)
```

## 8. Loss, Training, And Regularization

- Primary loss: existing balanced cross entropy on coarse binary labels.
- Optimizer: AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Batch size: `512`.
- Dropout: `0.1` in the MLP head.
- Determinism: fixed probe coefficients, fixed prime list, seed `42`.
- Fair comparison: same train/val/test split, same binary target, same 3-epoch budget, same artifact pipeline.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `residue_only` | Keep residue histograms and polynomial values, remove additive cos/sin and Legendre signs | Character phases/signs matter | If it matches, character theory is unnecessary. |
| `material_polynomial_only` | Polynomials use only piece counts/material codes, no rank/file terms | Board geometry matters | If it matches, arithmetic features are material shortcuts. |
| `random_residue_remap` | Randomly permute residues independently for each prime before characters | Finite-field arithmetic structure matters | If it matches, characters are random hash features. |
| `phase_batch_shuffle` | Shuffle character phase/sign features across samples | Phase/sign features are sample-specific evidence | If it matches, features are not tied to labels. |
| `single_prime` | Use only one prime | Multi-prime consistency matters | If it matches, keep simpler feature set. |
| `real_polynomial_mlp` | Use real-valued polynomial summaries without mod/characters | Finite-field nonlinearity matters | If it matches, ordinary polynomial features suffice. |
| `cnn_matched_params` | Small CNN with similar parameter count | Character features beat conventional capacity | If it matches, do not scale this idea. |

## 10. Benchmark And Falsification Criteria

Baselines:

- `configs/bench_cnn_small_simple18.yaml`
- `configs/bench_cnn_medium_simple18.yaml`
- `configs/bench_residual_small_simple18.yaml`
- `configs/bench_piece_token_cnn_hybrid_simple18.yaml` if implemented
- `configs/bench_bispectral_phase_simple18.yaml` if implemented

Metrics:

- AUROC.
- Accuracy and balanced accuracy.
- F1.
- Calibration.
- Required fine-label `0/1/2 -> predicted 0/1` confusion matrix for main and central ablations.
- Class `1` recall or precision at matched fine-label-`0` false-positive rate where available.

Additional diagnostics:

- Character-sum norm by fine label.
- Legendre sign distribution by fine label.
- Main versus `residue_only` delta.
- Main versus `random_residue_remap` delta.
- Probe importance by permutation.

Success threshold:

- Main model beats best same-budget CNN/residual baseline by at least `+0.5` AUROC point, or improves class-`1` recall at matched fine-label-`0` FPR by at least `+1.0` point.
- Main beats `residue_only`, `material_polynomial_only`, and `random_residue_remap` by at least `+0.5` AUROC point or clear diagnostic gain.

Failure threshold:

- Residue-only, material-only, random remap, or matched CNN controls match the main model.

Abandon if:

- Character phases/signs do not beat residue-only and random-remap controls.

Scale if:

- Multi-prime character features beat central controls and show stable fine-label diagnostics.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260424_character_sums/idea.yaml` | Create | Idea metadata copied from machine-readable block. |
| `ideas/20260424_character_sums/math_thesis.md` | Create | Section 6 mathematical thesis. |
| `ideas/20260424_character_sums/architecture.md` | Create | Section 7 architecture details. |
| `ideas/20260424_character_sums/ablations.md` | Create | Section 9 ablations. |
| `src/chess_nn_playground/models/character_sums.py` | Create | Finite-field feature extractor and MLP head. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `finite_field_character_sums`. |
| `configs/bench_character_sums_simple18.yaml` | Create | Main config. |
| `configs/bench_character_sums_residue_only.yaml` | Create | Central ablation config. |
| `configs/bench_character_sums_random_remap.yaml` | Create | Random-residue control. |
| `tests/test_character_sums_forward.py` | Create | Forward shape, finite logits, residue range, deterministic feature tests. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-24_2115_friday_shanghai_character_sums.md
  generated_at: 2026-04-24 21:15
  weekday: Friday
  timezone: Asia/Shanghai
  idea_slug: character_sums
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260424_character_sums
  name: Finite-Field Character-Sum Board Network
  slug: character_sums
  status: draft
  created_at: 2026-04-24
  author: Codex
  short_thesis: Puzzle-like positions may have distinctive additive and multiplicative character-sum signatures over finite-field board polynomials.
  novelty_claim: Uses finite-field harmonic analysis and character sums, not CNN depth, attention, residuals, graphs, move deltas, transport, topology, matrix decompositions, or real FFT spectra.
  expected_advantage: Tests a global arithmetic signature of board arrangement with strong residue-only and random-remap falsifiers.
  central_falsification_ablation: residue_only
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary logits
  compute_notes: Deterministic feature extraction over small primes and at most 32 occupied pieces; MLP head is trainable.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/bench_character_sums_simple18.yaml
  model_path: src/chess_nn_playground/models/character_sums.py
  latest_result_path: null
  notes: Must run residue-only, material-only, random-remap, and matched-CNN controls before scaling.
```

```yaml
config_yaml:
  run:
    name: bench_character_sums_simple18
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
    name: finite_field_character_sums
    input_channels: 18
    num_classes: 2
    primes: [11, 13, 17, 19, 23]
    probe_count: 128
    polynomial_degree: 2
    head_hidden: 192
    include_residue_histogram: true
    include_legendre: true
    include_material_summary: true
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
  model_name: finite_field_character_sums
  file_path: src/chess_nn_playground/models/character_sums.py
  builder_function: build_finite_field_character_sums_from_config
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18FiniteFieldEncoder
    - CharacterProbeTable
    - FiniteFieldCharacterFeatures
    - CharacterSumHead
  required_config_fields:
    - input_channels
    - num_classes
    - primes
    - probe_count
    - polynomial_degree
    - include_residue_histogram
    - include_legendre
    - ablation
  expected_parameter_count: 100000-400000
  expected_memory_notes: Feature tensor grows with primes * probe_count; no large activations.
```

```yaml
research_continuity:
  idea_fingerprint: current-board sparse polynomial over finite fields + additive/multiplicative character sums + cancellation/phase statistics + binary puzzle-likeness
  already_researched_family_overlap: None close in current repo memory; broad adjacency only to spectral methods because both use harmonic ideas, but this is finite-field arithmetic rather than real FFT.
  closest_duplicate_risk: Could be confused with bispectral phase; distinguish by finite-field character sums and residue-only/random-remap falsifiers.
  do_not_repeat_if_this_fails:
    - Finite-field character-sum board classifiers with only different prime lists, probe counts, or low-degree polynomial terms.
    - Arithmetic hash/character features rescued by larger MLPs without beating residue-only and random-remap controls.
  suggested_next_search_directions:
    - If character phases partly help, reduce to the most informative primes/probes before scaling.
    - If residue-only matches, abandon character theory and treat residue histograms as the useful feature.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add this packet to imported memory with fingerprint `finite-field board polynomials + additive/multiplicative character sums`. | Prevents repeats under number theory, character sums, Legendre symbols, or arithmetic harmonics terminology. | `Imported Research Memory` |
| Add anti-duplicate wording for finite-field character models unless the arithmetic object or falsifier changes materially. | Avoids future packets that only change primes, probe counts, or polynomial degree. | Anti-duplicate rules near spectral ideas |
| Require residue-only, material-polynomial-only, random-residue-remap, and matched-CNN controls for future arithmetic feature models. | These controls distinguish real arithmetic structure from random hashing and material shortcuts. | `Ablation Plan` requirements |

## 14. Final Sanity Check

- Stored as a Markdown file in `ideas/research/packets/classic/`: yes
- Really math-heavy concept selected: yes, finite-field character sums
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported and local packets completed: yes
- Distinct from real FFT bispectral phase packet: yes
- Distinct from high-level linear algebra packets: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Mobius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
- Not an exact near-duplicate of imported cubical Euler/Betti topology, Hall-defect overload, or king-cage/king-escape path-DP packets: yes
- Not an exact near-duplicate of imported FCA/Galois-closure, denoising-score-field, or non-backtracking-edge-walk packets: yes
