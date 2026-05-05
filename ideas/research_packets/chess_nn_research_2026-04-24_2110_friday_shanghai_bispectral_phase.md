# Codex Handoff Packet: Bispectral Phase-Coupling Board Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-24_2110_friday_shanghai_bispectral_phase.md`
- Generated at: 2026-04-24 21:10
- Weekday: Friday
- Timezone: Asia/Shanghai
- Idea slug: `bispectral_phase`
- Intended next consumer: Codex
- Status: draft research packet, not implemented

## 2. Executive Selection

- Idea name: Bispectral Phase-Coupling Board Network
- One-sentence thesis: Puzzle-like positions may have distinctive spatial phase-coupling patterns between piece planes, and a bispectral bottleneck can test board arrangement geometry that is invisible to magnitude-only Fourier summaries and different from CNN texture learning.
- Idea fingerprint: current-board planes + learned channel mixtures + 2D FFT + selected bispectrum phase-coupling terms + phase/magnitude diagnostics + binary puzzle-likeness head.
- Why this is novel relative to the current packet set: no imported or local packet treats the board as a small multichannel signal and classifies from higher-order Fourier phase coupling.
- Why this is not a common CNN/ResNet/Transformer variant: the central computation is a fixed spectral transform and third-order phase-coupling statistic, not learned convolution depth, residual blocks, attention, graph propagation, move deltas, transport, or matrix decompositions.
- Current-data minimal experiment: train on `simple_18` using the existing `crtk_sample_3class` train/val/test splits for 3 epochs, compare against same-budget CNN/residual baselines and magnitude-only spectral ablations.
- Smallest central falsification ablation: keep Fourier magnitude spectra, low-frequency energy, channel powers, and ordinary pooled board stats, but replace all bispectral phase-coupling features with zeros or batch-shuffled phase features.
- Expected information gain if it fails: a clean failure rules out FFT phase-coupling as a useful static-board inductive bias before trying larger spectral models.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- `0`: known non-puzzle
- `1`: verified near-puzzle
- `2`: verified puzzle

Fine labels are evaluation diagnostics only. The model must return logits shaped `(batch, 2)` and use the shared trainer/reporting pipeline.

Allowed neural inputs:

- Current-board `simple_18` tensor.
- Side-to-move, castling, and en-passant planes already present.
- Deterministic coordinate planes if used only as current-board features.
- Fixed FFT frequency indices and deterministic spectral transforms.

Forbidden neural inputs:

- Stockfish scores, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Legal move generation, search, checkmate/stalemate oracles, or future game outcomes.

Tensor contract:

```text
input:              (B, 18, 8, 8)
mixed_planes:       (B, Cmix, 8, 8)
fft_coeffs:         (B, Cmix, 8, 5) complex from rfft2
bispectral_terms:   (B, Cpair, T) complex or phase/value features
features:           (B, H)
logits:             (B, 2)
```

Leakage checklist:

- FFT and bispectrum operate only on current tensor planes.
- No engine/search/source metadata enters the model.
- No move generation or legal oracle is needed.

## 4. Research Map

External research anchors are conceptual only. No external citation was verified during generation.

| Source or concept | Borrowed | Not copied |
|---|---|---|
| Fourier phase analysis | Spatial arrangement information is encoded in phase, not only magnitude. | No image retrieval system and no claim of translation invariance as a chess rule. |
| Bispectrum / higher-order spectra | Third-order products `F(k) F(l) conj(F(k+l))` capture phase coupling and are less reducible to power spectra. | No signal-processing estimator over time series and no statistical stationarity assumption. |
| Cross-power spectrum | Phase relationships between channels can encode relative spatial alignment. | No registration or image-alignment algorithm. |

Candidate search trace:

| Candidate mechanism | Why not selected |
|---|---|
| Fourier magnitude-only classifier | Too weak; magnitude loses much spatial arrangement and has a clean ablation role instead. |
| Learned spectral convolution layer | Too close to ordinary CNN with FFT implementation details. |
| Full 2D complex Transformer over Fourier coefficients | Too broad and would make phase coupling hard to falsify. |
| Wavelet scattering | Already logged as a practical candidate; bispectrum phase coupling tests a different signal property. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Fourier phase | Angle of complex FFT coefficients after learned channel mixing | `(B, C, 8, 5)` complex | phase shuffle | Not CNN, wavelet scattering, attention, or matrix decomposition. |
| Bispectrum | `B(k,l)=F(k)F(l)conj(F(k+l))` for selected frequency pairs | `(B, C, T)` complex | magnitude-only spectra | Tests higher-order phase coupling. |
| Cross-channel phase coupling | Bispectral or cross-power terms between mixed piece groups | `(B, Cpair, T)` | channel-pair shuffle | Tests relative board arrangement across piece groups. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/cnn.py` | Already present and tests local learned filters, not spectral phase coupling. |
| Residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | More residual capacity is regular scaling. |
| LC0-style CNN/residual CNN | Existing 112-plane configs | Too close to engine-network conventions. |
| Vanilla ViT over 64 squares | Common square-token Transformer | Too broad and does not isolate phase coupling. |
| Wavelet scattering | Local 2026-04-24 batch candidate | Fixed multiscale wavelets are local-band features; this packet tests Fourier phase and bispectrum. |
| Harmonic potential solver | Local 2026-04-24 packet | Inverse-Laplacian fields differ from phase-coupling spectra. |
| Determinant/Grassmannian/matrix-pencil/polar packets | Local 2026-04-24 packets | Those are matrix geometry bottlenecks; this is signal phase coupling. |
| Hyperparameter tuning | Existing configs | Not a research architecture. |
| Ensembling | Any leaderboard ensemble | Would hide whether bispectral features matter. |

## 6. Mathematical Thesis

Let `x in R^{C x 8 x 8}` be the current-board tensor. A learned `1x1` channel mixer creates `Cmix` real board fields:

```text
z_c = sum_j W_{c,j} x_j
```

For each mixed channel, compute the 2D discrete Fourier transform:

```text
F_c(k) = FFT2(z_c)(k)
```

where `k` ranges over the small 8x8 frequency grid. The power spectrum `|F_c(k)|^2` captures frequency energy but discards spatial phase. The bispectrum keeps third-order phase coupling:

```text
Bis_c(k, l) = F_c(k) F_c(l) conj(F_c(k + l mod grid))
```

Its phase is:

```text
angle Bis_c(k, l) = phase F_c(k) + phase F_c(l) - phase F_c(k + l)
```

Core hypothesis:

Chess puzzle-likeness may depend on relative spatial arrangements among pieces that appear as consistent phase-coupling signatures after safe channel mixing. These signatures are not captured by magnitude-only frequency summaries and are not forced by ordinary CNN local filters.

Proposition:

The bispectral phase term is invariant to a global translation of the board field in the continuous periodic-grid sense, because translation adds a linear phase ramp to each `F(k)` and the three phases cancel in `phase F(k) + phase F(l) - phase F(k+l)`.

Proof sketch:

If `z'(s)=z(s-a)`, then `F'(k)=exp(-i k dot a) F(k)`. Therefore:

```text
F'(k)F'(l)conj(F'(k+l))
= exp(-i(k+l-(k+l)) dot a) F(k)F(l)conj(F(k+l))
= F(k)F(l)conj(F(k+l))
```

On the finite 8x8 grid this holds for circular translations. Chess is not truly translation-invariant, so this is not a desired full symmetry. The useful point is that bispectrum separates arrangement phase coupling from absolute placement enough to provide a distinct diagnostic; coordinate or low-frequency features can restore absolute-board context.

What is actually proven:

- Bispectral terms contain phase-coupling information not present in the power spectrum alone.
- Magnitude-only ablations remove the central third-order phase term.
- The transform uses only current-board tensors.

What remains hypothesized:

- That useful chess puzzle signals are visible in these small-grid spectral phase couplings.
- That learned channel mixing finds meaningful piece-group fields.
- That phase features improve over CNN baselines and magnitude-only spectral features.

Counterexamples:

- Labels are driven mostly by material, phase, or source artifacts.
- Chess tactics require exact move consequences not visible in global spectral arrangement.
- The 8x8 grid is too small for stable spectral statistics.
- Absolute square placement dominates, and translation-stable phase coupling washes out important board semantics.

Self-critique:

This is unconventional for chess and could become a gimmick if not ablated carefully. The mandatory controls are magnitude-only, phase-shuffled, channel-pair-shuffled, and coordinate-restored baselines. If magnitude-only or CNN baselines match the full model, do not rescue the idea by adding a large CNN wrapper.

## 7. Architecture Specification

Module names:

- `SpectralChannelMixer`
- `BoardFFTFeatureLayer`
- `BispectralPhaseCoupling`
- `BispectralPhaseHead`

Forward pass:

1. Input `(B, 18, 8, 8)`.
2. Optional append coordinate planes:

```text
rank, file, center distance, side-relative forward
```

3. Use `1x1` convolution to mix channels:

```text
(B, Cin, 8, 8) -> (B, Cmix, 8, 8)
```

Default `Cmix=16`.

4. Compute `torch.fft.rfft2`:

```text
F: (B, Cmix, 8, 5) complex
```

5. Select a fixed small set of frequency pairs `(k, l)`:

- low-low pairs,
- low-mid pairs,
- diagonal-sensitive pairs,
- file/rank directional pairs.

Default `T=48` terms per channel. Avoid all frequencies to keep the head small.

6. Compute bispectrum:

```text
Bis(k,l)=F(k)F(l)conj(F(k+l mod 8x8))
```

For `rfft2`, implementation can either reconstruct the full complex FFT with `fft2` for simplicity or use careful index mapping. First implementation should use `torch.fft.fft2` on real mixed planes to avoid rfft indexing mistakes.

7. Build features:

   - `cos(angle Bis)` and `sin(angle Bis)`,
   - `log(1 + abs(Bis))`,
   - power spectrum low-frequency magnitudes,
   - cross-channel phase terms for selected channel pairs,
   - optional raw material/count summaries.

8. MLP head returns `(B, 2)`.

Shapes:

```text
input:        (B, 18, 8, 8)
mixed:        (B, 16, 8, 8)
fft:          (B, 16, 8, 8) complex
bispectrum:   (B, 16, 48) complex
features:     (B, about 2000-3500)
logits:       (B, 2)
```

Parameter estimate:

- 80k to 300k depending on `Cmix`, number of selected terms, and head width.

Complexity:

- FFT over 8x8 grids is cheap.
- Feature vector can get large; use a projection MLP and keep selected terms small.

Required config fields:

```yaml
model:
  name: bispectral_phase_coupling
  input_channels: 18
  num_classes: 2
  mixed_channels: 16
  bispectrum_terms: 48
  head_hidden: 192
  use_coordinate_planes: true
  include_power_spectrum: true
  include_cross_channel_phase: true
  ablation: none
```

Encoding adapters:

- `simple_18`: supported first.
- `lc0_static_112`: possible through learned channel mixing, but deterministic interpretation of history/current channels must be documented.
- `lc0_bt4_112`: optional later; start with `simple_18` to avoid mixing zero-filled history planes into spectral features.

Pseudocode:

```python
if use_coordinate_planes:
    x = torch.cat([x, coord_planes.expand(batch, -1, -1, -1)], dim=1)
z = channel_mixer(x)
f = torch.fft.fft2(z, dim=(-2, -1))
features = []
for k, l in selected_pairs:
    kl = ((k[0] + l[0]) % 8, (k[1] + l[1]) % 8)
    b = f[..., k[0], k[1]] * f[..., l[0], l[1]] * f[..., kl[0], kl[1]].conj()
    features.append(torch.cos(torch.angle(b)))
    features.append(torch.sin(torch.angle(b)))
    features.append(torch.log1p(torch.abs(b)))
if include_power_spectrum:
    features.append(power_spectrum_features(f))
features = torch.cat([v.flatten(1) for v in features], dim=1)
features = apply_ablation(features, f, z)
return head(features)
```

## 8. Loss, Training, And Regularization

- Primary loss: existing balanced cross entropy on coarse binary labels.
- Optimizer: AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Batch size: `512`.
- Regularization:
  - dropout `0.1` in the head;
  - optional small L2 penalty on channel mixer weights;
  - no stochastic spectral augmentation in first benchmark.
- Determinism: fixed selected frequency pairs, seed `42`, deterministic true.
- Fair comparison: same splits, same binary target, same 3-epoch budget, same artifact pipeline.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `magnitude_only` | Keep power spectrum and `abs(Bis)`, remove phase `sin/cos` | Phase coupling matters | If it matches, bispectral phase is unnecessary. |
| `power_only` | Use only Fourier power spectrum, no bispectrum | Third-order coupling matters | If it matches, bispectrum is unnecessary. |
| `phase_batch_shuffle` | Shuffle phase-coupling features across samples inside a batch | Phase features are sample-specific evidence | If it matches, phase features are not tied to labels. |
| `random_frequency_pairs` | Use random fixed frequency pairs with same count | Selected frequency structure matters | If it matches, any spectral sampling is enough. |
| `channel_pair_shuffle` | Shuffle cross-channel phase terms across channel pairs | Piece-group relationships matter | If it matches, channel semantics are weak. |
| `cnn_matched_params` | Small CNN with similar parameter count | Spectral phase beats regular CNN capacity | If it matches, this is not useful as a new inductive bias. |
| `no_coordinate_planes` | Remove coordinate planes | Absolute board context matters | If it improves or matches, coordinates are unnecessary. |

## 10. Benchmark And Falsification Criteria

Baselines:

- `configs/bench_cnn_small_simple18.yaml`
- `configs/bench_cnn_medium_simple18.yaml`
- `configs/bench_residual_small_simple18.yaml`
- `configs/bench_multiscale_cnn_mixer_simple18.yaml` if implemented
- `configs/bench_piece_token_cnn_hybrid_simple18.yaml` if implemented

Metrics:

- AUROC.
- Accuracy and balanced accuracy.
- F1.
- Calibration.
- Required fine-label `0/1/2 -> predicted 0/1` confusion matrix for main and central ablations.
- Class `1` recall or precision at matched fine-label-`0` false-positive rate where available.

Additional diagnostics:

- Mean phase-coupling feature norms by fine label.
- Main versus `magnitude_only` delta.
- Main versus `phase_batch_shuffle` delta.
- Frequency-pair importance by learned head weights or permutation.

Success threshold:

- Main model beats the best same-budget CNN/residual baseline by at least `+0.5` AUROC point, or improves class-`1` recall at matched fine-label-`0` FPR by at least `+1.0` point.
- Main beats `magnitude_only` and `power_only` by at least `+0.5` AUROC point or a clear class-`1` diagnostic gain.

Failure threshold:

- Magnitude-only, power-only, phase-shuffled, or matched CNN controls match the main model.

Abandon if:

- Phase coupling does not beat magnitude-only and phase-shuffled controls.

Scale if:

- Phase-coupling features beat central controls and remain competitive with regular CNNs.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260424_bispectral_phase/idea.yaml` | Create | Idea metadata copied from machine-readable block. |
| `ideas/20260424_bispectral_phase/math_thesis.md` | Create | Section 6 mathematical thesis. |
| `ideas/20260424_bispectral_phase/architecture.md` | Create | Section 7 architecture details. |
| `ideas/20260424_bispectral_phase/ablations.md` | Create | Section 9 ablations. |
| `src/chess_nn_playground/models/bispectral_phase.py` | Create | FFT feature layer, bispectrum feature extractor, builder. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `bispectral_phase_coupling`. |
| `configs/bench_bispectral_phase_simple18.yaml` | Create | Main config. |
| `configs/bench_bispectral_phase_magnitude_only.yaml` | Create | Central ablation config. |
| `configs/bench_bispectral_phase_power_only.yaml` | Create | Power-only ablation config. |
| `tests/test_bispectral_phase_forward.py` | Create | Forward shape, finite logits, ablation smoke tests, fixed selected-pair count. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-24_2110_friday_shanghai_bispectral_phase.md
  generated_at: 2026-04-24 21:10
  weekday: Friday
  timezone: Asia/Shanghai
  idea_slug: bispectral_phase
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260424_bispectral_phase
  name: Bispectral Phase-Coupling Board Network
  slug: bispectral_phase
  status: draft
  created_at: 2026-04-24
  author: Codex
  short_thesis: Puzzle-like positions may have distinctive Fourier bispectral phase-coupling signatures across learned current-board piece fields.
  novelty_claim: Uses higher-order spectral phase coupling over board planes, not CNN depth, self-attention, residuals, token subspaces, graphs, move deltas, OT, topology, or matrix decompositions.
  expected_advantage: Captures spatial arrangement phase relationships that magnitude-only spectra and local CNN filters may miss.
  central_falsification_ablation: magnitude_only
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary logits
  compute_notes: FFT over 8x8 grids is cheap; feature size depends on selected bispectrum term count.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/bench_bispectral_phase_simple18.yaml
  model_path: src/chess_nn_playground/models/bispectral_phase.py
  latest_result_path: null
  notes: Must run magnitude-only, power-only, phase-shuffled, and matched-CNN controls before scaling.
```

```yaml
config_yaml:
  run:
    name: bench_bispectral_phase_simple18
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
    name: bispectral_phase_coupling
    input_channels: 18
    num_classes: 2
    mixed_channels: 16
    bispectrum_terms: 48
    head_hidden: 192
    use_coordinate_planes: true
    include_power_spectrum: true
    include_cross_channel_phase: true
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
  model_name: bispectral_phase_coupling
  file_path: src/chess_nn_playground/models/bispectral_phase.py
  builder_function: build_bispectral_phase_coupling_from_config
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - SpectralChannelMixer
    - BoardFFTFeatureLayer
    - BispectralPhaseCoupling
    - BispectralPhaseHead
  required_config_fields:
    - input_channels
    - num_classes
    - mixed_channels
    - bispectrum_terms
    - use_coordinate_planes
    - include_power_spectrum
    - include_cross_channel_phase
    - ablation
  expected_parameter_count: 80000-300000
  expected_memory_notes: Feature tensor grows with mixed_channels * bispectrum_terms; keep selected terms fixed and small.
```

```yaml
research_continuity:
  idea_fingerprint: current-board channel mixtures + 2D FFT + selected bispectral phase-coupling terms + binary puzzle-likeness
  already_researched_family_overlap: Adjacent only to wavelet/scattering at the broad signal-processing level; not graph, move-delta, OT, topology, attention, residual, token-subspace, or matrix-decomposition work.
  closest_duplicate_risk: Could be confused with wavelet scattering; distinguish by Fourier bispectrum phase coupling and magnitude-only falsifier.
  do_not_repeat_if_this_fails:
    - FFT bispectrum/phase-coupling board classifiers with only different frequency-pair lists or mixed-channel counts.
    - Spectral phase models rescued by large CNN wrappers without beating magnitude-only and phase-shuffled controls.
  suggested_next_search_directions:
    - If phase partly helps, try smaller hand-picked frequency-pair sets before scaling feature count.
    - If power-only matches, treat magnitude spectrum as the useful part and abandon phase coupling.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add this packet to imported memory with fingerprint `FFT bispectral phase coupling over learned board fields`. | Prevents repeats under Fourier phase, bispectrum, or higher-order spectra terminology. | `Imported Research Memory` |
| Add anti-duplicate wording for spectral phase-coupling classifiers unless the spectral statistic or falsifier changes materially. | Avoids future packets that only change frequency lists or channel mixer width. | Anti-duplicate rules near wavelet/spectral ideas |
| Require magnitude-only, power-only, phase-shuffled, and matched-CNN controls for future spectral phase ideas. | These controls isolate phase coupling from ordinary spectral energy and model capacity. | `Ablation Plan` requirements |

## 14. Final Sanity Check

- Stored as a Markdown file in `ideas/research_packets/`: yes
- Completely new relative to current imported/local packet set: yes, within repo memory
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported and local packets completed: yes
- Distinct from wavelet scattering: yes
- Distinct from high-level linear algebra packets: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Mobius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
- Not an exact near-duplicate of imported cubical Euler/Betti topology, Hall-defect overload, or king-cage/king-escape path-DP packets: yes
- Not an exact near-duplicate of imported FCA/Galois-closure, denoising-score-field, or non-backtracking-edge-walk packets: yes
