# Codex Research Synthesis: Best Candidates To Expand Next

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2113_friday_shanghai_best_expansions.md`
- Generated at: 2026-04-24 21:13
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: selection and expansion memo, not implemented

## Selection Summary

After reviewing the imported research memory and the local 2026-04-24 packets, the strongest path is not to implement the most abstract model first. The best sequence is:

1. Build a stronger regular comparator.
2. Test a compact token/attention bottleneck.
3. Test a residual-diagnostic model with observable failure modes.
4. Test one genuinely new signal-processing idea.
5. Only then spend effort on heavier pretrained or high-math mechanisms.

## Top Picks

| Rank | Idea | Source packet | Why it is one of the best |
|---|---|---|---|
| 1 | Piece-Token CNN Hybrid | `chess_nn_research_2026-04-24_2109_friday_shanghai_piece_token_cnn_hybrid.md` | Best practical baseline upgrade; easy to implement; directly tests whether occupied-piece tokens help beyond CNNs. |
| 2 | Set-Query Attention Bottleneck | `chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md` | Compact attention idea with clean controls; produces useful attention diagnostics without a full Transformer. |
| 3 | Fixed-Point Residual Defect Network | `chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md` | Best residual-inspired idea; the residual trajectory itself is measurable and falsifiable. |
| 4 | Bispectral Phase-Coupling Board Network | `chess_nn_research_2026-04-24_2110_friday_shanghai_bispectral_phase.md` | Most novel relative to current memory; cheap FFT computation; strong phase-vs-magnitude falsifier. |
| 5 | Masked Codec Interaction-Curvature Network | `chess_nn_research_2026-04-24_2051_friday_shanghai_researched_family_variants.md` | Best second-generation imported-family expansion, but requires pretraining and should come after simpler tests. |

## Why These Beat The Others

The high-level linear algebra ideas are interesting, especially Grassmannian angles and matrix pencils, but they are more abstract and likely to need careful numerical work before they pay off. The tropical circuit, determinant volume, harmonic potential, and polar Procrustes ideas are also legitimate, but their first failures would be harder to interpret unless stronger regular baselines are already established.

The two regular baselines are not glamorous, but they are strategically important. If future research packets cannot beat a piece-token CNN hybrid or a multi-scale mixer, they are probably not worth scaling.

## Implementation Queue

| Stage | Implement | Main question | Stop condition |
|---|---|---|---|
| 1 | Piece-Token CNN Hybrid | Do explicit occupied-piece tokens improve over CNN-only? | Stop if `cnn_only_matched` equals full hybrid. |
| 2 | Set-Query Attention Bottleneck | Do learned board queries add value beyond mean pooling? | Stop if `uniform_attention` or `random_frozen_queries` equals full model. |
| 3 | Fixed-Point Residual Defect Network | Are residual convergence defects diagnostic? | Stop if `final_latent_only` equals full trajectory model. |
| 4 | Bispectral Phase-Coupling | Does phase coupling beat magnitude-only spectra? | Stop if `magnitude_only` or `power_only` equals full model. |
| 5 | Masked Codec Interaction-Curvature | Does second-order masked surprise beat first-order surprise? | Stop if `first_order_only` equals curvature model. |

## Expanded Candidate 1: Piece-Token CNN Hybrid

### Why Expand First

This is the most useful near-term model. It can become a stronger regular baseline and it exercises infrastructure needed by many later ideas: occupied-token extraction, token masks, token pooling, and CNN/token fusion.

### Concrete First Implementation

Use a modest model so it is not just a parameter-count win:

```yaml
model:
  name: piece_token_cnn_hybrid
  input_channels: 18
  num_classes: 2
  cnn_width: 48
  cnn_blocks: 3
  token_dim: 48
  token_mixer_layers: 2
  fusion_hidden: 160
  dropout: 0.1
  use_batchnorm: true
  include_interaction: true
  ablation: none
```

Implementation details:

- Token extractor should produce a fixed `(B, 32, F)` tensor plus `(B, 32)` mask.
- Keep token features simple: piece type, own/opponent flag, color, rank/file, side-relative rank/file, castling/en-passant scalars.
- Token mixer should be MLP/set pooling only in the first implementation; do not add attention here.
- Fusion should include:
  - CNN mean pool,
  - CNN max pool,
  - token mean pool,
  - token max pool,
  - token sum pool,
  - small multiplicative interaction `cnn_proj * token_proj`.

### Required Files

| Path | Action |
|---|---|
| `src/chess_nn_playground/models/trunk/piece_token_cnn_hybrid.py` | Create model and builder. |
| `configs/bench_piece_token_cnn_hybrid_simple18.yaml` | Main config. |
| `configs/bench_piece_token_cnn_hybrid_cnn_only.yaml` | Remove token branch; match params. |
| `configs/bench_piece_token_cnn_hybrid_token_only.yaml` | Remove CNN branch. |
| `tests/test_piece_token_cnn_hybrid_forward.py` | Shape, finite logits, token mask tests. |

### Minimal Ablation Set

| Ablation | Why mandatory |
|---|---|
| `cnn_only_matched` | Proves tokens matter beyond capacity. |
| `token_only` | Shows whether dense board texture still matters. |
| `material_token_only` | Detects material shortcut in token branch. |
| `shuffle_token_coordinates` | Confirms token geometry matters. |

### Decision Rule

Promote this to a standard baseline if:

- It beats the best existing same-budget CNN/residual baseline by at least `0.5` AUROC point or improves class `1` recall at matched fine-label `0` false-positive rate.
- `cnn_only_matched` is clearly worse.
- Coordinate shuffle hurts the token branch.

## Expanded Candidate 2: Set-Query Attention Bottleneck

### Why Expand Second

This is the best attention-inspired idea because it does not require a full Transformer. It gives interpretable diagnostics: attention entropy, max mass, occupied mass, side-to-move mass, and best-second margins.

### Concrete First Implementation

```yaml
model:
  name: set_query_attention_bottleneck
  input_channels: 18
  num_classes: 2
  token_dim: 64
  query_count: 16
  head_count: 4
  head_hidden: 128
  attention_dropout: 0.0
  include_attention_diagnostics: true
  ablation: none
```

Implementation details:

- Use 64 square tokens, not occupied tokens, for the first version.
- Token feature at each square should include the 18 input channels plus rank/file/center-distance coordinate features.
- Use learned query vectors only; no token-to-token self-attention.
- Compute attention once:

```text
attention = softmax(query dot key / sqrt(d), over 64 squares)
attended = attention @ values
```

- Head input should include attended values and diagnostics.

### Required Files

| Path | Action |
|---|---|
| `src/chess_nn_playground/models/trunk/set_query_attention.py` | Create model and builder. |
| `configs/bench_set_query_attention_simple18.yaml` | Main config. |
| `configs/bench_set_query_attention_uniform.yaml` | Uniform attention ablation. |
| `configs/bench_set_query_attention_random_queries.yaml` | Frozen random query ablation. |
| `tests/test_set_query_attention_forward.py` | Shape, finite logits, attention sums to one. |

### Minimal Ablation Set

| Ablation | Why mandatory |
|---|---|
| `uniform_attention` | Tests selectivity. |
| `random_frozen_queries` | Tests learned queries. |
| `mean_pool_matched_params` | Tests against ordinary token pooling. |
| `value_only_no_diagnostics` | Tests whether attention-map diagnostics matter. |

### Decision Rule

Continue if:

- Full model beats uniform attention and mean pooling.
- Attention diagnostics differ between fine labels `0`, `1`, and `2`.
- It is competitive with Piece-Token CNN Hybrid.

Drop if:

- Random queries or mean pooling match it.

## Expanded Candidate 3: Fixed-Point Residual Defect Network

### Why Expand Third

This is the best residual idea because the residual is not just a skip connection. The model asks whether puzzle-like positions produce distinctive convergence defects under a shared latent update operator.

### Concrete First Implementation

```yaml
model:
  name: fixed_point_residual_defect
  input_channels: 18
  num_classes: 2
  latent_dim: 96
  steps: 5
  update_hidden: 192
  alpha: 0.5
  projection_dim: 12
  include_final_latent: true
  ablation: none
```

Implementation details:

- Use a small CNN stem to produce `global_board_embed`.
- Project to `h0`.
- Shared update MLP receives `[h_t, global_board_embed]`.
- Store every residual `r_t = F(h_t, x) - h_t`.
- Head sees:
  - final latent,
  - residual L2 path,
  - residual L1 path,
  - residual cosine similarities,
  - contraction ratios,
  - learned low-rank projections of residuals.

### Required Files

| Path | Action |
|---|---|
| `src/chess_nn_playground/models/trunk/fixed_point_residual.py` | Create model and builder. |
| `configs/bench_fixed_point_residual_simple18.yaml` | Main config. |
| `configs/bench_fixed_point_residual_final_only.yaml` | Final latent only ablation. |
| `configs/bench_fixed_point_residual_untied.yaml` | Ordinary untied residual block control. |
| `tests/test_fixed_point_residual_forward.py` | Shape, finite logits, residual path shape. |

### Minimal Ablation Set

| Ablation | Why mandatory |
|---|---|
| `final_latent_only` | Tests whether trajectory matters. |
| `untied_residual_blocks` | Tests against ordinary residual capacity. |
| `defect_norm_only` | Tests whether direction/oscillation adds signal. |
| `single_step` | Tests whether multi-step dynamics matter. |

### Decision Rule

Continue if:

- Full trajectory beats final-only and untied residual controls.
- Residual contraction/oscillation diagnostics differ by fine label.

Drop if:

- Final latent only matches full model.

## Expanded Candidate 4: Bispectral Phase-Coupling Board Network

### Why Expand Fourth

This is the most genuinely new idea in the local packet set. It is also cheap to compute and has crisp controls. It should not be implemented before stronger conventional baselines, because otherwise a failure is hard to interpret.

### Concrete First Implementation

```yaml
model:
  name: bispectral_phase_coupling
  input_channels: 18
  num_classes: 2
  mixed_channels: 12
  bispectrum_terms: 32
  head_hidden: 160
  use_coordinate_planes: true
  include_power_spectrum: true
  include_cross_channel_phase: false
  ablation: none
```

First version should skip cross-channel phase terms. Add them only if within-channel bispectrum beats controls.

Implementation details:

- Use `torch.fft.fft2`, not `rfft2`, to keep indexing simple.
- Hard-code selected low-frequency pair list.
- Feature for each term:

```text
cos(angle(Bis)), sin(angle(Bis)), log1p(abs(Bis))
```

- Include power spectrum features so magnitude-only ablation is fair.

### Required Files

| Path | Action |
|---|---|
| `src/chess_nn_playground/models/bispectral_phase.py` | Create FFT/bispectrum layer and builder. |
| `configs/bench_bispectral_phase_simple18.yaml` | Main config. |
| `configs/bench_bispectral_phase_magnitude_only.yaml` | Remove phase features. |
| `configs/bench_bispectral_phase_power_only.yaml` | Remove all bispectrum terms. |
| `tests/test_bispectral_phase_forward.py` | Shape, finite logits, selected-pair count. |

### Minimal Ablation Set

| Ablation | Why mandatory |
|---|---|
| `magnitude_only` | Tests phase coupling. |
| `power_only` | Tests third-order coupling. |
| `phase_batch_shuffle` | Tests sample-specific phase evidence. |
| `cnn_matched_params` | Tests against ordinary capacity. |

### Decision Rule

Continue if:

- Full model beats magnitude-only and power-only.
- It is competitive with regular CNNs.

Drop if:

- Magnitude-only or power-only matches it.

## Expanded Candidate 5: Masked Codec Interaction-Curvature Network

### Why Expand Later

This is the best expansion of an imported family, but it needs label-free pretraining and a more complex artifact flow. It is worth doing only after the regular baselines and one compact novel model are tested.

### Concrete First Implementation

Use a two-stage setup:

1. Train a small masked-board codec on train split positions only.
2. Freeze the codec.
3. Train a classifier from first-order surprise and second-order mask-interaction curvature.

First mask set:

- all own pieces,
- all opponent pieces,
- center 4x4,
- kings' local neighborhoods,
- each rank stripe group,
- each file stripe group.

Curvature:

```text
curv(A, B) = surprise(A union B) - surprise(A) - surprise(B)
```

### Required Files

| Path | Action |
|---|---|
| `src/chess_nn_playground/models/masked_codec_curvature.py` | Create codec/classifier modules. |
| `scripts/pretrain_masked_board_codec.py` | Pretrain frozen codec. |
| `configs/pretrain_masked_board_codec_simple18.yaml` | Pretraining config. |
| `configs/bench_masked_codec_curvature_simple18.yaml` | Main classifier config. |
| `configs/bench_masked_codec_curvature_first_order.yaml` | Parent-style ablation. |

### Minimal Ablation Set

| Ablation | Why mandatory |
|---|---|
| `first_order_only` | Tests curvature beyond parent masked-surprise idea. |
| `unigram_codec` | Tests learned codec. |
| `random_mask_pairs` | Tests semantic mask interactions. |
| `curvature_shuffled` | Tests sample-specific curvature. |

### Decision Rule

Continue if:

- Curvature beats first-order surprise.
- Unigram codec is worse.
- Curvature diagnostics differ by fine label.

Drop if:

- First-order surprise matches curvature.

## Candidates To Deprioritize For Now

| Candidate family | Reason to wait |
|---|---|
| Grassmannian, matrix-pencil, polar Procrustes | Interesting but numerically and conceptually abstract; better after token baselines exist. |
| Determinantal volume | Good idea, but overlaps enough with token/subspace work that it should wait behind Piece-Token and Set-Query baselines. |
| Harmonic potential | Easy but may be too generic; likely weaker than multi-scale CNN unless diagnostics prove otherwise. |
| Tropical circuit | Interesting but optimization/sparsity tuning may dominate early results. |
| Rank-quantile evidence | Useful regular idea, but less distinctive than Set-Query or Piece-Token. |
| Hall/king-path/sheaf/OT variants | Already heavily explored imported families; implement only if deliberately revisiting a family. |

## Recommended Experiment Suite

Create a small suite once the first two models are implemented:

```yaml
name: practical_plus_best_novel_suite
description: "Regular baselines plus compact attention/residual/spectral candidates."
configs:
  - configs/bench_cnn_small_simple18.yaml
  - configs/bench_cnn_medium_simple18.yaml
  - configs/bench_residual_small_simple18.yaml
  - configs/bench_piece_token_cnn_hybrid_simple18.yaml
  - configs/bench_piece_token_cnn_hybrid_cnn_only.yaml
  - configs/bench_set_query_attention_simple18.yaml
  - configs/bench_set_query_attention_uniform.yaml
  - configs/bench_fixed_point_residual_simple18.yaml
  - configs/bench_fixed_point_residual_final_only.yaml
  - configs/bench_bispectral_phase_simple18.yaml
  - configs/bench_bispectral_phase_magnitude_only.yaml
```

## Final Recommendation

Implement in this order:

1. `Piece-Token CNN Hybrid`
2. `Set-Query Attention Bottleneck`
3. `Fixed-Point Residual Defect Network`
4. `Bispectral Phase-Coupling Board Network`
5. `Masked Codec Interaction-Curvature Network`

This gives the cleanest research progression: practical baseline, attention bottleneck, residual diagnostic, novel phase signal, then second-generation masked-codec research.

## Final Sanity Check

- Reviewed generated and imported packet index: yes
- Selected best candidates by implementation value and falsifiability: yes
- Expanded the top candidates with concrete implementation details: yes
- Avoided forbidden engine/search/source features: yes
- Kept current-data minimal experiments possible: yes
- Included stop conditions and central ablations: yes
- Stored as a Markdown file in `ideas/research/packets/classic/`: yes
