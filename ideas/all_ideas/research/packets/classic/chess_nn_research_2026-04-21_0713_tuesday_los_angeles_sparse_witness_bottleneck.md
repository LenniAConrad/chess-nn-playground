# Codex Handoff Packet: Sparse Witness-Piece Bottleneck Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0713_tuesday_los_angeles_sparse_witness_bottleneck.md`
- Generated at: 2026-04-21 07:13 America/Los_Angeles
- Weekday: Tuesday
- Timezone: America/Los_Angeles
- Idea slug: `sparse_witness_bottleneck`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Sparse Witness-Piece Bottleneck Network, abbreviated `swpb`
- One-sentence thesis: Chess puzzle-likeness should often be decidable from a small witness set of occupied piece-squares, so force the classifier to see only a learned fixed-budget subset of pieces and test whether that sparse explanation carries more signal than matched random piece subsets.
- Idea fingerprint: `current-board occupied piece-square set -> hard binary top-k witness selector over occupied pieces only -> masked-board classifier sees selected pieces plus safe global side/castling/en-passant bits -> binary puzzle-likeness logits; no engine features, no move generation, no attack graph, no transport, no sheaf, no nuisance projection`.
- Why this is not a common CNN/ResNet/Transformer variant: the central operator is a discrete minimum-description bottleneck over occupied piece-squares; the downstream CNN is deliberately small and receives a censored board, not the full position.
- Current-data minimal experiment: train `SparseWitnessBottleneckNet` on `simple_18` for the existing `crtk_sample_3class` train/val/test split with witness budget `K=8`, balanced binary cross-entropy, and the normal shared trainer reports.
- Smallest central falsification ablation: replace the learned occupied-piece selector with a random occupied-piece selector that preserves the same per-position witness count `min(K, n_occupied)` and the same visible global planes; if learned and random witnesses perform the same, the sparse witness semantics are not doing work.
- Expected information gain if it fails: a clean failure says puzzle-likeness in this dataset is not captured by a tiny occupied-piece rationale, or the selector is too weak/noisy; future cycles should avoid top-k occupied-piece rationale bottlenecks and shift toward ambiguity/calibration or cross-environment invariance.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The task is binary chess puzzle-likeness classification from a single board position:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The default benchmark is binary, with the positive class formed from fine labels `1` and `2`. Reports must still include the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

The model must be a PyTorch `torch.nn.Module` accepting tensors shaped:

```text
(batch, C, 8, 8)
```

and returning logits shaped:

```text
(batch, 2)
```

Current benchmark split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The full Parquet dataset has roughly 45M rows, but this proposal must not point the current trainer directly at the full file until streaming support exists.

Allowed encodings already known to the project:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant
- `lc0_static_112`
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists

Leakage checklist:

- Safe as inputs or deterministic transforms: current board coordinates, piece occupancy, piece identity, side-to-move, castling/en-passant planes already present in the encoding, and deterministic pseudo-legal attack geometry derived only from the current board. This proposal does **not** use attack geometry.
- Leakage-prone unless separately justified, engine-free, label-independent, and ablated: full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences. This proposal uses none of them.
- Always forbidden as neural-network inputs: Stockfish or other engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, or unresolved candidate-pool status.
- Fine labels `0/1/2` may be used as supervised targets or diagnostics, not as input features.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic occupied-piece geometry may only be extracted from documented current-board piece channels. History channels may be consumed only by a learned neural adapter, never by deterministic rule geometry. If channel semantics are unknown, the adapter must fail closed.

## 4. Research Map

External ideas used:

| Source | Borrowed | Not copied |
|---|---|---|
| Tishby, Pereira, and Bialek, “The Information Bottleneck Method,” arXiv: `https://arxiv.org/abs/physics/0004057` | The variational idea that useful representations should preserve label-relevant information while compressing input information. | No Blahut-Arimoto solver, no claim that the implemented objective exactly solves the information-bottleneck equations. |
| Louizos, Welling, and Kingma, “Learning Sparse Neural Networks through L0 Regularization,” arXiv: `https://arxiv.org/abs/1712.01312` | The idea that discrete sparsity can be optimized with stochastic gates and an expected sparsity objective. | The proposed gates select occupied input pieces, not weights; pruning is not the goal. |
| Jang, Gu, and Poole, “Categorical Reparameterization with Gumbel-Softmax,” arXiv: `https://arxiv.org/abs/1611.01144` | Straight-through relaxed discrete selection for differentiable top-k training. | No generative categorical latent-variable model is copied. |
| Lei, Barzilay, and Jaakkola, “Rationalizing Neural Predictions,” ACL Anthology: `https://aclanthology.org/D16-1011/` | The generator/encoder split: select a short rationale, then classify only from that rationale. | No text rationale assumptions, no contiguity regularizer, and no natural-language explanation target. |

Candidate search trace: the internal pass considered more than twelve families, including ordinal heads, masked autoencoding, invariant risk, equivariant canonicalization, energy models, sparse rationales, set networks, graph networks, spectral attacks, contrastive views, selective prediction, and source-artifact adversaries. The serious candidates below were rejected in favor of the sparse occupied-piece witness bottleneck.

| Serious candidate mechanism | Why it lost to the final idea |
|---|---|
| Ordinal ambiguity/evidential head using fine labels `0 < 1 < 2` | Attractive for near-puzzle calibration, but mostly changes the loss/head; it does not force a new chess-specific representation and could be added later to any model. |
| Cross-encoding invariant risk minimization across `simple_18`, `lc0_static_112`, and `lc0_bt4_112` | Good anti-artifact direction, but it requires multi-encoding dataloader work before the central hypothesis can be cleanly tested; the current minimal experiment should be smaller. |
| Masked board autoencoder / MDL motif compressor | Conceptually close, but reconstruction pressure may reward material/phase regularities rather than puzzle-likeness; the sparse witness classifier has a sharper falsifier. |
| Exact color-swap/rank-flip equivariant logit decomposition | Safe and useful, but too close to augmentation/equivariance hygiene and too weak as a standalone research cycle. |
| Energy-based legal-position contrastive model | It risks learning dataset/source artifacts and ordinary board plausibility instead of puzzle-likeness, and negative sampling would introduce extra design choices. |
| Selective prediction/abstention for fine label `1` | Valuable reporting layer, but it dodges the representation question and may optimize ambiguity exposure rather than improving binary classification. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on full `simple_18` board | `src/chess_nn_playground/models/cnn.py` | Already present and sees the full board, so it does not test sparse tactical sufficiency. |
| Residual CNN on full board | `src/chess_nn_playground/models/residual_cnn.py` | Already present; adding residual capacity is not a new inductive bias. |
| LC0-style CNN or residual CNN on `lc0_bt4_112` | Existing LC0 BT4-style CNN/residual variants | Already covered by the baseline suite and too close to copying engine-input architecture. |
| Bigger/deeper/wider CNN | Small/medium/deep CNN variants | Ordinary capacity scaling is explicitly not a research idea here. |
| Ordinary ViT over 64 squares | Vanilla Transformer-over-squares baseline | Too generic, data-hungry, and not specifically tied to puzzle-likeness. |
| Plain GNN on 64 board squares | Square-neighborhood GNN | Without a sharper operator, it is just another message-passing backbone and overlaps graph-baseline territory. |
| Hyperparameter tuning only | Existing trainer/configs | Tuning optimizer, width, depth, or schedule cannot falsify a new mathematical claim. |
| Ensembling baselines | Any existing model ensemble | Likely improves metrics without explaining what structural signal matters; also explicitly disallowed as the core idea. |
| Data augmentation only | Existing CNN with flips/color transforms | Useful hygiene, but not enough novelty and could hide rather than explain failure modes. |
| Tactical attack/sheaf/Hodge graph | Imported tactical sheaf/Hodge packets | Already researched family; edge labels, curvature, tension, or Laplacian variants would be duplicate work. |
| One-ply move-delta set model | Imported counterfactual move-delta packets | Already researched family; this proposal deliberately avoids legal or pseudo-legal move bags. |
| Sinkhorn/optimal-transport piece-target model | Imported OT packets | Already researched family; this proposal has no transport plan, pressure map, or target measure. |
| Nuisance residualization/projection | Imported nuisance-orthogonal packet | Already researched family; this proposal constrains visible pieces rather than projecting latent features. |
| Source/provenance classifier with adversarial removal | None in baseline, but common domain-adversarial pattern | Source labels/provenance must not be inputs, and hidden source artifacts need safer tests than supervised source adversaries. |

## 6. Mathematical Thesis

### Input space definition

Let `B` be the set of legal or dataset-provided chess positions represented as tensors `x in R^{C x 8 x 8}`. For `simple_18`, define deterministic current-board piece planes

```text
P(x) in {0,1}^{12 x 8 x 8}
```

and a safe global vector

```text
g(x) in R^d
```

formed from side-to-move, castling, and en-passant planes by spatial averaging or direct extraction. Let

```text
T(x) = {(s_i, p_i)}_{i=1}^{n(x)}
```

be the occupied piece-square set, where `s_i in {1,...,64}`, `p_i in {1,...,12}`, and `n(x) <= 32` for ordinary legal chess positions.

### Label/target definition

The binary target is

```text
y_bin = 0  if fine_label = 0
y_bin = 1  if fine_label in {1, 2}
```

Fine labels remain available for diagnostics but are not inputs.

### Data distribution assumptions

The training, validation, and test rows are drawn from the provided split distribution. The split may contain source artifacts, material/phase shortcuts, and puzzle-family biases. The hypothesis is not that all puzzles are locally tactical; the hypothesis is that a meaningful fraction of puzzle-likeness signal is carried by a small set of occupied witness pieces, and that forcing sparse witness sufficiency reduces reliance on diffuse artifacts.

### Allowed symmetry or equivariance assumptions

Chess is not invariant under arbitrary rotations or reflections: pawn direction, castling, en-passant, and side-to-move matter. The only safe exact board symmetry used optionally is the color-swap plus rank-flip involution that maps the side-to-move and piece colors accordingly. This proposal does not require symmetry to work; symmetry consistency is optional regularization, not the central operator.

### Core hypothesis

There exists a small budget `K` such that for many puzzle-like positions there is a witness subset

```text
S*(x) subset T(x),       |S*(x)| <= K
```

for which the conditional distribution of the label is close to the conditional distribution given the full board:

```text
P(Y | X=x) approx P(Y | X_{S*(x)}, g(x)).
```

In words, a tactic-like puzzle usually has a compact cast of relevant pieces: kings, attacker, defender, pinned piece, overloaded piece, mating net piece, or promotion/capture target. Non-puzzle source artifacts and broad material/phase regularities may be less compressible into a fixed small occupied-piece set.

### Formal object introduced

Define the `K`-witness compression family

```text
W_K = { z = (S, x_S, g(x)) : S subset T(x), |S| <= K }.
```

A selector `q_theta(S | x)` chooses a subset of occupied pieces. A predictor `f_phi` maps the compressed witness representation to logits. The training objective is the empirical Lagrangian

```text
min_{theta, phi}  E_{(x,y)} E_{S ~ q_theta(.|x)} [ CE(f_phi(S, x_S, g(x)), y) ]
                 + lambda_budget E_x [ max(0, |S| - K)^2 ]
                 + lambda_consistency L_flip(theta, phi; x)
```

with the default implementation using hard fixed-budget top-k, so the budget term is zero except for safety/debugging. `L_flip` is optional and enforces consistency under the exact color-swap/rank-flip involution.

### Proposition: capacity-limited witness sufficiency

Assume there is a measurable selector `S*(x)` with `|S*(x)| <= K` such that

```text
Y independent of X given (S*(X), X_{S*(X)}, g(X))
```

up to error `epsilon` in total variation. Let `F_K` be a finite classifier family over `K`-witness representations. Then the best classifier in `F_K` has excess Bayes risk bounded by an approximation term plus `O(epsilon)`, and its representation support is bounded by

```text
log |support(Z_K)| <= log sum_{j=0}^K binom(32, j) + K log(12 * 64) + dim(g) * b_g,
```

where `b_g` is the effective bit budget for quantized global features. This is much smaller than the support of full piece boards when `K << n(x)`.

### Proof sketch or derivation

The conditional-independence assumption says the Bayes decision rule can be expressed through the witness representation `Z_K = (S*, X_{S*}, g(X))` with at most `epsilon` loss from dropping the rest of `X`. Restricting ERM to functions of `Z_K` therefore does not remove the assumed Bayes-relevant signal. The support bound follows by counting subset choices from at most 32 occupied pieces and piece-square identities for at most `K` selected pieces. Standard finite-class or compression-style generalization intuition then favors the smaller representation class, provided the assumption is approximately true.

### What is actually proven

Only the compression/counting statement and the conditional implication are proven: if a small sufficient witness exists and the model can learn it, the bottleneck can preserve label signal while limiting full-board information.

### What remains only hypothesized

It is not proven that CRTK puzzle-likeness has small occupied-piece witnesses, that the neural selector will find them, or that this improves benchmark metrics. It is also not proven that source artifacts require more than `K` pieces; some artifacts may be sparse and therefore survive the bottleneck.

### Counterexamples where the idea should fail

- Quiet endgame studies, zugzwang, fortress breaks, or opposition themes where empty squares and long-range constraints matter more than a small cast of pieces.
- Puzzles whose solution depends on many escape squares being covered; the absence of pieces can matter and is hidden by the occupied-piece bottleneck.
- Dataset artifacts expressible by a few pieces, such as “queen near king” or “low material plus side-to-move” shortcuts.
- Positions where castling/en-passant/global planes dominate labels; the bottleneck cannot suppress those if they are passed through unchanged.
- Very sparse endgames with fewer than `K` pieces, where the bottleneck degenerates into a full-board view.

### Self-critique

The strongest objection is selector steganography: a learned selector could encode class information in *which* pieces it selects rather than selecting a human-meaningful tactical witness. This is not forbidden leakage, because it is computed from the current board, but it would weaken interpretability. The design reduces this risk by selecting only occupied squares, using a hard binary mask, fixing the witness budget, passing no continuous gate values to the classifier, and requiring matched random/top-k/nuisance-preserving ablations. The minimal experiment is still worth running because the central question is empirical and sharply falsifiable: do learned occupied-piece witnesses outperform equally small nonsemantic witnesses?

## 7. Architecture Specification

### Module names

- `SparseWitnessBottleneckNet`
- `EncodingAdapter`
- `Simple18Adapter`
- `LC0CurrentBoardAdapter` with fail-closed channel-map validation
- `OccupiedPieceTopKSelector`
- `WitnessGridEncoder`
- `OptionalFlipConsistencyMixin`

### Forward-pass steps

Input:

```text
x: float tensor [B, C, 8, 8]
```

Step 1: encode current-board semantics.

For `simple_18`:

```text
piece = x[:, 0:12, :, :]                         # [B, 12, 8, 8]
global_planes = x[:, 12:C, :, :]                 # [B, G, 8, 8]
global_vec = mean(global_planes, dim=(2,3))       # [B, G]
occupied = clamp(sum(piece, dim=1, keepdim=True), 0, 1)  # [B, 1, 8, 8]
```

For `lc0_static_112` and `lc0_bt4_112`, the adapter must require explicit config fields:

```yaml
piece_plane_indices: [...]
global_plane_indices: [...]
history_plane_indices: [...]
```

If these are absent, raise a clear error. Deterministic occupancy must use only documented current-board piece channels. History channels may feed a learned adapter only when explicitly enabled; they must not define occupied-piece geometry.

Step 2: score occupied pieces.

```text
score_features = concat(piece, global_planes_selected_or_broadcast)  # [B, 12+G, 8, 8]
raw_scores = BoardScorer(score_features)                             # [B, 1, 8, 8]
raw_scores[occupied == 0] = -large_constant
```

`BoardScorer` default:

```text
Conv3x3(in=12+G, out=32) -> GELU -> Conv3x3(32,32) -> GELU -> Conv1x1(32,1)
```

Step 3: choose hard witnesses.

Training default:

```text
mask = straight_through_gumbel_topk(raw_scores, k=K, valid=occupied)
```

Evaluation default:

```text
mask = deterministic_topk(raw_scores, k=K, valid=occupied)
```

Shape:

```text
mask: [B, 1, 8, 8] hard binary, selecting min(K, n_occupied) occupied squares
```

No continuous gate probabilities may be concatenated to the classifier input.

Step 4: censor the board.

```text
witness_piece = piece * mask                      # [B, 12, 8, 8]
mask_plane = mask                                 # [B, 1, 8, 8]
global_broadcast = broadcast(global_vec)          # [B, G, 8, 8]
witness_grid = concat(witness_piece, mask_plane, global_broadcast)
```

Default shape for `simple_18` if `G=6`:

```text
witness_grid: [B, 19, 8, 8]
```

Step 5: classify.

`WitnessGridEncoder` default:

```text
Conv3x3(19, 48) -> GELU
3 x small residual blocks at width 48
Global average pool -> [B, 48]
Linear(48, 96) -> GELU -> Dropout(0.10)
Linear(96, 2) -> logits [B, 2]
```

The module returns only:

```text
logits: [B, 2]
```

For debugging/reporting, an optional method `forward_with_mask` may return `(logits, mask, raw_scores)` but the shared trainer path should call normal `forward` and receive logits only.

### Parameter-count estimate

For `simple_18`, width `48`, three residual blocks:

- Board scorer: roughly 15k parameters.
- Witness encoder stem and residual blocks: roughly 130k to 170k parameters depending on exact global channel count.
- MLP head: roughly 5k parameters.
- Total expected range: `0.15M` to `0.25M` parameters.

This is intentionally smaller than many residual CNN baselines; the experiment tests the bottleneck, not capacity.

### FLOP or complexity estimate

- Occupied scoring: `O(B * 64 * 32 * (12+G) * 3 * 3)` for the first scorer convolution plus small follow-up convolutions.
- Top-k selection: `O(B * 64 log K)` or `O(B * 64)` with `torch.topk` on an 8x8 map.
- Witness encoder: about `3M` to `7M` MACs per position depending on width/block details.
- Candidate-set memory: occupied token list, if materialized, is at most `[B, 32, d_token]`. With `d_token=32` and `B=512`, this is about `512 * 32 * 32 * 4 ~= 2 MB`.
- Grid memory for `witness_grid` at `B=512`, `19` channels: about `512 * 19 * 8 * 8 * 4 ~= 2.5 MB`.
- No chunking is needed for the current occupied-piece candidate set because it is bounded by 32. If Codex generalizes this to larger candidate pools, scoring should be chunked before top-k.

### Required config fields

```yaml
model:
  name: sparse_witness_bottleneck
  input_channels: 18
  num_classes: 2
  encoding: simple_18
  witness_budget: 8
  selector_temperature: 1.0
  selector_temperature_min: 0.3
  selector_hard: true
  selector_valid_only_occupied: true
  encoder_width: 48
  encoder_blocks: 3
  dropout: 0.10
  use_flip_consistency: false
  flip_consistency_weight: 0.0
  adapter:
    type: simple_18
    piece_plane_indices: [0,1,2,3,4,5,6,7,8,9,10,11]
    global_plane_indices: [12,13,14,15,16,17]
```

### Encoding support recommendation

First experiment should use only `simple_18`. Reason: it has explicit current-board piece channels, making occupied-piece masking unambiguous and minimizing adapter risk.

`lc0_static_112` and `lc0_bt4_112` support is feasible only after Codex confirms the channel schema. The adapter must fail closed when channel semantics are unknown. For BT4, unavailable history planes are currently zero-filled; they may be passed through a learned adapter in later experiments, but they must not affect deterministic occupied-piece extraction.

### Pseudocode, not final implementation

```text
forward(x):
    piece, global_vec, global_planes, occupied = adapter(x)
    score_features = concat(piece, selected_global_planes(global_planes))
    scores = board_scorer(score_features)
    scores = mask_invalid(scores, occupied == 0, value=-1e9)

    if training:
        witness_mask = straight_through_gumbel_topk(scores, K, valid=occupied, temperature=tau)
    else:
        witness_mask = deterministic_topk(scores, K, valid=occupied)

    witness_piece = piece * witness_mask
    global_broadcast = broadcast_to_board(global_vec)
    witness_grid = concat(witness_piece, witness_mask, global_broadcast)
    logits = witness_encoder(witness_grid)
    return logits
```

## 8. Loss, Training, And Regularization

Primary loss:

```text
balanced binary cross-entropy with logits
```

Binary labels are derived from fine labels as `0 -> 0`, `1/2 -> 1`.

Optional auxiliary losses:

- `flip_consistency_loss`: KL or MSE between logits for a position and the inverse-transformed logits for the exact color-swap/rank-flip transform. Default off for the minimal experiment.
- `selector_entropy_floor`: a small debugging regularizer to avoid all scores collapsing early. Default off unless instability is observed.

Class weighting:

- Use the existing `balanced` class weighting used by benchmark configs.

Batch size expectations:

- Default `batch_size: 512` on CPU/GPU if existing baselines use it.
- Reduce to `256` only if top-k implementation or debug mask export causes memory pressure.

Optimizer defaults:

```yaml
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
epochs: 3
early_stopping_patience: 2
mixed_precision: false
```

Regularizers:

- Fixed witness budget `K=8` is the main regularizer.
- Dropout `0.10` only in the small MLP head.
- No engine-derived regularization.
- No source/provenance adversary.

Determinism requirements:

- Set seed `42`.
- Use deterministic PyTorch settings already expected by the trainer when feasible.
- For validation/test, use deterministic top-k, not stochastic Gumbel selection.
- For training reproducibility, seed Gumbel noise through the normal PyTorch RNG.

What must stay unchanged for a fair comparison:

- Same train/val/test split.
- Same `simple_18` encoding for the main baseline comparison.
- Same binary label mapping.
- Same report generation, confusion matrices, predictions, and leaderboard update path.
- Same number of epochs, class weighting policy, and early-stopping policy as the selected baseline config unless the baseline config cannot run.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Random occupied top-k witness | Replaces learned scores with uniform random selection among occupied pieces, preserving `min(K, n_occupied)` and global planes | The learned witness subset has label-relevant semantics beyond sparse censoring | If equal to main model, the selector is not learning useful witness semantics. |
| Score-shuffled within batch | Computes learned scores, then randomly permutes score maps across batch items before top-k while preserving each board's occupancy mask | Board-specific selector semantics matter | If equal to main model, the scorer may only learn global priors, not position-specific witnesses. |
| Fixed centrality heuristic | Selects occupied pieces closest to kings or board center with no learning | Learned witnesses beat obvious chess/geometric heuristics | If heuristic matches main model, gains may come from trivial king/center proximity. |
| Full occupied board, same encoder | Sets mask to all occupied pieces and uses the same `WitnessGridEncoder` | Bottleneck helps or hurts compared with uncensored board at similar capacity | If full board is much better and random top-k is poor, sparse witness assumption may be false or `K` too small. |
| Material/count-only control | Replaces selected piece locations with aggregate material counts, side-to-move, castling, en-passant, and selected-count features through a small MLP | Model is not just exploiting material/phase/global shortcuts | If this matches main model, sparse witness geometry is unnecessary. |
| Piece-identity-preserving square shuffle | Keeps selected piece identities and count but randomly reassigns them to occupied selected squares or legal board squares within color/piece buckets | Board coordinates and geometry of witnesses matter | If this matches main model, the classifier may rely mostly on piece composition. |
| Source-square marginal shuffle | Samples witness squares from an empirical square-frequency table conditioned on piece type and side-to-move, preserving obvious marginals | The learned joint configuration matters beyond marginal piece-square priors | If this matches main model, the witness is mostly dataset prior. |
| Budget sweep `K in {4, 8, 12, all}` | Changes witness capacity | Puzzle-likeness has an informative compression curve | If only `all` works, abandon the small-witness thesis. If `4` works nearly as well as `8`, scaling should explore stronger bottlenecks. |
| No global planes | Removes castling/en-passant and optionally side-to-move from the witness grid, with side-to-move retained only if needed for legal interpretation | Global planes are not carrying the entire signal | If performance collapses only when globals are removed, inspect whether globals are legitimate or shortcut-heavy. |
| Continuous-mask leak test | Passes soft selector probabilities to the classifier intentionally | Whether hard binary gating is protecting against selector steganography | If soft masks greatly outperform but hard masks fail, the original design correctly avoided an untrustworthy shortcut. |

Smallest central falsification ablation: `Random occupied top-k witness`.

Because this model uses a rule-generated occupied-piece candidate set, the ablations above include count-only and nuisance-preserving controls that preserve obvious shortcuts: candidate count, material, side-to-move, moving/piece identity, source-square marginal, selected-count, and coarse capture-like material composition, while destroying the learned witness semantics. No move-set, attack graph, hypergraph, sheaf, transport, or search-surrogate object is introduced.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN with the closest parameter budget.
- Existing `simple_18` residual CNN small/medium baseline.
- Existing LC0 BT4-style CNN/residual CNN results only as secondary context, not as the main fairness comparison if the proposed model uses `simple_18`.
- All central ablations listed above, especially random occupied top-k and full occupied board.

Metrics to inspect:

- Test accuracy.
- Test F1 for binary positive class.
- Test AUROC if the shared reporter supports it.
- Test AUPRC if available.
- Calibration: expected calibration error if already implemented; otherwise reliability bins are optional.
- Required `3x2` diagnostic matrix: true fine label `0/1/2` by predicted binary output `0/1`.

Near-puzzle diagnostic:

- Report class-`1` recall at the threshold where fine-label-`0` false-positive rate matches the best existing `simple_18` residual CNN.
- Also report class-`1` precision among predicted positives if the reporter can compute it from saved predictions.

Required artifacts:

- Main model metrics JSON or CSV.
- Main model fine-label `3x2` confusion matrix.
- Prediction file with logits/probabilities if existing reporter supports it.
- Same artifacts for random top-k, score-shuffle, full-board, and material/count-only controls.
- A mask summary report: average selected pieces by piece type, square heatmap, average selected count, and examples are useful but must not be required for the shared leaderboard.

Success threshold:

- Main model improves binary positive F1 or AUROC by at least `+1.0` percentage point over the closest `simple_18` residual CNN at comparable training budget, **and** beats random occupied top-k by at least `+2.0` points in positive F1 or AUROC.
- Alternatively, if overall metrics are tied, success can be claimed if class-`1` recall improves by at least `+3.0` points at matched fine-label-`0` false-positive rate while random top-k does not.

Failure threshold:

- Main model is within `0.3` points of random occupied top-k on F1/AUROC, or worse than the simple CNN while full-board same-encoder performs normally.
- Main model improves only when soft masks are passed downstream, indicating selector steganography rather than sparse witness learning.

What result would make me abandon the idea:

- Learned hard top-k witnesses do not beat random, score-shuffled, or heuristic witnesses across at least two seeds, and increasing `K` only approaches the full-board baseline without a useful compression curve.

What result would justify scaling:

- A clear learned-vs-random witness gap, stable across seeds, plus improved class-`1` recall at matched fine-label-`0` false-positive rate. Scaling should then test `K` schedules, `lc0_static_112` with a verified channel map, and optional ordinal/evidential heads.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0713_sparse_witness_bottleneck/idea.yaml` | Create | Machine-readable idea metadata, status `draft`, fingerprint, intended configs, and central falsifier. |
| `ideas/20260421_0713_sparse_witness_bottleneck/math_thesis.md` | Create | Copy Section 6 with the formal `K`-witness compression family, proposition, proof sketch, and failure cases. |
| `ideas/20260421_0713_sparse_witness_bottleneck/architecture.md` | Create | Copy Section 7 with module names, shapes, adapter assumptions, and pseudocode. |
| `ideas/20260421_0713_sparse_witness_bottleneck/implementation_notes.md` | Create | Notes on hard binary top-k, no continuous mask leakage, occupied-only candidates, fail-closed LC0 adapter, and debug mask export. |
| `ideas/20260421_0713_sparse_witness_bottleneck/trainer_notes.md` | Create | Loss, optimizer, deterministic validation top-k, class weighting, and unchanged benchmark requirements. |
| `ideas/20260421_0713_sparse_witness_bottleneck/ablations.md` | Create | Section 9 ablation table plus exact central falsification instructions. |
| `ideas/20260421_0713_sparse_witness_bottleneck/train.py` | Create | Thin experiment entrypoint that loads the project trainer/config; should not duplicate trainer logic. |
| `ideas/20260421_0713_sparse_witness_bottleneck/config.yaml` | Create | Config equivalent to the `config_yaml` block below. |
| `ideas/20260421_0713_sparse_witness_bottleneck/report_template.md` | Create | Required report fields: metrics, `3x2` fine confusion, ablation deltas, mask summaries, failure/success call. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Preserve hard constraints and add an anti-duplicate note for sparse occupied-piece top-k witness bottlenecks after this packet is consumed. |
| `src/chess_nn_playground/models/sparse_witness_bottleneck.py` | Create | Implement `SparseWitnessBottleneckNet`, adapters, selector, and witness encoder. No engine/move/attack features. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register builder name `sparse_witness_bottleneck`. |
| `configs/sparse_witness_bottleneck_simple18.yaml` | Create | Main benchmark config using `simple_18`, `K=8`, width `48`, 3 epochs, balanced class weighting. |
| `configs/sparse_witness_bottleneck_random_topk_simple18.yaml` | Create | Central falsification ablation config. |
| `configs/sparse_witness_bottleneck_fullmask_simple18.yaml` | Create | Full occupied-board same-encoder ablation config. |
| `tests/test_sparse_witness_bottleneck.py` | Create | Focused tests: output shape `[B,2]`, mask selects only occupied squares, mask selects `min(K,n_occupied)`, empty invalid squares cannot be selected, eval deterministic top-k, unknown LC0 channel map raises. |
| `tests/test_model_registry_sparse_witness.py` | Create if registry tests exist | Confirms registry builder instantiates model from config and returns logits. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0713_tuesday_los_angeles_sparse_witness_bottleneck.md
  generated_at: 2026-04-21 07:13 America/Los_Angeles
  weekday: Tuesday
  timezone: America/Los_Angeles
  idea_slug: sparse_witness_bottleneck
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0713_sparse_witness_bottleneck
  name: Sparse Witness-Piece Bottleneck Network
  slug: sparse_witness_bottleneck
  status: draft
  created_at: 2026-04-21 07:13 America/Los_Angeles
  author: ChatGPT Pro
  short_thesis: Force puzzle classification through a hard top-k occupied-piece witness subset and test whether learned sparse witnesses beat matched random sparse witnesses.
  novelty_claim: Discrete occupied-piece minimum-description bottleneck over the current board; no move generation, attack/sheaf graph, Sinkhorn transport, or nuisance projection.
  expected_advantage: Reduces diffuse material/source shortcuts and encourages compact tactical evidence, with a sharp learned-vs-random witness falsifier.
  central_falsification_ablation: Random occupied top-k witness preserving per-position selected count and global planes.
  target_task: coarse_binary
  input_representation: simple_18 for first experiment; lc0_static_112/lc0_bt4_112 only with explicit fail-closed current-piece channel maps
  output_heads: binary logits only, shape [batch, 2]
  compute_notes: About 0.15M-0.25M parameters; occupied candidate set bounded by 32; no chunking needed for current experiment.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/sparse_witness_bottleneck_simple18.yaml
  model_path: src/chess_nn_playground/models/sparse_witness_bottleneck.py
  latest_result_path: null
  notes: Use hard binary masks downstream; do not pass continuous gate probabilities to classifier except in explicit leak-test ablation.
```

```yaml
config_yaml:
  run:
    name: sparse_witness_bottleneck_simple18
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
    name: sparse_witness_bottleneck
    input_channels: 18
    num_classes: 2
    encoding: simple_18
    witness_budget: 8
    selector_temperature: 1.0
    selector_temperature_min: 0.3
    selector_hard: true
    selector_valid_only_occupied: true
    encoder_width: 48
    encoder_blocks: 3
    dropout: 0.10
    use_flip_consistency: false
    flip_consistency_weight: 0.0
    adapter:
      type: simple_18
      piece_plane_indices: [0,1,2,3,4,5,6,7,8,9,10,11]
      global_plane_indices: [12,13,14,15,16,17]
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
  model_name: sparse_witness_bottleneck
  file_path: src/chess_nn_playground/models/sparse_witness_bottleneck.py
  builder_function: build_sparse_witness_bottleneck
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - EncodingAdapter
    - Simple18Adapter
    - LC0CurrentBoardAdapter
    - OccupiedPieceTopKSelector
    - WitnessGridEncoder
    - SparseWitnessBottleneckNet
  required_config_fields:
    - model.input_channels
    - model.num_classes
    - model.encoding
    - model.witness_budget
    - model.adapter.type
    - model.adapter.piece_plane_indices
    - model.adapter.global_plane_indices
  expected_parameter_count: 0.15M-0.25M for simple_18 width 48 with 3 residual blocks
  expected_memory_notes: Candidate set is occupied pieces only, at most B x 32 x d_token; witness grid about B x 19 x 8 x 8 for simple_18. No chunking needed unless future candidate pools exceed 32.
```

```yaml
research_continuity:
  idea_fingerprint: current-board occupied piece-square set + hard binary top-k sparse witness selector + masked-board classifier + learned-vs-random witness falsifier + no engine/move/attack/sheaf/transport/projection features
  already_researched_family_overlap: Low overlap with imported tactical sheaf/Hodge, move-delta, OT, and nuisance-projection packets; closest conceptual overlap is generic information bottleneck/rationale extraction, not any imported chess operator.
  closest_duplicate_risk: Could be mistaken for ordinary attention or saliency; avoid that by enforcing hard occupied-only top-k masks and ablations where the classifier cannot see the full board.
  do_not_repeat_if_this_fails:
    - hard top-k occupied-piece witness bottleneck as the central operator
    - sparse square/piece rationale classifier with fixed K and masked board input
    - learned selector versus random occupied top-k as the main claim
    - continuous gate/saliency maps presented as evidence without hard-mask falsification
  suggested_next_search_directions:
    - label-safe ordinal/evidential modeling of fine labels 0/1/2
    - cross-encoding or cross-phase invariant risk without source/provenance inputs
    - selective prediction calibrated for near-puzzle ambiguity
    - generative compression only if reconstruction is explicitly controlled against material/phase shortcuts
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Sparse Witness-Piece Bottleneck Network` to the imported research memory after implementation results are known. | Prevents future packets from repeating fixed-budget occupied-piece rationale selectors if this cycle fails or succeeds. | `Imported Research Memory` |
| Add anti-duplicate language for “hard top-k occupied-piece/square rationale bottlenecks with masked-board classifiers” unless the next proposal changes the formal object and falsifier. | This idea can be easily renamed as saliency, rationale, sparse attention, or witness selection; the prompt should block superficial repeats. | Anti-duplicate paragraph after nuisance-projection exclusions |
| Require future sparse-mask ideas to state whether the classifier sees continuous gate probabilities, binary masks, selected features only, or the full board. | Prevents selector steganography and attention-map-only proposals. | `What Counts As Creative Enough` or `Depth requirements` |
| Require count/material/source-square-marginal controls for any rule-generated candidate subset, not only move-set models. | The same shortcut risk exists for occupied-piece subsets. | `Ablation Plan` requirements |
| Record whether this experiment found a learned-vs-random witness gap, a useful `K` compression curve, and class-`1` recall gains. | The next research pass needs the empirical verdict, not just the idea name. | `Research Continuity` |
| Preserve all leakage, label, engine, and anti-duplicate constraints unchanged. | These rules are necessary for benchmark integrity. | No weakening; only additive edits |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0713_tuesday_los_angeles_sparse_witness_bottleneck.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes, `simple_18` on the current CRTK sample split
- Falsification criterion is concrete: yes, learned hard top-k witness must beat random occupied top-k and related controls
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
