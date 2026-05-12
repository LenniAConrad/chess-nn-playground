# Codex Handoff Packet: Ordinal Evidence Ladder Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0711_tuesday_los_angeles_ordinal_evidence_ladder.md`
- Generated at: 2026-04-21 07:11:58 America/Los_Angeles
- Weekday: Tuesday
- Timezone: los_angeles
- Idea slug: ordinal_evidence_ladder
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Ordinal Evidence Ladder Network
- One-sentence thesis: Treat `known non-puzzle -> verified near-puzzle -> verified puzzle` as an ordered ladder and force the classifier through two nested cumulative thresholds plus an evidential concentration score, so class `1` becomes a supervised middle-band ambiguity signal instead of being collapsed into an undifferentiated positive class.
- Idea fingerprint: `current-board tensor -> ordinary small board backbone -> scalar puzzle-potential bottleneck -> two monotone cumulative ordinal thresholds P(fine>=1), P(fine>=2) -> binary logits from P(fine>=1) + optional Dirichlet evidence for selective diagnostics; no engine metadata, no attack sheaf, no move-delta bag, no Sinkhorn/OT, no nuisance projection`.
- Why this is not a common CNN/ResNet/Transformer variant: the falsifiable operator is the constrained nested-survival head `q_2(x) <= q_1(x)` with all binary and fine-label predictions passing through a one-dimensional puzzle-potential score and two learned ordered thresholds; the convolutional trunk is deliberately baseline-sized and not the research claim.
- Current-data minimal experiment: train on `simple_18` using the existing `crtk_sample_3class` split, with binary cross-entropy for `fine>0`, cumulative ordinal BCE for `fine>=1` and `fine>=2`, and the same benchmark reports plus fine-label `3x2` diagnostics.
- Smallest central falsification ablation: replace the ordinal ladder head with an unconstrained same-parameter 3-class softmax auxiliary head whose binary probability is `p(fine=1)+p(fine=2)`; if it matches or beats the ladder on class-`1` recall/precision at matched fine-label-`0` false-positive rate, the ordered-middle-band hypothesis is not buying anything.
- Expected information gain if it fails: a failure would show that near-puzzle labels do not behave like an ordered margin between non-puzzle and puzzle under current encodings, so future work should avoid scalar ordinal/uncertainty heads and search for multi-factor or causal-invariance mechanisms instead.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification from board positions:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The binary target is `y_binary = 1[fine_label > 0]`. This packet uses fine labels only as supervised training targets and diagnostics, never as neural-network input features.

Allowed input tensors are current project encodings with shape:

```text
(batch, C, 8, 8)
```

The model must return logits with shape:

```text
(batch, 2)
```

The benchmark split must remain:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Current encodings:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant.
- `lc0_static_112`.
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists.

Leakage checklist:

- Safe: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board.
- Leakage-prone unless explicitly justified, engine-free, label-independent, and ablated: full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences.
- Never use as neural-network inputs: Stockfish or other engine evaluations, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, unresolved-pool status, dataset provenance, or any artifact derived from the verification pipeline.
- Fine label `1` and fine label `2` must not be fabricated. They may be used as existing supervised targets because they are part of the stated data contract, but they must not be inferred for unresolved positions.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry extraction is allowed only from explicitly documented current-board channels. History channels may be passed to a learned neural adapter, but handcrafted rule-derived features must fail closed when channel semantics are unknown.

## 4. Research Map

External anchors used for this idea:

1. CORAL ordinal regression: Wenzhi Cao, Vahid Mirjalili, Sebastian Raschka, “Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation,” arXiv:1901.07884, https://arxiv.org/abs/1901.07884. Borrowed: the view that ordered labels can be represented by cumulative binary subtasks with rank consistency. Not copied: age-estimation architecture, datasets, and the exact CORAL output-layer constraint.
2. CORN ordinal regression: Xintong Shi, Wenzhi Cao, Sebastian Raschka, “Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities,” arXiv:2111.08851, https://arxiv.org/abs/2111.08851. Borrowed: the warning that ordinal heads should avoid inconsistent rank probabilities. Not copied: conditional training-set construction.
3. Evidential deep learning: Murat Sensoy, Lance Kaplan, Melih Kandemir, “Evidential Deep Learning to Quantify Classification Uncertainty,” arXiv:1806.01768, https://arxiv.org/abs/1806.01768. Borrowed: a Dirichlet evidence interpretation for predicted class probabilities. Not copied: any claim of OOD detection success for chess or any OOD benchmark protocol.
4. Selective classification: Yonatan Geifman, Ran El-Yaniv, “Selective Classification for Deep Neural Networks,” arXiv:1705.08500, https://arxiv.org/abs/1705.08500. Borrowed: the coverage-risk framing and the idea that uncertainty can be evaluated by abstention curves. Not copied: their post-hoc risk guarantee as a central training method.
5. Variational information bottleneck: Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, Kevin Murphy, “Deep Variational Information Bottleneck,” arXiv:1612.00410, https://arxiv.org/abs/1612.00410. Borrowed: the principle that useful classifiers may benefit from compression of nuisance information. Not copied: stochastic VIB sampling or a KL-to-Gaussian latent objective.
6. Risk extrapolation / invariant risk: David Krueger et al., “Out-of-Distribution Generalization via Risk Extrapolation,” arXiv:2003.00688, https://arxiv.org/abs/2003.00688. Borrowed only for a rejected candidate involving encoding-family invariance. Not copied into the selected architecture.

Candidate search trace, including serious candidates not selected:

1. Cross-encoding invariant risk bottleneck: train synchronized `simple_18` and `lc0_bt4_112` views with a V-REx-style penalty so only encoding-stable signals survive. Rejected for this cycle because it requires a more invasive multi-view dataloader and could conflate encoding invariance with exporter bugs; it is a good next direction if the ordinal idea fails.
2. Masked board minimum-description-length motif model: pretrain a masked generative compressor and classify from reconstruction-free compressed latents. Rejected because the unsupervised objective may learn common opening/material statistics rather than puzzle-likeness, and a 3-epoch current-data test would be underpowered.
3. Color-swap / perspective-equivariant contrastive model: enforce consistency under legal color inversion and rank reflection. Rejected because it is useful hygiene but too close to ordinary augmentation unless paired with a stronger causal objective.
4. Sparse neural motif dictionary over board patches: learn a small dictionary of motifs and classify from sparse activations. Rejected because it risks becoming a renamed CNN filter bank and has weak falsification unless motifs are manually constrained.
5. Energy-based latent class model for `0/1/2`: fit an energy distribution over fine labels with calibrated free energy. Rejected because the selected cumulative-threshold model gives a simpler mathematical falsifier and integrates with the binary trainer more cleanly.
6. Selective-only conformal wrapper: keep the baseline model and add abstention calibration on validation data. Rejected because it is diagnostic, not a new inductive bias for learning puzzle structure.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN with binary cross-entropy | `src/chess_nn_playground/models/trunk/cnn.py` | Already present and collapses fine labels `1` and `2` into one positive class. |
| Residual CNN with binary cross-entropy | `src/chess_nn_playground/models/trunk/residual_cnn.py` | Already present; depth changes do not test a new hypothesis about near-puzzle ambiguity. |
| LC0-style CNN or residual CNN | Existing LC0 BT4-style CNN/residual variants | Already covered as an encoding/backbone family; copying LC0 does not explain the `0/1/2` diagnostic structure. |
| Ordinary ViT over 64 squares | No exact baseline, but common square-token model | Too generic and excluded by the prompt; it changes capacity and tokenization, not the target structure. |
| Plain GNN on square adjacency | Would resemble a generic board graph model | Too ordinary and not clearly different from a CNN over the 8x8 grid unless it introduces a distinct operator. |
| Hyperparameter tuning | All existing baselines | Excluded by prompt and unlikely to teach whether class `1` is an ordered middle state. |
| Ensembling multiple baselines | Any existing model ensemble | Excluded by prompt; improves variance but does not produce a falsifiable mechanism. |
| More data from the 45M-row Parquet | Existing trainer without streaming | Excluded as the core idea and currently unsafe until streaming support exists. |
| Static attack-defense graph/sheaf/Hodge model | Imported tactical sheaf/Hodge packets | Already researched family; adding edge types or changing pooling would be a duplicate. |
| One-ply move-delta bag or move-landscape model | Imported counterfactual move-delta packets | Already researched family and specifically disallowed unless the operator is genuinely different. |
| Piece-target Sinkhorn / transport bottleneck | Imported optimal-transport packets | Already researched family; temperature or target-bucket variations would be duplicates. |
| Deterministic nuisance residualization | Imported nuisance-orthogonal packet | Already researched; closed-form projection away from material/phase/king features is not this idea. |
| Post-hoc calibration only | Could wrap any existing baseline | Useful reporting, but not a model-side inductive bias. |
| Label smoothing only | Any classifier | Too generic; it does not use the ordinal meaning of `0 < 1 < 2`. |

## 6. Mathematical Thesis

### Input and targets

Let

```text
X_C = R^{C x 8 x 8}
```

be the space of encoded current-board tensors for a fixed encoding with `C` channels. The data distribution is a distribution `D` over `(X, Y)` where:

```text
Y in {0, 1, 2}
B = 1[Y > 0] in {0, 1}
```

Here `Y=0` is known non-puzzle, `Y=1` is verified near-puzzle, and `Y=2` is verified puzzle. The binary benchmark target is `B`.

### Distribution assumptions

The core assumption is not that all puzzle-likeness is one-dimensional. The weaker and testable assumption is:

> The fine labels contain an ordinal component: for many positions, near-puzzles occupy a middle band between obvious non-puzzles and verified puzzles along some learned puzzle-potential statistic.

There may be multiple tactical motifs, but the selected model assumes that the final decision boundary can benefit from compressing those motifs into a scalar potential before deciding `Y>=1` and `Y>=2`.

### Symmetry assumptions

Chess is not invariant to arbitrary rotations or reflections because pawn direction, castling, en-passant, and side-to-move matter. A safe optional symmetry is color-perspective inversion: swap piece colors, mirror ranks, swap castling rights, transform en-passant consistently, and toggle side-to-move. The label should be invariant under that exact rule-preserving transformation. This packet does not require that augmentation in the minimal experiment.

### Formal object: the ordinal evidence ladder

Let a backbone produce:

```text
h_theta(x) in R^d
```

The head computes a scalar puzzle potential and an evidence concentration:

```text
s_theta(x) = w_s^T h_theta(x) + b_s
kappa_theta(x) = kappa_min + softplus(w_k^T h_theta(x) + b_k)
```

Learn two ordered thresholds by parameterizing:

```text
delta = softplus(a_delta) + epsilon
center = a_center
tau_0 = center - delta / 2
tau_1 = center + delta / 2
rho = softplus(a_rho) + epsilon
```

Define cumulative survival probabilities:

```text
q_1(x) = P_theta(Y >= 1 | x) = sigmoid(rho * (s_theta(x) - tau_0))
q_2(x) = P_theta(Y >= 2 | x) = sigmoid(rho * (s_theta(x) - tau_1))
```

Because `tau_0 < tau_1` and `rho > 0`, `q_2(x) <= q_1(x)` for every input. Convert to fine-label probabilities:

```text
p_0(x) = 1 - q_1(x)
p_1(x) = q_1(x) - q_2(x)
p_2(x) = q_2(x)
```

The binary puzzle-like probability is exactly:

```text
P_theta(B=1 | x) = q_1(x)
```

The model returns binary logits compatible with the shared trainer:

```text
logits_binary(x) = [0, logit(q_1(x))]
```

For evidential diagnostics, define a Dirichlet distribution over fine-label probabilities:

```text
alpha_j(x) = 1 + kappa_theta(x) * p_j(x),  j in {0,1,2}
```

The total evidence is `S(x)=sum_j alpha_j(x)`, and vacuity can be reported as `3 / S(x)`.

### Optimization objective

Use the supervised cumulative ordinal loss:

```text
L_ord = BCE(logit(q_1), 1[Y >= 1]) + lambda_2 * BCE(logit(q_2), 1[Y >= 2])
```

Optionally add fine-label negative log-likelihood:

```text
L_fine = -log p_Y(x)
```

Optionally add the evidential expected cross-entropy from the Dirichlet parameters:

```text
L_evid = sum_j onehot(Y)_j * (psi(S) - psi(alpha_j))
```

The main current-data experiment should optimize:

```text
E_D[ L_bin + lambda_ord * L_ord + lambda_fine * L_fine + lambda_evid * L_evid ]
```

where `L_bin` is ordinary binary cross-entropy on `B`. Since `L_bin` and the first term of `L_ord` share `q_1`, Codex may set `lambda_ord=1` and `L_bin` weight small or use `L_bin` only for compatibility reporting. Do not double-count class weights accidentally.

### Proposition

For any input `x`, the ordinal ladder defines a valid probability vector `p(x) in Delta^2`. Moreover, if the true conditional cumulative probabilities satisfy the scalar-threshold realizability condition

```text
P(Y >= j | X=x) = sigmoid(rho* * (s*(x) - tau*_j)),  j in {1,2}, tau*_0 < tau*_1,
```

then minimizing expected cumulative binary cross-entropy over a rich enough backbone and the ladder head recovers the Bayes-optimal cumulative probabilities, and the binary decision rule based on `q_1` is Bayes-optimal for the coarse target under the usual threshold for the chosen cost ratio.

### Proof sketch

1. Since `tau_0 < tau_1` and `rho > 0`, `rho(s-tau_0) >= rho(s-tau_1)`. The sigmoid is monotone, so `q_1 >= q_2`.
2. Therefore `p_0=1-q_1 >= 0`, `p_1=q_1-q_2 >= 0`, `p_2=q_2 >= 0`, and `p_0+p_1+p_2=1`.
3. Binary cross-entropy is a strictly proper scoring rule for each nested event `A_j={Y>=j}`. For unrestricted measurable `q_j`, the expected BCE for event `A_j` is minimized at `q_j(x)=P(A_j|X=x)`.
4. The events are nested, so the true cumulative probabilities obey `P(Y>=2|x) <= P(Y>=1|x)`. If they are representable by the shared scalar ladder, the constrained optimum matches the true cumulative probabilities.
5. The binary benchmark target is exactly `A_1`, so the recovered `q_1` gives the Bayes-optimal binary posterior under the realizability assumption.

### What is actually proven

- The ladder output is always a valid ordered distribution over fine labels.
- The cumulative BCE objective is proper for the nested events.
- Under the explicit scalar-threshold realizability condition, the learned cumulative probabilities can represent the Bayes posterior and therefore the binary posterior.

### What remains hypothesized

- Chess puzzle-likeness in this dataset has a strong scalar ordinal component.
- Fine label `1` is mostly an intermediate tactical state rather than a source artifact or noisy mixture.
- Evidential concentration correlates with genuine ambiguity rather than class imbalance or backbone uncertainty shortcuts.
- Better class-`1` behavior will improve useful puzzle-likeness classification rather than merely reshuffling positives.

### Counterexamples where the idea should fail

- Fine label `1` is not between `0` and `2`; it is a separate source-specific bucket with unrelated visual statistics.
- Verified puzzles contain several disconnected tactical modes that require non-monotone fine-label probabilities in any one-dimensional score.
- Class `2` differs from class `1` mainly by engine verification depth or metadata not visible in the board tensor.
- The binary task is best solved by material/opening/source artifacts that are easier than ordinal tactical structure.
- The dataset has severe duplicate or near-duplicate leakage across splits; then any architecture can look good for the wrong reason.

### Self-critique

The strongest objection is that this may be “just a better loss head” on top of a CNN. That objection is fair, so the experiment must not claim success from raw binary accuracy alone. The idea survives only if the constrained ladder specifically improves the fine-label `3x2` behavior, especially class-`1` recall/precision at matched fine-label-`0` false-positive rate, and beats an unconstrained same-backbone 3-class softmax auxiliary head. Another risk is that the scalar bottleneck underfits real chess tactics; that is acceptable because underfitting would falsify the “near-puzzle as ordered middle band” hypothesis quickly and cheaply.

## 7. Architecture Specification

### Module names

- Model file: `src/chess_nn_playground/models/trunk/ordinal_evidence_ladder.py`
- Main module: `OrdinalEvidenceLadderNet`
- Submodules:
  - `EncodingSafeStem`
  - `TinyBoardBackbone`
  - `OrdinalLadderHead`
  - optional loss helper in idea folder: `OrdinalEvidenceLoss`

### Forward-pass steps and tensor shapes

Assume input `x` has shape `[B, C, 8, 8]`.

1. `EncodingSafeStem`
   - Validate `C == config.model.input_channels`.
   - Apply `1x1 Conv(C -> 32)`, normalization, activation.
   - Output shape: `[B, 32, 8, 8]`.
2. `TinyBoardBackbone`
   - `3x3 Conv(32 -> 64)`, normalization, activation.
   - Output shape: `[B, 64, 8, 8]`.
   - Two residual blocks at width `64`, each using two `3x3` convolutions.
   - Output shape: `[B, 64, 8, 8]`.
   - `3x3 Conv(64 -> 96)`, normalization, activation.
   - Output shape: `[B, 96, 8, 8]`.
   - Global average pooling over board squares.
   - Output shape: `[B, 96]`.
3. `OrdinalLadderHead`
   - Linear score: `[B, 96] -> [B, 1]`, yielding `s`.
   - Linear evidence: `[B, 96] -> [B, 1]`, yielding `kappa` after `softplus`.
   - Learned global ordered thresholds `tau_0 < tau_1` and positive slope `rho`.
   - Cumulative logits:
     - `ell_1 = rho * (s - tau_0)`, shape `[B, 1]`.
     - `ell_2 = rho * (s - tau_1)`, shape `[B, 1]`.
   - Cumulative probabilities: `q_1=sigmoid(ell_1)`, `q_2=sigmoid(ell_2)`, both `[B, 1]`.
   - Fine probabilities: `p_fine = concat(1-q_1, q_1-q_2, q_2)`, shape `[B, 3]`.
   - Evidence parameters: `alpha = 1 + kappa * p_fine`, shape `[B, 3]`.
   - Binary logits returned to shared trainer: `logits = concat(zeros_like(ell_1), ell_1)`, shape `[B, 2]`.

### Pseudocode

```text
forward(x, return_aux=False):
    z = stem(x)
    h = backbone(z)
    s = score_head(h)
    kappa = kappa_min + softplus(kappa_head(h))
    gap = softplus(raw_gap) + eps
    tau0 = center - 0.5 * gap
    tau1 = center + 0.5 * gap
    rho = softplus(raw_slope) + eps
    ell1 = rho * (s - tau0)
    ell2 = rho * (s - tau1)
    q1 = sigmoid(ell1)
    q2 = sigmoid(ell2)
    p_fine = [1 - q1, q1 - q2, q2]
    logits = [0, ell1]
    if not return_aux:
        return logits
    alpha = 1 + kappa * p_fine
    return {
        "logits": logits,
        "ordinal_logits": [ell1, ell2],
        "fine_probs": p_fine,
        "alpha": alpha,
        "score": s,
        "concentration": kappa,
        "thresholds": [tau0, tau1],
    }
```

This is pseudocode only; Codex should implement idiomatic PyTorch.

### Parameter-count estimate

For `simple_18`:

- `1x1 C->32`: about `18*32 = 576` weights plus bias/norm.
- `3x3 32->64`: about `18,432` weights.
- two residual blocks at width `64`: four `3x3 64->64` convolutions, about `147,456` weights.
- `3x3 64->96`: about `55,296` weights.
- heads and normalization: roughly `2,000` to `10,000` parameters depending on normalization choices.

Expected total: about `0.23M` to `0.27M` parameters. For `112` input channels, add only about `3,008` extra weights in the first projection, so expected total remains under `0.30M`.

### FLOP / complexity estimate

Per sample, the convolutional body costs roughly:

```text
O(8*8*(C*32 + 9*32*64 + 4*9*64*64 + 9*64*96))
```

For `C=18`, this is approximately `14M` multiply-adds per position. The ordinal/evidence head is negligible. Runtime should be close to or cheaper than many existing residual CNN variants.

### Candidate-set memory

This architecture generates no move set, candidate set, graph edge list, transport plan, or search surrogate. Memory is dominated by feature maps:

```text
O(B * 96 * 8 * 8)
```

The largest activation listed above is `[B, 96, 8, 8]`, about `6,144*B` floats before autograd storage. No chunking plan is needed.

### Required config fields

- `model.name: ordinal_evidence_ladder`
- `model.input_channels`
- `model.num_classes: 2`
- `model.backbone_width: 64`
- `model.embedding_dim: 96`
- `model.kappa_min: 2.0`
- `model.threshold_gap_init: 1.0`
- `model.slope_init: 1.0`
- `loss.binary_weight`
- `loss.ordinal_weight`
- `loss.fine_nll_weight`
- `loss.evidential_weight`
- `loss.class_weighting: balanced`
- `data.encoding`

### Encoding support

First experiment should use `simple_18` because its channel semantics are clear and it keeps the first test focused on the ordinal hypothesis.

Support plan:

- `simple_18`: fully supported. The adapter validates `input_channels=18` and treats all channels as learned tensor inputs. No extra handcrafted features are required.
- `lc0_static_112`: supported only as an opaque learned tensor when `input_channels=112`; deterministic geometry extraction must fail closed unless a documented current-board channel map is available.
- `lc0_bt4_112`: supported only as an opaque learned tensor when `input_channels=112`; zero-filled unavailable history channels are allowed to pass through the learned stem, but handcrafted geometry must not be derived from history planes.

The model returns ordinary binary logits, so existing reports, predictions, confusion matrices, and leaderboards should keep working. The idea-specific `train.py` may call `forward(return_aux=True)` to compute ordinal/evidential losses, but the default `forward(x)` must return only `[B, 2]` logits.

## 8. Loss, Training, And Regularization

### Primary loss

Use binary cross-entropy through two-class logits for the benchmark target:

```text
B = 1[fine_label > 0]
```

Class weighting should be balanced using the train split only.

### Auxiliary losses

Use existing fine labels as targets:

```text
t1 = 1[fine_label >= 1]
t2 = 1[fine_label >= 2]
```

Recommended loss:

```text
L = binary_weight * CE_binary(logits, B)
  + ordinal_weight * (BCE(ell1, t1) + lambda2 * BCE(ell2, t2))
  + fine_nll_weight * NLL(p_fine, fine_label)
  + evidential_weight * EvidentialExpectedCE(alpha, fine_label)
```

Initial weights:

```text
binary_weight: 0.5
ordinal_weight: 1.0
lambda2: 1.0
fine_nll_weight: 0.25
evidential_weight: 0.01
```

The first term and `BCE(ell1,t1)` duplicate the same event, so Codex should avoid excessive weighting. The duplication is intentional only to maintain direct binary-loss compatibility; ablations should report whether removing `binary_weight` changes results.

### Class weighting

- Binary CE: balanced weights for `B=0` and `B=1`.
- Cumulative BCE for `t2`: balanced positive/negative weights because verified puzzles may be rarer than non-puzzles plus near-puzzles.
- Fine NLL: optional inverse-frequency weights clipped to avoid unstable rare-class gradients.

### Batch size expectations

Start with batch size `512` on `simple_18`. If GPU memory is limited, reduce to `256`; do not change the split, labels, or epoch budget when comparing ablations.

### Optimizer defaults

- Optimizer: AdamW.
- Learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Epochs: `3` for the minimal current-data experiment.
- Early stopping patience: `2`.
- Mixed precision: off for first deterministic run.

### Regularizers

- Weight decay as above.
- Clamp probabilities for `log` operations with `eps=1e-6`.
- Threshold gap lower bound `epsilon=1e-4`.
- Optional mild penalty on extreme slope: `1e-4 * rho^2` if training becomes numerically sharp.
- No dropout in the first pass unless existing baselines use it in the matched configuration.

### Determinism requirements

Use seed `42`, deterministic PyTorch settings when available, deterministic dataloader order for evaluation, and persisted train/val/test predictions. Do not use stochastic data augmentation in the main comparison unless the same augmentation is applied to every baseline and ablation.

### Fair-comparison invariants

Keep unchanged from existing benchmark configs:

- train/validation/test split paths
- binary benchmark mode
- report format and confusion matrices
- prediction artifact schema where possible
- epoch budget for first pass
- optimizer family and learning-rate scale unless baseline configs differ for documented reasons
- encoding for a matched comparison, starting with `simple_18`

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Same-backbone unconstrained 3-class softmax | Replace ladder with a 3-logit fine-label head; binary probability is `p1+p2` | The nested scalar ordinal structure matters beyond merely using fine labels | If this matches or beats OEL on class-`1` diagnostics, abandon the scalar ladder claim. |
| Binary-only head | Remove `ell2`, fine NLL, and evidential losses; train same backbone on `fine>0` only | Fine-label ordinal supervision improves binary puzzle-likeness | If equal, class `1/2` supervision is not helping under current data. |
| Ordinal order permutation | Keep binary labels fixed but swap fine labels `1` and `2` for the `ell2`/fine auxiliary target | The semantic order `0 < 1 < 2` matters | If performance is unchanged, the ladder is exploiting class counts or regularization rather than order. |
| No scalar bottleneck | Predict `ell1` and `ell2` from two independent linear heads with a soft monotonic penalty instead of shared `s` and thresholds | A single puzzle-potential axis is useful | If independent heads win strongly, puzzle-likeness may be multi-axis. |
| No evidential concentration | Keep ordinal ladder but remove `kappa`, `alpha`, and evidential loss | Evidence helps calibration/selective diagnostics, not core classification | If classification is same but ECE/coverage improves without evidence, the evidence head is noise. |
| Fixed thresholds | Set `tau0=-0.5`, `tau1=0.5`, learn only score and slope | Learned separation between near-puzzle and puzzle matters | If fixed thresholds match, the adaptive threshold parameters are unnecessary. |
| Material-only nuisance control | Train a tiny MLP on deterministic material counts, side-to-move, castling, and en-passant only | OEL is not merely exploiting material/side shortcuts | If material-only approaches OEL, the dataset has severe superficial shortcuts. |
| Frozen random backbone + trained ladder | Freeze random convolutional trunk, train only ladder head | Learned board features are required | If competitive, the labels are likely dominated by simple input-channel statistics or leakage. |
| Class-1-as-positive hardening | Treat labels `1` and `2` identically everywhere, but keep the same parameter count | The middle-band distinction improves near-puzzle behavior | If equal on class-`1` matched-FPR metrics, the fine ordinal split is not useful. |
| Calibration-only post-hoc temperature | Train binary baseline and temperature-scale on validation | OEL gains are not just better calibration | If temperature scaling closes the gap on near-puzzle diagnostics, prefer simpler calibration. |

There is no generated move set or candidate set. Therefore count-only, source-square marginal, moving-piece identity, and capture-histogram ablations are not applicable. The order-permutation ablation is the semantics-destroying ablation for this label-structure operator.

## 10. Benchmark And Falsification Criteria

### Baselines to compare against

Use the strongest available matched-encoding baselines already in the repo:

- simple CNN on `simple_18`
- residual CNN on `simple_18`
- if feasible after the minimal pass: LC0 BT4-style CNN/residual on `lc0_bt4_112`
- same-backbone binary-only ablation
- same-backbone unconstrained 3-class softmax auxiliary ablation

### Metrics to inspect

- binary accuracy
- binary balanced accuracy
- binary AUROC
- binary AUPRC
- binary F1
- calibration: ECE and Brier score
- fine-label rectangular `3x2` confusion matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

- class-`1` recall at a matched fine-label-`0` false-positive rate
- class-`1` precision at the same operating threshold
- class-`2` recall to verify that better class-`1` behavior does not sacrifice verified puzzles
- optional selective curves: risk versus coverage using `vacuity = 3/sum(alpha)` or max binary probability

### Required fine-label diagnostic

For the main model and every central ablation, Codex must report the `3x2` matrix on validation and test. Use the same thresholding rule for all models. In addition, compute a threshold chosen on validation to match the best residual CNN's fine-label-`0` false-positive rate, then evaluate class-`1` recall and precision at that threshold on test.

### Required artifacts

- trained checkpoint
- config snapshot
- metrics JSON
- validation and test predictions Parquet or CSV including:
  - `fine_label`
  - `binary_label`
  - `binary_prob`
  - `binary_pred`
  - `q_ge1`
  - `q_ge2`
  - `p_fine_0`, `p_fine_1`, `p_fine_2`
  - `evidence_concentration`
  - `vacuity`
- confusion matrices for main and ablations
- report Markdown with success/failure decision

### Success threshold

Call the idea successful for this cycle if, on the test split:

1. OEL improves class-`1` recall by at least `+2.0` percentage points at matched fine-label-`0` false-positive rate versus the strongest matched `simple_18` residual CNN, and
2. binary AUROC is not worse by more than `0.005`, and
3. class-`2` recall is not worse by more than `1.0` percentage point, and
4. the same-backbone unconstrained 3-class softmax ablation does not match the class-`1` gain.

### Failure threshold

Call the idea failed if any of these occur:

- OEL is within `±0.5` percentage points of the binary-only same-backbone ablation on class-`1` recall at matched fine-label-`0` FPR.
- The unconstrained 3-class softmax ablation is equal or better on the main near-puzzle diagnostic.
- The order-permutation ablation is equal or better, suggesting the ordinal semantics do not matter.
- OEL reduces class-`2` recall by more than `2.0` percentage points while only improving class `1`.

### Result that would make me abandon the idea

Abandon scalar ordinal/evidential ladder heads if the order-permutation or unconstrained-softmax ablation matches OEL while binary metrics remain similar. That would mean the near-puzzle label is not behaving like an ordered middle band in the current data.

### Result that would justify scaling

Scale to `lc0_bt4_112` and longer training only if the `simple_18` experiment shows a clear class-`1` matched-FPR improvement and the central ablations support the ordinal semantics. Scaling without that ablation win would just be capacity chasing.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_ordinal_evidence_ladder/idea.yaml` | Create | Machine-readable idea metadata from the `idea_yaml` block below. |
| `ideas/20260421_ordinal_evidence_ladder/math_thesis.md` | Create | Section 6 mathematical thesis, proposition, proof sketch, counterexamples, and self-critique. |
| `ideas/20260421_ordinal_evidence_ladder/architecture.md` | Create | Section 7 architecture, shapes, pseudocode, parameter count, encoding-adapter rules. |
| `ideas/20260421_ordinal_evidence_ladder/implementation_notes.md` | Create | Notes on safe fine-label target use, fail-closed encoding adapters, no engine inputs, no generated move sets. |
| `ideas/20260421_ordinal_evidence_ladder/trainer_notes.md` | Create | Loss composition, class weighting, deterministic settings, expected artifacts. |
| `ideas/20260421_ordinal_evidence_ladder/ablations.md` | Create | Full ablation table and exact falsification criteria. |
| `ideas/20260421_ordinal_evidence_ladder/train.py` | Create | Thin idea-specific training entrypoint that loads existing splits, requests `forward(return_aux=True)`, computes auxiliary losses, and emits standard report artifacts. |
| `ideas/20260421_ordinal_evidence_ladder/config.yaml` | Create | Config from the `config_yaml` block below. |
| `ideas/20260421_ordinal_evidence_ladder/report_template.md` | Create | Template requiring binary metrics, `3x2` fine-label confusion, matched-FPR class-`1` diagnostics, ablation comparison, and abandon/scale decision. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Preserve hard constraints and add an anti-duplicate note for scalar ordinal/evidential ladder heads after this packet is consumed. Also clarify whether future packets may use existing fine labels as auxiliary targets. |
| `src/chess_nn_playground/models/trunk/ordinal_evidence_ladder.py` | Create | Implement `OrdinalEvidenceLadderNet`, `EncodingSafeStem`, `TinyBoardBackbone`, `OrdinalLadderHead`; default `forward(x)` returns `[B,2]`; optional `return_aux=True` returns auxiliary dictionary. |
| `src/chess_nn_playground/models/registry.py` | Update | Register builder function `build_ordinal_evidence_ladder` or equivalent model name `ordinal_evidence_ladder`. |
| `configs/ordinal_evidence_ladder_simple18.yaml` | Create | Shared-trainer-compatible config with `encoding: simple_18`, `input_channels: 18`, model name, seed, batch size, and benchmark paths. |
| `tests/test_ordinal_evidence_ladder.py` | Create | Focused tests: output shape `[B,2]`, `q_ge2 <= q_ge1`, fine probabilities nonnegative and sum to 1, thresholds ordered, default forward compatible with trainer, adapter rejects mismatched channel count. |
| `tests/test_ordinal_evidence_loss.py` | Create if loss helpers are outside `train.py` | Check cumulative targets for fine labels `0/1/2`, finite loss values, and deterministic behavior on fixed tensors. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0711_tuesday_los_angeles_ordinal_evidence_ladder.md
  generated_at: "2026-04-21 07:11:58 America/Los_Angeles"
  weekday: Tuesday
  timezone: los_angeles
  idea_slug: ordinal_evidence_ladder
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_ordinal_evidence_ladder
  name: Ordinal Evidence Ladder Network
  slug: ordinal_evidence_ladder
  status: draft
  created_at: "2026-04-21 07:11:58 America/Los_Angeles"
  author: ChatGPT Pro
  short_thesis: Treat fine labels 0<1<2 as a supervised ordinal ladder and classify puzzle-likeness through nested cumulative thresholds plus evidence concentration.
  novelty_claim: A scalar puzzle-potential bottleneck with ordered thresholds tests whether verified near-puzzles are a middle band rather than just positives collapsed into binary labels.
  expected_advantage: Better near-puzzle recall/precision at matched non-puzzle false-positive rate, with improved calibration and no engine-derived inputs.
  central_falsification_ablation: Same-backbone unconstrained 3-class softmax auxiliary head with binary probability p1+p2.
  target_task: coarse_binary
  input_representation: simple_18 first; opaque learned 112-channel adapter optional after simple_18 succeeds
  output_heads: binary logits plus optional ordinal/evidential auxiliary outputs during training
  compute_notes: About 0.25M parameters and roughly 14M multiply-adds per simple_18 position; no candidate-set memory.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/ordinal_evidence_ladder_simple18.yaml
  model_path: src/chess_nn_playground/models/trunk/ordinal_evidence_ladder.py
  latest_result_path: null
  notes: Must report fine-label 3x2 confusion and matched-FPR class-1 diagnostics for main model and central ablations.
```

```yaml
config_yaml:
  run:
    name: ordinal_evidence_ladder_simple18
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
    name: ordinal_evidence_ladder
    input_channels: 18
    num_classes: 2
    backbone_width: 64
    embedding_dim: 96
    kappa_min: 2.0
    threshold_gap_init: 1.0
    slope_init: 1.0
  loss:
    binary_weight: 0.5
    ordinal_weight: 1.0
    lambda2: 1.0
    fine_nll_weight: 0.25
    evidential_weight: 0.01
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
  model_name: ordinal_evidence_ladder
  file_path: src/chess_nn_playground/models/trunk/ordinal_evidence_ladder.py
  builder_function: build_ordinal_evidence_ladder
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - EncodingSafeStem
    - TinyBoardBackbone
    - OrdinalLadderHead
    - OrdinalEvidenceLadderNet
  required_config_fields:
    - model.name
    - model.input_channels
    - model.num_classes
    - model.backbone_width
    - model.embedding_dim
    - model.kappa_min
    - model.threshold_gap_init
    - model.slope_init
  expected_parameter_count: 0.23M-0.30M depending on input channels and normalization
  expected_memory_notes: No generated candidates; largest forward activation is approximately B*96*8*8 floats before autograd overhead.
```

```yaml
research_continuity:
  idea_fingerprint: current-board tensor + small board backbone + scalar puzzle-potential bottleneck + two ordered cumulative thresholds + optional Dirichlet evidence + binary logits from P(fine>=1)
  already_researched_family_overlap: Does not use imported sheaf/Hodge, one-ply move-delta, Sinkhorn/OT, or deterministic nuisance-projection families.
  closest_duplicate_risk: CORAL/CORN-style ordinal regression head on a CNN, extended here with chess-specific fine-label semantics and evidence/selective diagnostics.
  do_not_repeat_if_this_fails:
    - scalar cumulative ordinal threshold ladder over fine labels 0<1<2
    - class-1-as-middle-band ambiguity target for puzzle-likeness
    - Dirichlet/evidential concentration attached to ordinal fine probabilities as the central novelty
    - near-puzzle selective diagnostics as the only claimed contribution without a new learning mechanism
  suggested_next_search_directions:
    - cross-encoding causal invariance using synchronized simple_18 and lc0 views with risk-variance penalties
    - masked generative compression or MDL motif models with strong source-artifact controls
    - multi-axis latent ordinal models only if scalar ladder fails but unordered 3-class auxiliary helps
    - conformal/selective wrappers as evaluation tools, not standalone research mechanisms
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add this packet to the imported research memory as `Ordinal Evidence Ladder Network`. | Prevents the next research pass from repeating scalar cumulative ordinal thresholds with evidential/selective wrapping. | Imported Research Memory |
| Clarify that existing fine labels `0/1/2` may be used as supervised targets and diagnostics, but never as input features or fabricated labels. | Future agents may otherwise avoid useful label-safe ordinal ideas or accidentally misuse labels as features. | Project Context You Must Respect / Non-Negotiable Constraints |
| Add a requirement that any future uncertainty/selective idea specify its training signal, uncertainty score, and matched-FPR diagnostic. | Avoids vague “better calibration” proposals without falsification. | Required Markdown File Content / Benchmark And Falsification Criteria |
| Add an anti-duplicate rule: do not repeat scalar `0<1<2` cumulative ordinal heads, CORAL/CORN-style label ladders, or Dirichlet evidence heads unless the operator changes substantially. | Makes this idea part of continuity whether it succeeds or fails. | Imported Research Memory / anti-duplicate paragraph |
| If Codex finds that class `1` is not present or not available in train batches, require future prompts to state the exact parquet label columns. | The idea depends on safe use of real fine labels; missing columns would change the feasible research space. | Project Context You Must Respect |
| Add matched fine-label-`0` false-positive-rate reporting as a default near-puzzle diagnostic for future packets. | It compares near-puzzle behavior without letting models win by simply predicting more positives. | Benchmark requirements |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
