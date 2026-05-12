# Codex Handoff Packet: Nuisance-Orthogonal Puzzle Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0508_tuesday_local_nuisance_orthogonal.md`
- Generated at: 2026-04-21 05:08:09 America/Los_Angeles
- Weekday: Tuesday
- Timezone: local, America/Los_Angeles
- Idea slug: `nuisance_orthogonal`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Nuisance-Orthogonal Puzzle Bottleneck, abbreviated `NOPB`.
- One-sentence thesis: A chess puzzle-like position should remain recognizable after the model's latent representation is explicitly projected away from deterministic material/phase/side-to-move nuisance directions, because tactical interest is partly a residual structural property rather than merely a material-profile shortcut.
- Idea fingerprint: `current-board tensor -> deterministic material/phase/king/castling/ep nuisance vector -> fixed normalized nuisance feature matrix -> batchwise ridge orthogonal projection of CNN latent off nuisance span -> binary puzzle-likeness logits; no engine metadata, no source metadata, no move-delta bag, no attack/sheaf/Hodge operator`.
- Why this is not a common CNN/ResNet/Transformer variant: the central operator is a nonparametric batchwise residualization map `(I - P_Q)H`, where `Q` is a fixed rule-derived nuisance design matrix and `H` is the learned latent matrix; changing depth, width, attention, or residual blocks without this projection destroys the proposed mechanism.
- Current-data minimal experiment: train `NOPB` on `simple_18` with the existing `crtk_sample_3class` train/val/test parquet split, using binary coarse labels where fine label `0 -> 0` and fine labels `1,2 -> 1`, then report ordinary binary metrics plus the required fine-label `3x2` diagnostic matrix.
- Smallest central falsification ablation: set projection strength `gamma=0` while leaving the trunk, parameter count, optimizer, batch size, deterministic nuisance extraction, and classifier head unchanged.
- Expected information gain if it fails: failure would show that material/phase/king-side nuisance correlations are not the main shortcut limiting this benchmark, or that puzzle-likeness signal is itself too entangled with those variables for hard residualization to help.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The task is binary chess puzzle-likeness classification from a single board tensor:

- output `0`: non-puzzle;
- output `1`: puzzle-like.

The available fine labels are:

- fine label `0`: known non-puzzle;
- fine label `1`: verified near-puzzle;
- fine label `2`: verified puzzle.

For the default coarse binary benchmark, use the existing project mapping. If Codex needs the mapping explicitly for this idea, use `fine_label == 0 -> binary 0` and `fine_label in {1, 2} -> binary 1`. Do not invent or relabel fine labels.

Available encodings:

- `simple_18`: 12 piece planes plus side-to-move, castling, and en-passant planes;
- `lc0_static_112`;
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists.

The implementation target is PyTorch. The model must accept a tensor shaped:

```text
(batch, C, 8, 8)
```

and return logits shaped:

```text
(batch, 2)
```

The benchmark split is fixed:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Do not point the current trainer directly at the roughly 45M-row full parquet file until streaming support exists.

Leakage checklist:

- Safe: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and deterministic current-board geometry derived only from those planes.
- Safe for this idea: piece counts, material totals, phase proxies, pawn-file counts, king coordinates, castling flags, en-passant presence/file, side-to-move, and other deterministic summaries computed inside the model from the current board tensor.
- Risky unless separately justified and ablated: full legal-move generation, legal move counts, checkmate/stalemate oracles, forced-line detection, and move-tree consequences. This idea intentionally does not use them.
- Forbidden as neural-network inputs: engine evaluations, Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, or anything derived from the dataset construction process rather than the current board.
- Labels are supervision only. Fine labels may be used for diagnostics and, if the existing training stack supports it, optional supervised diagnostics; they must never be passed as input features.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic nuisance extraction may use only known current-board piece/side/castling/en-passant channels. History channels may be consumed by a learned neural adapter, but they must not be parsed for deterministic nuisance geometry unless their semantics are explicitly declared by the encoding registry.
- Adapters must fail closed: if channel semantics are unknown, raise a clear error or disable this model/config rather than silently extracting wrong features.

## 4. Research Map

This packet borrows ideas from representation invariance, information bottlenecks, and residualization, but it applies them to a board-only chess classification setting without engine targets or move-search supervision.

| Source | What is borrowed | What is not copied |
|---|---|---|
| Ganin et al., "Domain-Adversarial Training of Neural Networks," JMLR 2016, https://jmlr.org/papers/v17/15-239.html | The representation-learning goal that useful predictions can improve under shift when features cannot easily identify nuisance/domain variables. | No gradient reversal layer, no domain labels, no adversarial classifier, no source-provenance input. |
| Arjovsky et al., "Invariant Risk Minimization," arXiv:1907.02893, https://arxiv.org/abs/1907.02893 | The causal framing that stable predictive structure should survive changes in spurious correlations. | No IRM penalty, no multi-environment objective, no claim that the split provides true causal environments. |
| Alemi et al., "Deep Variational Information Bottleneck," arXiv:1612.00410 / ICLR 2017, https://arxiv.org/abs/1612.00410 | The bottleneck framing: retain label-relevant information while suppressing nuisance information. | No variational latent distribution, no KL-to-prior objective, no stochastic encoder requirement. |
| Achille and Soatto, "Emergence of Invariance and Disentanglement in Deep Representations," JMLR 2018 / arXiv:1706.01350, https://arxiv.org/abs/1706.01350 | The connection between nuisance invariance and information minimality in learned representations. | No use of their specific theory as a guarantee for this chess task; no stochastic quantization mechanism. |
| Basu, "The Yule-Frisch-Waugh-Lovell Theorem," arXiv:2307.00369, https://arxiv.org/abs/2307.00369 | The residualization geometry: project variables away from the column span of nuisance covariates before fitting a target relation. | No econometric coefficient inference, no claim of unbiased treatment-effect estimation. |
| Chernozhukov et al., "Double/debiased machine learning for treatment and structural parameters," arXiv:1608.00060, https://arxiv.org/abs/1608.00060 | The broad lesson that orthogonalizing against nuisance functions can reduce sensitivity to nuisance estimation. | No causal treatment effect, no sample-split DML estimator, no asymptotic inference claim. |
| Oprescu, Syrgkanis, and Wu, "Orthogonal Random Forest for Causal Inference," arXiv:1806.03467, https://arxiv.org/abs/1806.03467 | The term "orthogonal" as a design principle for reducing nuisance sensitivity. | No random forest, no CATE target, no causal identification assumption. |

The new contribution here is not another attack graph, sheaf, Hodge, curvature, or one-ply move-delta landscape. It is a hard, differentiable latent projection against a deterministic chess nuisance design matrix.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN over `simple_18` | Existing simple CNN in `src/chess_nn_playground/models/cnn.py` | Too ordinary; it can learn material/source shortcuts but gives no direct test of whether puzzle signal survives nuisance removal. |
| Residual CNN over `simple_18` | Existing residual CNN in `src/chess_nn_playground/models/residual_cnn.py` | Already covered; more residual blocks change capacity, not the inductive bias. |
| LC0-style CNN or residual CNN over 112 planes | Existing LC0 BT4-style CNN/residual variants | Already covered by the baseline suite and risks becoming an encoding-capacity test rather than a new mathematical mechanism. |
| Ordinary ViT over 64 squares | Generic square-token Transformer | Too common and data-hungry; vanilla square attention does not specifically target puzzle-likeness or nuisance separation. |
| Plain GNN on board squares | Standard graph neural network on 8-neighbor or line-of-sight edges | Either too generic if it uses square adjacency, or too close to imported attack-defense graph families if it uses chess attacks. |
| Hyperparameter tuning | Any current baseline | Explicitly disallowed as the core idea; it is useful after a mechanism works, not a research hypothesis. |
| Ensembling baselines | Any collection of existing CNN/residual/LC0 models | Explicitly disallowed and would hide whether a single inductive bias is responsible for gains. |
| More data or full parquet training | Existing trainer plus larger dataset | Not valid as the central idea; current trainer should not target the full 45M-row file before streaming support exists. |
| Engine-score distillation | Stockfish/LC0 supervised teacher | Forbidden because engine scores, PVs, node counts, mate scores, and verification metadata cannot be neural-network inputs or targets here. |
| Tactical sheaf/Hodge/attack-incidence variant | Imported tactical sheaf/Hodge packets | Already researched; adding edge types, curvature names, or pooling variants would be a duplicate. |
| One-ply move-delta bag/set/spectrum model | Imported counterfactual move-delta packets | Already researched; this packet intentionally avoids move generation and move-delta multisets. |
| Label smoothing or ordinary calibration | Standard classification post-processing | Useful but not enough; calibration alone does not impose a falsifiable chess-specific nuisance-residual structure. |

## 6. Mathematical Thesis

### Input space definition

Let `B` denote a legal chess board state represented by one of the allowed encodings. For an encoding `e`, let

```text
X_e(B) in R^{C_e x 8 x 8}
```

be the tensor passed to the model. For the minimal experiment, `e = simple_18` and `C_e = 18`.

Let

```text
F in {0,1,2}
```

be the verified fine label and

```text
Y = 1{F in {1,2}} in {0,1}
```

be the coarse binary puzzle-likeness label, unless the existing project already defines the same `coarse_binary` mapping internally.

### Data distribution assumptions

The observed split is sampled from a mixture of chess-position sources and puzzle-curation processes. Let `N(B)` be a deterministic nuisance summary of the current board: material profile, phase, side-to-move, castling/en-passant flags, pawn-file profile, king coordinates, and a small number of edge/center occupancy summaries. The working assumption is:

```text
P(Y | B) depends on both T(B) and N(B),
```

where `T(B)` is latent tactical/puzzle structure, but `N(B)` is more likely to contain shortcuts that vary with source, phase, and curation. The goal is not to prove `N` is irrelevant. The goal is to test whether a classifier using the part of its learned representation orthogonal to `N` generalizes better and treats near-puzzles more coherently.

### Allowed symmetry or equivariance assumptions

Chess is not invariant under arbitrary rotations or reflections. Pawns move directionally, castling is side-specific, en-passant is directional, and side-to-move matters. This idea does not assume full dihedral symmetry. It uses ordinary convolution only as a local feature extractor and includes side-to-move/castling/en-passant in the nuisance vector where channel semantics are known.

A color-swap plus rank-flip canonicalization could be a separate future experiment, but it is not part of the central mechanism here and should not be mixed into the first falsification run.

### Core hypothesis

Let `H = f_theta(X)` be a learned latent representation. Let `Q = rho(N(B))` be a fixed, normalized nuisance feature design matrix for a mini-batch. If puzzle-likeness contains a stable tactical component not linearly explained by `rho(N)`, then a classifier trained on

```text
Z = (I - P_Q) H
```

should improve near-puzzle and puzzle discrimination at a matched non-puzzle false-positive rate compared with the same trunk/head trained on `H` directly.

### Formal operator

For a mini-batch of size `b`, define:

```text
H in R^{b x d},     H_i = f_theta(X_i)
Q in R^{b x k},     Q_i = rho(N(B_i))
```

where columns of `H` and `Q` are centered across the batch. For ridge parameter `lambda > 0`, define the nuisance projection:

```text
P_Q^lambda H = Q (Q^T Q + lambda I_k)^{-1} Q^T H.
```

The projected latent is:

```text
Z = H - gamma P_Q^lambda H,
```

with main experiment `gamma = 1`. The classifier is:

```text
logits = c_phi(Z) in R^{b x 2}.
```

### Proposition: empirical nuisance orthogonality

Assume `lambda = 0`, `rank(Q)=k`, and `gamma=1`. Then:

```text
Q^T Z = 0.
```

Thus every latent coordinate of `Z` has zero empirical linear covariance with every nuisance feature in `Q` on that mini-batch.

For `lambda > 0`,

```text
Q^T Z = lambda (Q^T Q + lambda I_k)^{-1} Q^T H,
```

so the residual correlation is controlled by `lambda` and the conditioning of `Q^T Q`.

### Proof sketch or derivation

For `lambda=0`:

```text
Z = H - Q(Q^TQ)^{-1}Q^T H.
```

Multiplying by `Q^T` gives:

```text
Q^T Z = Q^T H - Q^T Q (Q^TQ)^{-1} Q^T H = 0.
```

For `lambda>0`, the same multiplication gives:

```text
Q^T Z = Q^T H - Q^TQ(Q^TQ + lambda I)^{-1}Q^TH.
```

Using

```text
I - A(A + lambda I)^{-1} = lambda(A + lambda I)^{-1}
```

with `A = Q^TQ` yields the ridge expression above.

If the latent decomposes as

```text
H = QA + E,   with Q^T E = 0,
```

then the exact projection recovers `Z = E`. The classifier is therefore forced to rely on the residual component `E`, not the nuisance component `QA`.

### Variational principle

The projection is the closed-form solution to:

```text
Z* = argmin_Z ||H - Z||_F^2
     subject to Q^T Z = 0.
```

With ridge, it is the stable shrinkage version of the same residualization. Training minimizes:

```text
min_{theta,phi} E[ CE(c_phi(Proj_perp_Q(f_theta(X))), Y) ]
              + alpha ||Q^T Z / b||_F^2
              + beta R(theta,phi),
```

where the covariance penalty is optional because the projection already enforces the condition up to ridge and numerical precision.

### What is actually proven

The projection provably removes mini-batch empirical linear dependence between `Z` and the fixed nuisance feature map `Q` when `lambda=0`, and approximately removes it when `lambda>0`. It also gives a clear ablation: if this removal is responsible for gains, `gamma=0` or nuisance-shuffled projections should underperform.

### What remains only hypothesized

It is not proven that puzzle-likeness is independent of material, phase, king position, or side-to-move. It is not proven that the finite sample split has harmful nuisance shift. It is not proven that mini-batch residualization approximates population conditional independence. Those are empirical hypotheses for Codex to test.

### Counterexamples where the idea should fail

- If the dataset's puzzle-like label is genuinely determined mostly by material/phase/king-location profiles, projection will remove signal and reduce accuracy.
- If the deterministic nuisance vector omits the real shortcut, the projection will not address the failure mode.
- If the nuisance vector is too rich, it may strip valid tactical signal that happens to correlate with king/pawn structure.
- If batches are too small or class composition is unstable, the projection matrix will be noisy.
- If the encoding adapter parses channels incorrectly, all results are invalid; this is why adapters must fail closed.
- If near-puzzles are intrinsically label-ambiguous rather than nuisance-confounded, a hard bottleneck may not improve class-1 diagnostics.

## 7. Architecture Specification

### Module names

Implement one model file:

```text
src/chess_nn_playground/models/nuisance_orthogonal_bottleneck.py
```

Suggested classes/functions:

- `NuisanceOrthogonalPuzzleNet(nn.Module)`
- `EncodingSemanticAdapter(nn.Module or helper class)`
- `DeterministicNuisanceExtractor(nn.Module)`
- `FixedNuisanceFeatureMap(nn.Module)`
- `BatchRidgeOrthogonalProjector(nn.Module)`
- `ConvResidualTrunk(nn.Module)` or reuse a small existing residual block internally
- builder function: `build_nuisance_orthogonal_bottleneck(config)`

The model must return only logits by default so the shared trainer remains compatible.

### Forward-pass steps

Input:

```text
x: FloatTensor[batch, C, 8, 8]
```

Step 1: semantic adapter validation.

- If encoding is `simple_18`, assert `C == 18` and known channel order.
- If encoding is `lc0_static_112` or `lc0_bt4_112`, require config-provided current-board semantic indices. If missing, raise a clear `ValueError`.
- The learned convolutional trunk may consume all `C` channels. The deterministic nuisance extractor may use only known current-board channels.

Step 2: deterministic nuisance extraction.

Output:

```text
n: FloatTensor[batch, m]
```

Recommended first-pass nuisance components:

- 12 normalized piece counts from the current-board piece planes.
- 6 material summaries: white material, black material, material balance, absolute material imbalance, non-pawn material phase, total piece count.
- 1 side-to-move scalar, encoded as `+1` for white-to-move and `-1` for black-to-move where available.
- 4 castling-right scalars when available, else zeros only if the encoding explicitly lacks them; do not guess.
- 9 en-passant features: presence scalar plus file one-hot if available.
- 8 king-coordinate features: white king file/rank normalized to `[-1,1]`, black king file/rank normalized, and four edge-distance summaries.
- 16 pawn-file counts: white and black pawns per file, normalized.
- 8 occupancy marginal summaries: total occupancy per rank group and file group, kept coarse to avoid encoding the entire board as nuisance.

Expected `m`: about `64`. Codex may trim or pad to the configured `nuisance_dim`, but must document the exact vector in `implementation_notes.md`.

Step 3: fixed nuisance feature map.

Output:

```text
q: FloatTensor[batch, k]
```

Use deterministic normalized features, not a learned nuisance encoder, to avoid collapse. Recommended:

```text
q = LayerNormNoAffine([n, selected_pairwise_products(n), selected_squares(n)])
```

Then keep the first `k = min(config.nuisance_rank, q_dim)` columns, where default `nuisance_rank = 64`. If a random projection is used to reduce dimension, it must be a fixed registered buffer generated from `seed=42`, not a trainable parameter.

Step 4: convolutional trunk.

A compact trunk is enough because the projection is the research mechanism:

```text
Conv2d(C, 64, kernel_size=3, padding=1)
BatchNorm2d(64)
GELU
4 x ResidualBlock(64 -> 64, 3x3 convs)
Conv2d(64, 96, kernel_size=3, padding=1)
BatchNorm2d(96)
GELU
AdaptiveAvgPool2d(1)
Flatten
Linear(96, latent_dim=256)
GELU
LayerNorm(256)
```

Shapes:

```text
x                       [b, C, 8, 8]
conv stem               [b, 64, 8, 8]
residual stack          [b, 64, 8, 8]
projection conv         [b, 96, 8, 8]
global pooled           [b, 96]
h = latent              [b, 256]
q = nuisance features   [b, k]
z = projected latent    [b, 256]
logits                  [b, 2]
```

Step 5: batchwise ridge orthogonal projection.

Do not instantiate a `[b,b]` projection matrix. Compute:

```text
Hc = H - mean_batch(H)
Qc = Q - mean_batch(Q)
G  = Qc.T @ Qc + ridge_lambda * I_k      # [k,k]
A  = solve(G, Qc.T @ Hc)                 # [k,d]
Proj = Qc @ A                            # [b,d]
Z = Hc - gamma * Proj
```

Default:

```text
ridge_lambda = 1e-3
gamma = 1.0
```

If `batch_size <= nuisance_rank + 2`, either reduce effective rank to `batch_size - 2` by slicing `Q` or skip projection with a logged warning in training. The recommended benchmark batch size is `512`, so this should not occur except for a small last batch. Prefer `drop_last=true` for training if supported.

Step 6: classifier head.

```text
LayerNorm(256)
Dropout(p=0.05)
Linear(256, 128)
GELU
Linear(128, 2)
```

Return:

```text
logits: FloatTensor[batch, 2]
```

### Parameter-count estimate

For `simple_18`:

- convolution stem: about 10.5k parameters;
- four residual blocks at width 64: about 295k parameters plus normalization;
- 64-to-96 conv: about 55k parameters;
- latent linear and head: about 58k parameters;
- total expected: roughly 0.42M to 0.48M parameters depending on normalization/bias choices.

For `lc0_static_112` or `lc0_bt4_112`, the input stem grows by about `94 * 64 * 3 * 3 = 54,144` parameters, so total expected is roughly 0.48M to 0.54M.

The nuisance extractor and projection have no trainable parameters unless Codex adds a fixed registered random projection buffer, which should not count as learned capacity.

### FLOP or complexity estimate

Convolutional FLOPs dominate. The residual stack is approximately:

```text
O(batch * 8 * 8 * 4 blocks * 2 convs * 64 * 64 * 3 * 3)
```

The projection cost is:

```text
O(batch * k^2 + k^3 + batch * k * latent_dim)
```

With `batch=512`, `k=64`, `latent_dim=256`, the projection is small compared with the CNN and should be comfortably under a few million multiply-adds per batch, excluding the solve overhead.

### Candidate-set memory and chunking

This model generates no move candidate set, no attack graph, no hypergraph, and no one-ply board-delta multiset.

Projection memory:

```text
H: batch * latent_dim floats
Q: batch * nuisance_rank floats
G: nuisance_rank^2 floats
A: nuisance_rank * latent_dim floats
Proj: batch * latent_dim floats
```

For `batch=512`, `latent_dim=256`, `nuisance_rank=64`, this is under 2 MB in float32 for the projection tensors, not counting autograd overhead. No chunking is needed. If future experiments raise `latent_dim` or `nuisance_rank`, keep `Q` rank below `min(batch/2, 128)` and use `torch.linalg.solve` or Cholesky solve rather than forming a batch-by-batch matrix.

### Required config fields

```yaml
model:
  name: nuisance_orthogonal_bottleneck
  input_channels: 18
  num_classes: 2
  encoding: simple_18
  latent_dim: 256
  trunk_channels: 64
  trunk_blocks: 4
  nuisance_rank: 64
  ridge_lambda: 0.001
  projection_gamma: 1.0
  dropout: 0.05
  piece_plane_order: simple_18_default
  fail_closed_semantics: true
```

### Encoding support

First experiment should use `simple_18`. It is the safest because current-board semantics are explicit and deterministic nuisance extraction can be audited.

Support plan:

- `simple_18`: fully supported in first implementation. The extractor reads known piece planes, side-to-move, castling, and en-passant channels.
- `lc0_static_112`: support only if registry/config provides exact indices for current piece planes and state channels. The trunk may consume all 112 channels. Nuisance extraction uses only current-board channels.
- `lc0_bt4_112`: same as `lc0_static_112`; unavailable history planes are not parsed as deterministic geometry. They may be consumed by the learned trunk, because that is already part of the encoding, but not by the deterministic nuisance extractor.

Adapters must fail closed when channel semantics are unknown.

### Pseudocode, not final implementation

```python
class NuisanceOrthogonalPuzzleNet(nn.Module):
    def forward(self, x):
        self.adapter.validate(x)
        n = self.nuisance_extractor(x)          # [b, m], deterministic
        q = self.nuisance_feature_map(n)        # [b, k], fixed/non-trainable
        h = self.trunk(x)                       # [b, d]
        z = self.projector(h, q)                # [b, d]
        logits = self.head(z)                   # [b, 2]
        return logits
```

The model must not require labels during `forward`.

## 8. Loss, Training, And Regularization

Primary loss:

```text
CrossEntropyLoss(logits, binary_target)
```

Use class weighting:

```text
class_weighting: balanced
```

Optional auxiliary regularization:

```text
cov_loss = || Qc.T @ Z / batch ||_F^2
```

Default `cov_loss_weight = 0.0` for the first run, because the projection already imposes the condition. If numerical leakage appears, run a follow-up with `cov_loss_weight = 0.01`.

Optional variance floor to prevent representational collapse:

```text
var_loss = mean_j max(0, min_std - std_batch(Z_j))^2
```

Default `var_loss_weight = 0.0`. Enable only if logs show near-zero latent variance.

Batch size expectations:

- Recommended: `512`.
- Minimum practical: at least `2 * nuisance_rank` for stable projection.
- If the last batch is small, use `drop_last=true` in training or reduce effective rank for that batch.

Optimizer defaults:

```text
AdamW
learning_rate = 0.001
weight_decay = 0.0001
epochs = 3
early_stopping_patience = 2
mixed_precision = false for the first deterministic run
```

Regularizers:

- dropout `0.05` in the final head;
- weight decay `1e-4`;
- ridge projection parameter `1e-3`;
- no data augmentation in the first benchmark unless the current baseline config already uses it.

Determinism requirements:

- seed `42`;
- deterministic mode enabled where supported;
- fixed nuisance feature map, no trainable nuisance encoder in the central experiment;
- log the exact nuisance vector definition and feature rank.

What must stay unchanged for fair comparison:

- train/val/test split paths;
- binary label mapping;
- encoding for the main comparison, preferably `simple_18`;
- epochs, batch size, optimizer, learning rate, weight decay, class weighting, and early-stopping policy, except for any unavoidable `drop_last` setting needed by the projection;
- reporting pipeline, confusion matrices, predictions, and leaderboard schema.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `gamma=0` no-projection | Uses the same trunk and head but feeds `H` directly to the classifier. | Tests the central claim that nuisance orthogonalization, not trunk capacity, improves puzzle-likeness classification. | If equal or better than `gamma=1`, the projection is not helping and the idea should not be scaled. |
| Shuffled nuisance alignment | Randomly permutes `Q` across samples inside each batch before projection, preserving nuisance marginal distribution and rank. | Tests whether sample-aligned chess nuisance semantics matter. | If shuffled `Q` matches the main model, the gain is probably generic random rank removal or regularization. |
| Random rank-matched projection | Replaces `Q` with fixed Gaussian noise of shape `[batch,k]`, centered and scaled like real `Q`. | Tests whether any low-rank projection bottleneck helps. | If random projection matches the main model, chess-specific nuisance design is not the mechanism. |
| Columnwise nuisance permutation | Independently permutes each nuisance feature column across the batch, preserving per-feature histograms but destroying coherent board-level nuisance vectors. | Tests whether the joint material/phase/king profile matters rather than marginal feature distributions. | If performance is unchanged, the nuisance vector semantics are not being used. |
| Material-only nuisance | Keeps only piece counts, material totals, and phase in `Q`; drops castling, en-passant, pawn-file, and king-coordinate terms. | Tests whether material/phase shortcuts dominate the nuisance effect. | If material-only performs as well as full `Q`, future work can simplify the extractor. |
| Side/rights-only nuisance | Keeps only side-to-move, castling, and en-passant features. | Tests whether label artifacts are tied mostly to state flags rather than material. | If this matches full `Q`, the model may be exploiting metadata-like state channels more than board content. |
| Nuisance-only classifier | Trains a small MLP on `N(B)` alone, no board trunk. | Measures the shortcut ceiling from deterministic nuisance variables. | If nuisance-only is close to the full model, this dataset may be dominated by material/phase artifacts. |
| Projection with excessive nuisance rank | Raises `nuisance_rank` to 128 if batch size allows. | Tests whether too-rich nuisance removal strips valid tactical signal. | If performance drops sharply, keep the nuisance map compact and avoid encoding full board geometry as nuisance. |
| Projection after classifier head | Applies residualization to logits or penultimate two-class features instead of the 256-d latent. | Tests whether the latent-level operator is necessary. | If late projection works equally well, implementation can be simpler; if it fails, latent residualization matters. |
| Baseline plus covariance penalty only | Removes exact projection and uses `||Q^T H||_F^2` as a soft penalty. | Tests whether hard closed-form residualization is better than a standard regularizer. | If soft penalty matches hard projection, the projection operator may not be necessary. |

The smallest central falsification ablation is `gamma=0` with everything else unchanged.

This idea has no graph, hypergraph, sheaf, transport plan, counterfactual move-set, or search-surrogate object. The semantics-destroying ablations are therefore the shuffled, columnwise-permuted, and random rank-matched nuisance projections, each of which preserves obvious nuisance counts or rank while destroying sample-aligned nuisance semantics.

Because no move-generated candidate set exists, no count-only move ablation is needed. The nuisance-only classifier still checks the analogous shortcut: whether deterministic material/state counts can solve the task by themselves.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- existing simple CNN on `simple_18`;
- existing residual CNN on `simple_18`;
- existing LC0-style CNN/residual only as secondary context if Codex has stable configs;
- `NOPB gamma=0` no-projection ablation;
- nuisance-only classifier;
- shuffled and random-projection ablations.

Metrics to inspect:

- validation and test accuracy;
- balanced accuracy;
- AUROC if already available in the report stack;
- F1 and precision/recall for binary class `1`;
- Brier score and/or ECE if available;
- required rectangular `3x2` matrix: true fine label `0/1/2 -> predicted binary output 0/1`.

Required fine-label matrix for main model and every central ablation:

```text
rows: true fine label 0, 1, 2
cols: predicted binary 0, 1
```

Near-puzzle diagnostic:

- choose a threshold on validation that matches the fine-label-`0` false-positive rate of the strongest existing `simple_18` baseline;
- at that matched FPR, report fine-label-`1` recall and fine-label-`2` recall on validation and test;
- also report fine-label-`1` precision among predicted positives if the report stack can compute it.

Required artifacts:

- model config YAML;
- training logs;
- validation and test metrics JSON/CSV;
- predictions parquet/CSV with identifiers consistent with existing reports;
- confusion matrices, including the fine-label `3x2` matrices;
- ablation report table;
- nuisance diagnostics: nuisance-only performance, `||Q^T Z / b||_F`, latent variance statistics, and projection rank/condition-number summaries.

Success threshold:

- Main `NOPB` beats the strongest existing `simple_18` baseline by at least `+1.0` percentage point AUROC or balanced accuracy on test, or improves fine-label-`1` recall by at least `+2.0` points at matched fine-label-`0` FPR, while losing no more than `0.5` points of fine-label-`2` recall.
- Main `NOPB` must also beat `gamma=0`, shuffled nuisance, and random projection on the near-puzzle matched-FPR diagnostic.

Failure threshold:

- Main `NOPB` is within noise of `gamma=0`, shuffled nuisance, or random projection on all key metrics;
- or it trails the strongest `simple_18` baseline by more than `1.0` point AUROC/balanced accuracy;
- or nuisance-only performance is close enough to full-model performance that the dataset appears dominated by deterministic shortcuts.

Abandon the idea if:

- `gamma=0` is equal or better across two deterministic seeds;
- shuffled or random projection matches the main model;
- projection causes latent collapse or unstable training that is not fixed by ridge `1e-3` to `1e-2`;
- class-1 near-puzzle diagnostics worsen at matched fine-label-0 FPR while class-2 recall does not improve.

Justify scaling if:

- the main projection beats all central ablations;
- fine-label-1 recall improves at matched fine-label-0 FPR;
- the nuisance-only classifier is substantially weaker than the full model, proving the trunk is not merely reproducing the nuisance vector;
- results remain stable across at least two seeds or one additional encoding with verified channel semantics.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0508_nuisance_orthogonal/idea.yaml` | Create | Machine-readable idea metadata matching the `idea_yaml` block below. |
| `ideas/20260421_0508_nuisance_orthogonal/math_thesis.md` | Create | Copy Section 6, including projection proposition, proof sketch, hypotheses, and counterexamples. |
| `ideas/20260421_0508_nuisance_orthogonal/architecture.md` | Create | Copy Section 7 with exact module names, tensor shapes, adapter rules, and pseudocode. |
| `ideas/20260421_0508_nuisance_orthogonal/implementation_notes.md` | Create | Document exact `simple_18` channel assumptions, nuisance vector entries, projection math, numerical stability choices, and fail-closed behavior. |
| `ideas/20260421_0508_nuisance_orthogonal/trainer_notes.md` | Create | Copy Section 8 plus any repo-specific notes on class weighting, `drop_last`, deterministic mode, and shared report compatibility. |
| `ideas/20260421_0508_nuisance_orthogonal/ablations.md` | Create | Copy Section 9 and mark `gamma=0`, shuffled `Q`, random `Q`, and nuisance-only as central required ablations. |
| `ideas/20260421_0508_nuisance_orthogonal/train.py` | Create | Thin idea-local entry point that calls the existing shared trainer with `configs/nuisance_orthogonal_simple18.yaml`; do not fork the whole training stack unless needed for ablations. |
| `ideas/20260421_0508_nuisance_orthogonal/config.yaml` | Create | Idea-local copy of the benchmark config using `simple_18`, `batch_size=512`, and `model.name=nuisance_orthogonal_bottleneck`. |
| `ideas/20260421_0508_nuisance_orthogonal/report_template.md` | Create | Template requiring baseline comparison, fine-label `3x2` matrices, matched-FPR near-puzzle diagnostic, projection diagnostics, and ablation table. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported research memory after implementation; add anti-duplicate rules for deterministic nuisance orthogonalization and batchwise residual projection if it fails. |
| `src/chess_nn_playground/models/nuisance_orthogonal_bottleneck.py` | Create | Implement `NuisanceOrthogonalPuzzleNet`, semantic adapter, deterministic nuisance extractor, fixed nuisance feature map, batch ridge projector, and builder. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `nuisance_orthogonal_bottleneck` builder while preserving existing model names. |
| `configs/nuisance_orthogonal_simple18.yaml` | Create | Main benchmark config matching the `config_yaml` block below. |
| `configs/nuisance_orthogonal_simple18_no_projection.yaml` | Create | Ablation config with `projection_gamma=0.0`. |
| `configs/nuisance_orthogonal_simple18_shuffled_q.yaml` | Create | Ablation config that shuffles nuisance rows before projection during training and evaluation. |
| `configs/nuisance_orthogonal_simple18_random_q.yaml` | Create | Ablation config replacing `Q` with fixed rank-matched random features. |
| `configs/nuisance_orthogonal_simple18_nuisance_only.yaml` | Create | Nuisance-only shortcut baseline config if the trainer can support it; otherwise implement as idea-local script. |
| `tests/models/test_nuisance_orthogonal_bottleneck.py` | Create | Focused shape test: `[2,18,8,8] -> [2,2]`; projection numerical test with synthetic `H,Q`; fail-closed test for unknown encoding semantics. |
| `tests/models/test_nuisance_features_simple18.py` | Create | Unit tests for deterministic piece counts, side-to-move, castling/en-passant extraction, and king coordinates using simple synthetic tensors. |

For `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md`, Codex should preserve all hard leakage, label, falsification, and anti-duplicate constraints. It should add this idea to the researched-memory section after results are known, with the outcome and whether batchwise nuisance projection should be avoided in future cycles.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0508_tuesday_local_nuisance_orthogonal.md
  generated_at: "2026-04-21 05:08:09 America/Los_Angeles"
  weekday: Tuesday
  timezone: local
  idea_slug: nuisance_orthogonal
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0508_nuisance_orthogonal
  name: Nuisance-Orthogonal Puzzle Bottleneck
  slug: nuisance_orthogonal
  status: draft
  created_at: "2026-04-21 05:08:09 America/Los_Angeles"
  author: ChatGPT Pro
  short_thesis: Project the learned board latent away from deterministic material/phase/side-to-move nuisance directions so puzzle-likeness must be predicted from residual tactical structure.
  novelty_claim: Uses a fixed current-board nuisance design matrix and differentiable batchwise ridge residualization rather than a bigger CNN, attack/sheaf graph, or one-ply move-delta set.
  expected_advantage: Better near-puzzle recall at matched non-puzzle false-positive rate by suppressing material and phase shortcuts.
  central_falsification_ablation: Same trunk and head with projection_gamma=0.0.
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112 and lc0_bt4_112 only with verified current-board channel semantics
  output_heads: binary logits only, shape [batch, 2]
  compute_notes: Projection cost O(batch*k^2 + k^3 + batch*k*latent_dim), with default batch=512, k=64, latent_dim=256.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/nuisance_orthogonal_simple18.yaml
  model_path: src/chess_nn_playground/models/nuisance_orthogonal_bottleneck.py
  latest_result_path: null
  notes: Do not use engine scores, source labels, verification metadata, or move-generation features; deterministic nuisance extraction must fail closed on unknown channel semantics.
```

```yaml
config_yaml:
  run:
    name: nuisance_orthogonal_simple18
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
    name: nuisance_orthogonal_bottleneck
    input_channels: 18
    num_classes: 2
    encoding: simple_18
    latent_dim: 256
    trunk_channels: 64
    trunk_blocks: 4
    nuisance_rank: 64
    ridge_lambda: 0.001
    projection_gamma: 1.0
    dropout: 0.05
    piece_plane_order: simple_18_default
    fail_closed_semantics: true
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
  model_name: nuisance_orthogonal_bottleneck
  file_path: src/chess_nn_playground/models/nuisance_orthogonal_bottleneck.py
  builder_function: build_nuisance_orthogonal_bottleneck
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - EncodingSemanticAdapter
    - DeterministicNuisanceExtractor
    - FixedNuisanceFeatureMap
    - BatchRidgeOrthogonalProjector
    - ConvResidualTrunk
    - NuisanceOrthogonalPuzzleNet
  required_config_fields:
    - model.name
    - model.input_channels
    - model.num_classes
    - model.encoding
    - model.latent_dim
    - model.nuisance_rank
    - model.ridge_lambda
    - model.projection_gamma
    - model.fail_closed_semantics
  expected_parameter_count: approximately 0.42M-0.48M for simple_18; approximately 0.48M-0.54M for 112-channel encodings
  expected_memory_notes: Projection tensors are O(batch*latent_dim + batch*nuisance_rank + nuisance_rank*latent_dim); avoid forming a batch-by-batch projection matrix.
```

```yaml
research_continuity:
  idea_fingerprint: current-board deterministic material/phase/king/castling/en-passant nuisance vector + fixed normalized nuisance feature matrix + batchwise ridge orthogonal projection of CNN latent away from nuisance span + binary puzzle-likeness target + no engine metadata or move-delta set
  already_researched_family_overlap: Does not overlap with imported tactical sheaf/Hodge/attack graph family or one-ply move-delta family; partial conceptual overlap with causal invariant representation learning and information bottlenecks.
  closest_duplicate_risk: Domain-adversarial or information-bottleneck debiasing; this idea differs by using a closed-form residual projection rather than learned adversarial nuisance removal.
  do_not_repeat_if_this_fails:
    - deterministic material/phase nuisance-vector projection
    - batchwise ridge residualization of chess latents against handcrafted nuisance features
    - covariance-only nuisance penalties using the same material/phase/king feature map
    - adversarial nuisance-invariant classifier using the same nuisance vector unless the failure analysis specifically shows hard projection was the only problem
  suggested_next_search_directions:
    - label-safe selective prediction or ordinal uncertainty focused on class-1 near-puzzle ambiguity
    - multi-encoding invariance using simple_18 versus lc0_static_112 as environments without using source labels as inputs
    - optimal transport over piece-square distributions that avoids attack graphs and move-delta sets
    - calibration diagnostics that separate true tactical ambiguity from material/source artifacts
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Nuisance-Orthogonal Puzzle Bottleneck` to imported research memory after implementation, including result status. | Prevents future ChatGPT passes from proposing another material/phase nuisance residualization packet as fresh. | `Imported Research Memory` |
| If this fails, add an anti-duplicate rule: do not repeat deterministic material/phase/king nuisance-vector projection, covariance penalty, or adversarial nuisance removal unless the operator is fundamentally different and the prior failure is addressed. | Avoids superficial variants such as replacing projection with gradient reversal or HSIC while testing the same hypothesis. | Anti-duplicate paragraph after the move-delta family rules |
| Add a reusable requirement that any debiasing/bottleneck idea must include a nuisance-only shortcut baseline and a semantics-destroying nuisance shuffle. | Makes future causal/invariance ideas easier to falsify. | `What Counts As Creative Enough` or `Required Markdown File Content / Ablation Plan` |
| Add a note that deterministic current-board summaries are allowed only when computed from known encoding semantics and must fail closed on unknown channel orders. | Prevents accidental leakage or invalid LC0-channel parsing in future packets. | `Problem Restatement And Data Contract` |
| If NOPB succeeds, add a prompt hint that future ideas should inspect fine-label-1 matched-FPR behavior, not just overall binary accuracy. | Near-puzzle behavior is the most informative diagnostic for ambiguity-sensitive puzzle-likeness models. | `Benchmark And Falsification Criteria` |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: Yes.
- Filename follows required date/time/day/timezone/slug pattern: Yes, `chess_nn_research_2026-04-21_0508_tuesday_local_nuisance_orthogonal.md`.
- No forbidden engine features used as inputs: Yes.
- Does not fabricate labels: Yes.
- Not a routine CNN/ResNet/Transformer variant: Yes; the central mechanism is batchwise nuisance orthogonal projection.
- Minimal current-data experiment exists: Yes; `simple_18` on the fixed `crtk_sample_3class` split.
- Falsification criterion is concrete: Yes; `projection_gamma=0`, shuffled `Q`, random `Q`, nuisance-only, and matched-FPR near-puzzle diagnostics.
- Codex can implement without asking for missing architecture details: Yes.
- Prompt maintenance notes included for Codex: Yes.
- Repetition check against imported research packets completed: Yes.
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: Yes.
