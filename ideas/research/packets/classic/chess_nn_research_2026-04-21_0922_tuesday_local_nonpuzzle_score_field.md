# Codex Handoff Packet: Non-Puzzle Score-Field Bottleneck Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0922_tuesday_local_nonpuzzle_score_field.md`
- Generated at: `2026-04-21 09:22 UTC-07:00`
- Weekday: `Tuesday`
- Timezone: `local`
- Idea slug: `nonpuzzle_score_field`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: **Non-Puzzle Score-Field Bottleneck Network**
- One-sentence thesis: Train a rule-safe denoising score prior only on verified non-puzzle boards, then classify puzzle-likeness from the current board plus a bottlenecked vector field estimating how the non-puzzle manifold would locally “repair” that board.
- Idea fingerprint: `current-board simple_18 tensor + class-0-only Gaussian denoising score prior + clean-board residual score maps at fixed noise scales + small bottleneck classifier -> binary puzzle logits; no engine, no move tree, no attack graph, no Sinkhorn, no mask-code likelihood`.
- Why this is not a common CNN/ResNet/Transformer variant: the central signal is not extra depth, attention, or board convolution; it is the estimated input score field of the **non-puzzle** distribution, exposed through a low-dimensional bottleneck and tested against same-compute score-prior ablations.
- Current-data minimal experiment: use `simple_18` on `data/splits/crtk_sample_3class/{split_train,split_val,split_test}.parquet`, train the score prior on training rows whose binary label is `0`, train/fine-tune the classifier on the same binary benchmark, and report the usual test metrics plus the rectangular `3x2` fine-label diagnostic matrix.
- Smallest central falsification ablation: replace the class-0-only score prior with an all-training-positions denoising prior, keeping architecture, noise schedule, bottleneck size, supervised classifier, training epochs, class weights, and reports unchanged.
- Expected information gain if it fails: a failure says puzzle-likeness in this split is not captured by local deviation from the verified non-puzzle board manifold; the next cycle should stop trying class-conditional denoising ordinaryness and instead search for a supervised tactical relation that is not anomaly-based.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The current task is board-position chess puzzle-likeness classification:

- output `0`: non-puzzle
- output `1`: puzzle-like

The source fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The benchmark target is binary, usually `0 -> 0` and `1/2 -> 1`. Reports must still include the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Allowed input representations already available in the repo are:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant
- `lc0_static_112`
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from one FEN, with unavailable history planes currently zero-filled until exporter support exists

The implementation target is PyTorch. The model must accept:

```text
(batch, C, 8, 8)
```

and return logits:

```text
(batch, num_classes)
```

Default split paths for the minimal experiment:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Do not point the trainer directly at the roughly 45M-row full Parquet source until streaming support exists.

Leakage checklist:

- Use only current-board tensor channels and labels as training targets.
- Do not use Stockfish scores, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, or dataset provenance as neural-network inputs.
- Do not fabricate fine label `1` or fine label `2` examples.
- Do not treat unresolved candidates as verified near-puzzles or verified puzzles.
- Do not use full legal move generation, move counts, checkmate/stalemate oracles, forced-line search, or move-tree consequences in this idea.
- The denoising prior is trained from `x` and the binary training label only; at inference it sees only `x`.

Boundary between safe rule-derived features and leakage:

- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board are allowed in general. This idea intentionally does **not** require pseudo-legal attack geometry.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated. This idea avoids them.
- Engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance must never be neural-network inputs.
- For `lc0_static_112` and `lc0_bt4_112`, distinguish current-board channels used for deterministic geometry from history channels used only by learned neural adapters. The score-prior branch must fail closed if it cannot map channels to a known current-board `simple_18` canonical tensor. History channels may be consumed only by a learned supervised adapter in later experiments, not by the deterministic score-prior target.

## 4. Research Map

External ideas used:

| Source | What is borrowed | What is not copied |
|---|---|---|
| Aapo Hyvärinen, “Estimation of Non-Normalized Statistical Models by Score Matching,” JMLR 2005. URL: `https://jmlr.org/papers/v6/hyvarinen05a.html` | The idea that a model can learn gradients of log-density without estimating a normalized probability. | No continuous image-generation sampler, no MCMC, no engine-derived energy. |
| Pascal Vincent, “A Connection Between Score Matching and Denoising Autoencoders,” Neural Computation 2011. URL: `https://direct.mit.edu/neco/article/23/7/1661/7677/A-Connection-Between-Score-Matching-and-Denoising` | The denoising residual identity: under Gaussian corruption, the optimal denoiser residual estimates a smoothed data score. | No layerwise unsupervised pretraining recipe, no masked-square prediction, no likelihood/code-length score. |
| Tommi Jaakkola and David Haussler, “Exploiting Generative Models in Discriminative Classifiers,” NeurIPS 1998. URL: `https://proceedings.neurips.cc/paper/1998/hash/db1915052d15f7815c8b88e879465a1e-Abstract.html` | The philosophy of using generative-model statistics inside a discriminative classifier. | No SVM kernel implementation and no parameter-gradient Fisher kernel; this proposal uses input score maps as neural features. |
| Yang Song and Stefano Ermon, “Generative Modeling by Estimating Gradients of the Data Distribution,” NeurIPS 2019. URL: `https://arxiv.org/abs/1907.05600` | Multi-noise score estimation as a stable way to learn score fields on low-dimensional data manifolds. | No Langevin sampling, no image-generation objective, no diffusion model rollout. |
| Ruff et al., “Deep One-Class Classification,” ICML 2018. URL: `https://proceedings.mlr.press/v80/ruff18a.html` | The one-class learning stance: learn structure of nominal data and use deviations as useful evidence. | No hypersphere/SVDD objective; puzzle classification remains supervised and uses a score-field bottleneck. |
| LeCun et al., “A Tutorial on Energy-Based Learning,” 2006. URL: `https://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf` | The broad energy-model view that configurations can be ranked by compatibility without normalized likelihood. | No structured-output inference, no search over moves, no energy minimization at test time. |

Candidate search trace:

| Candidate mechanism considered | Why it lost to the selected idea |
|---|---|
| Differentiable Horn-clause tactical motif network over pins, forks, and skewers | Too close to another static attack/defense relation graph unless the clauses became so abstract that Codex could not test them quickly. |
| Learned invariant-risk objective over unsupervised “source style” clusters | The current prompt forbids source labels as inputs, and unsupervised clusters could accidentally rediscover provenance artifacts instead of chess structure. |
| Selective-prediction or abstention head for near-puzzle ambiguity | Useful for deployment calibration, but it changes the decision policy more than the board operator and is adjacent to imported ordinal/credal evidence packets. |
| Vector-quantized motif dictionary with MDL usage penalty | Interesting, but too near to generative code-length and masked-codec families unless it included a new chess object; implementation risk is higher than the score-field test. |
| Fixed wavelet/scattering transform over piece planes | Distinct from ResNets, but likely to behave as a handcrafted shallow CNN texture baseline on an 8x8 board. |
| Neural cellular automaton relaxation over board occupancy | Too unconstrained; without rule semantics it is just a recurrent CNN, and with rule semantics it drifts toward imported kinematic operator families. |
| Causal data augmentation by color/side/file transformations | Already covered by orbit, tempo, and rule-partition invariance families; not enough new falsifiable content. |
| Board-edit finite-difference saliency from deleting or swapping pieces | Close to imported sparse witness and masked-board surprise packets. |
| Spectral compression of attack maps | Too close to static attack-defense graph, Hodge, and Betti/pressure-field packets. |
| Differentiable SAT/constraint satisfaction over legal motifs | Would require hand-labeling or rule templates whose failures are hard to interpret; also risks move-oracle creep. |
| One-class SVDD on CNN latents | Simpler anomaly baseline, but it lacks the local vector-field object that can be falsified with an all-class denoising prior. |
| Denoising score prior on non-puzzle boards | Selected: it is label-safe, current-data testable, outside the imported attack/move/OT/topology families, and has a precise mathematical identity to ablate. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Score matching | Noise-conditional denoiser residual `S_sigma(x) = (D_theta(x, sigma) - x) / sigma^2` trained on class-0 boards | input `x: [B,18,8,8]`; output score stack `[B,K*18,8,8]` | Train the same score prior on all classes instead of class `0` | It estimates a continuous smoothed input score, not a sheaf tension, attack graph, move-delta set, or masked-square code length. |
| Fisher-style generative statistic | Bottlenecked score maps concatenated with a small board encoder before binary logits | score bottleneck `[B,K*18,8,8] -> [B,24,8,8]`; logits `[B,2]` | Replace score maps with matched random fields of the same channel-wise norm distribution | It is not an SVM Fisher kernel and not a generic ensemble; the statistic is the learned non-puzzle repair field. |
| One-class ordinaryness | Auxiliary denoising prior trained only on verified non-puzzle training rows | training filter `y_binary=0`; no inference-time label | Compare with ordinary autoencoder and all-class prior | It is not Deep SVDD/hypersphere anomaly detection; supervised classification still decides the label. |
| Information bottleneck | Low-dimensional score-field bottleneck and optional norm-only diagnostic scalars | `[B,K*18,8,8] -> [B,score_bottleneck_channels,8,8]` | Increase bottleneck to equal raw score maps or remove score branch | It suppresses nuisance reconstructions instead of projecting out a hand-coded nuisance vector. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on `simple_18` | `src/chess_nn_playground/models/trunk/cnn.py` | Already present; it tests generic local texture and piece-pattern learning without the new non-puzzle score-field operator. |
| Residual CNN | `src/chess_nn_playground/models/trunk/residual_cnn.py` | Already present; more residual blocks would be routine architecture tuning. |
| LC0-style CNN or residual CNN on `lc0_bt4_112` | Existing LC0 BT4-style CNN/residual variants | Already present and currently history planes are zero-filled; copying LC0 does not test a new puzzle-likeness hypothesis. |
| Ordinary ViT over 64 squares | Common vanilla Transformer baseline | Too generic and explicitly disallowed as a core idea; it would mostly test token attention capacity. |
| Plain GNN on board squares | Generic graph neural network over 8x8 adjacency | Too ordinary and likely weaker than CNNs unless augmented with relation edges, which would drift toward imported attack-defense graph families. |
| Hyperparameter tuning | Existing trainer configs | Disallowed as the research idea; it would not explain puzzle-likeness. |
| Ensembling several existing models | Existing leaderboard candidates | Disallowed as the core idea; improves variance but provides little scientific signal. |
| Bigger/deeper CNN | Small/medium/deep variants | Explicitly disallowed and not scientifically distinct. |
| Add more data or train on the full 45M file | Current sampled split versus full Parquet | Disallowed as a core idea and unsafe until streaming support exists. |
| Tactical sheaf/Hodge/curvature variant | Imported sheaf and Hodge packets | Already heavily represented; changing edge types or pooling would be a near-duplicate. |
| One-ply move-delta landscape | Imported counterfactual move-delta packets | The prompt forbids another move-delta set/multiset spectrum or DeepSets variant. |
| Sinkhorn or piece-target transport | Imported optimal-transport packets | Already represented; changing target buckets or temperatures is not a new mechanism. |
| Masked-board likelihood/surprise codec | Imported masked board code-length packet | Adjacent generative idea, but the selected mechanism uses continuous denoising score fields and a class-0-only prior rather than masked token prediction or code length. |
| Static pseudo-likelihood ratio | Imported geometry-conditioned pseudo-likelihood packet | Already represented; the selected score field avoids class-conditioned per-square conditional likelihoods. |
| Cubical topology, Hall defect, or king-cage path DP | Imported topology/matroid/path packets | Already represented and not needed for this experiment. |

## 6. Mathematical Thesis

### Input space definition

For the minimal experiment, let

```text
X_simple = {x in {0,1}^{18 x 8 x 8} satisfying the exported simple_18 channel convention}.
```

The first 12 channels are piece occupancy planes. The remaining 6 channels encode side-to-move, castling rights, and en-passant information. During score-prior training, embed `X_simple` into the continuous cube `[0,1]^{18 x 8 x 8}` and corrupt it with Gaussian noise. This continuous relaxation is a learning device, not a claim that illegal fractional boards are chess positions.

### Label/target definition

Let the fine label be

```text
f in {0,1,2}
```

and the binary target be

```text
y = 1[f >= 1].
```

The classifier outputs logits `g(x) in R^2` for `y in {0,1}`. Fine labels are used for diagnostics, not as neural-network input features. The denoising score prior uses only training rows with `y=0`, which are exactly the known non-puzzle rows in the current mapping.

### Data distribution assumptions

Let `P_0` denote the training distribution of verified non-puzzle boards over the continuous embedding of `simple_18`. Let `P_+` denote the distribution of verified near-puzzle or puzzle-like boards. The empirical split is assumed to be label-independent except for the intended train/val/test separation; no position from validation or test is used in the score-prior pretraining objective.

The working statistical assumption is not that puzzles are arbitrary anomalies. The stronger and more useful assumption is:

```text
some puzzle-like positions are local departures from the high-density manifold of ordinary non-puzzle positions, and the direction of the class-0 denoising repair field contains discriminative tactical evidence not captured by material counts alone.
```

### Allowed symmetry or equivariance assumptions

No global rotation/reflection equivariance is imposed. Chess is not fully rotation/reflection invariant because pawns, castling, en-passant, and side-to-move have direction. The model may learn translation-like local filters on the 8x8 board through convolution, but there is no exact dihedral, side-flip, file-mirror, or orbit quotient assumption in the central mechanism. This avoids duplicating imported orbit and side-canonical packets.

### Core hypothesis

Let `q_sigma(u | x) = N(x, sigma^2 I)` and let

```text
p_{0,sigma}(u) = integral q_sigma(u | x) dP_0(x)
```

be the Gaussian-smoothed non-puzzle density. Define the non-puzzle score field

```text
s_{0,sigma}(u) = grad_u log p_{0,sigma}(u).
```

The core hypothesis is:

```text
A bottlenecked stack {s_{0,sigma_k}(x)}_{k=1..K}, estimated from class-0-only denoising, improves binary puzzle classification and fine-label-1 diagnostics beyond an equal-capacity board-only classifier and beyond an all-class denoising prior.
```

### Formal object introduced

The introduced object is the **non-puzzle repair score stack**:

```text
S_0(x) = concat_k ((D_theta(x, sigma_k) - x) / sigma_k^2) in R^{(K*18) x 8 x 8},
```

where `D_theta(u, sigma)` is trained to reconstruct a clean class-0 board from a Gaussian-corrupted class-0 board:

```text
L_DSM(theta) = E_{x ~ P_0, sigma ~ pi, epsilon ~ N(0,I)}
               [ ||D_theta(x + sigma epsilon, sigma) - x||_2^2 / (2 sigma^2) ].
```

The classifier is

```text
g_phi(x) = h_phi(x, B_psi(S_0(x))),
```

where `B_psi` is a small convolutional bottleneck. The score prior may be frozen after pretraining or updated only through the denoising loss; the minimal experiment should freeze it for the cleanest falsification.

### Proposition and derivation

**Proposition.** Assume `X ~ P_0`, `U = X + sigma epsilon`, and `epsilon ~ N(0,I)`. For squared-error denoising over all measurable functions, the optimal denoiser satisfies

```text
D^*(u, sigma) = E[X | U = u]
              = u + sigma^2 grad_u log p_{0,sigma}(u).
```

Therefore

```text
(D^*(u, sigma) - u) / sigma^2 = grad_u log p_{0,sigma}(u).
```

**Proof sketch.** The squared-error minimizer is the conditional expectation `E[X | U=u]`. For Gaussian corruption,

```text
p_{0,sigma}(u) = integral p_0(x) N(u; x, sigma^2 I) dx.
```

Differentiating under the integral gives

```text
grad_u p_{0,sigma}(u)
= integral p_0(x) N(u; x, sigma^2 I) (x-u)/sigma^2 dx
= p_{0,sigma}(u) (E[X | U=u] - u)/sigma^2.
```

Dividing by `p_{0,sigma}(u)` yields the identity. This is the Tweedie/denoising-score identity underlying the architecture.

### Optimization objective

The full training objective for the idea-specific trainer is:

```text
min_{theta,psi,phi}  E_{(x,y)} CE(g_phi(x), y)
                   + lambda_dsm E_{x:y=0,sigma,epsilon}
                     ||D_theta(x + sigma epsilon, sigma) - x||_2^2 / (2 sigma^2)
                   + lambda_bottleneck ||B_psi(S_0(x))||_1
                   + lambda_tv TV(B_psi(S_0(x))).
```

For the cleanest minimal test, use two stages:

1. train `theta` on class-0 training rows with the denoising objective;
2. freeze `theta` and train `psi, phi` with balanced cross-entropy.

The auxiliary regularizers are optional and should be small. If they complicate the first implementation, omit `lambda_tv` and keep only the bottleneck dimensionality as the main regularizer.

### What is actually proven

The denoiser residual estimates the smoothed input score of the class-0 distribution under the idealized conditions of infinite data, sufficient model capacity, Gaussian corruption, and squared-error denoising. This justifies the score-field object mathematically.

### What remains only hypothesized

It is not proven that puzzle-likeness is detectable from non-puzzle score fields. It is also not proven that the empirical `simple_18` distribution has enough class-0 coverage for the score prior to learn useful ordinaryness rather than material/phase shortcuts. These are exactly what the ablations must test.

### Counterexamples where the idea should fail

- A puzzle position whose board is completely ordinary under the class-0 distribution but whose puzzle-likeness depends on a deep forced line invisible to current-board ordinaryness.
- A non-puzzle position from an unusual opening, study-like material imbalance, or rare castling/en-passant configuration that lies off the class-0 manifold and is falsely marked puzzle-like.
- A dataset where class `0` and classes `1/2` differ mostly by source artifacts or material distributions; the score prior could learn those artifacts unless material/nuisance ablations catch it.
- A near-puzzle class `1` whose labels are intentionally heterogeneous; class-0 score magnitude may correlate weakly with that boundary.

### Self-critique

The strongest objection is that “puzzle” is not the same as “anomalous under non-puzzles.” Many tactical puzzles are normal-looking positions, and many weird positions are not puzzles. The experiment is still worth running because the idea does not rely on raw anomaly score alone: it gives the supervised classifier a local **directional repair field** from the non-puzzle manifold, not merely a scalar outlier score. The all-class-prior ablation and material/nuisance controls make this falsifiable. If the class-0 score field adds nothing over the all-class or material-only prior, abandon this mechanism rather than scaling it.

## 7. Architecture Specification

### Module names

- `Simple18CanonicalAdapter`
- `NoiseLevelEmbedding`
- `OrdinaryScoreDenoiser`
- `ScoreFieldBottleneck`
- `NonPuzzleScoreFieldNet`
- optional trainer utility: `NonPuzzleScorePriorPretrainer`

### Encoding adapters

`Simple18CanonicalAdapter`:

- accepts `x: [B,18,8,8]` when `encoding=simple_18`;
- returns `x18: [B,18,8,8]` unchanged after validating channel count;
- may optionally clamp/float-cast to `[0,1]` during denoising-target preparation, but do not silently alter supervised inputs.

`lc0_static_112` and `lc0_bt4_112` adapters:

- first experiment should use only `simple_18` because channel semantics are unambiguous;
- a later adapter may map documented current-board LC0 channels into canonical `simple_18`;
- if the channel map is missing or ambiguous, fail closed with a clear error;
- history planes may be used only by a learned supervised adapter, not as score-prior denoising targets, unless Codex can prove the current-board channel map is documented and correct.

### Forward-pass steps

Default hyperparameters:

```text
K = 3
noise_sigmas = [0.05, 0.10, 0.20]
score_prior_hidden = 32
score_prior_blocks = 3
score_bottleneck_channels = 24
board_hidden = 32
fusion_hidden = 56
num_classes = 2
```

Forward pass for `NonPuzzleScoreFieldNet`:

1. Input `x: [B,C,8,8]`.
2. Adapter returns `x18: [B,18,8,8]`.
3. For each `sigma_k`, evaluate the frozen or denoising-loss-trained prior on the clean board:

   ```text
   recon_k = OrdinaryScoreDenoiser(x18, sigma_k)       # [B,18,8,8]
   score_k = (recon_k - x18) / (sigma_k^2)             # [B,18,8,8]
   ```

4. Concatenate score maps:

   ```text
   score_stack = concat(score_1, score_2, score_3, dim=channel)  # [B,54,8,8]
   ```

5. Score bottleneck:

   ```text
   z_score = ScoreFieldBottleneck(score_stack)  # [B,24,8,8]
   ```

6. Board encoder:

   ```text
   z_board = BoardStem(x18)  # [B,32,8,8]
   ```

7. Fusion:

   ```text
   z = concat(z_board, z_score, dim=channel)  # [B,56,8,8]
   z = FusionBlocks(z)                        # [B,56,8,8]
   pooled = concat(global_avg_pool(z), global_max_pool(z))  # [B,112]
   logits = MLP(pooled)                       # [B,2]
   ```

8. Return `logits` only from `forward` so the shared trainer, reports, confusion matrices, predictions, and leaderboards keep working. A separate method such as `forward_with_aux` may return score maps and denoising reconstructions for the idea-specific trainer.

### OrdinaryScoreDenoiser structure

Pseudocode only:

```text
sigma_emb = NoiseLevelEmbedding(log(sigma))          # [B,16]
sigma_map = broadcast(sigma_emb, 8, 8)               # [B,16,8,8]
h = concat(x_noisy_or_clean, sigma_map)              # [B,34,8,8]
h = Conv3x3(34, hidden=32) + SiLU + GroupNorm
repeat score_prior_blocks times:
    h = h + ResidualBlock32(h)
recon = Conv3x3(32, 18)(h)                           # [B,18,8,8]
return recon
```

Use real-valued reconstruction output for the score identity. For optional diagnostics, a separate piece-category head can be added later, but do not make it part of the minimal experiment.

### ScoreFieldBottleneck structure

```text
score_stack: [B,54,8,8]
z = Conv1x1(54,24) + GroupNorm + SiLU
z = DepthwiseConv3x3(24,24) + PointwiseConv1x1(24,24) + SiLU
return z: [B,24,8,8]
```

### Parameter-count estimate

With `score_prior_hidden=32`, `score_prior_blocks=3`, `board_hidden=32`, and two fusion residual blocks:

- `OrdinaryScoreDenoiser`: about `70k-90k` parameters.
- `ScoreFieldBottleneck`: about `2k-4k` parameters.
- Board encoder and fusion classifier: about `130k-220k` parameters depending on exact residual block implementation.
- Total active parameters: about `0.25M-0.35M`, comfortably smaller than many medium/deep CNN variants.

Codex should compute exact counts in the report.

### FLOP and complexity estimate

For one denoiser sigma with hidden width `H=32` and `R=3` residual blocks, approximate multiply-adds per sample are:

```text
64 * (34*H*9 + 2*R*H*H*9 + H*18*9)
```

With `H=32, R=3`, this is roughly `4.5M` MACs per sigma per sample. With `K=3`, score extraction is roughly `13.5M` MACs per sample before the smaller classifier. On an 8x8 board this is acceptable, but it is more expensive than a tiny CNN. Report wall-clock time.

### Candidate-set memory estimate and chunking plan

This idea has no generated legal-move candidate set. The main generated tensor is the score stack:

```text
score_stack memory = B * K * 18 * 8 * 8 * bytes_per_float.
```

For `B=512`, `K=3`, and fp32 this is about `7 MB`. Denoiser activations dominate training memory. If memory is tight:

- evaluate score maps one sigma at a time;
- freeze the prior and wrap score extraction in `torch.no_grad()` during classifier training;
- store only the concatenated score maps or bottlenecked maps;
- expose config `score_eval_chunk_size`, default `1` sigma at a time.

### Required config fields

Add idea-specific fields in `configs/nonpuzzle_score_field_simple18.yaml` or the idea config:

```text
model.name = nonpuzzle_score_field
model.input_channels = 18
model.num_classes = 2
model.noise_sigmas = [0.05, 0.10, 0.20]
model.score_prior_hidden = 32
model.score_prior_blocks = 3
model.score_bottleneck_channels = 24
model.freeze_score_prior_after_pretrain = true
model.score_prior_pretrain_epochs = 1
model.score_prior_train_on_binary_zero_only = true
model.fail_closed_on_unknown_encoding = true
```

Keep the public `forward(x)` contract returning logits `[B,2]`.

## 8. Loss, Training, And Regularization

Primary supervised loss:

```text
CrossEntropyLoss(logits, binary_target, weight=balanced_class_weights)
```

Auxiliary denoising loss for score-prior pretraining:

```text
L_DSM = mean_{x:y=0, sigma, epsilon}
        ||D_theta(x + sigma epsilon, sigma) - x||_2^2 / (2 sigma^2).
```

Recommended minimal schedule:

1. Pretrain `OrdinaryScoreDenoiser` for `1` epoch on training rows with binary label `0` only.
2. Freeze the denoiser.
3. Train `NonPuzzleScoreFieldNet` classifier for the same supervised epoch count as baselines, default `3`, with balanced class weighting.
4. Run the all-class-prior ablation with the same pretraining epoch count and same supervised epoch count.

Optional auxiliary during supervised training:

- If Codex implements joint training, keep `lambda_dsm <= 0.1` and apply the denoising loss only to binary-zero samples in the current batch.
- Do not backpropagate supervised classification loss into the score prior in the minimal experiment; freezing makes the central falsification cleaner.

Batch size expectations:

- Default `512` for classifier training.
- If pretraining memory is tight, use `512` or `256`; report any change.

Optimizer defaults:

```text
AdamW
learning_rate = 0.001
weight_decay = 0.0001
betas = (0.9, 0.999)
```

Regularizers:

- Balanced class weighting in the supervised loss.
- Optional dropout `0.05-0.10` in the final MLP only.
- Optional small `L1` penalty on `z_score` if score maps dominate; default off for first run.
- No data augmentation in the first run unless the baseline configs already use the same augmentation.

Determinism requirements:

- Use seed `42`.
- Use deterministic dataloader ordering when requested by existing configs.
- Store the sampled `noise_sigmas` policy and random seed in the report.
- For deterministic comparison, draw Gaussian denoising noise from the seeded PyTorch generator.

What must stay unchanged from existing benchmark configs for fair comparison:

- Split paths.
- Binary target mapping.
- Fine-label diagnostic reporting.
- Main supervised epochs, batch size, optimizer family, learning rate, weight decay, class weighting, and early stopping policy unless memory forces a documented change.
- Evaluation metrics and report artifact formats.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| **AllClassScorePrior** | Train the denoising prior on all training rows instead of binary-zero rows; keep everything else fixed. | The class-0 ordinaryness score field matters specifically, not just generic denoising features. | If equal or better, the selected class-0 score thesis is falsified; use generic self-supervised features or abandon score ordinaryness. |
| **NoScoreBranchMatchedParams** | Remove `S_0(x)` and replace the score branch with an equal-parameter CNN branch on `x18`. | Score maps add information beyond extra capacity. | If equal, the score field is not doing useful work. |
| **RandomScoreFieldMarginals** | Replace score maps with random fields matched per batch to each sigma/channel mean and standard deviation. | Spatial/directional score semantics matter. | If equal, the classifier uses nuisance statistics or capacity, not the repair vector field. |
| **SquarePermutedScoreField** | Apply a fixed random permutation of 64 squares to score maps, preserving channel/sigma values and global norms. | Board-local alignment of score vectors with pieces matters. | If equal, local repair geometry is irrelevant. |
| **MaterialBroadcastPrior** | Train a denoising branch that sees only material counts, side-to-move, castling, and en-passant broadcast to 8x8. | The prior is not merely material/phase/castling nuisance. | If equal, score prior likely learned low-level nuisance only. |
| **SigmaLessAutoencoder** | Use a standard clean/noisy autoencoder at one fixed noise level and feed reconstruction error instead of multi-sigma score stack. | Multi-noise score-field direction matters more than generic reconstruction. | If equal, denoising-score math may be unnecessary. |
| **NormOnlyScoreScalars** | Replace score maps with per-sigma scalar norms: mean, max, occupied-square mean, empty-square mean. | Spatial vector fields matter beyond anomaly magnitude. | If equal, a cheap scalar anomaly feature suffices. |
| **FrozenRandomDenoiser** | Keep architecture but use an untrained random denoiser; bottleneck and classifier train normally. | The learned non-puzzle prior is necessary. | If close to main, the branch is acting as random features. |
| **ScoreOnlyClassifier** | Use only bottlenecked score maps, no board encoder. | Score field alone is sufficient or complementary. | If strong, the score prior carries major signal; if weak but main strong, score maps need board context. |
| **ClassZeroSVDDControl** | Optional: train a simple one-class latent-distance branch on class `0` with similar parameter count. | Local score fields beat scalar one-class distance. | If SVDD control matches main, the new score-field object is overkill. |

Structured-object hard controls: this idea does not use a graph, hypergraph, sheaf, transport plan, legal move set, or search surrogate. The semantics-destroying controls above preserve obvious shortcuts such as tensor shape, channel count, sigma count, score-magnitude marginals, material counts, side-to-move, castling/en-passant availability, and parameter count while destroying the proposed score-field semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- existing simple CNN on `simple_18`;
- existing residual CNN on `simple_18`;
- best already-reported LC0 BT4-style CNN/residual CNN if available in the current leaderboard;
- `NoScoreBranchMatchedParams`;
- `AllClassScorePrior`;
- `RandomScoreFieldMarginals`;
- `MaterialBroadcastPrior`.

Metrics to inspect:

- validation and test accuracy;
- ROC-AUC if already available;
- PR-AUC if already available;
- binary precision/recall/F1;
- calibration metrics if already available, but do not make calibration the success criterion;
- rectangular `3x2` diagnostic matrix for every main and central ablation run;
- training time and parameter count.

Required fine-label diagnostic:

```text
For the main model and every central ablation, report:
true fine label 0 -> predicted 0 / predicted 1
true fine label 1 -> predicted 0 / predicted 1
true fine label 2 -> predicted 0 / predicted 1
```

Near-puzzle diagnostic:

- Compute class-`1` recall at a matched fine-label-`0` false-positive rate.
- Recommended fixed operating point: choose the threshold on validation so fine-label-`0` FPR is as close as possible to `10%`, then report class-`1` recall and class-`2` recall on test at that threshold.
- Also report class-`1` precision among predicted positives if the report pipeline supports it.

Required artifacts:

- trained model checkpoint;
- score-prior checkpoint;
- YAML config used;
- training log;
- validation and test metrics JSON/Markdown;
- fine-label `3x2` matrix for test;
- ablation comparison table;
- optional score-map visualizations for a few validation examples, with no engine information.

Success threshold:

- Main model improves test ROC-AUC or balanced accuracy over the best `simple_18` CNN/residual baseline by at least `1.0` percentage point, **and**
- main model improves class-`1` recall at matched fine-label-`0` FPR by at least `2.0` percentage points over the best `simple_18` baseline, **and**
- main model beats `AllClassScorePrior` and `NoScoreBranchMatchedParams` on the same near-puzzle diagnostic.

Failure threshold:

- Main model is within `0.3` percentage points of `AllClassScorePrior` and `NoScoreBranchMatchedParams` on balanced accuracy and class-`1` matched-FPR recall, or
- `MaterialBroadcastPrior` matches the main model, or
- `RandomScoreFieldMarginals` matches the main model.

What result would make me abandon the idea:

- Any central ablation that destroys class-0 score semantics while preserving compute and nuisance statistics matches or beats the main model across both binary metrics and class-`1` matched-FPR recall.

What result would justify scaling:

- Main model beats all central ablations and existing `simple_18` baselines on test, with a visible gain on fine label `1`, and the score-map visualizations are not dominated by material/castling/en-passant artifacts. Scaling would then mean trying a documented `lc0_static_112` current-board adapter or a longer score-prior pretrain, not immediately increasing CNN depth.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_nonpuzzle_score_field/idea.yaml` | Create | Machine-readable idea metadata, novelty claim, central ablation, config path, and status. |
| `ideas/20260421_nonpuzzle_score_field/math_thesis.md` | Create | Section 6 mathematical thesis, denoising score derivation, and failure counterexamples. |
| `ideas/20260421_nonpuzzle_score_field/architecture.md` | Create | Module descriptions, tensor shapes, parameter estimates, adapter assumptions, and pseudocode. |
| `ideas/20260421_nonpuzzle_score_field/implementation_notes.md` | Create | Notes on freezing the score prior, score extraction chunking, deterministic Gaussian noise, and fail-closed LC0 adapters. |
| `ideas/20260421_nonpuzzle_score_field/trainer_notes.md` | Create | Two-stage training plan, class-0 filtering, balanced CE, report requirements, and fairness controls. |
| `ideas/20260421_nonpuzzle_score_field/ablations.md` | Create | Ablation table from Section 9 plus exact config variants. |
| `ideas/20260421_nonpuzzle_score_field/train.py` | Create | Idea-specific wrapper that pretrains the score prior on binary-zero train rows, freezes it, then invokes or mirrors the shared trainer for supervised training and reports. |
| `ideas/20260421_nonpuzzle_score_field/config.yaml` | Create | Minimal experiment config using `simple_18`, split paths, model name, training defaults, and score-prior fields. |
| `ideas/20260421_nonpuzzle_score_field/report_template.md` | Create | Template requiring main metrics, `3x2` diagnostic matrix, matched-FPR near-puzzle diagnostic, ablations, and failure decision. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported memory after implementation, including anti-duplicate notes about class-conditional denoising score-field bottlenecks. Preserve all hard leakage and anti-duplicate constraints. |
| `src/chess_nn_playground/models/nonpuzzle_score_field.py` | Create | `Simple18CanonicalAdapter`, `NoiseLevelEmbedding`, `OrdinaryScoreDenoiser`, `ScoreFieldBottleneck`, `NonPuzzleScoreFieldNet`; forward returns logits. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register builder function `build_nonpuzzle_score_field` or equivalent model name `nonpuzzle_score_field`. |
| `configs/nonpuzzle_score_field_simple18.yaml` | Create | Shared-trainer-compatible config for the main run. Include score-prior fields if the config system allows extension. |
| `configs/nonpuzzle_score_field_simple18_allclass_prior.yaml` | Create | Central ablation config. |
| `configs/nonpuzzle_score_field_simple18_no_score.yaml` | Create | Matched-parameter no-score ablation config. |
| `configs/nonpuzzle_score_field_simple18_random_score.yaml` | Create | Random-score-marginal ablation config. |
| `tests/test_nonpuzzle_score_field.py` | Create | Focused tests for tensor shapes, fail-closed adapter behavior, deterministic sigma handling, logits shape `[B,2]`, and score-stack memory shape. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0922_tuesday_local_nonpuzzle_score_field.md
  generated_at: "2026-04-21 09:22 UTC-07:00"
  weekday: Tuesday
  timezone: local
  idea_slug: nonpuzzle_score_field
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_nonpuzzle_score_field
  name: Non-Puzzle Score-Field Bottleneck Network
  slug: nonpuzzle_score_field
  status: draft
  created_at: "2026-04-21 09:22 UTC-07:00"
  author: ChatGPT Pro
  short_thesis: Train a class-0-only denoising score prior and classify puzzle-likeness from bottlenecked non-puzzle repair vector fields.
  novelty_claim: Uses a smoothed input score field of the verified non-puzzle board distribution, not attack graphs, move deltas, Sinkhorn transport, nuisance projection, masked code length, or ordinary CNN capacity.
  expected_advantage: Should improve detection of near-puzzle and puzzle-like boards that locally deviate from ordinary non-puzzle structure while preserving the standard binary trainer interface.
  central_falsification_ablation: AllClassScorePrior with identical architecture and training budget.
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary_logits
  compute_notes: Score extraction evaluates a small denoiser at three noise levels; freeze and chunk by sigma if memory is tight.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/nonpuzzle_score_field_simple18.yaml
  model_path: src/chess_nn_playground/models/nonpuzzle_score_field.py
  latest_result_path: null
  notes: Minimal experiment should avoid LC0 encodings until a fail-closed current-board channel adapter is implemented.
```

```yaml
config_yaml:
  run:
    name: nonpuzzle_score_field_simple18
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
    name: nonpuzzle_score_field
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

```yaml
model_spec:
  model_name: nonpuzzle_score_field
  file_path: src/chess_nn_playground/models/nonpuzzle_score_field.py
  builder_function: build_nonpuzzle_score_field
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18CanonicalAdapter
    - NoiseLevelEmbedding
    - OrdinaryScoreDenoiser
    - ScoreFieldBottleneck
    - NonPuzzleScoreFieldNet
  required_config_fields:
    - input_channels
    - num_classes
    - noise_sigmas
    - score_prior_hidden
    - score_prior_blocks
    - score_bottleneck_channels
    - freeze_score_prior_after_pretrain
    - score_prior_train_on_binary_zero_only
    - fail_closed_on_unknown_encoding
  expected_parameter_count: "0.25M-0.35M with default hidden widths"
  expected_memory_notes: "Score stack memory is B*K*18*8*8 floats; for B=512,K=3,fp32 this is about 7 MB, while denoiser activations dominate. Evaluate sigmas sequentially when frozen."
```

```yaml
research_continuity:
  idea_fingerprint: "current-board simple_18 + class-0-only Gaussian denoising score prior + bottlenecked non-puzzle repair vector fields + binary classifier"
  already_researched_family_overlap: "Adjacent only to broad generative/MDL interests; deliberately not a masked-board code-length, pseudo-likelihood ratio, attack graph, move-delta, Sinkhorn, nuisance projection, symmetry quotient, topology, Hall-defect, or king-path model."
  closest_duplicate_risk: "Masked Board Code-Length Surprise Network, because both use label-free board reconstruction ideas; this packet differs by using class-0-only continuous denoising score fields and all-class-prior falsification, not masked token likelihood or code length."
  do_not_repeat_if_this_fails:
    - class-0-only denoising score-field bottlenecks on current board tensors
    - non-puzzle repair residual maps as puzzle-likeness evidence
    - Fisher-style discriminative classifiers over denoising residual score stacks
    - scalar anomaly variants of the same non-puzzle ordinaryness prior unless a new chess object is introduced
  suggested_next_search_directions:
    - label-safe selective prediction that changes deployment policy rather than board operator, only after core classifier gains plateau
    - genuinely new causal environments not based on source labels or imported material/phase/color partitions
    - supervised motif mechanisms that do not use attack graphs, move deltas, mask-code likelihood, or sparse witness piece selection
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Non-Puzzle Score-Field Bottleneck Network` to imported research memory after implementation, including the fingerprint `class-0-only denoising score prior + bottlenecked repair vector fields`. | Prevents the next research pass from proposing the same score-prior ordinaryness mechanism under a different name. | `Imported Research Memory` |
| Add an anti-duplicate rule: do not propose another class-conditional denoising autoencoder, diffusion-score, or Fisher-score bottleneck over current-board tensors unless it introduces a new formal observable and a stronger falsifier than all-class-prior and score-randomization controls. | Keeps future work from recycling denoising-score variants after this experiment resolves the question. | Anti-duplicate paragraph after masked-codec and pseudo-likelihood restrictions |
| Add a reusable ablation requirement for generative-prior ideas: include an all-class-prior control, a nuisance-only prior, and a semantics-destroyed random-field or permutation control. | This packet shows that generative priors can shortcut through material and phase; the prompt should demand hard controls. | `Ablation Plan` requirements |
| Add a note that class labels may be used as training targets or to filter auxiliary training subsets, but must never be inference-time inputs or provenance features. | Clarifies that class-0-only auxiliary training is allowed while preserving leakage rules. | `Problem Restatement And Data Contract` or `Non-Negotiable Constraints` |
| Add `score matching / denoising ordinaryness` to the list of explored concept families if the result is negative or inconclusive. | Avoids spending another deep-research pass on class-conditional score estimation if it fails. | `Research Continuity` |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0922_tuesday_local_nonpuzzle_score_field.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes, `simple_18` on the provided train/val/test split
- Falsification criterion is concrete: yes, all-class score prior plus matched no-score and randomized-score controls
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
- Not an exact near-duplicate of imported cubical Euler/Betti topology, Hall-defect overload, or king-cage/king-escape path-DP packets: yes
