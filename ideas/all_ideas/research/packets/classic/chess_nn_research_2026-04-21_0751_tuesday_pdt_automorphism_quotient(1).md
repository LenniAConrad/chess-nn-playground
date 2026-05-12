# Codex Handoff Packet: Rule-Automorphism Quotient Bottleneck Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0751_tuesday_pdt_automorphism_quotient.md`
- Generated at: 2026-04-21 07:51 PDT
- Weekday: Tuesday
- Timezone: PDT / America/Los_Angeles
- Idea slug: `automorphism_quotient`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Rule-Automorphism Quotient Bottleneck Network, abbreviated `RAQ-Net`.
- One-sentence thesis: A chess puzzle-like position should remain puzzle-like under the exact color/turn reversal symmetry of chess, and usually under file mirror when castling rights are absent, so training a classifier on the quotient of these safe rule automorphism orbits should suppress color, side, and file-orientation shortcuts without using engines, move trees, attack graphs, sheaves, or transport.
- Idea fingerprint: finite safe chess automorphism groupoid over the current board + shared convolutional encoder + masked Reynolds/orbit latent average + orbit-consistency no-collapse regularizer + per-transform risk-variance penalty + binary logits.
- Why this is not a common CNN/ResNet/Transformer variant: the central object is not a deeper feature extractor; it is an exact, label-preserving quotient operator over rule automorphism orbits, with explicit falsifiers that preserve the number of views and nuisance statistics while destroying chess-rule semantics.
- Current-data minimal experiment: train `RAQ-Net-simple18` on `data/splits/crtk_sample_3class/split_train.parquet`, validate on `split_val.parquet`, test on `split_test.parquet`, using only `simple_18` because its channel semantics are small enough to transform safely.
- Smallest central falsification ablation: replace the legal automorphism orbit with same-sized pseudo-orbits made by rank/file flips that do **not** swap color/side/castling consistently, while keeping the same encoder, parameter count, view count, CE loss, and regularizer weights.
- Expected information gain if it fails: if legal-orbit quotienting does not beat both a simple/residual CNN and the semantics-destroyed pseudo-orbit control, then this benchmark split probably does not reward suppressing color/turn/file artifacts; future cycles should not spend more effort on finite chess automorphism quotient bottlenecks unless a source-shift split is introduced.

## 3. Problem Restatement And Data Contract

The task is board-position classification for `chess-nn-playground`.

- Binary output `0`: non-puzzle.
- Binary output `1`: puzzle-like.
- Fine source labels:
  - fine `0`: known non-puzzle.
  - fine `1`: verified near-puzzle.
  - fine `2`: verified puzzle.
- Training target for the default binary run: `y_binary = 0` for fine `0`, and `y_binary = 1` for fine `1` or fine `2`.
- Diagnostic reporting must still include the rectangular `3x2` matrix: true fine label `0/1/2` by predicted binary output `0/1`.
- Model input contract: a PyTorch tensor shaped `(batch, C, 8, 8)`.
- Model output contract: logits shaped `(batch, num_classes)`, with `num_classes = 2`.
- Benchmark split:
  - train: `data/splits/crtk_sample_3class/split_train.parquet`
  - validation: `data/splits/crtk_sample_3class/split_val.parquet`
  - test: `data/splits/crtk_sample_3class/split_test.parquet`
- Do not point the existing trainer directly at the roughly 45M-row full Parquet dataset until streaming support exists.

Leakage checklist:

- Safe neural inputs: deterministic board coordinates, piece occupancy, side-to-move, castling rights, en-passant plane, and tensor views produced by rule symmetries from the same current board.
- Safe rule-derived features for this idea: the transform masks derived from whether castling rights are absent; the masks are used only to decide whether file mirror is an exact automorphism for that sample.
- Allowed but not central here: pseudo-legal attack geometry derived only from the current board.
- Leakage-prone and not used here: full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, move-tree consequences, or any rule oracle whose output is effectively a tactical solution label.
- Strictly forbidden as neural-network inputs: Stockfish or other engine evaluations, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, unresolved-pool membership, and dataset provenance.
- Fine labels are used only to create the supervised binary target and the required diagnostic matrix, never as input features.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry may be derived only from channels whose current-board semantics are explicitly known. History channels may be passed through learned neural adapters, but must not be transformed by hand unless their semantic mapping is registered and tested.

Boundary between safe rule-derived features and leakage:

- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board are allowed.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated.
- Engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance must never be neural-network inputs.
- For `lc0_static_112` and `lc0_bt4_112`, distinguish current-board channels used for deterministic geometry from history channels used only by learned neural adapters.

## 4. Research Map

External ideas used, with exact borrow/copy boundaries:

| Source | What is borrowed | What is not copied |
|---|---|---|
| Cohen and Welling, “Group Equivariant Convolutional Networks,” ICML 2016, https://proceedings.mlr.press/v48/cohenc16.html | The principle that known symmetries can reduce sample complexity through weight sharing or equivariance. | No group convolution kernels, no full image dihedral symmetry, no claim that chess is rotation/reflection invariant. |
| Cohen, Geiger, and Weiler, “A General Theory of Equivariant CNNs on Homogeneous Spaces,” arXiv 1811.02017, https://arxiv.org/abs/1811.02017 | The representation-theoretic distinction between equivariant feature maps and invariant readouts. | No homogeneous-space machinery is implemented; chess with castling is treated as a partial groupoid, not a homogeneous image domain. |
| Sannai et al., “Invariant and Equivariant Reynolds Networks,” JMLR 2024, https://www.jmlr.org/papers/volume25/22-0891/22-0891.pdf | The Reynolds idea: average a model or representation over a finite group to obtain an invariant function. | No universal approximation construction or large-group reduction is copied; `RAQ-Net` uses a tiny chess-specific orbit. |
| Benton et al., “Learning Invariances in Neural Networks,” arXiv 2010.11882, https://arxiv.org/pdf/2010.11882 | The idea that output averaging over transformations can make a classifier invariant. | No learned augmentation distribution; the transforms are fixed by chess rules and are ablated against semantics-destroyed pseudo-transforms. |
| Arjovsky et al., “Invariant Risk Minimization,” arXiv 1907.02893, https://arxiv.org/abs/1907.02893 | The causal-learning motivation for predictors whose optimal classifier is stable across environments. | No strong causal identification claim; group elements are generated environments, not real source domains. |
| Krueger et al., “Out-of-Distribution Generalization via Risk Extrapolation,” ICML 2021, https://proceedings.mlr.press/v139/krueger21a.html | The practical penalty on risk variance across environments. | No extrapolated-domain robust optimization claim; the penalty is only a training regularizer over automorphism views. |
| Bardes, Ponce, and LeCun, “VICReg,” arXiv 2105.04906, https://arxiv.org/abs/2105.04906 | The collapse-preventing combination of invariance, variance floor, and covariance decorrelation terms for paired views. | No self-supervised image pretraining protocol; paired views are deterministic chess-rule orbit elements of the same labeled board. |

Candidate search trace: serious mechanisms considered but not selected.

| Candidate mechanism | Why it was serious | Why it lost to `RAQ-Net` |
|---|---|---|
| Encoding-family invariant bottleneck across `simple_18`, `lc0_static_112`, and `lc0_bt4_112` | Puzzle-likeness should be independent of representation format, and multi-view invariance could suppress encoding artifacts. | It depends on reliable multi-encoding access and channel semantics inside the current training loop; minimal current-data implementation is less certain than a `simple_18` orbit experiment. |
| Label-safe evidential/selective boundary model using fine label `1` as an ambiguity diagnostic | It directly targets near-puzzle ambiguity without fabricating labels. | It risks becoming a head/loss tweak close to ordinal or calibration work, and it does not introduce a new board operator. |
| Masked generative compression or MDL motif code over board tensors | Puzzle-like positions may be unusually compressible under tactical motifs. | The imported pseudo-likelihood/description-length packet already covers a nearby static-board generative ratio family; novelty would be hard to defend. |
| Neural cellular automaton over attacks-to-kings without move generation | Iterated local message passing could approximate tactical pressure without engines. | It is too close to static attack-defense graph/sheaf families unless it uses a substantially new formal object, which this pass did not find. |
| Spectral harmonic classifier on square-color and rank/file parity irreps | Low-frequency board harmonics might separate tactical geometry from material clutter. | It risks being a shallow feature-engineering baseline and overlaps high-order constellation/spectral compression ideas without a strong falsifier. |
| Conformal/selective abstention calibrated on fine label `1` | It would improve operational reliability and expose ambiguous examples. | It is better as a reporting layer after a stronger representation is found; alone it is not a puzzle-likeness architecture. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Chess rule symmetry | Safe automorphism groupoid with color/turn rank reversal always valid, and file mirror only when castling rights are absent | `(B,C,8,8) -> (B,K_max,C,8,8), mask (B,K_max)` | Replace transforms with same-sized pseudo-transforms that preserve counts but violate color/side/castling semantics | Not an attack graph, sheaf, Hodge operator, move-delta set, or transport coupling |
| Reynolds quotient | Masked orbit average of latent vectors | `(B,K,D), mask -> (B,D)` | Keep augmentation but remove masked latent averaging and classify only the original view | Uses a chess-specific partial groupoid rather than standard full image D4 pooling |
| Invariant risk | Auxiliary CE risk variance across valid transform environments | per-view logits `(B,K,2) -> K valid risks` | Set `lambda_rex=0` while preserving all orbit views | Not nuisance projection; no closed-form residualization and no source/domain labels |
| No-collapse paired-view bottleneck | VICReg-style orbit projection loss with invariance, variance floor, and covariance decorrelation | projection vectors `(B*K_valid,P)` | Pair each board with another same-label/material-bucket board instead of its own transform | Not a sparse witness, ordinal ladder, pseudo-likelihood, or ANOVA constellation model |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Plain simple CNN | `src/chess_nn_playground/models/cnn.py` | Already present and tests generic local convolution, not the quotient hypothesis. |
| Plain residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | Already present and would only change capacity/depth, not the causal symmetry shortcut question. |
| LC0-style CNN or residual CNN | Existing LC0 BT4-style CNN/residual variants | Already represented; copying LC0-style planes or residual blocks is not a new research mechanism. |
| Ordinary vanilla ViT over 64 squares | Standard square-token Transformer | Too generic, compute-heavier than needed, and explicitly excluded as a core idea. |
| Plain GNN on squares | Basic square-neighbor or piece-neighbor graph network | Too ordinary and easily drifts into static attack-defense graph variants already researched. |
| Hyperparameter tuning | Any existing baseline with changed LR/depth/width | Explicitly excluded and would not explain a new puzzle-likeness inductive bias. |
| Ensembling several baselines | Ensemble of CNN/residual/LC0 models | Explicitly excluded and would obscure whether any operator learned something new. |
| Another attack-defense graph/sheaf/Hodge model | Imported tactical sheaf/Hodge packets | Already researched family; adding edge labels or changing pooling would be a near-duplicate. |
| One-ply move-delta bag, landscape, entropy, or spectrum | Imported counterfactual move-delta packets | Already researched family and close to move-tree consequences; not needed for this quotient test. |
| Entropic piece-target transport | Imported optimal-transport packets | Already researched family; changing costs or temperatures would be duplicate work. |
| Ordinal ladder over fine labels | Imported ordinal evidence ladder | Useful diagnostic but already researched and does not create a new board operator. |
| Pseudo-likelihood or MDL board-ratio model | Imported geometry-conditioned pseudo-likelihood packet | Strong candidate, but already covered closely enough to reject for novelty. |
| Sparse witness-piece bottleneck | Imported sparse witness packet | It answers “which pieces witness the tactic,” whereas this idea answers “which representation survives rule automorphism quotienting.” |
| Ray-language automaton | Imported weighted finite automaton packet | Uses oriented rays and formal languages already represented in imported memory. |

## 6. Mathematical Thesis

### Input space definition

Let `X` be the set of current-board tensor encodings that can be decoded into at least:

\[
B=(O, s, c, e),
\]

where `O` is piece occupancy by type and color on `8 x 8`, `s` is side to move, `c` is the four castling-right bits, and `e` is the en-passant square or empty. For the minimal experiment, `X = {simple_18 tensors}` with a registered channel specification.

### Label/target definition

Fine labels are `F in {0,1,2}`. The binary label is

\[
Y = \mathbf{1}\{F \in \{1,2\}\}.
\]

The network predicts logits for `Y`. The fine label is retained only for diagnostics.

### Data distribution assumptions

Samples are drawn from the fixed benchmark split distribution `P(X,F)`. The core assumption is not that train and test are out-of-distribution; it is that the empirical distribution may contain non-causal artifacts correlated with labels, such as side-to-move, piece-color frequency, or file orientation, and that these artifacts should not be necessary for puzzle-likeness.

### Allowed symmetry or equivariance assumptions

Chess is not fully invariant to rotations or reflections. Pawns, side-to-move, castling, and en-passant matter. This idea uses only the following safe transformations:

1. Identity `I`.
2. Color/turn reversal `C`: rank `r -> 7-r`, swap white and black piece planes, toggle side-to-move, swap white castling rights with black castling rights on the same files, and rank-flip the en-passant square. This is treated as an exact chess-rule automorphism.
3. File mirror `H`: file `f -> 7-f`, piece colors and side-to-move unchanged, en-passant file mirrored. This is used only when all castling rights are absent. With castling rights present, file mirror is not assumed exact because orthodox castling is tied to the e-file and asymmetric rook distances.
4. Composition `HC` only when `H` is valid for that sample.

Thus each sample has a valid transform set

\[
G_x \subseteq \{I, C, H, HC\}, \qquad \{I,C\}\subseteq G_x,
\]

with a mask for invalid file-mirror views. This is a small groupoid rather than a global full-image symmetry group.

### Core hypothesis

For rule-safe transforms `g in G_x`, puzzle-likeness is invariant:

\[
P(Y=1\mid X=x)=P(Y=1\mid X=T_gx).
\]

A representation that factors through the quotient orbit `x / G_x` should discard shortcuts that change under color/turn reversal or file mirror while preserving board relations that are truly relevant to puzzle-likeness.

### Formal object introduced

Let `phi_theta: X -> R^D` be a shared encoder and `p_theta: R^D -> R^P` a projection head. Define valid orbit latents

\[
z_g(x)=\phi_\theta(T_gx),\quad g\in G_x.
\]

Define the masked Reynolds quotient latent

\[
\bar z(x)=\frac{1}{|G_x|}\sum_{g\in G_x} z_g(x).
\]

The main classifier is

\[
f_{\theta,w}(x)=W\bar z(x)+b \in R^2.
\]

Auxiliary per-view logits are

\[
f^g_{\theta,w}(x)=Wz_g(x)+b.
\]

The training objective is

\[
\min_{\theta,w}\; \mathbb{E}\left[\ell_{CE}(f_{\theta,w}(X),Y)\right]
+\lambda_{inv}\,\mathcal{L}_{orbit}
+\lambda_{rex}\,\mathcal{L}_{risk}
+\lambda_{kl}\,\mathcal{L}_{small},
\]

where `L_small` is an optional mild latent-norm or variational bottleneck term; it should default to `0` in the first run.

The orbit consistency term is

\[
\mathcal{L}_{orbit}=\mathbb{E}\left[\frac{1}{|G_X|}\sum_{g\in G_X}\left\|p_\theta(z_g(X))-\bar p(X)\right\|_2^2\right]
+\alpha_v\mathcal{L}_{variance}+\alpha_c\mathcal{L}_{covariance},
\]

with

\[
\bar p(X)=\frac{1}{|G_X|}\sum_{g\in G_X}p_\theta(z_g(X)).
\]

`L_variance` and `L_covariance` are VICReg-style no-collapse terms computed over all valid projection vectors in the batch.

The risk-variance term is

\[
\mathcal{L}_{risk}=\operatorname{Var}_{g\in\mathcal{G}_{batch}}\left(
\frac{1}{|\mathcal{B}_g|}\sum_{i:g\in G_{x_i}}\ell_{CE}(f^g_{\theta,w}(x_i),y_i)
\right),
\]

where `G_batch` is the set of transform labels valid for at least one sample in the minibatch.

### Proposition

If `L_orbit = 0` without collapse and `p_theta` is injective on the classifier-relevant subspace, then the projected representation used by the classifier is constant on every valid orbit:

\[
p_\theta(\phi_\theta(x))=p_\theta(\phi_\theta(T_gx)) \quad \forall g\in G_x.
\]

Consequently, any classifier depending only on the quotient latent cannot distinguish two boards related by a valid rule automorphism. In particular, any scalar shortcut `a(x)` with nonzero orbit variance, `a(x) != a(T_gx)` for some valid `g`, cannot be represented as a deterministic function of the quotient projection.

### Proof sketch or derivation

`L_orbit = 0` implies every valid projected latent equals its within-orbit mean, so projected latents are equal along orbit morphisms. Equality along equivalence classes is exactly the factorization condition: there exists a function `h` on quotient orbits such that `p_theta(phi_theta(x)) = h([x])`. Since the final logits use either the orbit mean or the same classifier on orbit-equal latents, the logits are invariant. A non-invariant shortcut cannot be a function of the quotient because it assigns different values to members of the same orbit.

The risk penalty does not prove invariance by itself. It encourages the same readout to incur similar loss on each valid transform environment, reducing incentives to use features that work only for one orientation or side/color naming convention.

### What is actually proven

The quotient construction proves exact invariance of the output if the model uses the masked orbit mean at inference. The consistency loss proves projected latent equality only when optimized to zero and only on valid transform pairs seen during training. The proposition proves removal of non-invariant shortcuts from the quotient representation, not better accuracy.

### What remains only hypothesized

It is a hypothesis that this benchmark has harmful non-invariant shortcuts and that removing them improves test AUROC, balanced accuracy, or fine-label-`1` recall at matched fine-label-`0` false-positive rate. It is also a hypothesis that the learned encoder keeps enough invariant tactical information after quotienting.

### Counterexamples where the idea should fail

- If labels or sampling pipelines contain real asymmetry not preserved by color/turn reversal, quotienting can remove predictive information.
- If the random train/test split has the same shortcut distribution, a baseline CNN may already exploit shortcuts without test penalty.
- If `simple_18` channel semantics are misregistered, transforms can corrupt positions and make the experiment invalid.
- If many positions have castling rights and the implementation mistakenly applies file mirror anyway, the transform is not rule-safe.
- If all useful signal is already invariant and the baseline already learns it, `RAQ-Net` may only add compute.

### Self-critique

The strongest objection is that finite automorphism invariance may be too weak for puzzle-likeness: tactics depend on pins, overloads, king exposure, and move consequences, none of which are explicitly modeled here. The counterargument is that the experiment is cheap, label-safe, and has unusually clean falsifiers. If legal-orbit quotienting beats semantics-destroyed pseudo-orbits while preserving the fine-label confusion profile, it demonstrates that even a small rule quotient removes harmful artifacts. If it fails, the lab learns to avoid this symmetry-bottleneck family and move toward semantic or ambiguity-focused mechanisms.

## 7. Architecture Specification

### Module names

- `RuleAutomorphismQuotientNet`
- `Simple18AutomorphismOrbit`
- `Lc0AutomorphismOrbit` with fail-closed semantics checks; optional future scale-up only.
- `SharedBoardEncoder`
- `MaskedReynoldsPool`
- `OrbitProjectionHead`
- `QuotientClassifier`

### Forward-pass steps

Default `forward(x)` must return logits only, shaped `(B,2)`, to remain compatible with the shared trainer. A custom training path may call `forward(x, return_aux=True)` to obtain auxiliary tensors.

1. Input:
   - `x`: `(B,C,8,8)`.
2. Orbit generation:
   - `x_orbit, orbit_mask = orbit_adapter(x)`.
   - `x_orbit`: `(B,K_max,C,8,8)`, with `K_max=4` for `[I,C,H,HC]`.
   - `orbit_mask`: `(B,K_max)`, bool or float. `I` and `C` always valid; `H` and `HC` valid only when all castling rights are absent and `use_file_mirror_if_castling_absent=true`.
3. Flatten valid view dimension for convolution:
   - reshape to `(B*K_max,C,8,8)`.
4. Shared encoder:
   - `h = SharedBoardEncoder(x_orbit_flat)`.
   - Recommended first version: stem `Conv2d(C,64,3,padding=1)`, 4 small residual blocks at 64 channels, global average pool.
   - output `z_flat`: `(B*K_max,D)`, with `D=128`.
5. Reshape:
   - `z`: `(B,K_max,D)`.
6. Masked Reynolds pool:
   - `z_bar = sum_k mask[:,k]*z[:,k,:] / sum_k mask[:,k]`.
   - `z_bar`: `(B,D)`.
7. Classifier:
   - `logits = Linear(D,2)(LayerNorm(z_bar))`.
   - output `(B,2)`.
8. Auxiliary tensors when `return_aux=True`:
   - per-view logits: `(B,K_max,2)` from the same classifier applied to each `z[:,k,:]`.
   - projection vectors: `(B,K_max,P)`, with `P=64`.
   - orbit mask: `(B,K_max)`.
   - optional orbit variance diagnostic per sample: `(B,)`.

### Parameter-count estimate

For `simple_18`, using width `64`, depth `4`, `D=128`, `P=64`:

- stem: about `18*64*3*3 = 10,368` weights.
- four residual blocks, two `3x3` convs each: about `4*2*64*64*3*3 = 294,912` weights.
- pooling projection `64 -> 128`: about `8,320` weights.
- projection head `128 -> 128 -> 64`: about `24,704` weights.
- classifier `128 -> 2`: about `258` weights.
- normalization and biases: under `5,000` parameters.
- Total: roughly `0.35M` to `0.45M` parameters depending on exact block normalization.

This should be comparable to small baselines, not a bigger-network proposal.

### FLOP and complexity estimate

Let `K` be the maximum orbit views, `K=4`, and `V_i` be valid views per sample, normally `2` or `4`. Convolutional cost is approximately `K` times the underlying encoder if implemented naively. With an 8x8 board and width 64, this is still small.

Memory for generated orbits:

\[
\text{bytes} \approx B \cdot K \cdot C \cdot 8 \cdot 8 \cdot \text{bytes_per_float}.
\]

For `B=512`, `K=4`, `C=18`, fp32: about `9.4 MB` for the orbit input tensor. For `C=112`, fp32: about `58.7 MB`. Latent memory is `B*K*D*4`, about `1 MB` for `D=128`.

Chunking plan:

- Add config `orbit_chunk_size` default `0` meaning no chunking.
- If `B*K` is too large, process orbit views in chunks along flattened dimension and concatenate latent outputs.
- The orbit adapter itself is cheap; memory pressure comes from feature maps inside the encoder.

### Required config fields

- `model.name: rule_automorphism_quotient_net`
- `model.input_channels: 18` for the first run.
- `model.num_classes: 2`
- `model.encoding: simple_18`
- `model.hidden_channels: 64`
- `model.latent_dim: 128`
- `model.projection_dim: 64`
- `model.num_res_blocks: 4`
- `model.use_color_turn_reversal: true`
- `model.use_file_mirror_if_castling_absent: true`
- `model.fail_closed_unknown_channels: true`
- `loss.lambda_orbit: 0.1`
- `loss.lambda_rex: 0.05`
- `loss.vicreg_variance_weight: 0.01`
- `loss.vicreg_covariance_weight: 0.01`
- `loss.lambda_latent_small: 0.0`

### Encoding support

First experiment should use only `simple_18` because the channel inventory is small and matches the required rule transforms. `RAQ-Net` can later support LC0 encodings only if Codex adds explicit channel specs and tests.

Encoding-adapter assumptions:

- `simple_18`:
  - Needs a registered plane map for 12 piece planes, side-to-move, four castling planes, and en-passant plane.
  - Do not silently assume plane order unless the repository already documents it. Add a `Simple18ChannelSpec` and tests using known FENs.
  - Color/turn reversal swaps white and black piece planes, flips ranks, toggles side-to-move, swaps castling planes `K<->k` and `Q<->q`, and rank-flips the en-passant plane.
  - File mirror flips files and en-passant file; it is valid only if all four castling planes are zero for that sample.
- `lc0_static_112`:
  - If current-board piece, side, castling, and en-passant channels are explicitly mapped, the same transforms can be applied to those channels.
  - If channel semantics are unknown, the adapter must raise a clear error when orbit training is requested.
- `lc0_bt4_112`:
  - History planes must not be transformed by hand unless every time-slice piece plane and auxiliary plane is mapped.
  - If history support is absent, do not mix transformed current channels with untransformed history channels. Fail closed.
  - A future safe option is to transform every piece-history slice consistently, but that must be separately tested.

### Pseudocode

```python
# Pseudocode only; do not paste as final implementation.
class RuleAutomorphismQuotientNet(nn.Module):
    def forward(self, x, return_aux=False):
        x_orbit, mask = self.orbit_adapter(x)       # (B,K,C,8,8), (B,K)
        B, K, C, H, W = x_orbit.shape
        z_flat = self.encoder(x_orbit.view(B*K,C,H,W))
        z = z_flat.view(B,K,self.latent_dim)        # (B,K,D)
        z_bar = masked_mean(z, mask, dim=1)         # (B,D)
        logits = self.classifier(self.norm(z_bar)) # (B,2)
        if not return_aux:
            return logits
        view_logits = self.classifier(self.norm(z)) # (B,K,2), broadcast LayerNorm
        proj = self.projection_head(z)              # (B,K,P)
        return {"logits": logits, "view_logits": view_logits,
                "proj": proj, "orbit_mask": mask, "z": z}
```

## 8. Loss, Training, And Regularization

Primary loss:

- Balanced cross-entropy on quotient logits `(B,2)` using `y_binary`.
- Class weighting should match existing benchmark behavior, default `balanced`.

Auxiliary losses:

- `L_orbit_inv`: mean squared deviation of each valid projection vector from its within-sample orbit mean.
- `L_vicreg_var`: variance floor over valid projection vectors in the minibatch to prevent collapse.
- `L_vicreg_cov`: covariance off-diagonal penalty over valid projection vectors.
- `L_rex`: variance of per-transform CE risks computed from per-view logits.
- `L_latent_small`: optional latent norm or variational bottleneck term, default `0.0` for the first experiment.

Recommended total loss:

\[
L = L_{CE} + 0.1L_{orbit\_inv}+0.01L_{vicreg\_var}+0.01L_{vicreg\_cov}+0.05L_{rex}.
\]

Batch size expectations:

- Start with `batch_size=512` for `simple_18`.
- If memory is tight, reduce to `256` before reducing model width.
- Keep the benchmark epochs at `3` initially to match the quick-cycle setup.

Optimizer defaults:

- AdamW.
- learning rate `1e-3`.
- weight decay `1e-4`.
- no mixed precision in the first deterministic run.

Regularizers:

- Existing weight decay.
- Orbit consistency as the central regularizer.
- VICReg-style variance/covariance only on projection vectors, not logits.
- Do not add label smoothing in the first run because it would confound near-puzzle diagnostics.

Determinism requirements:

- Set seed `42`.
- Use deterministic PyTorch settings if the repository supports them.
- Orbit transforms must be deterministic, with no stochastic augmentation in the main model.
- The pseudo-orbit falsification ablation may use seeded random permutations generated once and logged.

What must stay unchanged for fair comparison:

- Same train/val/test split.
- Same binary target construction.
- Same evaluation scripts, prediction artifacts, 3x2 diagnostic matrices, and leaderboard hooks.
- Same number of epochs for the first benchmark comparison.
- Same input encoding for primary comparison: `simple_18`.
- No extra data, no engine outputs, no source labels, no unresolved-pool labels.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Central pseudo-orbit falsifier | Replace `C/H/HC` with same-count transforms that flip ranks/files without consistent color, side, castling, and en-passant updates | Legal rule semantics, not merely extra views, cause any gain | If pseudo-orbit matches main model, the quotient mechanism is not using chess-rule semantics |
| Augmentation-only control | Train on transformed views with CE but classify original view; no masked Reynolds pool, no orbit loss, no risk variance | Quotient bottleneck matters beyond ordinary data augmentation | If it matches main model, the main contribution is just augmentation |
| Reynolds-only control | Use masked orbit mean at inference but set `lambda_orbit=lambda_rex=0` | Explicit orbit consistency/risk penalties matter | If it matches main model, averaging alone is sufficient |
| No-risk-variance control | Set `lambda_rex=0`, keep orbit consistency and Reynolds pool | Risk invariance across transform environments adds value | If it matches, REx-style term is unnecessary |
| No-VICReg control | Remove variance/covariance no-collapse terms, keep pairwise orbit MSE | Collapse prevention matters for stable orbit projection | If metrics are unchanged, simplify future implementation |
| Color/turn-only control | Disable file mirror entirely; use only `I` and `C` | Optional castling-safe file mirror adds information | If better than full model, file mirror is harmful or too noisy |
| File-mirror-unsafe stress test, report-only | Apply `H` even with castling rights, but do not use as default | Tests whether the exactness restriction matters | If this improves, inspect for castling artifacts before trusting it; it may be exploiting invalid transforms |
| Orbit-count nuisance control | Feed only `|G_x|` or the valid-view mask to a tiny head, plus baseline CNN logits frozen | Variable orbit count could be a shortcut | If count-only helps, main results must report count-stratified metrics |
| Same-label/material paired-view control | Pair each board with another board of same binary label and material bucket instead of its transform | The paired-view loss needs same-position orbit semantics, not just same-label clustering | If it matches, the loss is acting like supervised contrastive regularization rather than automorphism quotienting |
| Side/color shortcut audit | Train a small probe from frozen latents to predict side-to-move and dominant material color | Quotienting reduces non-causal side/color information | If probes remain as strong as baseline, the bottleneck did not remove intended shortcuts |

The smallest ablation that falsifies the central mathematical claim is the central pseudo-orbit falsifier: it preserves view count, encoder capacity, loss shape, and nuisance statistics while breaking the rule-correct transform semantics.

Because this is a structured operator over a transform groupoid, Codex must include at least one semantics-destroying randomized ablation. The pseudo-orbit and same-label/material paired-view controls satisfy this requirement. There is no generated move-set or candidate move set in this idea, so move-count and capture-histogram controls are not applicable.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN.
- Existing `simple_18` residual CNN of comparable parameter count.
- Existing `lc0_static_112` or `lc0_bt4_112` CNN/residual only as context, not as the primary fairness comparison unless `RAQ-Net` is also implemented safely for that encoding.
- `RAQ-Net` central pseudo-orbit falsifier.
- `RAQ-Net` augmentation-only control.
- `RAQ-Net` Reynolds-only control.

Metrics to inspect:

- Test accuracy.
- Balanced accuracy.
- AUROC.
- AUPRC.
- Cross-entropy / negative log likelihood.
- Brier score and expected calibration error if available.
- Binary `2x2` confusion matrix.
- Required fine-label `3x2` diagnostic matrix for the main model and every central ablation.
- Per-transform consistency: mean KL or probability difference between `f(x)` and `f(T_Cx)`.

Near-puzzle diagnostic:

- On validation, choose thresholds that match a fixed fine-label-`0` false-positive rate, preferably `5%` and `10%`.
- On test, report fine-label-`1` recall and fine-label-`2` recall at those matched thresholds.
- Also report class-`1` precision among predicted positives if the reporting code supports it.

Required artifacts:

- Model config YAML.
- Training log.
- Best checkpoint path.
- Predictions Parquet or CSV with true fine label, binary target, logits/probabilities, prediction, and split.
- Main report Markdown.
- Confusion matrices including `3x2` fine-label diagnostics.
- Ablation reports for pseudo-orbit and augmentation-only at minimum.
- Transform audit report verifying involutions and channel-plane preservation.

Success threshold:

- Primary: improve simple_18 residual CNN test AUROC by at least `+1.0` percentage point, or improve fine-label-`1` recall at matched fine-label-`0` FPR by at least `+3.0` percentage points.
- Secondary: beat augmentation-only and pseudo-orbit controls by at least `+0.5` AUROC point or `+1.5` fine-label-`1` recall points at matched FPR.
- Do not accept a model that gains class-`1` recall by collapsing class-`0` specificity; matched-FPR reporting is mandatory.

Failure threshold:

- Main model is within `±0.3` AUROC points of residual CNN and within `±1.0` class-`1` matched-FPR recall point.
- Pseudo-orbit control matches or beats the legal-orbit model.
- Fine-label `3x2` matrix shows class-`2` recall drops by more than `2` points without a compensating class-`1` diagnostic gain.

What result would make us abandon the idea:

- Legal-orbit `RAQ-Net`, pseudo-orbit falsifier, and augmentation-only control are statistically indistinguishable, and transform consistency does not correlate with errors. Do not repeat finite automorphism quotient bottlenecks in the next cycle if this happens.

What result would justify scaling:

- Legal-orbit model beats both residual CNN and pseudo-orbit control on AUROC or matched-FPR class-`1` recall, with no class-`2` recall collapse. Then scale to wider `simple_18`, and only then attempt `lc0_static_112` or `lc0_bt4_112` after channel-semantic tests pass.

## 11. Implementation Plan For Codex

Use idea id `20260421_0751_automorphism_quotient`.

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0751_automorphism_quotient/idea.yaml` | Create | Machine-readable idea metadata from Section 12. |
| `ideas/20260421_0751_automorphism_quotient/math_thesis.md` | Create | Section 6, including groupoid definition, objective, proposition, and self-critique. |
| `ideas/20260421_0751_automorphism_quotient/architecture.md` | Create | Section 7 with tensor contracts, modules, pseudocode, and parameter estimates. |
| `ideas/20260421_0751_automorphism_quotient/implementation_notes.md` | Create | Channel-spec rules, fail-closed behavior, transform tests, and chunking notes. |
| `ideas/20260421_0751_automorphism_quotient/trainer_notes.md` | Create | Section 8 plus how custom training consumes `return_aux=True` while shared trainer receives logits by default. |
| `ideas/20260421_0751_automorphism_quotient/ablations.md` | Create | Section 9 ablation table and required controls. |
| `ideas/20260421_0751_automorphism_quotient/train.py` | Create | Thin experiment entrypoint that loads config, trains `RAQ-Net` with auxiliary orbit losses, and writes standard predictions/reports. Reuse existing dataset, evaluation, and reporting utilities wherever possible. |
| `ideas/20260421_0751_automorphism_quotient/config.yaml` | Create | First-run simple_18 config with `rule_automorphism_quotient_net`, batch size 512, 3 epochs, balanced weighting. |
| `ideas/20260421_0751_automorphism_quotient/report_template.md` | Create | Report shell requiring main metrics, `3x2` matrix, matched-FPR near-puzzle diagnostics, and ablation comparison. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported memory after implementation; add anti-duplicate rule for finite chess automorphism quotient bottlenecks if it fails. Preserve hard leakage and label constraints. |
| `src/chess_nn_playground/models/rule_automorphism_quotient.py` | Create | `RuleAutomorphismQuotientNet`, orbit adapters, masked pool, and builder function. Default forward returns logits. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register `rule_automorphism_quotient_net`. Keep old model names intact. |
| `configs/rule_automorphism_quotient_simple18.yaml` | Create | Repo-level config for standard trainer compatibility and idea train script. |
| `tests/test_rule_automorphism_quotient.py` | Create | Focused tests: output shape, transform involution for `C`, piece-count preservation, file mirror disabled with castling rights, no silent LC0 use without channel spec, default forward returns tensor logits. |
| `tests/test_simple18_channel_spec.py` | Create if needed | Validate channel maps on hand-built positions or known FEN encoder outputs. |

For `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md`, Codex should update the prompt after consuming this output. The update should preserve hard constraints while adding reusable lessons, new anti-duplicate rules, clearer output requirements, and failure-mode guidance discovered from this research pass.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0751_tuesday_pdt_automorphism_quotient.md
  generated_at: 2026-04-21 07:51 PDT
  weekday: tuesday
  timezone: pdt
  idea_slug: automorphism_quotient
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0751_automorphism_quotient
  name: Rule-Automorphism Quotient Bottleneck Network
  slug: automorphism_quotient
  status: draft
  created_at: 2026-04-21 07:51 PDT
  author: ChatGPT Pro
  short_thesis: Learn puzzle-likeness from a masked quotient latent over exact chess rule automorphism orbits rather than from orientation, color, or side-to-move artifacts.
  novelty_claim: Uses a chess-specific safe automorphism groupoid with Reynolds pooling, orbit consistency, and transform-risk variance; avoids attack graphs, sheaves, move-delta sets, Sinkhorn transport, ordinal heads, pseudo-likelihood, and deterministic nuisance projection.
  expected_advantage: Better fine-label-1 recall at matched fine-label-0 false-positive rate and lower side/color shortcut sensitivity on simple_18.
  central_falsification_ablation: Same-sized pseudo-orbit transforms that preserve view count and nuisance statistics but violate color/side/castling semantics.
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112 and lc0_bt4_112 only with explicit fail-closed channel specs
  output_heads: binary logits plus optional auxiliary per-view logits and projection vectors for custom training
  compute_notes: About 0.4M parameters; up to four 8x8 orbit views per sample; chunk flattened orbit dimension if needed.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/rule_automorphism_quotient_simple18.yaml
  model_path: src/chess_nn_playground/models/rule_automorphism_quotient.py
  latest_result_path: null
  notes: Default forward must return logits only; idea train.py may request auxiliary tensors with return_aux=True.
```

```yaml
config_yaml:
  run:
    name: rule_automorphism_quotient_simple18
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
    name: rule_automorphism_quotient_net
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
  model_name: rule_automorphism_quotient_net
  file_path: src/chess_nn_playground/models/rule_automorphism_quotient.py
  builder_function: build_rule_automorphism_quotient_net
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18AutomorphismOrbit
    - SharedBoardEncoder
    - MaskedReynoldsPool
    - OrbitProjectionHead
    - QuotientClassifier
  required_config_fields:
    - input_channels
    - num_classes
    - hidden_channels
    - latent_dim
    - projection_dim
    - num_res_blocks
    - use_color_turn_reversal
    - use_file_mirror_if_castling_absent
    - fail_closed_unknown_channels
  expected_parameter_count: approximately 0.35M to 0.45M for simple_18 width 64 depth 4
  expected_memory_notes: Orbit tensor memory is B*K*C*8*8 floats; for B=512,K=4,C=18,fp32 this is about 9.4 MB before encoder activations. For C=112 it is about 58.7 MB, so chunk flattened orbit views if scaling.
```

```yaml
research_continuity:
  idea_fingerprint: finite safe chess automorphism groupoid over current board + masked Reynolds latent quotient + orbit consistency/VICReg no-collapse + per-transform risk variance + binary puzzle-likeness logits
  already_researched_family_overlap: Uses neither sheaf/Hodge/attack incidence, one-ply move deltas, Sinkhorn transport, deterministic nuisance projection, ordinal ladders, sparse witnesses, ray automata, Möbius constellations, nor board pseudo-likelihood ratios.
  closest_duplicate_risk: Could be mistaken for ordinary data augmentation or group-invariant CNN; the required pseudo-orbit, augmentation-only, and Reynolds-only ablations distinguish it.
  do_not_repeat_if_this_fails:
    - finite chess rule-automorphism quotient bottlenecks over color/turn reversal and castling-safe file mirror
    - orbit consistency losses over deterministic board symmetries as the central mechanism
    - REx/VICReg regularization over generated transform environments without new semantic board operators
  suggested_next_search_directions:
    - label-safe ambiguity and selective prediction specifically for fine label 1 without ordinal ladders
    - causal invariance across real source or temporal splits if source-shift metadata becomes safely available for evaluation only
    - masked generative compression with stronger controls that avoid the imported pseudo-likelihood-ratio family
    - semantic latent-variable models that do not enumerate moves, attacks, transport couplings, or sheaf incidence objects
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Rule-Automorphism Quotient Bottleneck Network` to imported research memory after implementation, with fingerprint: safe color/turn reversal and castling-gated file mirror + Reynolds orbit pooling + orbit consistency/risk variance. | Prevents the next research cycle from repeating finite chess automorphism quotienting under a new name. | `Imported Research Memory` |
| Add anti-duplicate warning: do not propose another finite board-symmetry quotient, orbit-average, or automorphism-consistency model unless it introduces a new falsifiable object beyond color/turn/file symmetries. | Tightens novelty after this packet is consumed. | Anti-duplicate rules following imported fingerprints |
| Add a requirement that any symmetry-based future idea must state exactly which chess rules break which rotations/reflections, especially castling and pawns. | Avoids invalid claims of full board reflection or rotation invariance. | `Research Mode` or `What Counts As Creative Enough` |
| Add a reusable ablation requirement for symmetry ideas: include augmentation-only, invariant-pooling-only, and semantics-destroyed pseudo-transform controls. | Makes symmetry results interpretable and prevents ordinary augmentation from being mislabeled as a new mechanism. | `Ablation Plan` requirements |
| Record whether `simple_18` channel order is documented after Codex inspects the repo; if not, require a channel-spec registry before future transform-based ideas. | Reduces implementation risk and avoids silent plane corruption. | `Project Context You Must Respect` under encodings |
| If this idea fails, add a note that finite automorphism quotienting should not be repeated without a source-shift benchmark or a new semantic operator. | Preserves negative results across cycles. | `Research Continuity` |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes.
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0751_tuesday_pdt_automorphism_quotient.md`.
- No forbidden engine features used as inputs: yes.
- Does not fabricate labels: yes.
- Not a routine CNN/ResNet/Transformer variant: yes; the core is a safe chess automorphism quotient with falsification controls.
- Minimal current-data experiment exists: yes; `simple_18` on the existing train/val/test split.
- Falsification criterion is concrete: yes; same-sized semantics-destroyed pseudo-orbit control plus augmentation-only control.
- Codex can implement without asking for missing architecture details: yes.
- Prompt maintenance notes included for Codex: yes.
- Repetition check against imported research packets completed: yes.
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes.
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes.
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes.
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: yes.
