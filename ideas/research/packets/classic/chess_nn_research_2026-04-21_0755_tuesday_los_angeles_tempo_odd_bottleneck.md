# Codex Handoff Packet: Tempo-Odd Bottleneck Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0755_tuesday_los_angeles_tempo_odd_bottleneck.md`
- Generated at: 2026-04-21 07:55 America/Los_Angeles
- Weekday: Tuesday
- Timezone: America/Los_Angeles
- Idea slug: `tempo_odd_bottleneck`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Tempo-Odd Bottleneck Network, abbreviated `TempoOddBottleneckNet`.
- One-sentence thesis: Puzzle-likeness is often a side-to-move interaction property, so a model should isolate the latent component that changes under a rule-only side-to-move intervention instead of relying mainly on side-blind static board shortcuts.
- Idea fingerprint: current board tensor -> deterministic side-to-move toggle view `tau(x)` with en-passant sanitized -> shared board encoder -> two-point Walsh even/odd latent split `(h(x)+h(tau x))/2`, `(h(x)-h(tau x))/2` -> classifier bottleneck emphasizing the odd component -> binary puzzle-like logits.
- Why this is not a common CNN/ResNet/Transformer variant: the central operator is not a deeper trunk, attention over squares, or generic augmentation; it is a causal/representation-theoretic projection onto the odd component of a semantic involution that flips only the side-to-move variable while preserving board occupancy.
- Current-data minimal experiment: train on `data/splits/crtk_sample_3class/` with `simple_18`, binary labels `fine_label == 0 -> 0` and `fine_label in {1,2} -> 1`, for the same 3-epoch benchmark budget as existing small CNN/residual baselines.
- Smallest central falsification ablation: replace `tau(x)` by an identity paired view `x` while keeping the same shared encoder calls, bottleneck sizes, classifier, optimizer, and candidate count; this makes the odd component identically zero and tests whether the tempo-odd channel, not extra compute, explains any gain.
- Expected information gain if it fails: a clean failure says side-to-move interaction isolation is not a useful inductive bias on this split, so future cycles should stop proposing side-to-move-toggle Walsh/decomposition models and should look instead at uncertainty, generative compression, or cross-encoding invariance.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The supervised task is chess puzzle-likeness classification from a single board position. The model returns binary logits with shape `(batch, 2)` for outputs `0 = non-puzzle` and `1 = puzzle-like`. The fine labels remain diagnostic:

- fine label `0`: known non-puzzle;
- fine label `1`: verified near-puzzle;
- fine label `2`: verified puzzle.

The default binary target should remain the repository target mapping. If Codex must spell it out for this idea, use `coarse_y = 0` for fine label `0` and `coarse_y = 1` for fine labels `1` and `2`; do not relabel, upsample into fake classes, or infer missing labels.

Allowed input tensor contract:

- a PyTorch module accepting `(batch, C, 8, 8)`;
- returning logits `(batch, num_classes)` with `num_classes = 2`;
- first minimal experiment uses `simple_18` with `C = 18`.

Benchmark split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Do not point the trainer at the roughly 45M-row full Parquet dataset until streaming support exists.

Leakage checklist:

- Allowed: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and rule-only tensor transformations derived from the current encoded board.
- Allowed with care: pseudo-legal attack geometry derived only from the current board. This idea does not use attack geometry.
- Leakage-prone unless separately justified and ablated: full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences. This idea uses none of these.
- Forbidden as neural-network inputs: Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance.
- For `lc0_static_112` and `lc0_bt4_112`, current-board channels may be consumed by learned adapters, but deterministic geometry or side-to-move toggling must be used only when channel semantics are explicitly declared. History channels must not be interpreted by hard-coded logic unless their semantics are known. Unknown LC0 channel semantics must fail closed.

Safe-rule boundary for this idea:

- The transformation `tau` toggles the side-to-move plane and optionally zeros en-passant history channels. It does not generate moves, score moves, count moves, or ask whether a move is legal.
- The label of `tau(x)` is not assumed to equal the label of `x`. `tau(x)` is a representation probe, not a supervised augmented example.
- Castling rights are copied because they are part of the current FEN state for both sides; en-passant is sanitized because it is history-sensitive and can become inconsistent under a pure side-to-move toggle.

## 4. Research Map

External ideas used, with citations or URLs:

| Source | What is borrowed | What is not copied |
|---|---|---|
| Ryan O'Donnell, *Analysis of Boolean Functions*, arXiv: https://arxiv.org/abs/2105.10386 | The two-point Fourier/Walsh idea: decompose a function under a binary variable into even and odd components. | No Boolean-function learning algorithm, no high-order Fourier expansion over all board variables, and no imported Möbius piece-constellation mechanism. |
| Judea Pearl, *Causality: Models, Reasoning, and Inference*, Cambridge/PDF mirror: https://archive.illc.uva.nl/cil/uploaded_files/inlineitem/Pearl_2009_Causality.pdf | The intervention viewpoint: treat side-to-move as a variable that can be intervened on to reveal interaction dependence. | No causal identification claim from observational data, no do-calculus estimator, and no claim that `tau(x)` has a true observed label. |
| Cohen and Welling, “Group Equivariant Convolutional Networks,” arXiv: https://arxiv.org/abs/1602.07576 | The discipline of tying model behavior to a known transformation rather than hoping SGD learns it. | No full group convolution, no D4 chess symmetry assumption, no rotation/reflection invariance, and no copied LC0-style architecture. |
| Arjovsky et al., “Invariant Risk Minimization,” arXiv: https://arxiv.org/abs/1907.02893 | The causal motivation that stable signal should survive nuisance changes. | No IRM penalty, no environment classifier, and no use of source labels or provenance. |
| Bardes, Ponce, LeCun, “VICReg,” arXiv: https://arxiv.org/abs/2105.04906 | Optional variance floor as an anti-collapse regularizer for the odd latent. | No self-supervised joint-embedding objective and no replacement of the supervised binary loss. |

Candidate search trace:

| Candidate mechanism considered | Why it lost to Tempo-Odd Bottleneck |
|---|---|
| Multi-encoding causal invariance between `simple_18` and `lc0_bt4_112` | Promising but needs paired multi-encoding batches or loader changes; not the smallest current-data experiment. |
| Evidential/abstention model for fine label `1` ambiguity | Useful diagnostic direction, but it changes calibration/decision policy more than board representation and risks becoming an ordinal-head near-duplicate. |
| Masked generative MDL compression of boards | Interesting, but too close to the imported pseudo-likelihood/description-length family unless redesigned around a genuinely different observable. |
| Rule-orbit file-mirror/color-swap equivariant CNN | Safer symmetry, but too close to ordinary augmentation or group-equivariant CNNs; less chess-specific than a side-to-move interaction probe. |
| Domain-adversarial material-phase invariance | Could remove source artifacts, but material phase is also real chess signal; it also resembles nuisance-suppression work already imported. |
| Persistent homology or cubical topology of occupancy planes | Mathematically distinct, but the expected link to tactical puzzle-likeness is weak and hard to falsify cleanly. |
| Spectral king-zone operator | Likely drifts into attack-defense graph or sheaf/Hodge territory, already heavily represented. |
| Low-rank polynomial interactions over pieces and side-to-move | The side-to-move part is interesting, but full polynomial constellation modeling is already covered by the Möbius/ANOVA packet. |
| Rank/file formal-language model with side-to-move gates | Too close to the imported ray-language automaton family. |
| Move-count or legal-move entropy bottleneck | Leakage-prone and adjacent to the one-ply move-delta landscape family. |
| Source-invariant adversary | Source labels and provenance must not become neural-network inputs or auxiliary targets. |
| Ordinary selective prediction without a new board operator | Useful reporting layer, not enough as the central research idea. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Two-point Boolean Fourier/Walsh decomposition | `z_even = 0.5*(h(x)+h(tau(x)))`, `z_odd = 0.5*(h(x)-h(tau(x)))` | two encoder outputs `(batch, d)` -> two latents `(batch, d)` | identity-pair ablation makes `z_odd = 0` | Not a piece-constellation ANOVA; only decomposes the semantic side-to-move intervention. |
| Causal intervention | `tau` toggles the current side-to-move plane and sanitizes en-passant | input `(batch,C,8,8)` -> paired input `(batch,C,8,8)` | batch-permuted side-bit ablation preserves marginals but destroys paired intervention | Not move-delta, no legal move tree, no engine or oracle. |
| Bottleneck against static shortcuts | classifier receives high-capacity odd branch and low-capacity/stop-gradient even context | `z_odd -> odd_dim`, `z_even -> even_dim` | even-only and unrestricted-concat ablations | Not deterministic nuisance-vector residualization; no closed-form projection over material/phase vectors. |
| Anti-collapse regularization | optional variance floor on odd projection dimensions | `(batch, odd_dim)` -> scalar penalty | run with `lambda_odd_var = 0` | Optional regularizer, not the core operator. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/trunk/cnn.py` | Already exists and does not test a new chess-specific hypothesis. |
| Residual CNN | `src/chess_nn_playground/models/trunk/residual_cnn.py` | Already exists; adding residual depth is ordinary capacity tuning. |
| LC0-style CNN or residual CNN | LC0 BT4-style CNN/residual variants | Already represented; copying LC0-style static processing does not isolate a new cause of puzzle-likeness. |
| Bigger CNN/deeper trunk | small/medium/deep existing variants | Violates the “not just depth/width tuning” constraint. |
| Ordinary ViT over 64 squares | vanilla Transformer-style baseline | Too generic and explicitly disallowed as the core idea. |
| Plain GNN on board squares | square-grid graph model | Mostly repackages local convolution/message passing and lacks a falsifiable chess-specific operator. |
| Hyperparameter search | all existing baselines | Not a research idea and would not explain a failure. |
| Ensembling | any combination of current models | Explicitly disallowed as the core idea and confounds architecture with variance reduction. |
| Full group-equivariant CNN over rotations/reflections | group CNN/common augmentation | Chess is not D4-invariant because pawns, castling, and side-to-move break many image symmetries. |
| Static attack-defense graph model | imported sheaf/Hodge/attack packets | Already covered by multiple imported tactical sheaf and attack-defense graph families. |
| One-ply move-delta set model | imported counterfactual move-delta packets | Already covered and too close to legal move landscape pooling. |
| Sinkhorn or piece-target transport model | imported optimal-transport packets | Already covered by several transport bottleneck variants. |
| Ordinal ladder head | imported ordinal evidence ladder | Fine-label monotonicity is already researched; this packet needs a different formal observable. |
| Sparse witness-piece bottleneck | imported sparse witness packet | Already represented and would duplicate the witness bottleneck idea. |
| Pseudo-likelihood board-ratio model | imported geometry-conditioned pseudo-likelihood packet | Generative compression is interesting but the existing packet already owns that nearest family. |

## 6. Mathematical Thesis

### Input space definition

For the minimal `simple_18` experiment, write a position tensor as

\[
x = (B, S, R, E) \in \mathcal X,
\]

where `B` contains the 12 piece occupancy planes, `S` is the side-to-move plane encoded as a constant binary plane, `R` contains castling-right planes, and `E` is the en-passant plane. Let `s in {-1,+1}` denote the scalar side-to-move variable corresponding to `S`.

Define a deterministic intervention

\[
\tau(B,S,R,E) = (B, 1-S, R, 0),
\]

with the convention that the real input is also sanitized to `(B,S,R,0)` when `zero_ep_for_tau = true`. The en-passant choice is conservative: en-passant is rare, history-sensitive, and can become semantically inconsistent after a pure side-to-move intervention.

### Label/target definition

Let `F in {0,1,2}` be the fine label and `Y in {0,1}` be the benchmark target, normally `Y = 1[F in {1,2}]`. The model learns `p_theta(Y | x)` and returns two logits.

### Data distribution assumptions

The training, validation, and test splits are sampled from the current CRTK sample split. Source processes may imprint static artifacts such as material balance, phase, castling availability, side-to-move proportions, and encoding quirks. The central assumption is not that these artifacts are absent; it is that many artifacts are approximately invariant under `tau`, while real puzzle-likeness often depends on which side has the move.

### Allowed symmetry or equivariance assumptions

This idea does **not** assume chess is invariant under arbitrary rotations or reflections. Pawns, castling rights, en-passant, and side-to-move matter. It also does **not** assume `Y(tau x) = Y(x)`. The only exact algebraic fact used is that `tau` is an involution on the sanitized tensor space: `tau(tau(x)) = x`.

### Core hypothesis

There exists a useful latent decomposition of puzzle-likeness into a static context and a side-to-move interaction:

\[
\eta(x) = \log \frac{P(Y=1 \mid x)}{P(Y=0 \mid x)} \approx a(B,R) + s\,b(B,R),
\]

where `a` contains side-blind static context and spurious shortcuts, while `s b(B,R)` captures tempo-dependent tactical agency. The model should emphasize `s b(B,R)` without throwing away all static context.

### Formal object introduced

For any learned encoder `h_theta: X -> R^d`, define the `tau`-even and `tau`-odd projections:

\[
P_+ h(x) = \frac{h(x)+h(\tau x)}{2}, \qquad
P_- h(x) = \frac{h(x)-h(\tau x)}{2}.
\]

The architecture exposes `P_- h` as the high-capacity predictive path and exposes `P_+ h` only through a small, optionally stop-gradient context bottleneck.

### Proposition

For any encoder `h` and involution `tau`, the projections satisfy:

\[
P_+ h(\tau x)=P_+ h(x), \qquad P_- h(\tau x)=-P_- h(x).
\]

If `h(B,s,R)=u(B,R)+s v(B,R)` for `s in {-1,+1}`, then:

\[
P_+ h(B,s,R)=u(B,R), \qquad P_- h(B,s,R)=s v(B,R).
\]

Thus the odd projection exactly cancels any side-blind term expressible inside the encoder and exactly recovers the first-order side-to-move interaction term.

### Proof sketch or derivation

The first identities follow from `tau^2 = identity`:

\[
P_+h(\tau x)=\frac{h(\tau x)+h(\tau^2x)}{2}=\frac{h(\tau x)+h(x)}{2}=P_+h(x),
\]

and similarly

\[
P_-h(\tau x)=\frac{h(\tau x)-h(x)}{2}=-P_-h(x).
\]

For `h(B,s,R)=u(B,R)+s v(B,R)`, substituting `-s` for `s` gives the second pair of identities.

### Optimization objective

The main objective is supervised cross-entropy on the real position only:

\[
\min_{\theta,\psi}\;\mathbb E_{(x,y)}\left[\operatorname{CE}\left(g_\psi\left(A P_-h_\theta(x),\; \operatorname{sg}(B P_+h_\theta(x))\right), y\right)\right] + \lambda_{var} L_{var},
\]

where `A` is the odd projection head, `B` is the small even-context head, `sg` means stop-gradient when enabled, and `L_var` is an optional batch variance floor on odd features. The stop-gradient is not a theorem; it is an engineering constraint that prevents label gradients from optimizing the encoder through the side-blind context route.

### What is actually proven

Only the even/odd algebra is proven. In particular, if the encoder represents a side-blind nuisance term equally for `x` and `tau(x)`, that term cancels from `P_-h`.

### What remains only hypothesized

It is a hypothesis that CRTK puzzle-likeness has enough side-to-move interaction signal for `P_-h` to improve generalization, near-puzzle recall, or calibration. It is also a hypothesis that the stop-gradient even context improves robustness rather than discarding useful static information.

### Counterexamples where the idea should fail

- Positions where both sides have nearly symmetric tactical opportunities.
- Puzzles whose label is mostly source-specific or composition-specific rather than tempo-specific.
- Static mating nets where the side-to-move toggle does not change the relevant visual pattern much.
- Endgame zugzwang-like positions where legal-move availability matters, but this model intentionally does not generate legal moves.
- En-passant-specific tactics, because the minimal safe version sanitizes en-passant.

### Self-critique

The strongest objection is that the model may learn a brittle “side-to-move bit interaction detector” rather than genuine tactical agency. Toggling side-to-move without generating legal moves is also an artificial intervention: it is algebraically clean, but not a reachable game transition. The experiment is still worth running because the central ablation is unusually sharp. If identity-pair, batch-permuted side-bit, or even-only controls match the main model, the idea is dead. If the main model improves near-puzzle recall at matched non-puzzle false-positive rate while those controls fail, the result is informative and not explainable by extra compute alone.

## 7. Architecture Specification

### Module names

- `TempoOddBottleneckNet`: top-level `torch.nn.Module`.
- `Simple18TempoToggle`: deterministic adapter for `simple_18`.
- `FailClosedTempoToggle`: adapter for encodings whose channel semantics are not explicitly declared.
- `SharedBoardEncoder`: small CNN/residual encoder reused on `x` and `tau(x)`.
- `OddEvenWalshBottleneck`: computes `z_even` and `z_odd`.
- `TempoOddHead`: final classifier returning `(batch, 2)` logits.

### Forward-pass steps and shapes

For the first experiment, use `encoding = simple_18`, `C = 18`, width `W = 64`, encoder output dimension `D = 128`, odd dimension `D_odd = 64`, and even context dimension `D_even = 16`.

1. Input: `x`, shape `(B, 18, 8, 8)`.
2. Adapter sanitization: `x0 = sanitize(x)` with en-passant channel zeroed if configured; shape `(B, 18, 8, 8)`.
3. Side-to-move intervention: `xt = tau(x0)` with side-to-move plane replaced by `1 - side_to_move_plane`; shape `(B, 18, 8, 8)`.
4. Shared encoder on both views:
   - concatenate for efficient encoding as `(2B, 18, 8, 8)`;
   - stem conv: `(2B, 64, 8, 8)`;
   - 4 small residual blocks: `(2B, 64, 8, 8)`;
   - global average pool: `(2B, 64)`;
   - projection MLP: `(2B, 128)`;
   - split into `h0`, `ht`, each `(B, 128)`.
5. Walsh split:
   - `z_even = 0.5 * (h0 + ht)`, shape `(B, 128)`;
   - `z_odd = 0.5 * (h0 - ht)`, shape `(B, 128)`.
6. Bottleneck projections:
   - `u = LinearNoBias(128, 64)(LayerNorm(z_odd))`, shape `(B, 64)`;
   - `m = abs(u)`, shape `(B, 64)`;
   - `v = Linear(128, 16)(LayerNorm(stopgrad(z_even)))` when `stopgrad_even_context = true`, shape `(B, 16)`.
7. Classifier input: `cat([u, m, v])`, shape `(B, 144)`.
8. MLP head: `144 -> 64 -> 2`, output logits `(B, 2)`.

### Parameter-count estimate

Approximate parameter count for `simple_18`:

- stem `18 -> 64` 3x3 conv: about 10.4k weights;
- four residual blocks, two 3x3 `64 -> 64` convs each: about 295k weights;
- projection and bottleneck/head MLPs: about 30k to 45k weights;
- total: roughly 340k to 370k parameters, depending on normalization and bias choices.

For `lc0_*_112` with the same trunk, the stem grows by about 54k weights, giving roughly 390k to 430k parameters. The first experiment should not use LC0 unless channel semantics are configured.

### FLOP or complexity estimate

The model performs two encoder passes per board. With width 64, four residual blocks, and 8x8 spatial maps, complexity is about twice a same-width small residual CNN. The paired-view overhead is intentional and controlled by identity-pair and random-side ablations.

### Candidate-set memory and chunking plan

The generated candidate set has constant size `K = 2`: the real sanitized board and its tempo intervention.

- paired input memory in float32: `B * K * C * 8 * 8 * 4` bytes;
- for `B = 512`, `C = 18`: about 2.36 MB;
- for `B = 512`, `C = 112`: about 14.7 MB;
- latent memory: `B * K * D * 4`, about 0.5 MB for `D = 128` and `B = 512`.

No chunking is needed for `simple_18`. If LC0 encodings are later used and GPU memory is tight, encode the `(2B, C, 8, 8)` paired batch in chunks of size `B` and concatenate latents before the Walsh split.

### Required config fields

- `model.name: tempo_odd_bottleneck`
- `model.input_channels: 18`
- `model.num_classes: 2`
- `model.encoding: simple_18` or equivalent data config field
- `model.side_to_move_channel: 12`
- `model.en_passant_channels: [17]`
- `model.zero_en_passant_for_tau: true`
- `model.encoder_width: 64`
- `model.encoder_blocks: 4`
- `model.latent_dim: 128`
- `model.odd_dim: 64`
- `model.even_dim: 16`
- `model.stopgrad_even_context: true`
- `model.odd_variance_weight: 0.001` initially, with an ablation at `0.0`
- `model.tau_mode: semantic_toggle`; ablations use `identity`, `batch_permuted_side`, or `random_side_marginal`.

### Encoding-adapter assumptions

- `simple_18`: adapter may assume channels `0..11` are piece planes, channel `12` is side-to-move, channels `13..16` are castling, and channel `17` is en-passant. Codex should verify against the existing exporter before hard-coding. If the repository defines a channel map elsewhere, use that map instead of duplicating constants.
- `lc0_static_112`: fail closed by default. Enable only if a config-provided channel map identifies the current side-to-move plane and any en-passant/history channels to sanitize.
- `lc0_bt4_112`: fail closed by default. Because BT4 history planes from a single FEN may be zero-filled until exporter support exists, deterministic toggling must not reinterpret unknown history. A learned encoder may consume the tensor, but `tau` needs declared semantics.

### Pseudocode, not final implementation

```text
forward(x):
    x0 = adapter.sanitize(x)
    xt = adapter.make_tau(x0, mode=config.tau_mode)

    paired = concat_batch([x0, xt])
    h = shared_encoder(paired)          # (2B, D)
    h0, ht = split_batch(h)             # each (B, D)

    z_even = 0.5 * (h0 + ht)
    z_odd  = 0.5 * (h0 - ht)

    u = odd_linear(layer_norm(z_odd))   # no bias preferred
    m = abs(u)
    if stopgrad_even_context:
        z_even_for_head = stop_gradient(z_even)
    else:
        z_even_for_head = z_even
    v = even_linear(layer_norm(z_even_for_head))

    logits = head(concat([u, m, v]))
    return logits
```

The shared trainer, reports, confusion matrices, predictions, and leaderboards remain compatible because the module returns ordinary binary logits.

## 8. Loss, Training, And Regularization

- Primary loss: weighted cross-entropy on binary logits, using the repository’s balanced class weighting for `coarse_binary`.
- Optional auxiliary loss: odd variance floor, VICReg-style but much smaller:

  \[
  L_{var}=\frac{1}{D_{odd}}\sum_j \max(0, \gamma - \operatorname{std}_B(u_j))^2,
  \]

  with `gamma = 0.2` and `odd_variance_weight = 0.001`. This is optional and must be ablated at `0.0`.
- Do not add a supervised loss on `tau(x)` because `Y(tau x)` is unknown.
- Do not use source labels, engine outputs, move counts, PVs, verification metadata, or proposed labels in any loss.
- Batch size expectation: `512` for `simple_18`; reduce only if GPU memory requires it, and keep baselines comparable.
- Optimizer default: AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Epochs: `3` for the minimal benchmark, matching the current config block.
- Early stopping: patience `2` on validation loss or the project’s existing validation criterion.
- Determinism: set seed `42`, enable deterministic PyTorch options where already supported, and record exact config.
- Regularizers: ordinary weight decay, dropout only in the final MLP if existing configs use it, stop-gradient even context by default, optional odd variance floor.
- Fair comparison requirements: keep benchmark split, binary target mapping, data encoding, batch size, epoch budget, class weighting, metrics, report format, and confusion-matrix logic unchanged relative to comparable `simple_18` baselines.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Identity-pair central ablation | Set `tau(x)=x`, encode two identical views, keep all dimensions and classifier unchanged; `z_odd=0`. | Any gain requires the tempo-odd component rather than paired compute or head size. | If identity-pair matches main, abandon the central claim. |
| Batch-permuted side-bit tau | Keep piece/castling/en-passant tensors from `x`, but use a complemented side-to-move plane from another batch element; preserves side marginal and candidate count but destroys paired intervention. | The exact paired side intervention matters, not just injecting side-bit noise. | If this matches main, the model exploits side-bit distribution artifacts. |
| Random-side marginal tau | Replace `tau` side plane with Bernoulli samples matching the batch side-to-move frequency. | The odd channel needs semantic complement, not any binary contrast. | If this matches main, the odd branch is acting as noise regularization. |
| Even-only | Use `z_even` bottleneck only, with parameter count roughly matched by widening its head. | Static side-blind context alone is insufficient. | If it matches main, puzzle signal is mostly static or the odd path is unnecessary. |
| Odd-only, no even context | Drop `z_even` entirely and classify from signed/magnitude odd features. | Static context is only a small helper, not the main route. | If odd-only beats main, even context was a shortcut; keep odd-only in later scaling. |
| Unrestricted concatenation | Classify from `cat(h(x), h(tau x))` with matched parameter count, no Walsh split. | The even/odd projection matters beyond giving the network a second view. | If concat matches or beats main, the mathematical bottleneck is not buying structure. |
| Side-plane-zero control | Zero the side-to-move plane in both `x` and `tau(x)`. | The mechanism uses side-to-move interaction. | If this matches main, the claimed tempo signal is not being used. |
| No odd variance regularizer | Set `odd_variance_weight=0`. | Optional anti-collapse regularizer is not the core idea. | If performance changes greatly, report separately; do not attribute gain solely to the Walsh operator. |
| En-passant retained | Do not zero en-passant. | En-passant sanitization is not hiding useful signal or causing artifacts. | If retained-EP wins only on rare EP cases, inspect leakage risk before scaling. |
| File-mirror augmentation added | Add safe horizontal mirror augmentation to both views. | Legal symmetry augmentation is complementary but not central. | If only this version wins, the idea may reduce to ordinary augmentation. |
| Count-only/candidate-count control | Replace latents by a constant candidate-count feature `K=2` plus side-to-move scalar. | Constant two-view generation is not the source of improvement. | If surprisingly competitive, dataset has severe side-to-move or class-prior artifacts. |
| Material-preserving nuisance control | Feed material counts, side-to-move, castling flags, and EP flag to a tiny MLP baseline. | The model beats obvious deterministic shortcuts. | If this matches main, the learned board encoder adds little beyond nuisance statistics. |

This model is not a move-set model, so source-square marginals and capture histograms are not generated. The nuisance-preserving controls above preserve candidate count, material, side-to-move marginal, castling, and en-passant availability while destroying the proposed paired-intervention semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- existing `simple_18` simple CNN with similar or smaller parameter count;
- existing `simple_18` residual CNN small/medium variant closest in parameter count;
- deterministic material/side/castling MLP baseline if available or easy to add;
- every central ablation in Section 9, at least identity-pair, batch-permuted side-bit, even-only, unrestricted concatenation, and no-odd-variance.

Metrics to inspect:

- validation and test cross-entropy;
- accuracy and balanced accuracy;
- ROC-AUC and PR-AUC if existing reporting supports them;
- precision, recall, and F1 for binary puzzle-like class;
- calibration/ECE if already available;
- rectangular fine-label diagnostic matrix: true fine label `0/1/2` -> predicted binary `0/1`.

Required fine-label diagnostic:

- produce the `3x2` diagnostic matrix for the main model;
- produce the same matrix for identity-pair, batch-permuted side-bit, even-only, and unrestricted-concat ablations;
- report class `1` and class `2` recall separately, not only combined positive recall.

Near-puzzle diagnostic:

- At a threshold chosen on validation to match the best comparable `simple_18` residual baseline’s fine-label-`0` false-positive rate, report fine-label-`1` recall on test.
- Also report fine-label-`2` recall at the same threshold, so a gain on near-puzzles is not hiding a loss on verified puzzles.

Required artifacts:

- trained checkpoint path;
- config YAML used for main and ablations;
- train/val/test metrics JSON or CSV;
- confusion matrices including `3x2` diagnostics;
- predictions file with logits/probabilities and true fine labels;
- report comparing the main model to baselines and ablations;
- parameter count and throughput estimate.

Success threshold:

- Main model improves test balanced accuracy or ROC-AUC by at least `+1.0` percentage point over the best comparable `simple_18` baseline under the same training budget; and
- main model improves fine-label-`1` recall at matched fine-label-`0` FPR by at least `+2.0` percentage points; and
- identity-pair and batch-permuted side-bit ablations trail the main model by at least `0.7` percentage points in balanced accuracy or ROC-AUC.

Failure threshold:

- Main model is within `±0.3` percentage points of the best comparable baseline and central ablations; or
- main gains appear only in fine label `2` while fine label `1` recall stagnates or drops; or
- random/batch-permuted side controls match the main model.

What result would make this idea abandoned:

- Identity-pair or batch-permuted side-bit ablation matches or beats the main model across two seeds; or
- side-plane-zero control remains competitive, proving the intended side-to-move interaction is unnecessary.

What result would justify scaling:

- A clean win over comparable `simple_18` baselines plus clear separation from semantic-destroying ablations, especially if fine-label-`1` recall improves at matched fine-label-`0` FPR. Scaling steps should then test a larger encoder and, only after channel maps are explicit, an LC0 adapter.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_tempo_odd_bottleneck/idea.yaml` | Create | Machine-readable idea metadata from the `idea_yaml` block below. |
| `ideas/20260421_tempo_odd_bottleneck/math_thesis.md` | Create | Section 6 math thesis, including the involution, even/odd proposition, and falsifiers. |
| `ideas/20260421_tempo_odd_bottleneck/architecture.md` | Create | Section 7 architecture specification and tensor shapes. |
| `ideas/20260421_tempo_odd_bottleneck/implementation_notes.md` | Create | Adapter channel-map assumptions, fail-closed behavior, pseudocode, and leakage notes. |
| `ideas/20260421_tempo_odd_bottleneck/trainer_notes.md` | Create | Loss, optimizer, determinism, fair-comparison requirements, and no-label-for-`tau(x)` warning. |
| `ideas/20260421_tempo_odd_bottleneck/ablations.md` | Create | Section 9 ablation table and required central controls. |
| `ideas/20260421_tempo_odd_bottleneck/train.py` | Create | Thin wrapper or entrypoint that calls the shared trainer with this idea’s config; do not fork the trainer unless necessary. |
| `ideas/20260421_tempo_odd_bottleneck/config.yaml` | Create | Minimal experiment config using `simple_18`, batch size 512, 3 epochs, balanced class weighting. |
| `ideas/20260421_tempo_odd_bottleneck/report_template.md` | Create | Template requiring baseline comparison, central ablation comparison, and `3x2` fine-label matrices. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this idea to imported memory after implementation; add anti-duplicate notes for side-to-move-toggle Walsh bottlenecks if it fails. |
| `src/chess_nn_playground/models/trunk/tempo_odd_bottleneck.py` | Create | Implement `TempoOddBottleneckNet`, adapters, Walsh bottleneck, and optional odd variance loss hook if the trainer supports model auxiliary losses. |
| `src/chess_nn_playground/models/registry.py` | Update | Register builder function, for example `build_tempo_odd_bottleneck`. |
| `configs/tempo_odd_bottleneck_simple18.yaml` | Create | Top-level shared-trainer config for the main run. |
| `configs/tempo_odd_bottleneck_identity_ablation.yaml` | Create | Same as main but `tau_mode: identity`. |
| `configs/tempo_odd_bottleneck_batch_permuted_side_ablation.yaml` | Create | Same as main but `tau_mode: batch_permuted_side`. |
| `configs/tempo_odd_bottleneck_even_only_ablation.yaml` | Create | Same as main but classifier receives only even bottleneck with parameter-matched head. |
| `tests/test_tempo_odd_bottleneck.py` | Create | Focused tests for output shape, side-to-move toggle, EP zeroing, `tau(tau(x))=x` after sanitization, even/odd algebra, and fail-closed LC0 behavior. |

For `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md`, preserve all leakage, label, falsification, and anti-duplicate constraints. Add this packet’s fingerprint after Codex consumes it so the next research cycle does not rediscover the same side-to-move intervention idea.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0755_tuesday_los_angeles_tempo_odd_bottleneck.md
  generated_at: 2026-04-21 07:55 America/Los_Angeles
  weekday: Tuesday
  timezone: America/Los_Angeles
  idea_slug: tempo_odd_bottleneck
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_tempo_odd_bottleneck
  name: Tempo-Odd Bottleneck Network
  slug: tempo_odd_bottleneck
  status: draft
  created_at: 2026-04-21 07:55 America/Los_Angeles
  author: ChatGPT Pro
  short_thesis: Isolate the side-to-move odd latent component by comparing a board to a rule-only side-to-move intervention, then classify puzzle-likeness through that tempo bottleneck.
  novelty_claim: Uses a two-point Walsh projection under a semantic side-to-move involution, not attack graphs, move-delta sets, transport, ordinal heads, or generic CNN scaling.
  expected_advantage: Better near-puzzle recall and fewer static material/source shortcuts at matched non-puzzle false-positive rate.
  central_falsification_ablation: Identity-pair ablation with tau(x)=x and all compute/head dimensions preserved.
  target_task: coarse_binary
  input_representation: simple_18 first; LC0 encodings fail closed unless channel semantics are explicitly declared
  output_heads: binary logits [batch, 2]
  compute_notes: Two shared encoder passes per board; constant candidate set K=2; roughly 340k-370k parameters for simple_18.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/tempo_odd_bottleneck_simple18.yaml
  model_path: src/chess_nn_playground/models/trunk/tempo_odd_bottleneck.py
  latest_result_path: null
  notes: Do not supervise tau(x) with the original label; tau is a representation probe only.
```

```yaml
config_yaml:
  run:
    name: tempo_odd_bottleneck_simple18
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
    name: tempo_odd_bottleneck
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
  model_name: tempo_odd_bottleneck
  file_path: src/chess_nn_playground/models/trunk/tempo_odd_bottleneck.py
  builder_function: build_tempo_odd_bottleneck
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - TempoOddBottleneckNet
    - Simple18TempoToggle
    - SharedBoardEncoder
    - OddEvenWalshBottleneck
    - TempoOddHead
  required_config_fields:
    - input_channels
    - num_classes
    - side_to_move_channel
    - en_passant_channels
    - zero_en_passant_for_tau
    - encoder_width
    - encoder_blocks
    - latent_dim
    - odd_dim
    - even_dim
    - stopgrad_even_context
    - tau_mode
  expected_parameter_count: approximately 340k-370k for simple_18 with width 64 and 4 residual blocks
  expected_memory_notes: Constant two-view candidate set K=2; paired input memory is B*K*C*8*8 floats; no chunking needed for simple_18 at batch 512.
```

```yaml
research_continuity:
  idea_fingerprint: current-board side-to-move intervention tau + two-point Walsh even/odd latent split + high-capacity odd tempo bottleneck + binary puzzle-like logits
  already_researched_family_overlap: touches causal invariance and bottlenecks, but does not use sheaf/Hodge, attack graphs, one-ply move-delta sets, Sinkhorn/OT, nuisance-vector projection, ordinal ladders, sparse witnesses, ray automata, piece-constellation ANOVA, or pseudo-likelihood ratios
  closest_duplicate_risk: Siamese augmentation or group-equivariant CNN; distinguish by no label invariance assumption for tau(x), explicit odd/even projection, and central identity/random-side falsifiers
  do_not_repeat_if_this_fails:
    - side-to-move toggle paired-view classifiers
    - tau-even/tau-odd Walsh bottlenecks over side-to-move
    - stop-gradient even context plus high-capacity odd side-to-move branch
    - semantic-vs-random side-bit ablations as the central operator
  suggested_next_search_directions:
    - label-safe uncertainty or selective prediction focused on fine label 1 without ordinal-ladder duplication
    - masked generative compression with stronger source-artifact controls and not pseudo-likelihood ratio reuse
    - cross-encoding invariance once paired multi-encoding batches are available
    - causal environment tests over deterministic board strata that do not use provenance labels
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Tempo-Odd Bottleneck Network` to imported research memory after implementation, with fingerprint “side-to-move toggle intervention + two-point Walsh odd/even latent bottleneck.” | Prevents the next ChatGPT Pro pass from proposing the same tempo-intervention decomposition under a new name. | `Imported Research Memory` |
| Add an anti-duplicate rule: do not propose side-to-move-toggle Siamese/Walsh/odd-even bottlenecks unless the formal intervention or falsifier changes substantially. | Makes future novelty checks sharper if this idea fails. | `Research Continuity` or anti-duplicate constraints |
| Require future counterfactual-board ideas to state whether they assume label invariance for the counterfactual view. | Avoids accidental misuse of generated board views as labeled examples. | `Non-Negotiable Constraints` or `Problem Restatement` guidance |
| Add a reminder that LC0 deterministic adapters must fail closed when channel semantics are unknown. | Prevents hard-coded channel mistakes and hidden leakage through history planes. | Encoding section under `Project Context You Must Respect` |
| Record whether the identity-pair and random-side ablations separated from the main model. | Gives the next research pass actionable evidence about whether tempo interaction was real or just extra compute. | `Research Continuity` |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0755_tuesday_los_angeles_tempo_odd_bottleneck.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes, `simple_18` on the current CRTK sample split
- Falsification criterion is concrete: yes, identity-pair and batch-permuted side-bit ablations
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: yes
