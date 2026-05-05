# Codex Handoff Packet: Ray-Language Automaton Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0719_tuesday_local_ray_language_automaton.md`
- Generated at: 2026-04-21 07:19 UTC-07:00
- Weekday: Tuesday
- Timezone: local / America-Los-Angeles / UTC-07:00
- Idea slug: `ray_language_automaton`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Ray-Language Automaton Network, abbreviated `RLAN`.
- One-sentence thesis: Many chess puzzle-like positions contain short, ordered, gapped piece strings along ranks, files, and diagonals, so a differentiable weighted finite automaton over board rays should learn line-language motifs such as pins, skewers, batteries, back-rank constraints, and discovered-alignment patterns more sample-efficiently than an unconstrained 2D CNN.
- Idea fingerprint: `current-board side-relative piece-token strings over 92 oriented board rays + learned weighted finite automata over regular line languages + log-sum/max pooling over ray accept scores + binary puzzle-likeness logits; no attacks, no legal moves, no engine data, no sheaf/Hodge, no move-delta bag, no Sinkhorn/OT`.
- Why this is not a common CNN/ResNet/Transformer variant: the central learned operator is a family of weighted finite-state acceptors on one-dimensional chess-ray strings, with sequential state recurrences over piece tokens and gaps; convolutional layers, residual blocks, square attention, and square graph message passing are not the core mechanism.
- Current-data minimal experiment: train `ray_language_automaton_simple18` for the existing coarse binary task on `data/splits/crtk_sample_3class/{split_train,split_val,split_test}.parquet` using `simple_18`, 3 epochs, balanced class weighting, and the existing 3x2 fine-label diagnostic matrix.
- Smallest central falsification ablation: keep material, side-to-move, castling/en-passant metadata, line lengths, number of rays, automaton parameter count, and training recipe fixed, but randomly permute piece tokens across the 64 squares independently per position before ray-string extraction; if this ablation matches the main model, ordered ray language is not carrying the useful signal.
- Expected information gain if it fails: failure cleanly says that puzzle-likeness in this sample is not captured by side-relative regular languages over chess rays, or that CNN baselines already learn those line motifs well enough; either result rules out a broad family of future soft line motif models.

## 3. Problem Restatement And Data Contract

The task is chess puzzle-likeness classification from a single board position. The coarse output is binary:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The default benchmark remains binary, but every main result and central ablation must report the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Allowed tensor input shape is:

```text
(batch, C, 8, 8)
```

The module must return logits compatible with the shared trainer:

```text
(batch, 2)
```

Current encodings are:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant.
- `lc0_static_112`.
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists.

Benchmark split to use for the minimal experiment:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Do not point the current trainer directly at the roughly 45M-row full Parquet dataset until streaming support exists.

Leakage checklist:

- Safe as deterministic inputs or derived features: board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and deterministic current-board ray strings derived only from the current board.
- Safe but not central here: pseudo-legal attack geometry derived only from the current board. `RLAN` deliberately avoids it to stay outside the imported sheaf/attack families.
- Leakage-prone unless separately justified, engine-free, rule-only, and ablated: full legal-move generation, legal move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences.
- Never neural-network inputs: Stockfish or other engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved-candidate flags, or any future verification state.
- Fine labels may be used only as supervised targets or diagnostics, not as input features. This packet’s minimal experiment uses only the coarse binary target for the primary loss and the fine labels only for reporting.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic ray tokens may be extracted only from explicitly known current-board piece channels. History channels may be used only by a learned adapter in later experiments, not by deterministic geometry. If channel semantics are unknown, the adapter must fail closed with a clear error.

Boundary between rule-derived features and leakage: a ray string such as `friendly king, empty, friendly rook, empty, enemy rook` is a deterministic restatement of current occupancy along board coordinates. It is not a legal move, an attack edge, a generated variation, an engine score, or a label-derived fact. The model is allowed to learn that some strings are useful, but it must not be given whether the string is a pin, check, mate threat, or tactic according to an oracle.

## 4. Research Map

External ideas used:

1. Weighted finite automata and semiring recurrences.
   - Source: Mehryar Mohri, “Weighted Automata Algorithms,” Handbook of Weighted Automata, 2009. URL: https://research.google.com/pubs/archive/35076.pdf and DOI page https://link.springer.com/chapter/10.1007/978-3-642-01492-5_6
   - Borrowed: the formal view of a weighted automaton as initial weights, symbol-conditioned transition matrices, final weights, and a semiring dynamic program over sequences.
   - Not copied: determinization, minimization, transducer composition, shortest-distance algorithms, or any speech/NLP pipeline.

2. Rational kernels and automata as sequence feature maps.
   - Source: Cortes, Haffner, and Mohri, “Rational Kernels: Theory and Algorithms,” JMLR 2004. URL: https://www.jmlr.org/papers/v5/cortes04a.html
   - Borrowed: the idea that variable-length sequences can be represented through weighted automata/rational series and then used for classification.
   - Not copied: kernel SVM training, transducer kernels, or generic string-kernel feature engineering.

3. Soft-pattern neural WFSAs.
   - Source: Schwartz, Thomson, and Smith, “SoPa: Bridging CNNs, RNNs, and Weighted Finite-State Machines,” ACL 2018. URL: https://aclanthology.org/P18-1028/
   - Borrowed: the notion that WFSAs can act as trainable soft surface-pattern detectors and can sit between local convolution and fully recurrent sequence models.
   - Not copied: NLP tokenization, sentence classification architecture, epsilon-transition design, or code.

4. Differentiable weighted automata.
   - Source: Balakrishnan, “Differentiable Weighted Automata,” OpenReview PDF. URL: https://openreview.net/pdf?id=k2hIQYqHTh
   - Borrowed: a contemporary pointer that weighted automata can be embedded in autodiff pipelines.
   - Not copied: the paper’s particular framework, examples, or implementation. Venue status was not treated as a proven quality signal; it is a useful design reference only.

Candidate search trace. I screened at least twelve mechanisms internally, then selected the ray-language automaton because it is chess-specific, label-safe, small enough for the current split, and clearly falsifiable. Serious candidates not selected:

1. Encoding-invariant ordinal puzzle potential.
   - Mechanism: train paired `simple_18`/`lc0_bt4_112` views with a monotone ordinal head over labels 0/1/2.
   - Why it lost: good future direction, but the core novelty is mostly a training objective/head; it is less directly chess-structural and depends on paired encoding plumbing.

2. Sparse masked minimum-description-length motif bottleneck.
   - Mechanism: learn a small stochastic square mask that must classify and reconstruct masked board occupancy.
   - Why it lost: attractive but ablation is harder to interpret because the mask may simply learn material/source shortcuts; it also risks becoming a generic attention bottleneck.

3. Exact color-side equivariant CNN.
   - Mechanism: enforce invariance under 180-degree color swap and side-to-move canonicalization.
   - Why it lost: likely useful but too close to data augmentation or partial-equivariant CNN design; it does not introduce a strong new operator beyond symmetry handling.

4. Selective/abstaining near-puzzle calibration model.
   - Mechanism: train a calibrated classifier that exposes ambiguity of fine label 1 and optionally abstains.
   - Why it lost: valuable reporting upgrade, but the benchmark asks for binary classification; abstention quality could improve calibration without improving the central board-representation problem.

5. Energy-based board consistency model.
   - Mechanism: learn an energy for tactically compressed positions with contrastive board corruptions.
   - Why it lost: promising but negative sampling choices could dominate the result and create unclear failure modes.

6. Domain-adversarial source-artifact suppressor.
   - Mechanism: adversarially remove dataset-source/provenance signals.
   - Why it lost: source labels and provenance must never be neural-network inputs; using them even as adversary targets is too close to the forbidden boundary for this cycle.

Why the selected idea survived: chess rays are primitive current-board geometry, not engine output or legal-move search. Regular languages over rays are exactly the right mathematical object for ordered gapped alignments: king-piece-attacker, rook-empty-king, queen-bishop battery, back-rank pawn shell, and similar motifs. The failure ablation is clean: destroy line order while preserving material and obvious counts.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple 2D CNN on `simple_18` | `src/chess_nn_playground/models/cnn.py` | Already exists and does not test a new chess-structural hypothesis. |
| Residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | Already exists; deeper residual local filters are not a new research idea. |
| LC0-style CNN or residual CNN over `lc0_bt4_112` | Existing LC0 BT4-style CNN/residual variants | Already covered by current baseline family and risks becoming “copy LC0 but smaller.” |
| Ordinary ViT over 64 squares | No exact baseline, but common Transformer baseline | Too generic; square attention alone does not encode the ray-language thesis and is explicitly disfavored. |
| Plain GNN on squares | Common square graph baseline | Too ordinary; unless edges encode attacks it is just a board-neighborhood model, and attack edges would overlap imported sheaf/attack families. |
| Hyperparameter tuning of current models | Existing CNN/residual configs | Disallowed as a core research idea and would give low information gain. |
| Ensembling existing models | Existing model zoo | Disallowed as core idea; improves scores without explaining puzzle structure. |
| Bigger/deeper CNN | Existing CNN/residual variants | Explicitly disallowed and not mathematically distinct. |
| Tactical sheaf/Hodge/Laplacian/curvature model | Imported sheaf/Hodge packets | Already heavily researched; ray automata deliberately avoid attack/defense incidence and sheaf operators. |
| One-ply move-delta bag or move landscape | Imported counterfactual move-delta packets | Already researched and outside this idea; `RLAN` enumerates rays, not moves or move consequences. |
| Piece-target or material-target Sinkhorn transport | Imported optimal-transport packets | Already researched; no transport plan, cost matrix, or Sinkhorn coupling appears here. |
| Deterministic nuisance residualization | Imported nuisance-orthogonal packet | Already researched; no closed-form projection of latents away from material/phase vectors is used. |
| Generic masked autoencoder pretraining | No current baseline | Too generic and likely to learn board-distribution reconstruction rather than puzzle-likeness unless redesigned. |
| Line-wise 1D convolution | Partial close cousin of this idea | A width-k convolution only detects fixed contiguous windows; it lacks automaton state, variable gaps, and regular-language semantics. |
| Hand-coded tactic detectors | No current neural baseline | Too brittle and risks encoding chess-engine-like labels; this packet learns soft line languages from labels instead. |

## 6. Mathematical Thesis

### Input space definition

For an encoding family `e`, let

```text
X_e \subseteq R^{C_e \times 8 \times 8}
```

be the set of valid tensors produced from a single current chess position. For `simple_18`, the first 12 planes are piece occupancy, followed by side-to-move and castling/en-passant metadata. Define a deterministic parser

```text
\tau_e : X_e -> (A^{64}, m)
```

when channel semantics are known. Here `A` is a finite alphabet of side-relative square tokens:

```text
A = {empty,
     friend_king, friend_queen, friend_rook, friend_bishop, friend_knight, friend_pawn,
     enemy_king, enemy_queen, enemy_rook, enemy_bishop, enemy_knight, enemy_pawn,
     pad}
```

and `m` is a small safe metadata vector containing side-to-move/castling/en-passant values already present in the encoding.

Let `\mathcal L` be the fixed set of maximal rank, file, diagonal, and anti-diagonal board lines, each taken in both orientations. On an 8x8 board this gives at most `S = 92` oriented ray strings padded to length `T = 8`. For each line `\ell`, define:

```text
s_\ell(x) \in A^T
c_\ell \in R^{d_c}
```

where `c_\ell` is deterministic line context: axis type, orientation, length, side-relative center rank/file, and edge flags.

### Label/target definition

Fine label:

```text
Y \in {0,1,2}
```

Binary target:

```text
Z = 1[Y > 0] \in {0,1}
```

The model returns binary logits for `Z`. Fine labels are diagnostics in the minimal experiment, not input features.

### Data distribution assumptions

The current train/validation/test split is assumed to be sampled from a fixed empirical distribution `P(X,Y)`. The key modeling assumption is not that all tactics are line tactics. It is weaker:

```text
I(Z ; \Phi_{ray}(X) | nuisance) > 0
```

where `\Phi_{ray}` is a learned feature map of ordered ray strings and `nuisance` includes obvious material and side-to-move summaries. In words: ordered side-relative piece strings along rays carry additional puzzle-likeness information beyond material counts and global metadata.

### Allowed symmetry or equivariance assumptions

Chess is not fully rotation/reflection invariant. Pawns, castling, en-passant, board edges, and side-to-move matter.

`RLAN` assumes only a conservative color-side canonicalization: tokens are expressed as `friend_*` or `enemy_*` relative to the side to move, and line context is side-relative. This shares patterns between White-to-move and Black-to-move analogues. It does not assume arbitrary 90-degree rotations, horizontal file reflection, or full dihedral symmetry. Board-edge and line-position context remain available so back-rank and promotion-rank effects are not erased.

### Core hypothesis

Puzzle-like positions are enriched for a finite family of short ordered ray motifs. Examples include, without hand-coding them: `king ... blocker ... rook/queen`, `king ... diagonal piece ... bishop/queen`, `rook/queen ... empty ... back-rank king`, and two-piece batteries on the same line. Such motifs are naturally regular languages over piece-token strings with gaps.

### Formal object introduced by the idea

For each automaton `r = 1,...,R`, define a weighted finite automaton over the log semiring:

```text
A_r = (\alpha_r, {T_{r,a}}_{a \in A}, \omega_r)
```

with start weights `\alpha_r \in R^Q`, transition matrices `T_{r,a} \in R^{Q \times Q}`, and final weights `\omega_r \in R^Q`.

For a ray string `s = (a_1,...,a_T)`, define the log-semiring recurrence:

```text
h_0(j) = \alpha_r(j)

h_t(j) = logsumexp_i [ h_{t-1}(i) + T_{r,a_t}(i,j) ]

score_r(s) = logsumexp_j [ h_T(j) + \omega_r(j) ]
```

A context-conditioned score can add a learned affine context term:

```text
\tilde score_r(s_\ell,c_\ell) = score_r(s_\ell) + b_r(c_\ell)
```

Board-level ray features are pooled as:

```text
\Phi_r(x) = [ max_\ell \tilde score_r(s_\ell,c_\ell),
              logsumexp_\ell \tilde score_r(s_\ell,c_\ell),
              axis-wise max/logsumexp summaries ]
```

The final classifier is a small MLP:

```text
logits(x) = g_\theta(\Phi(x), m)
```

### Proposition

For any finite collection of side-relative line motifs expressible as regular languages over `A` with deterministic line-context predicates, there exists a finite `R`, finite state count `Q`, and a linear classifier on max-pooled automaton scores that computes whether any motif appears on any board ray.

### Proof sketch or derivation

A regular language over a finite alphabet is recognized by a finite automaton. A Boolean finite automaton can be embedded in the log-semiring weighted automaton family by assigning large positive transition weights to allowed transitions and large negative weights to disallowed transitions. The log-sum-exp recurrence then approximates logical OR over possible automaton paths. Max pooling over the finite set of oriented board rays approximates existential quantification over board lines. Context predicates such as “back rank” or “diagonal axis” can be represented by context-specific biases or by splitting automata by context type. A linear classifier can combine accept scores for multiple motif languages.

### What is actually proven

The operator can represent regular line-language detectors over current-board ray strings and can pool them over the board. It can exactly represent the presence of any finite set of regular ray motifs in the limit of sufficiently separated weights, and approximately represent them with finite differentiable weights.

### What remains only hypothesized

It is not proven that the CRTK puzzle labels are generated by line motifs. It is also not proven that learned automata will find semantically clean motifs rather than material or source artifacts. Those are empirical questions. The central ablation is designed to test them.

### Counterexamples where the idea should fail

- Pure knight-fork puzzles where the decisive geometry is not a rank/file/diagonal line.
- Zugzwang, opposition, fortress, or tablebase-like endgame puzzles where line motifs are weak.
- Mating nets requiring interactions among several non-collinear pieces.
- Positions whose puzzle/non-puzzle label in the sample is dominated by source artifacts rather than board geometry.
- Any distribution where a CNN already learns the same ray motifs perfectly from the available sample size.

### Self-critique

The strongest objection is that an 8x8 board is tiny: a standard CNN may already learn line alignments, and a WFA over rays might overfit to superficial strings like “king plus rook on same rank” without understanding legality, defense, or move consequences. Another objection is that pooling independent ray scores may miss cross-line coordination, which many real tactics require. The experiment is still worth running because the falsifier is sharp. If ordered ray strings matter, line-token permutation should hurt class-1 and class-2 puzzle detection after preserving material and metadata. If it does not hurt, future research should stop spending cycles on soft line-language detectors and move toward a different mechanism.

## 7. Architecture Specification

### Module names

Recommended new model file:

```text
src/chess_nn_playground/models/ray_language_automaton.py
```

Recommended classes/functions:

- `RayLanguageAutomatonNet(torch.nn.Module)`
- `BoardTokenParser`
- `RayIndexBuilder`
- `WeightedRayAutomata`
- `RayScorePooler`
- `build_ray_language_automaton(config)`

### Forward-pass steps

Input:

```text
x: (B, C, 8, 8)
```

Default minimal experiment:

```text
encoding = simple_18
C = 18
```

Step 1: deterministic token parsing.

- Parse piece planes into side-relative token IDs.
- Output square tokens:

```text
tokens_square: (B, 64) int64 values in [0, |A|-1]
metadata: (B, d_m)
```

For `simple_18`, `d_m` should include side-to-move, four castling-right flags, and en-passant summary already present in the encoding. If en-passant is represented as a plane, use safe reductions such as any/en-passant-file/rank encoding derived from that plane.

Step 2: ray gathering.

- Precompute fixed line indices once as a registered buffer:

```text
ray_indices: (S, T) with S <= 92, T = 8
ray_mask: (S, T) boolean, false on padding
ray_context: (S, d_c)
```

- Gather tokens:

```text
ray_tokens: (B, S, T)
```

Step 3: weighted automaton recurrence.

Default hyperparameters:

```text
num_automata R = 32
num_states Q = 8
alphabet_size |A| = 14
```

Parameters:

```text
start: (R, Q)
transitions: (R, |A|, Q, Q)
final: (R, Q)
context_bias_mlp: d_c -> R
```

Output:

```text
ray_scores: (B, S, R)
```

Step 4: pooling.

Pool over all rays and optionally over axis subsets:

```text
global_max: (B, R)
global_lse: (B, R)
axis_max: (B, 4, R)
axis_lse: (B, 4, R)
pooled: (B, 2*R*(1+4)) = (B, 320) when R=32
```

Step 5: metadata append and classifier.

```text
features = concat(pooled, metadata)  # about (B, 326) for simple_18
hidden = Linear -> GELU -> Dropout -> Linear -> GELU
logits = Linear(hidden, 2)
```

Output:

```text
logits: (B, 2)
```

### Parameter-count estimate

For `R=32`, `Q=8`, `|A|=14`:

- Automaton transitions: `32 * 14 * 8 * 8 = 28,672`.
- Start/final weights: `2 * 32 * 8 = 512`.
- Context bias MLP: roughly `1k-3k` depending on `d_c`.
- Classifier from about 326 to 128 to 64 to 2: roughly `50k`.
- Total expected parameters: about `80k-120k`, depending on metadata and context MLP details.

This is intentionally smaller than many CNN baselines. If underfitting is obvious, Codex may increase to `R=48, Q=8` as a scale experiment, but the first falsification should use the compact default.

### FLOP and complexity estimate

The main recurrence cost is:

```text
O(B * S * T * R * Q^2)
```

With `B=512`, `S=92`, `T=8`, `R=32`, `Q=8`, this is about `772 million` small log-sum-exp transition operations per batch. The operation is dense but tiny and should be practical on GPU; on CPU it may be slow.

Memory for dynamic states is:

```text
O(B * S * R * Q)
```

With the default values and float32 this is about:

```text
512 * 92 * 32 * 8 * 4 bytes ≈ 48 MB
```

Ray token memory is negligible:

```text
O(B * S * T)
```

Chunking plan: if memory or log-sum-exp kernels are slow, process rays in chunks of 16 or 23 rays and concatenate/accumulate pooled results. Automata can also be chunked over `R`, but ray chunking should be enough.

### Required config fields

```yaml
model:
  name: ray_language_automaton
  input_channels: 18
  num_classes: 2
  encoding: simple_18
  num_automata: 32
  num_states: 8
  hidden_dim: 128
  dropout: 0.1
  side_relative_tokens: true
  include_axis_pools: true
  include_line_context: true
  token_shuffle_ablation: false
  line_order_ablation: false
```

### Encoding support

First experiment should use `simple_18` only because its channel semantics are explicit in the prompt and deterministic token parsing can be made fail-safe.

Adapter assumptions:

- `simple_18`: fully supported in the minimal experiment. Parse the 12 piece planes and safe metadata. Fail if `input_channels != 18` or if the configured channel order is missing.
- `lc0_static_112`: support only after Codex adds an explicit current-board piece-channel map. Deterministic ray tokens must use current-board channels only. Non-current channels may be ignored or passed through a learned scalar adapter in a later noncentral experiment.
- `lc0_bt4_112`: same as `lc0_static_112`; unavailable history planes are not deterministic geometry. The model must not infer move history, legal move counts, or engine information from history planes. If the channel map is unknown, raise a clear configuration error.

### Pseudocode, not implementation

```text
forward(x):
    tokens_square, metadata = BoardTokenParser(encoding).parse(x)
    ray_tokens = gather(tokens_square, ray_indices)       # B,S,T
    if token_shuffle_ablation:
        ray_tokens = gather(shuffle_square_tokens(tokens_square), ray_indices)
    scores = WeightedRayAutomata(ray_tokens, ray_mask)    # B,S,R
    scores = scores + context_bias(ray_context)[None,:,:]
    pooled = RayScorePooler(scores, axis_ids)             # B,F
    logits = classifier(concat(pooled, metadata))         # B,2
    return logits
```

The implementation should not add legal move generation, attack maps, check detection, engine calls, or label-derived features.

## 8. Loss, Training, And Regularization

Primary loss:

```text
CrossEntropyLoss(logits, binary_label)
```

Class weighting: use the project’s existing balanced class weighting for coarse binary mode.

Optional auxiliary regularizers, off by default for the first fair comparison:

- Small transition L2 penalty through standard weight decay.
- Automaton diversity penalty on pooled automaton features, e.g. penalize large off-diagonal correlations within a batch. Use only after the main/ablation comparison; do not let this become the core idea.
- Transition sparsity L1 can improve interpretability but should be disabled in the minimal benchmark unless Codex has time for a separate ablation.

Batch size expectations:

- Start with `batch_size: 512`, matching the current config style.
- If the recurrence is slow on available hardware, reduce batch size before changing model semantics.

Optimizer defaults:

```text
AdamW, learning_rate = 0.001, weight_decay = 0.0001
```

Epochs:

```text
3 for the minimal benchmark
```

Regularizers:

- Use dropout `0.1` in the classifier MLP.
- Use gradient clipping at `1.0` if log-space recurrence produces unstable gradients.
- Avoid stochastic token sampling in the main model; deterministic training makes the ablation cleaner.

Determinism requirements:

- Set the same seed as current configs, default `42`.
- Enable deterministic mode when supported.
- The token-shuffle ablation must use a seeded per-position/per-epoch generator so results are reproducible.

What must stay unchanged for fair comparison:

- Same train/validation/test split.
- Same coarse binary target definition.
- Same reporting stack and 3x2 diagnostic matrix.
- Same epoch budget for first comparison.
- Same class weighting policy.
- No use of the full 45M-row dataset until streaming support exists.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Token-permutation central falsifier | Randomly permute piece tokens across 64 squares per position before ray extraction; preserve material multiset, side metadata, ray count, ray lengths, model size, and training recipe | Ordered ray placement and line language matter beyond material | If performance is unchanged, the automata are not using ray semantics; abandon this family. |
| Bag-of-line-tokens | Replace automaton recurrence with unordered per-line token histograms plus same pooler/MLP | Sequential order and gaps matter, not just line material | If unchanged, regular-language sequence structure is unnecessary. |
| Fixed random automata | Freeze random transition/start/final weights and train only classifier | Learned automaton languages matter | If unchanged, improvements come from random projections or classifier capacity, not learned line motifs. |
| Rank/file only | Remove diagonal and anti-diagonal rays | Bishop/queen diagonal languages carry signal | If unchanged, diagonal tactical motifs are not useful in this sample or are captured elsewhere. |
| Diagonal only | Remove rank/file rays | Rook/queen/file/rank languages carry signal | If unchanged with rank/file removed, selected motifs may be mostly diagonal or the model is underusing axes. |
| No side-relative canonicalization | Use raw white/black tokens with side-to-move metadata instead of friend/enemy tokens | Sharing color-swapped tactical languages improves sample efficiency | If raw tokens match or beat main, canonicalization may be too restrictive or the dataset has color-specific artifacts. |
| No line context | Remove axis, length, edge, and side-relative coordinate context | Position-sensitive line motifs such as back-rank patterns matter | If unchanged, motif content is enough or context was not used. |
| Material-only/nuisance baseline | Use piece counts, side-to-move, castling/en-passant, and no ray order | Ray strings add signal beyond obvious global shortcuts | If material-only matches main, the model is not adding chess geometry. |
| Parameter-matched small CNN | Train a small CNN with roughly the same parameter count on `simple_18` | Automaton bias beats generic local filters at similar capacity | If CNN matches main and token permutation does not hurt, prefer CNN. |
| Axis-label scramble | Keep ray strings but randomly permute axis/context labels while preserving line lengths | Axis/context semantics matter | If unchanged, context is irrelevant or overparameterized. |

The central semantics-destroying ablation is the token-permutation falsifier. It preserves candidate count, material, side-to-move, castling/en-passant metadata, moving-piece identity distribution in the weak sense of board material, source-square marginal only as a uniform board multiset, and all model dimensions while destroying line order and ray placement. There is no move-set, capture histogram, or generated legal candidate set in this idea.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN, matching the current best small/medium/deep config available in the repo.
- Existing `simple_18` residual CNN.
- Existing LC0-style CNN/residual results may be listed for context, but the primary fair comparison is `simple_18` because the first `RLAN` experiment uses `simple_18`.
- Material-only/nuisance baseline from the ablation table.
- Parameter-matched small CNN if quick to implement.

Metrics to inspect:

- Validation and test cross-entropy.
- Accuracy.
- ROC-AUC if existing reports support it.
- F1 for puzzle-like class.
- Calibration metrics if already available, such as Brier score or ECE; do not block the experiment if not available.
- Required 3x2 fine-label matrix: true fine label `0/1/2` vs predicted binary output `0/1`.

Near-puzzle diagnostic:

- Report fine-label-`1` recall at a matched fine-label-`0` false-positive rate.
- Use either the best existing simple CNN’s fine-label-`0` false-positive rate or a fixed 5% fine-label-`0` FPR threshold, whichever is easier to compute consistently.
- Also report class-`2` recall at the same threshold so near-puzzle gains do not hide true-puzzle degradation.

Required artifacts:

- Main model checkpoint and config.
- Main validation/test metrics JSON or markdown report.
- 3x2 diagnostic confusion matrix for main model.
- 3x2 diagnostic confusion matrix for token-permutation central ablation.
- At least one table comparing main, token-permutation, material-only, and best simple/residual CNN baseline.
- Saved predictions for main and central ablation if the existing reporting stack supports prediction exports.

Success threshold:

- Main `RLAN` improves test ROC-AUC by at least `+0.01` absolute over the best comparable `simple_18` non-ensemble CNN/residual baseline, or improves fine-label-`1` recall by at least `+0.03` absolute at matched fine-label-`0` FPR while keeping fine-label-`2` recall within `0.015` absolute of the best comparable baseline.
- Token-permutation central ablation must be worse than main by at least `0.01` ROC-AUC or at least `0.02` fine-label-`1` recall at matched FPR. Otherwise the line-language mechanism is not validated even if the main score is good.

Failure threshold:

- Main is within `±0.003` ROC-AUC of the token-permutation ablation and has no meaningful near-puzzle diagnostic gain.
- Main is materially worse than existing simple/residual CNN baselines after the same epoch budget.
- Material-only baseline matches main, implying ray order did not add value.

Abandon the idea if:

- Token-permutation and bag-of-line-tokens both match main within noise.
- The 3x2 matrix shows gains only from predicting nearly everything as puzzle-like.
- The model is too slow to train on the current split even with ray chunking and `R=24,Q=6`.

Justify scaling if:

- Main beats the strongest comparable `simple_18` baseline and the central ablation drops clearly.
- Fine-label-`1` recall improves at matched fine-label-`0` FPR without sacrificing fine-label-`2` recall.
- Learned automata have interpretable high-scoring ray strings when Codex inspects top-scoring lines.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_ray_language_automaton/idea.yaml` | Create | Machine-readable idea summary copied from the `idea_yaml` block below. |
| `ideas/20260421_ray_language_automaton/math_thesis.md` | Create | Section 6 expanded with any implementation-relevant notation changes. |
| `ideas/20260421_ray_language_automaton/architecture.md` | Create | Section 7 with exact tensor shapes and parser assumptions. |
| `ideas/20260421_ray_language_automaton/implementation_notes.md` | Create | Notes on deterministic token parsing, ray index buffers, log-space recurrence, chunking, and fail-closed adapters. |
| `ideas/20260421_ray_language_automaton/trainer_notes.md` | Create | Loss, optimizer, class weighting, deterministic mode, and unchanged benchmark requirements. |
| `ideas/20260421_ray_language_automaton/ablations.md` | Create | Section 9 ablation table plus commands/config names after implementation. |
| `ideas/20260421_ray_language_automaton/train.py` | Create | Thin entrypoint that invokes the shared trainer with the idea config; avoid custom trainer unless needed for central ablation wiring. |
| `ideas/20260421_ray_language_automaton/config.yaml` | Create | Copy of the `config_yaml` block below. |
| `ideas/20260421_ray_language_automaton/report_template.md` | Create | Template requiring main metrics, central-ablation metrics, and 3x2 matrices. |
| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported memory after consumption; add anti-duplicate rule for ray-language/WFSA line scanners if it fails or succeeds. Preserve hard constraints. |
| `src/chess_nn_playground/models/ray_language_automaton.py` | Create | Implement `RayLanguageAutomatonNet`, parser, ray index builder, WFA recurrence, pooler, and builder function. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register `ray_language_automaton` builder without breaking existing models. |
| `configs/ray_language_automaton_simple18.yaml` | Create | Minimal train config for main model. |
| `configs/ray_language_automaton_simple18_token_shuffle.yaml` | Create | Central falsification ablation config with `token_shuffle_ablation: true`. |
| `configs/ray_language_automaton_simple18_material_only.yaml` | Create | Nuisance/material-only baseline config if the repo supports ablation configs. |
| `tests/test_ray_language_automaton.py` | Create | Focused tests: output shape `(B,2)`, deterministic ray index shape, parser fail-closed on unknown encoding, token-shuffle preserves material counts, no legal-move/engine dependencies. |

For `ideas/chatgpt_pro_deep_math_research_prompt.md`, Codex should add a concise memory entry after running this experiment. If `RLAN` fails, future prompts should forbid near-duplicates such as “soft automata over ranks/files/diagonals,” “regular-language ray motif scanner,” “gapped line-string WFA,” and “line order motif pooler” unless the next proposal changes the formal object beyond automata over current-board rays.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0719_tuesday_local_ray_language_automaton.md
  generated_at: "2026-04-21 07:19 UTC-07:00"
  weekday: Tuesday
  timezone: local_america_los_angeles_utc_minus_07
  idea_slug: ray_language_automaton
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_ray_language_automaton
  name: Ray-Language Automaton Network
  slug: ray_language_automaton
  status: draft
  created_at: "2026-04-21 07:19 UTC-07:00"
  author: ChatGPT Pro
  short_thesis: Learn weighted finite automata over side-relative rank/file/diagonal piece-token strings to detect gapped chess ray motifs predictive of puzzle-likeness.
  novelty_claim: Uses regular-language automata on deterministic current-board ray strings rather than CNN filters, square attention, attack/sheaf incidence, move-delta sets, or transport couplings.
  expected_advantage: Better sample efficiency and near-puzzle recall for positions whose puzzle structure is driven by pins, skewers, batteries, back-rank alignments, and other ordered ray motifs.
  central_falsification_ablation: Randomly permute piece tokens across squares per position before ray extraction while preserving material, metadata, ray count, line lengths, model size, and training recipe.
  target_task: coarse_binary
  input_representation: simple_18_initial
  output_heads: binary_logits
  compute_notes: Default R=32 automata, Q=8 states, S<=92 rays, T=8 tokens; recurrence cost O(B*S*T*R*Q^2), chunk rays if needed.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/ray_language_automaton_simple18.yaml
  model_path: src/chess_nn_playground/models/ray_language_automaton.py
  latest_result_path: null
  notes: Fine labels are diagnostics only in the minimal experiment; no engine features, legal moves, attack graphs, sheaves, move deltas, OT, or source metadata.
```

```yaml
config_yaml:
  run:
    name: ray_language_automaton_simple18
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
    name: ray_language_automaton
    input_channels: 18
    num_classes: 2
    encoding: simple_18
    num_automata: 32
    num_states: 8
    hidden_dim: 128
    dropout: 0.1
    side_relative_tokens: true
    include_axis_pools: true
    include_line_context: true
    token_shuffle_ablation: false
    line_order_ablation: false
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
  model_name: ray_language_automaton
  file_path: src/chess_nn_playground/models/ray_language_automaton.py
  builder_function: build_ray_language_automaton
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - BoardTokenParser
    - RayIndexBuilder
    - WeightedRayAutomata
    - RayScorePooler
    - RayLanguageAutomatonNet
  required_config_fields:
    - input_channels
    - num_classes
    - encoding
    - num_automata
    - num_states
    - hidden_dim
    - side_relative_tokens
    - include_axis_pools
    - include_line_context
  expected_parameter_count: 80000_to_120000_default
  expected_memory_notes: Alpha state memory O(B*S*R*Q), about 48MB for B=512,S=92,R=32,Q=8,float32; chunk rays if needed.
```

```yaml
research_continuity:
  idea_fingerprint: current-board side-relative piece-token strings over oriented rank/file/diagonal rays + differentiable weighted finite automata regular-language scoring + ray max/logsum pooling + binary puzzle-likeness logits
  already_researched_family_overlap: Avoids imported tactical sheaf/Hodge, move-delta, optimal-transport, and nuisance-projection families; closest non-imported cousin is soft-pattern WFA sequence classification.
  closest_duplicate_risk: A line-wise 1D CNN or hand-coded pin/skewer detector; this packet differs by using learned gapped regular languages via automaton state recurrences.
  do_not_repeat_if_this_fails:
    - Soft weighted finite automata over current-board rank/file/diagonal strings
    - Regular-language ray motif scanners
    - Gapped line-string pattern poolers over board rays
    - Line-order-only puzzle classifiers preserving material and metadata
  suggested_next_search_directions:
    - Encoding-invariant ordinal puzzle potential using fine label 1 as an ambiguity bridge
    - Masked generative compression with stronger source-artifact controls
    - Selective prediction or calibration focused on near-puzzle diagnostics
    - Causal invariance across encodings or side-to-move transforms without using source provenance
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Ray-Language Automaton Network` to imported research memory after implementation and benchmarking. | Prevents the next research pass from rediscovering soft automata over ranks/files/diagonals. | `Imported Research Memory` |
| Add an anti-duplicate fingerprint for “current-board ray strings + WFA/regular-language/gapped line motif pooling + binary puzzle-likeness.” | This is distinct from sheaf/move-delta/OT but still broad enough to catch renamed duplicates. | `Research Continuity` / anti-duplicate rules |
| Record whether token-permutation ablation hurt main performance and by how much. | The next model should know whether ordered ray semantics survived the central falsifier. | `Imported Research Memory` result summary |
| Add a standing requirement that future structured current-board operators include a semantics-destroying ablation preserving material and side-to-move. | This packet’s central falsifier is useful and should become standard for non-move structured models. | `Depth requirements` or `Ablation Plan` instructions |
| If LC0 channel parsing was difficult, add a prompt note requiring future ideas to state exact fail-closed adapter assumptions for each encoding. | Avoids accidental use of unknown history/current-board channel semantics. | `Project Context` / encoding notes |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes.
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0719_tuesday_local_ray_language_automaton.md`.
- No forbidden engine features used as inputs: yes.
- Does not fabricate labels: yes.
- Not a routine CNN/ResNet/Transformer variant: yes.
- Minimal current-data experiment exists: yes, `simple_18` on the existing CRTK sample split.
- Falsification criterion is concrete: yes, material-preserving token permutation before ray extraction.
- Codex can implement without asking for missing architecture details: yes.
- Prompt maintenance notes included for Codex: yes.
- Repetition check against imported research packets completed: yes.
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes.
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes.
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes.
