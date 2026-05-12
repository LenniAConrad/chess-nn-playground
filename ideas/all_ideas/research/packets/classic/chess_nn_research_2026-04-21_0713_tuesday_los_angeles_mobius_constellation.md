# Codex Handoff Packet: Möbius Piece-Constellation Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0713_tuesday_los_angeles_mobius_constellation.md`
- Generated at: 2026-04-21 07:13 America/Los_Angeles
- Weekday: Tuesday
- Timezone: America/Los_Angeles
- Idea slug: `mobius_constellation`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Möbius Piece-Constellation Network, abbreviated `MPCN`.
- One-sentence thesis: Chess puzzle-likeness is often carried by sparse, unordered, high-order constellations of current-board piece-square facts; explicitly modeling degree-2 and degree-3 ANOVA/Möbius interactions over occupied square tokens should expose that signal without engine analysis, move enumeration, attack graphs, sheaves, or transport.
- Idea fingerprint: `current board -> occupied piece-square tokens -> degree-isolated elementary symmetric / ANOVA interaction embeddings Φ1, Φ2, Φ3 -> sparse gated binary classifier -> puzzle-like logit`.
- Why this is not a common CNN/ResNet/Transformer variant: It performs no convolution, residual spatial stack, square-to-square attention, graph message passing, pseudo-legal move pooling, attack/defense incidence construction, Sinkhorn transport, or nuisance projection; the central operator is a low-rank polynomial set functional over the multiset of occupied piece-square tokens.
- Current-data minimal experiment: Train `mobius_piece_constellation` on `simple_18` using the existing CRTK sample train/val/test Parquet split for 3 epochs, binary target `fine_label > 0`, balanced cross-entropy, and the same report/prediction/confusion tooling as current baselines.
- Smallest central falsification ablation: Replace `Φ2` and `Φ3` by zeros while keeping the same tokenizer, state adapter, classifier width, optimizer, data split, and parameter-matched filler MLP; if this degree-1-only model matches the main model, the high-order constellation claim is not supported.
- Expected information gain if it fails: A clean failure says that explicit low-rank current-board piece-square interactions, at least up to triples, are not adding signal beyond material/square marginals and existing CNN baselines; future cycles should avoid polynomial/factorization-machine constellation variants unless they add a genuinely new causal or generative test.

## 3. Problem Restatement And Data Contract

The task is chess position classification for `chess-nn-playground`. The model receives a board encoding tensor shaped `(batch, C, 8, 8)` and returns binary logits shaped `(batch, 2)`, where output `0` means non-puzzle and output `1` means puzzle-like. The fine labels are real dataset labels, not model inputs: fine label `0` is known non-puzzle, fine label `1` is verified near-puzzle, and fine label `2` is verified puzzle. The default benchmark is binary with target `coarse = 1[fine_label > 0]`, while diagnostics must report the rectangular `3x2` matrix `true fine label 0/1/2 -> predicted binary output 0/1`.

The current split contract is fixed:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The full roughly 45M-row Parquet dataset must not be used directly until streaming support exists. The first experiment should use `simple_18` because its channel semantics are explicit: 12 piece planes, side-to-move, castling, and en-passant. `lc0_static_112` and `lc0_bt4_112` can be supported later only through fail-closed channel maps.

Leakage checklist:

- Safe neural inputs: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and deterministic current-board tokenization derived only from those planes.
- Safe rule-derived features, if later added: pseudo-legal attack geometry derived only from the current board is allowed by project policy, but this packet intentionally does not use it.
- Leakage-prone unless explicitly justified and ablated: full legal-move generation, legal move counts, checkmate/stalemate oracles, forced-line search, or move-tree consequences.
- Never allowed as neural-network inputs: engine evaluation, Stockfish/LC0 scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, unresolved candidate-pool status, dataset provenance, or any label-derived metadata.
- For `lc0_static_112` and `lc0_bt4_112`: deterministic geometry may use only explicitly mapped current-board piece planes. History planes, if ever used, must go only through a learned adapter and must not be interpreted as current-board rule geometry. If the channel map is missing or ambiguous, the adapter must raise an error rather than guessing.

## 4. Research Map

External ideas used, with what is borrowed and what is not copied:

| Source | URL | Borrowed | Not copied |
|---|---|---|---|
| Steffen Rendle, “Factorization Machines,” ICDM 2010 | https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf | The principle that sparse categorical feature interactions can be modeled by low-rank factors instead of explicit combinatorial tables. | No recommender-system objective, no user/item structure, no second-order-only restriction. |
| Mathieu Blondel, Akinori Fujino, Naonori Ueda, Masakazu Ishihata, “Higher-Order Factorization Machines,” NeurIPS 2016 | https://arxiv.org/abs/1607.07195 | The ANOVA-kernel view of efficient higher-order feature interactions. | No dynamic-programming HOFM solver is copied; Codex should implement a small PyTorch elementary-symmetric recurrence over 64 board tokens. |
| Ninh Pham and Rasmus Pagh, “Fast and Scalable Polynomial Kernels via Explicit Feature Maps,” KDD 2013 | https://dl.acm.org/doi/10.1145/2487575.2487591 | Motivation that polynomial interactions can be made tractable without enumerating all monomials. | No TensorSketch hashing is required in the first experiment; the proposed operator uses learned low-rank coordinates instead. |
| Yang Gao et al., “Compact Bilinear Pooling,” CVPR 2016 | https://openaccess.thecvf.com/content_cvpr_2016/html/Gao_Compact_Bilinear_Pooling_CVPR_2016_paper.html | The idea that compact bilinear/higher-order statistics can capture fine-grained interactions that first-order features miss. | No visual bilinear CNN backbone is copied; this is a board-token set functional. |
| Michael Tsang et al., “How does this interaction affect me? Interpretable Attribution for Feature Interactions,” NeurIPS 2020 | https://papers.neurips.cc/paper_files/paper/2020/file/443dec3062d0286986e21dc0631734c9-Paper.pdf | The vocabulary that higher-order interactions are effects not decomposable into additive lower-order terms. | No post-hoc attribution method is copied; the interaction decomposition is built into the model. |
| Christos Louizos, Max Welling, Diederik Kingma, “Learning Sparse Neural Networks through L0 Regularization,” ICLR 2018 | https://arxiv.org/abs/1712.01312 | Motivation for sparsity pressure on learned interaction channels. | The first implementation should use a simple deterministic L1 or sigmoid-gate penalty, not hard-concrete stochastic gates, unless Codex chooses it as a later optional extension. |
| Taco Cohen and Max Welling, “Group Equivariant Convolutional Networks,” ICML 2016 | https://arxiv.org/abs/1602.07576 | A cautionary reference: symmetry sharing can reduce sample complexity when the symmetry is valid. | No group convolution is used; chess is not fully rotation/reflection invariant because pawns, castling, and side-to-move break most board symmetries. |
| Cao, Mirjalili, and Raschka, “Rank Consistent Ordinal Regression for Neural Networks,” Pattern Recognition Letters 2020 | https://arxiv.org/abs/1901.07884 | Considered as a serious alternate path for fine-label `0/1/2` structure. | Not selected because the central idea would be a label-head/loss change rather than a new board-position operator. |

Candidate search trace, including serious mechanisms not selected:

| Candidate mechanism screened | Decision | Reason it lost to MPCN |
|---|---|---|
| Coherent ordinal survival head for labels `0 < 1 < 2` | Not selected | Attractive for near-puzzle calibration, but too head/loss-centric and does not test a new board-structure hypothesis. |
| Multi-encoding invariant-risk model across `simple_18` and LC0-style encodings | Not selected | Useful later, but it requires more data-loader infrastructure and the synthetic environments may not correspond to real causal environments. |
| Masked board autoencoding / MDL motif compressor | Not selected | Strong self-supervised idea, but the first falsifier is murkier: good reconstruction may reward material/popularity patterns rather than puzzle structure. |
| Discrete chess wavelet/scattering features | Not selected | More geometry-aware than a CNN, but still close to fixed convolutional feature extraction and fragile under chess-specific orientation constraints. |
| Rule-only legality energy without engine search | Not selected | It risks drifting into legal move counts, check status, or mate/stalemate oracles; leakage policing would dominate the experiment. |
| Hypergraph over piece-square triples with learned message passing | Not selected | Close to graph/hypergraph neural networks and could be confused with imported incidence/sheaf families; MPCN uses no edges or message passing. |
| Energy-based binary model over board corruption | Not selected | Interesting but benchmark integration and calibration would be heavier than a direct logits model. |
| Selective/abstention classifier focused on class `1` ambiguity | Not selected | Diagnostic value is high, but it does not directly improve the representation of puzzle-like structure. |
| Source-artifact adversarial suppression | Not selected | Dataset provenance/source labels are forbidden as inputs; even as adversarial labels this would need careful policy review and is not minimal. |
| Plain square-token MLP mixer | Not selected | It is likely to collapse into an ordinary dense baseline without a crisp mathematical falsifier. |
| Spectral Laplacian on current occupancy grid | Not selected | Too close to generic image spectral features and weaker chess inductive bias than explicit piece constellations. |
| Learned sparse polynomial interactions over piece-square tokens | Selected | It is simple, label-safe, current-data testable, and has a clear ablation: remove or randomize degree-2/3 interaction semantics. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on `simple_18` | `src/chess_nn_playground/models/cnn.py` | Already exists and mostly tests local spatial filters, not explicit sparse high-order constellations. |
| Residual CNN small/medium/deep | `src/chess_nn_playground/models/residual_cnn.py` | Already exists and scaling residual depth would be routine architecture tuning. |
| LC0-style CNN on `lc0_static_112` or `lc0_bt4_112` | Existing LC0-style CNN variants | Already covered as an encoding/backbone family and too close to “copy LC0” if made central. |
| LC0-style residual CNN | Existing LC0 BT4-style residual variants | Repeating residual blocks over LC0 planes is a baseline extension, not a new falsifiable operator. |
| Ordinary ViT over 64 square tokens | Common transformer baseline | Too generic, attention-heavy, and explicitly disallowed as a core idea. |
| Plain GNN-on-squares | Common graph neural network | Too close to generic square adjacency message passing and less targeted than degree-isolated interactions. |
| Attack/defense graph, sheaf, Hodge, curvature, or tension model | Imported tactical sheaf/Hodge packets | Explicitly duplicate with the imported family and not allowed unless the formal operator changes substantially. |
| One-ply move-delta set, spectrum, entropy, or DeepSets model | Imported counterfactual move-delta packets | Explicitly duplicate with the imported move-delta family and would also require careful move-generation policing. |
| Piece-target/material-target Sinkhorn or transport bottleneck | Imported optimal-transport packets | Explicitly duplicate with the imported OT family. |
| Deterministic nuisance-vector residualization/projection | Imported nuisance-orthogonal packet | Explicitly duplicate with the imported projection family. |
| Hyperparameter tuning, optimizer tuning, or batch-size search | All existing baselines | Too ordinary and does not introduce a new inductive bias. |
| Ensembling existing models | Existing model suite | Likely improves leaderboard numbers without explaining whether a new structural signal exists. |
| Add more data from the full 45M-row Parquet | Data pipeline, not model baseline | Disallowed as a core idea until streaming support exists and not a representation hypothesis. |
| Coarse-only MLP on material counts and side-to-move | Nuisance/count baseline | Useful as an ablation but too weak and shortcut-prone as the main research idea. |
| Rank-consistent ordinal head only | No exact existing baseline | Worth future study, but here it would be a label-geometry idea rather than a current-board structural operator. |

## 6. Mathematical Thesis

### Input space definition

Let a board position be represented by a tensor `x ∈ X_C = {0,1 or real planes}^{C×8×8}` from an allowed encoding. For the first experiment, `C=18` and the first 12 channels are piece occupancy planes. Define the finite token universe

```text
A = {piece_type_color p ∈ {1,...,12}} × {square s ∈ {1,...,64}}.
```

For a board `b`, let `T(b) = {t_1, ..., t_n}` be the multiset of occupied piece-square tokens, where `n ≤ 32` in legal chess positions but the implementation may safely process all 64 squares with an occupancy mask. Let `u(b)` denote safe non-piece state derived from the encoding: side-to-move, castling rights, and en-passant plane information.

### Label/target definition

The fine label is `Y_f ∈ {0,1,2}`. The benchmark target is

```text
Y = 1[Y_f > 0] ∈ {0,1}.
```

Fine labels may be used by reports and optional diagnostics, but the main model returns binary logits and must not receive fine labels or any label-derived metadata as inputs.

### Data distribution assumptions

Assume train/validation/test positions are sampled from an empirical distribution `D` over `(b, Y_f)`. The source process may contain nuisance correlations such as material, phase, and common opening/endgame patterns. The hypothesis is not that these nuisances are absent; it is that verified and near-verified puzzle-likeness has an additional component expressible by a sparse set of low-order piece-square interactions.

### Allowed symmetry or equivariance assumptions

Chess is not invariant under arbitrary board rotations or reflections. Pawns, promotion direction, castling, en-passant, and side-to-move break most image symmetries. A 180-degree rotation plus color swap is a rule-level symmetry only if all state planes and labels are transformed consistently. MPCN therefore should not hard-tie rotations/reflections in the first experiment. It may include both raw square embeddings and side-to-move state embeddings, but no compulsory group-equivariant sharing.

### Core hypothesis

There exists a useful component of the Bayes logit `η(b)=log(P(Y=1|b)/P(Y=0|b))` that is close to a sparse bounded-degree set function of occupied piece-square tokens:

```text
η(b) ≈ c + g(u(b))
       + Σ_i a_1(t_i)
       + Σ_{i<j} a_2(t_i,t_j)
       + Σ_{i<j<k} a_3(t_i,t_j,t_k).
```

The degree-1 term captures material and individual square facts. The degree-2 and degree-3 terms capture conjunctions such as “king-like target square plus attacker-like piece plus loose high-value piece” without calculating attacks, legal moves, or engine scores. The model does not claim that all tactics reduce to triples; it tests whether triples are enough to add measurable signal on this benchmark.

### Formal object introduced

Let `ψ_θ: A -> R^d` be a learned embedding of piece-square tokens. Define degree-`r` elementary symmetric interaction embeddings

```text
Φ_r(T) = Σ_{1≤i_1<...<i_r≤n} ψ(t_{i_1}) ⊙ ψ(t_{i_2}) ⊙ ... ⊙ ψ(t_{i_r}) ∈ R^d,
```

where `⊙` is elementwise product. MPCN uses `Φ_1`, `Φ_2`, and `Φ_3`, normalized by simple count-dependent factors to reduce trivial dependence on the number of occupied pieces:

```text
H_1 = Φ_1 / sqrt(max(n,1))
H_2 = Φ_2 / sqrt(max(C(n,2),1))
H_3 = Φ_3 / sqrt(max(C(n,3),1)).
```

A sparse gate `γ_r ∈ [0,1]^d` may modulate each degree: `γ_r ⊙ H_r`. The classifier is a small MLP over `[H_1, H_2, H_3, state_embedding(u)]`.

### Proposition

For any order-`r` symmetric piece-token interaction tensor with CP rank at most `d`,

```text
A_r(t_1,...,t_r) = Σ_{m=1}^d w_{r,m} Π_{j=1}^r ψ_m(t_j),
```

the aggregate interaction score over all unordered `r`-tuples in a board is exactly linear in `Φ_r(T)`:

```text
Σ_{i_1<...<i_r} A_r(t_{i_1},...,t_{i_r}) = w_r^T Φ_r(T).
```

Therefore a linear head on `[Φ_1, Φ_2, Φ_3]`, and hence an MLP head on their normalized/gated versions, can represent any Bayes-logit component that decomposes into a sum of CP-rank-`d` interactions of degree at most 3 over occupied piece-square tokens.

### Proof sketch or derivation

Expand the right-hand side:

```text
w_r^T Φ_r(T)
= Σ_m w_{r,m} Σ_{i_1<...<i_r} Π_j ψ_m(t_{i_j})
= Σ_{i_1<...<i_r} Σ_m w_{r,m} Π_j ψ_m(t_{i_j})
= Σ_{i_1<...<i_r} A_r(t_{i_1},...,t_{i_r}).
```

The efficient recurrence is the standard elementary-symmetric recurrence. Initialize `E_0=1` and `E_1=E_2=E_3=0`; for each token vector `v`, update descending in degree:

```text
E_3 <- E_3 + E_2 ⊙ v
E_2 <- E_2 + E_1 ⊙ v
E_1 <- E_1 + E_0 ⊙ v
```

After all tokens, `E_r = Φ_r`. Descending updates prevent a token from interacting with itself.

### What is actually proven

The proposition proves representability and efficient computation for low-rank bounded-degree unordered token interactions. It also proves that the central operator is not merely pooling first-order material facts: `Φ_2` and `Φ_3` contain multiplicative conjunctions that cannot generally be reduced to an additive sum of token-wise terms.

### What remains only hypothesized

It is not proven that puzzle-likeness in this dataset is generated by low-rank degree-2 or degree-3 interactions. It is also not proven that learned embeddings will find semantically meaningful chess motifs rather than shortcut correlations. These are empirical claims to be tested by the ablations.

### Counterexamples where the idea should fail

- A puzzle whose key property requires a forced line several plies deep with no distinctive current-board constellation.
- Tablebase-like or zugzwang-like positions where legal tempo and full legal move consequences dominate static piece facts.
- Positions where a tactical-looking constellation exists but a hidden legal defense refutes the puzzle; MPCN has no engine or legal move tree to detect that.
- Motifs requiring high-order interactions beyond triples, such as multi-piece mating nets with many defenders and escape squares.
- Dataset splits where source artifacts or material/phase shortcuts dominate the label more than puzzle structure.
- LC0 encodings with unknown channel maps; the deterministic tokenizer must fail closed rather than guess.

### Self-critique

The strongest objection is that explicit piece-square triples may still be a fancy way to memorize material, phase, and square-frequency artifacts. A second objection is that tactics are about legal moves and attacks, while MPCN intentionally refuses both. The experiment is still worth running because the central falsifier is sharp: if degree-2/3 features and piece-square binding semantics matter, the main model should beat degree-1, material-count, and binding-shuffle ablations, especially on class `1` near-puzzle recall at matched fine-label-`0` false-positive rate. If those ablations match it, the mechanism should be abandoned rather than patched with more width.

## 7. Architecture Specification

### Module names

- `SafeBoardStateAdapter`
- `PieceSquareTokenizer`
- `ElementarySymmetricInteractionBlock`
- `DegreeGate`
- `ConstellationClassifierHead`
- `MobiusPieceConstellationNet`

### Forward-pass steps

Default config: `encoding=simple_18`, `input_channels=18`, `embedding_dim=96`, `state_dim=96`, `hidden_dim=192`, `max_degree=3`, `num_classes=2`.

1. Input tensor:

   ```text
   x: (B, C, 8, 8)
   ```

2. Safe adapter extracts piece planes and state planes.

   For `simple_18`:

   ```text
   piece_planes = x[:, 0:12, :, :]       -> (B, 12, 8, 8)
   state_planes = x[:, 12:18, :, :]      -> (B, 6, 8, 8)
   ```

   For `lc0_static_112` and `lc0_bt4_112`, Codex must require an explicit `channel_map.current_piece_planes` before deterministic tokenization. If the map is absent, raise a clear error. History planes must not be interpreted as current-board geometry.

3. Tokenization over all 64 squares.

   Let `S=64`, `D=embedding_dim`. Flatten board squares in a documented rank-file order. Compute:

   ```text
   occ:        (B, S)       = clamp(sum over 12 piece planes, 0, 1)
   piece_part: (B, S, D)    = piece one-hot at each square times learned piece embedding
   square_part:(B, S, D)    = learned raw square embedding, multiplied by occ
   ps_part:    (B, S, D)    = learned piece-square embedding table, selected by piece and square
   tokens V:   (B, S, D)    = occ[...,None] * layer_norm(piece_part + square_part + ps_part)
   ```

   This can be implemented without `argmax` by multiplying flattened piece planes against embedding tables. Empty squares produce zero vectors and `occ=0`.

4. State embedding.

   Flatten the whitelisted state planes and project them:

   ```text
   state_flat: (B, 6*8*8) for simple_18
   state_emb:  (B, state_dim)
   ```

   The state adapter must not read labels, source fields, legal move counts, or any metadata outside the input tensor.

5. Degree interaction block.

   Use the descending elementary-symmetric recurrence over `S=64` token vectors, with the occupancy mask already baked into `V`:

   ```text
   H1, H2, H3: each (B, D)
   ```

   Normalize:

   ```text
   n = occ.sum(dim=1)
   H1 = H1 / sqrt(clamp(n, min=1))
   H2 = H2 / sqrt(clamp(n*(n-1)/2, min=1))
   H3 = H3 / sqrt(clamp(n*(n-1)*(n-2)/6, min=1))
   ```

6. Optional degree gates.

   ```text
   G1, G2, G3: learned sigmoid gates, each (D,)
   Z = concat(LN(G1*H1), LN(G2*H2), LN(G3*H3), LN(state_emb)) -> (B, 3D + state_dim)
   ```

   The L1 gate penalty is optional but recommended at small weight. It should be possible to disable gates for an ablation.

7. Classifier head.

   ```text
   hidden = GELU(Linear(Z, hidden_dim))
   hidden = Dropout(p=0.10)(hidden)
   logits = Linear(hidden, num_classes) -> (B, 2)
   ```

   `forward(x)` returns only `logits` so the shared trainer, reports, confusion matrices, predictions, and leaderboard code can keep working. If Codex adds diagnostics, use a separate method such as `forward_with_features(x)` rather than changing the default return type.

### Parameter-count estimate

For `D=96`, `state_dim=96`, `hidden_dim=192`:

- Piece embedding: `12*96 ≈ 1.2k`.
- Square embedding: `64*96 ≈ 6.1k`.
- Piece-square embedding: `12*64*96 ≈ 73.7k`.
- State linear adapter from `384` to `96`: `≈36.9k` plus bias.
- Degree gates and layer norms: `<1k`.
- Classifier from `384` to `192` to `2`: `≈74.3k`.
- Total expected parameters: roughly `190k` to `220k`, depending on exact layer-norm and adapter choices.

This is intentionally smaller than many CNN baselines; a later scaled variant may use `D=128` only if the small model passes ablations.

### FLOP and complexity estimate

The interaction recurrence costs `O(B*S*D*max_degree)` elementwise operations. With `S=64`, `D=96`, `max_degree=3`, this is about `18,432` multiply/add-style operations per sample before the classifier. The classifier costs about `(3D+state_dim)*hidden_dim ≈ 73,728` multiply-adds per sample. The model is CPU/GPU friendly and should not require mixed precision.

### Candidate-set memory and chunking

MPCN deliberately does not enumerate pairs or triples. There is no generated move set, target set, graph edge set, or candidate list. Main activation memory is approximately:

```text
V token tensor: B * 64 * D * bytes_per_float
H degree tensors: B * 3 * D * bytes_per_float
state/head activations: B * O(hidden_dim)
```

At `B=512`, `D=96`, float32 token memory is about `512*64*96*4 ≈ 12.6 MB`. No chunking is needed for the default. If Codex later implements explicit diagnostic enumeration of pairs/triples, it must be disabled during normal training and chunked by tuple count.

### Required config fields

```yaml
model:
  name: mobius_piece_constellation
  input_channels: 18
  num_classes: 2
  encoding: simple_18
  embedding_dim: 96
  state_dim: 96
  hidden_dim: 192
  max_degree: 3
  dropout: 0.10
  use_degree_gates: true
  gate_l1_weight: 0.00001
  normalize_by_tuple_count: true
  channel_map: null
```

### Encoding support and fail-closed adapter assumptions

- `simple_18`: fully supported in the first experiment. Piece planes are `0:12`; state planes are `12:18`.
- `lc0_static_112`: not enabled by default. It may be enabled only if Codex provides an explicit current-board piece-plane map and tests it. Any history or auxiliary channels are learned features only, not deterministic geometry.
- `lc0_bt4_112`: not enabled by default. Because unavailable history planes may be zero-filled, the deterministic tokenizer must use only current-board piece channels from a verified map. If no verified map exists, raise an error.

## 8. Loss, Training, And Regularization

- Primary loss: weighted cross-entropy on binary target `Y = 1[fine_label > 0]` using logits `(B,2)`.
- Auxiliary loss: optional degree-gate sparsity, `λ * (mean(sigmoid(gate_1)) + mean(sigmoid(gate_2)) + mean(sigmoid(gate_3)))`; default `λ=1e-5`. This regularizes interaction channels but must not dominate training.
- Class weighting: use the existing balanced class-weighting path for the binary target.
- Batch size expectations: default `512`; if memory is tight, `256` is acceptable but the same value must be used for central ablations.
- Optimizer defaults: AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Epochs: `3` for the minimal current-data experiment, matching the provided config template. Later scale only if ablations support the mechanism.
- Regularizers: dropout `0.10` in classifier head, weight decay, optional gate sparsity. Do not add heavy augmentation in the first run because it would confound the mechanism test.
- Determinism requirements: seed `42`, deterministic PyTorch mode where supported, fixed data split, fixed threshold-selection protocol, and logged config hashes.
- Fair-comparison invariants: same train/val/test Parquet files, same binary target mapping, same `simple_18` encoding, same reporting scripts, same class weighting policy, and no use of full dataset or engine metadata.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Degree-1 only, parameter-matched | Zero `Φ2` and `Φ3`; optionally add a same-size MLP filler so parameter count is similar. | Central claim that multiplicative higher-order constellations add signal beyond material/square marginals. | If it matches the main model, abandon the high-order interaction thesis for this task. |
| Degree ≤2 only | Use `Φ1` and `Φ2`, zero `Φ3`. | Whether triples add value beyond pair interactions. | If no drop from main but both beat degree-1, prefer a simpler pair-only MPCN. |
| Binding-shuffle semantics ablation | For each sample, preserve occupied square set, material multiset, side-to-move, castling, en-passant, and piece count, but randomly reassign piece identities to occupied squares before building `ps_part`. | Whether exact piece-square binding matters, not just material and square occupancy marginals. | If it matches main, the model is exploiting material/square marginals or count-like artifacts. |
| Square-randomization ablation | Preserve material and side-state, but replace occupied square embeddings with a fresh random permutation per sample/epoch. | Whether board geometry and square identity matter. | If it matches main, spatial constellation semantics are not being used. |
| Interaction shuffle within material buckets | Compute normal `Φ2/Φ3`, then shuffle those degree features across batch items with the same coarse material-count bucket and side-to-move when possible. | Whether high-order features are label-aligned beyond nuisance bucket distribution. | If it matches main, interaction features likely act as nuisance regularizers rather than semantic signal. |
| Piece-type-only FM | Remove square and piece-square embeddings; keep only piece/color embeddings and state. | Whether material composition alone explains gains. | If close to main, MPCN is not learning chess-location structure. |
| No piece-square table | Use separate piece and square embeddings but remove the learned joint `piece_square[p,s]` embedding. | Whether gains require memorizing specific piece-square categories or only factored piece/square effects. | If this is better, the joint table may overfit source artifacts. |
| Gates disabled | Set all degree gates to one and remove L1 gate loss. | Whether sparse channel selection improves generalization. | If disabled gates are better, keep the simpler ungated version. |
| Count/material/state MLP baseline | Feed only material counts, occupied count, side-to-move, castling, and en-passant summary to a small MLP. | Shortcut floor from obvious safe nuisance features. | If near main, do not trust MPCN gains as tactical. |
| Label permutation smoke test | Permute training labels while leaving validation/test unchanged; train briefly. | Implementation sanity and leakage check. | Any meaningful validation performance suggests leakage or a reporting bug. |

The smallest central falsifier is the degree-1-only ablation. The strongest semantics-destroying ablation is binding-shuffle, because it preserves material, occupied-square set, side-state, piece count, and capture-free static marginals while destroying which piece is on which square.

## 10. Benchmark And Falsification Criteria

Codex should benchmark the main model and the central ablations on the same split and reporting pipeline.

Baselines to compare against:

- Existing `simple_18` simple CNN small/default.
- Existing `simple_18` residual CNN small/medium if available in current configs.
- Existing LC0-style CNN/residual results may be listed for context, but the primary fair comparison is against `simple_18` models because MPCN first uses `simple_18`.
- Count/material/state MLP ablation from this packet.

Metrics to inspect:

- Accuracy, balanced accuracy, AUROC, AUPRC, F1, precision, recall.
- Calibration/ECE if the reporting stack already supports it; do not block the run if absent.
- Fine-label rectangular confusion matrix `0/1/2 -> predicted 0/1` for the main model and every central ablation.
- Near-puzzle diagnostic: choose a threshold on validation to match the best baseline’s fine-label-`0` false-positive rate, then report class-`1` recall and class-`2` recall on test at that threshold. Also report class-`1` precision among predicted positives if supported.

Required artifacts:

- Main config YAML and resolved config.
- Model parameter count.
- Training log with seed and deterministic flag.
- Validation and test reports.
- Test predictions with fine labels retained for diagnostics.
- `3x2` diagnostic confusion matrices for main, degree-1-only, degree≤2-only, binding-shuffle, and count/material/state MLP.
- Leaderboard update entry.
- Short ablation report explaining whether `Φ2/Φ3` survived.

Success threshold:

- Main MPCN improves over the best fair `simple_18` CNN/residual baseline by at least one of: `+0.010 AUROC`, `+0.010 balanced accuracy`, or `+0.020 class-1 recall at matched fine-label-0 FPR`, without increasing the matched fine-label-0 FPR by more than `0.005` absolute.
- At least one semantics-preserving interaction model, main or degree≤2, must beat the count/material/state MLP by `≥0.020 AUROC`.
- Binding-shuffle must be measurably worse than main, preferably by `≥0.005 AUROC` or `≥0.010 class-1 recall at matched fine-label-0 FPR`.

Failure threshold:

- Main MPCN is statistically indistinguishable from degree-1-only and binding-shuffle on AUROC and near-puzzle recall.
- Count/material/state MLP is within `0.005 AUROC` of main.
- Fine-label `1` recall does not improve over baseline at matched fine-label-`0` FPR.

What result would make me abandon the idea:

- Degree-1-only, binding-shuffle, and count/material/state ablations all match main within noise, or the main model only wins by increasing fine-label-`0` false positives. In that case, do not repeat polynomial piece-square constellation models next cycle.

What result would justify scaling:

- Main or degree≤2 MPCN beats fair `simple_18` CNN/residual baselines and all nuisance-preserving ablations, with a clear class-`1` near-puzzle diagnostic gain. Then scale `D` from 96 to 128, test `lc0_static_112` only with verified channel maps, and consider combining MPCN features with a small CNN as a second-stage experiment.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_mobius_constellation/idea.yaml` | Create | Machine-readable idea summary, status, config path, model path, and latest result path placeholders. |
| `ideas/20260421_mobius_constellation/math_thesis.md` | Create | Mathematical thesis from Sections 6 and 9, including proposition, proof sketch, and ablation logic. |
| `ideas/20260421_mobius_constellation/architecture.md` | Create | Architecture details, tensor shapes, parameter count, FLOP estimate, and adapter fail-closed rules. |
| `ideas/20260421_mobius_constellation/implementation_notes.md` | Create | Vectorized tokenization notes, elementary-symmetric recurrence, deterministic tests, and no-leakage checklist. |
| `ideas/20260421_mobius_constellation/trainer_notes.md` | Create | Loss, class weighting, deterministic settings, benchmark invariants, and report requirements. |
| `ideas/20260421_mobius_constellation/ablations.md` | Create | Ablation table, exact config toggles, and expected interpretations. |
| `ideas/20260421_mobius_constellation/train.py` | Create | Thin entrypoint that invokes the shared trainer with this idea’s config, plus optional ablation config generation if the repo pattern supports it. |
| `ideas/20260421_mobius_constellation/config.yaml` | Create | Copy of the `config_yaml` block below, with model-specific fields added under `model`. |
| `ideas/20260421_mobius_constellation/report_template.md` | Create | Template requiring main metrics, `3x2` confusion, near-puzzle matched-FPR diagnostic, and ablation comparisons. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add anti-duplicate guidance for polynomial/ANOVA/factorization-machine piece-constellation networks after this packet is consumed. Preserve all hard leakage and label rules. |
| `src/chess_nn_playground/models/mobius_piece_constellation.py` | Create | `SafeBoardStateAdapter`, `PieceSquareTokenizer`, `ElementarySymmetricInteractionBlock`, `DegreeGate`, `ConstellationClassifierHead`, and `MobiusPieceConstellationNet`. |
| `src/chess_nn_playground/models/registry.py` | Update | Register builder name `mobius_piece_constellation`. Ensure `forward(x)` returns `(B,2)` logits. |
| `configs/mobius_piece_constellation_simple18.yaml` | Create | Shared-trainer compatible config for the minimal experiment. |
| `configs/mobius_piece_constellation_simple18_degree1.yaml` | Create | Central degree-1-only ablation config. |
| `configs/mobius_piece_constellation_simple18_degree2.yaml` | Create | Degree≤2 ablation config. |
| `configs/mobius_piece_constellation_simple18_binding_shuffle.yaml` | Create | Semantics-destroying material/square-marginal-preserving ablation config. |
| `configs/mobius_piece_constellation_simple18_count_mlp.yaml` | Create | Count/material/state nuisance baseline config. |
| `tests/test_mobius_piece_constellation.py` | Create | Shape tests, deterministic recurrence tests against explicit enumeration on tiny synthetic boards, gradient-flow test, and logits shape test. |
| `tests/test_mobius_safe_adapters.py` | Create | `simple_18` channel extraction test and LC0 fail-closed test when no channel map is supplied. |
| `tests/test_mobius_ablations.py` | Create | Verify degree masks, binding shuffle preservation of material/occupied-square counts, and label permutation smoke-test plumbing if feasible. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0713_tuesday_los_angeles_mobius_constellation.md
  generated_at: "2026-04-21 07:13 America/Los_Angeles"
  weekday: Tuesday
  timezone: America/Los_Angeles
  idea_slug: mobius_constellation
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_mobius_constellation
  name: "Möbius Piece-Constellation Network"
  slug: mobius_constellation
  status: draft
  created_at: "2026-04-21 07:13 America/Los_Angeles"
  author: ChatGPT Pro
  short_thesis: "Explicit degree-2 and degree-3 ANOVA interactions over occupied current-board piece-square tokens should expose puzzle-like tactical constellations without moves, engines, sheaves, transport, or nuisance projection."
  novelty_claim: "Applies a degree-isolated low-rank polynomial set functional to chess piece-square tokens as the central classifier operator, with semantics-destroying material-preserving ablations."
  expected_advantage: "Better near-puzzle recall and binary AUROC than simple_18 CNN baselines if puzzle-likeness depends on sparse piece constellations that local convolutions underuse."
  central_falsification_ablation: "Degree-1-only parameter-matched model with Φ2 and Φ3 removed."
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: "binary logits [batch, 2]"
  compute_notes: "No pair/triple enumeration; O(batch * 64 * embedding_dim * max_degree), default about 190k-220k parameters."
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/mobius_piece_constellation_simple18.yaml
  model_path: src/chess_nn_playground/models/mobius_piece_constellation.py
  latest_result_path: null
  notes: "Fail closed for lc0_static_112 and lc0_bt4_112 unless explicit current-board piece-plane channel maps are implemented and tested."
```

```yaml
config_yaml:
  run:
    name: mobius_piece_constellation_simple18
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
    name: mobius_piece_constellation
    input_channels: 18
    num_classes: 2
    embedding_dim: 96
    state_dim: 96
    hidden_dim: 192
    max_degree: 3
    dropout: 0.10
    use_degree_gates: true
    gate_l1_weight: 0.00001
    normalize_by_tuple_count: true
    channel_map: null
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
  model_name: mobius_piece_constellation
  file_path: src/chess_nn_playground/models/mobius_piece_constellation.py
  builder_function: build_mobius_piece_constellation
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - SafeBoardStateAdapter
    - PieceSquareTokenizer
    - ElementarySymmetricInteractionBlock
    - DegreeGate
    - ConstellationClassifierHead
    - MobiusPieceConstellationNet
  required_config_fields:
    - input_channels
    - num_classes
    - embedding_dim
    - state_dim
    - hidden_dim
    - max_degree
    - use_degree_gates
    - normalize_by_tuple_count
  expected_parameter_count: "190k-220k with embedding_dim=96, state_dim=96, hidden_dim=192"
  expected_memory_notes: "No candidate enumeration. Token tensor memory is batch*64*embedding_dim floats; about 12.6 MB at batch=512, embedding_dim=96, float32."
```

```yaml
research_continuity:
  idea_fingerprint: "current-board occupied piece-square tokens + degree-isolated ANOVA/Möbius elementary symmetric embeddings Φ1/Φ2/Φ3 + sparse gates + binary puzzle-likeness head"
  already_researched_family_overlap: "Not a tactical sheaf/Hodge/attack-defense incidence model; not a one-ply move-delta bag/spectrum/landscape; not a Sinkhorn/OT transport model; not a deterministic nuisance projection model."
  closest_duplicate_risk: "Higher-order factorization machine or polynomial feature network over chess piece-square categories."
  do_not_repeat_if_this_fails:
    - "Do not propose another current-board piece-square ANOVA, polynomial-kernel, factorization-machine, TensorSketch, compact bilinear, or low-rank monomial constellation classifier unless it changes the falsifiable object beyond degree/order/width/gating."
    - "Do not rescue this mechanism merely by increasing embedding_dim, max_degree, or MLP width."
    - "Do not repackage binding-shuffle-sensitive failures as tactical motif discovery without new evidence."
  suggested_next_search_directions:
    - "Coherent ordinal or selective prediction for fine labels 0/1/2 if representation gains fail but class-1 ambiguity remains important."
    - "Masked generative compression with strict nuisance controls if static supervised interactions fail."
    - "Causal invariance across verified non-source environments only if safe environment labels can be defined without provenance leakage."
    - "Encoding-consistency regularization between simple_18 and verified LC0 current-board adapters after fail-closed channel maps exist."
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Möbius Piece-Constellation Network` to imported research memory after implementation. | Prevents future ChatGPT passes from repeating polynomial/ANOVA piece-square token interactions as fresh. | `Imported Research Memory` |
| Add anti-duplicate fingerprint: `current-board occupied piece-square tokens + low-rank degree-2/3 polynomial, ANOVA, TensorSketch, compact bilinear, or factorization-machine pooling + binary puzzle-likeness`. | Captures nearby variants that only change degree, rank, gates, or pooling names. | `Research Continuity` or anti-duplicate rules |
| Add a required ablation for any future token-interaction model: material/count/state-preserving binding shuffle. | Forces distinction between real piece-square semantics and material/square-frequency shortcuts. | `Depth requirements` and `Ablation Plan` guidance |
| Clarify that using fine labels for auxiliary losses is allowed only when they are genuine dataset labels and never neural inputs; unresolved candidates must remain unresolved. | Helps future ordinal/calibration proposals stay label-safe. | `Non-Negotiable Constraints` |
| Add LC0 adapter fail-closed rule: deterministic channel semantics must be verified before deriving rule geometry from 112-plane encodings. | Prevents silent misuse of history or unknown planes as current-board features. | `Problem Restatement And Data Contract` |
| If MPCN fails, add a note discouraging more static piece-square polynomial/factorization-machine variants unless combined with a distinct causal, generative, or uncertainty falsifier. | Avoids width/order tuning disguised as research. | `Research Continuity` |

## 14. Final Sanity Check

- Downloadable Markdown file created: Yes.
- Filename follows required date/time/day/timezone/slug pattern: Yes, `chess_nn_research_2026-04-21_0713_tuesday_los_angeles_mobius_constellation.md`.
- No forbidden engine features used as inputs: Yes.
- Does not fabricate labels: Yes.
- Not a routine CNN/ResNet/Transformer variant: Yes.
- Minimal current-data experiment exists: Yes.
- Falsification criterion is concrete: Yes.
- Codex can implement without asking for missing architecture details: Yes.
- Prompt maintenance notes included for Codex: Yes.
- Repetition check against imported research packets completed: Yes.
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: Yes.
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: Yes.
- Not a deterministic nuisance-orthogonal projection bottleneck variant: Yes.
