# Codex Handoff Packet: Directed Attack-Sheaf Tension Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0427_tuesday_local_attack_sheaf_tension.md`
- Generated at: `2026-04-21T04:27:51-07:00`
- Weekday: `tuesday`
- Timezone: `local`, America/Los_Angeles
- Idea slug: `attack_sheaf_tension`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Directed Attack-Sheaf Tension Network
- One-sentence thesis: A chess position is puzzle-like when its static attack geometry contains localized, asymmetric, color-sensitive inconsistencies that are better exposed by a learned sheaf Laplacian and tension-energy readout on directed chess attack relations than by translation kernels or square-to-square attention.
- Idea fingerprint: `occupancy_gated_directed_chess_attack_graph + nontrivial_learned_sheaf_restrictions + laplacian_tension_energy_readout + no_engine + no_full_D4_symmetry`
- Why this is not a common CNN/ResNet/Transformer variant: The core computation is not convolution over the 8x8 grid, residual stacking, or all-pairs token attention; it constructs a fixed typed chess attack/ray incidence complex, learns relation-specific restriction maps, applies a state-gated sheaf diffusion operator, and explicitly feeds sheaf inconsistency energies to the classifier.
- Current-data minimal experiment: Train `AttackSheafTensionNet` on `simple_18` using the existing `crtk_sample_3class` train/val/test split, compare against the best existing non-ensemble `simple_18` baseline and the `identity_sheaf_same_edges` ablation, and report binary metrics plus fine-label `0/1/2 -> predicted 0/1` confusion.
- Expected information gain if it fails: A clean failure against the identity-sheaf and edge-shuffle ablations would show that explicit one-ply attack geometry and nontrivial relation restrictions do not add usable puzzle-likeness signal beyond local board features on the current dataset; future cycles should then avoid attack-graph/sheaf variants and look for non-engine differentiable search surrogates or causal dataset-invariance tests instead.

## 3. Problem Restatement And Data Contract

Task: classify chess board positions as binary puzzle-likeness.

Binary outputs:

- `0`: non-puzzle
- `1`: puzzle-like

Available source fine labels, used only by the dataset loader/loss target creation and reporting code:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

Binary target convention for the current benchmark:

- fine label `0` maps to binary class `0`
- fine labels `1` and `2` map to binary class `1`
- fine labels remain available for evaluation reports such as `true fine label 0/1/2 -> predicted binary output 0/1`
- fine labels must not be passed to the neural network as input features

Allowed neural-network inputs:

- Board-position tensors already emitted by the existing encoders:
  - `simple_18`
  - `lc0_static_112`
  - `lc0_bt4_112`
- Deterministic features derived only from the input tensor and chess-board geometry, such as square coordinates, candidate ray relations, between-square masks, and soft occupancy estimated from input planes.
- Optional encoding metadata that tells the model which input channels correspond to piece occupancy or side-to-move, if that metadata is already part of the repository's encoder definitions. This metadata is structural, not a label.

Forbidden neural-network inputs:

- Stockfish scores
- principal variations
- engine node counts
- tablebase outcomes
- verification metadata
- source-label identifiers
- proposed labels
- unresolved-candidate status
- any feature computed from labels or from the verification process

Tensor contract:

- Model type: PyTorch `nn.Module`
- Input: `(batch, C, 8, 8)`
- Output: logits `(batch, num_classes)`
- Default `num_classes`: `2`
- Internal flattening may use `(batch, 64, C)` with deterministic square order, but the public interface must remain unchanged.

Benchmark split:

- Train: `data/splits/crtk_sample_3class/split_train.parquet`
- Validation: `data/splits/crtk_sample_3class/split_val.parquet`
- Test: `data/splits/crtk_sample_3class/split_test.parquet`

Leakage checklist for Codex:

- [ ] The model constructor does not accept engine fields.
- [ ] The dataloader passes only board encodings and binary labels to training.
- [ ] Fine labels are used only for binarization and reporting, not as input features.
- [ ] Attack/ray edges are generated from board coordinates and optional input-derived occupancy only.
- [ ] No unresolved candidates are relabeled or fabricated.
- [ ] No augmentation uses engine analysis or puzzle verification metadata.
- [ ] No prompt-derived assumption is used to synthesize class `1` or class `2` examples.
- [ ] Any optional color-rotation tying permutes only board/input channels known from encoder metadata and never touches labels.

## 4. Research Map

This idea borrows mathematical machinery, not code or benchmark conclusions, from the following sources.

1. Taco Cohen and Max Welling, "Group Equivariant Convolutional Networks," ICML 2016 / arXiv:1602.07576. URL: https://arxiv.org/abs/1602.07576  
   Borrowed: the principle that known symmetries can reduce sample complexity.  
   Not copied: the proposed model does not use group convolution and does not impose full image-plane rotation/reflection invariance, because chess pawns, castling rights, and side-to-move break most dihedral symmetries.

2. Michael Bronstein, Joan Bruna, Taco Cohen, and Petar Veličković, "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges," arXiv:2104.13478. URL: https://arxiv.org/abs/2104.13478  
   Borrowed: the design pattern of matching architecture to the domain's geometric structure.  
   Not copied: this packet instantiates a chess-specific directed attack sheaf rather than a generic grid, graph, or transformer architecture.

3. Jakob Hansen and Thomas Gebhart, "Sheaf Neural Networks," arXiv:2012.06333. URL: https://arxiv.org/abs/2012.06333  
   Borrowed: replacing trivial graph diffusion with sheaf Laplacian diffusion using relation-dependent restriction maps.  
   Not copied: the graph, relation types, gating, and tension readout here are specialized to chess attack/ray geometry.

4. Cristian Bodnar, Francesco Di Giovanni, Benjamin Paul Chamberlain, Pietro Liò, and Michael Bronstein, "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs," arXiv:2202.04579. URL: https://arxiv.org/abs/2202.04579  
   Borrowed: the observation that nontrivial sheaves can control diffusion behavior better than ordinary GNN Laplacians, especially when relations are not homogeneous.  
   Not copied: no citation result is assumed for chess; this packet makes a separate, falsifiable hypothesis about tactical tension.

5. Thomas Kipf and Max Welling, "Semi-Supervised Classification with Graph Convolutional Networks," arXiv:1609.02907. URL: https://arxiv.org/abs/1609.02907  
   Borrowed: the baseline notion of Laplacian-style message propagation on graphs.  
   Not copied: the proposed model is explicitly not a plain GCN; the identity-sheaf ablation is included to test whether the nontrivial sheaf part matters.

6. Chess attack-map idea from standard chess rules. URL for rule reference if Codex wants one: https://www.chessprogramming.org/Attack_and_Defend_Maps  
   Borrowed: the fact that candidate attacks/defenses can be computed from piece movement geometry and occupancy.  
   Not copied: no chess-engine evaluation, search, score, PV, or node count is used.

Unverifiable citations: none intentionally. Codex should still verify URLs during implementation if repository policy requires pinned references.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN over `8x8` planes | simple CNN | Already represented; local translation filters do not distinguish directed pins, skewers, overloaded defenders, or long-range ray tension except by learning them indirectly. |
| Ordinary residual CNN | residual CNN | Residual stacking improves optimization but does not add a new chess-specific relational object; this would violate the no-standard-ResNet core constraint. |
| Small/medium/deep CNN scaling | small/medium/deep CNN variants | Depth/width sweeps are hyperparameter tuning, not a research idea; the desired information gain is structural. |
| LC0-style CNN or LC0-style residual CNN | LC0 BT4-style CNN and residual CNN variants | An LC0 clone is explicitly disallowed as the core idea and would mainly test whether a known chess-network prior transfers. |
| Ordinary ViT on 64 square tokens | no direct baseline, closest to square Transformer | Vanilla all-pairs attention is a generic token mixer; it ignores the typed sparse structure of chess attacks and is disallowed as a core idea. |
| Plain GNN-on-squares with adjacency by king moves or legal moves | no direct baseline, closest to a graph baseline | A plain square GNN uses scalar edge weights and trivial restrictions; it is likely a weaker restatement of message passing and is included only as an ablation. |
| Hyperparameter tuning of optimizer, LR, dropout, batch size, or scheduler | all current baselines | The prompt forbids ordinary tuning as the core contribution; training choices must stay stable for a fair comparison. |
| Ensembling multiple encodings or model families | possible ensemble of existing baselines | Ensembling is explicitly disallowed and would hide whether the proposed operator has signal. |
| More data or relabeling unresolved candidates | none | The prompt forbids fabricating labels and treating unresolved candidates as resolved; current-data falsification is required. |
| Engine-evaluation auxiliary target | none | Stockfish scores, PVs, node counts, and verification metadata are forbidden as model inputs or training signals for this pass. |
| Full dihedral-equivariant board CNN | no direct baseline | Chess is not fully invariant to rotations/reflections because pawn direction, castling side, en-passant context, and side-to-move matter. |
| Pure handcrafted tactical feature classifier | no direct baseline | Handcrafted counts alone would not learn relation-specific representations and would risk becoming a brittle feature-engineering baseline rather than a neural architecture. |

## 6. Mathematical Thesis

### Input space

Let

\[
X \subset \mathbb{R}^{C \times 8 \times 8}
\]

be the set of encoded chess positions emitted by one of the supported encoders. The model receives only `x in X`. Let

\[
V = \{0,\ldots,7\} \times \{0,\ldots,7\}
\]

be the 64 board squares.

For a batch element, flatten the tensor to square features

\[
x_v \in \mathbb{R}^C, \quad v \in V.
\]

A learned pointwise map produces node features

\[
z_v = \phi_0(x_v) \in \mathbb{R}^d.
\]

### Target definition

The benchmark target is

\[
y =
\begin{cases}
0, & \text{fine label } 0, \\
1, & \text{fine label } 1 \text{ or } 2.
\end{cases}
\]

The network estimates logits for \(p(y\mid x)\). Fine labels are not network inputs.

### Distribution assumptions

1. Train, validation, and test examples are drawn from the provided split family, not necessarily from all possible chess positions.
2. Fine labels `1` and `2` are both positive for the binary task, but may represent different positive submodes; evaluation should preserve per-fine-label reporting.
3. Puzzle-likeness is assumed to be partly visible from static board geometry, but not fully determined by it.
4. No assumption is made that the board distribution is invariant under arbitrary rotations or reflections.

### Symmetry and equivariance assumptions

Chess has limited useful symmetries:

- Board translations are not valid symmetries because square identity matters for promotion, castling, and edge effects.
- Full `D4` image symmetry is invalid because pawns have direction, castling has kingside/queenside structure, and side-to-move matters.
- A color-swap plus 180-degree rotation is closer to a chess-rule symmetry if the encoding channels are also permuted correctly, but even that can be broken by dataset conventions or unavailable castling/en-passant planes.

Therefore the proposed operator is not globally rotation/reflection invariant. It is only partially equivariant at the level of relation templates: opposite ray directions may optionally share parameters under a configured color-180 channel permutation. The default minimal experiment should set `tie_color180: false` unless the repository already has robust encoding-channel permutations.

### Core hypothesis

Puzzle-like positions contain concentrated tactical tension: pieces, attack rays, blockers, and defenders impose asymmetric constraints that are locally inconsistent under relation-specific projections. A nontrivial sheaf Laplacian on directed chess attack relations should expose this tension more directly than either:

- a CNN, which must discover long rays and asymmetric piece relations through stacked local filters, or
- a plain GNN, which treats all node states as living in the same relation space.

### Formal object

Define a typed directed candidate edge set

\[
E = E_{king} \cup E_{knight} \cup E_{pawn}^W \cup E_{pawn}^B \cup E_{rook\_ray} \cup E_{bishop\_ray}.
\]

Each edge

\[
e = (u, v, \tau, q)
\]

has source square \(u\), destination square \(v\), movement family/type \(\tau\), and geometry descriptor \(q\) containing direction, distance, and a between-squares mask for sliding rays.

The graph is a candidate attack graph, not a legal-move generator. For sliding rays, all aligned ordered square pairs are included; occupancy controls the gate rather than deleting the edge nondifferentiably.

For each edge type/direction key \(\kappa(e)\), learn two restriction maps

\[
A_{\kappa(e)} \in \mathbb{R}^{r \times d},
\quad
B_{\kappa(e)} \in \mathbb{R}^{r \times d}.
\]

Given input-derived soft occupancy and endpoint features, compute a gate

\[
g_e(x,z) \in [0,1].
\]

The sheaf coboundary residual is

\[
(\delta_x z)_e =
\sqrt{g_e(x,z)}\left(A_{\kappa(e)}z_u - B_{\kappa(e)}z_v\right)
\in \mathbb{R}^r.
\]

The corresponding sheaf energy is

\[
\mathcal{E}_x(z)
=
\sum_{e \in E}
g_e(x,z)
\left\|
A_{\kappa(e)}z_u - B_{\kappa(e)}z_v
\right\|_2^2.
\]

The weighted sheaf Laplacian is

\[
L_x = \delta_x^\ast \delta_x.
\]

A sheaf diffusion layer is

\[
z^{(\ell+1)} =
\operatorname{LN}\left(
z^{(\ell)} -
\eta_\ell \widetilde{L}_x z^{(\ell)}
\right)
+
\psi_\ell(z^{(\ell)}),
\]

where \(\widetilde{L}_x\) is degree-normalized by gated incidence degree, \(\eta_\ell > 0\) is a small learned or clamped step, and \(\psi_\ell\) is a squarewise feed-forward residual with no spatial mixing.

The readout pools both final node states and energy statistics:

\[
h(x) =
\operatorname{MLP}\left(
\operatorname{pool}(z^{(L)}),
\operatorname{pool}(\mathcal{E}_{node}),
\operatorname{logsumexp}_\tau(\mathcal{E}_{edge})
\right).
\]

### Proposition or objective

**Proposition: positive semidefinite tension.**  
If \(g_e(x,z) \ge 0\) for all edges and restriction maps are finite matrices, then the sheaf Laplacian \(L_x=\delta_x^\ast\delta_x\) is positive semidefinite for fixed gates. Moreover,

\[
\mathcal{E}_x(z) = \langle z, L_x z \rangle \ge 0.
\]

For fixed gates, a gradient step

\[
z' = z - \eta L_xz
\]

weakly decreases \(\mathcal{E}_x(z)\) for sufficiently small \(\eta\), for example \(0 < \eta < 2/\lambda_{max}(L_x)\) in the unnormalized Euclidean setting.

### Proof sketch

By construction, \(\delta_x\) is a linear operator in \(z\) when gates are fixed. Therefore

\[
\langle z, L_x z\rangle
=
\langle z, \delta_x^\ast\delta_x z\rangle
=
\langle \delta_x z, \delta_x z\rangle
=
\|\delta_x z\|_2^2
\ge 0.
\]

The energy identity follows from expanding the edge residual norm. For fixed gates, the update \(z' = z - \eta L_x z\) is standard gradient descent on the quadratic energy \(\frac12 z^T L_x z\). Quadratic gradient descent decreases the objective for a step below the inverse smoothness bound, giving the stated sufficient condition.

### What is proven

- The sheaf energy is nonnegative.
- The fixed-gate sheaf Laplacian is positive semidefinite.
- A small fixed-gate diffusion step is an energy-smoothing step.
- The operator is differentiable almost everywhere when gates are produced by smooth functions and occupancy summaries use differentiable products/sums.

### What is hypothesized

- Puzzle-like positions have a learnable signature in the distribution of sheaf residuals and node-localized tension.
- Nontrivial typed restriction maps outperform identity restrictions on the same chess attack graph.
- Occupancy-gated ray relations expose pins, skewers, discovered attacks, overloaded defenders, and king-zone pressure enough to improve binary classification.
- Fine label `1` near-puzzles will benefit more than fine label `2` if current baselines overfit obvious tactical motifs.

### Counterexamples

- A composed study or deep tactic whose puzzle-likeness depends on a long forcing line invisible to one-ply attack geometry.
- A position with many attacks but no tactical puzzle because every tension is trivially resolved.
- A quiet endgame position that is puzzle-like due to zugzwang rather than attack tension.
- Positions where castling rights or en-passant details matter but are absent or ambiguous in the selected encoding.
- Dataset artifacts where puzzle-likeness is mostly determined by source-generation bias rather than board geometry.
- Non-puzzle blunder positions with high material tension that resemble puzzles statically.

## 7. Architecture Specification

### Module name

`AttackSheafTensionNet`

Suggested source file:

`src/chess_nn_playground/models/attack_sheaf_tension.py`

### High-level components

1. `SquareEncoder`
   - Input: `(B, C, 8, 8)`
   - Flatten: `(B, 64, C)`
   - Apply learned squarewise projection `Linear(C, d)` or `1x1 Conv2d(C, d)` followed by LayerNorm and GELU.
   - Output: `z0` with shape `(B, 64, d)`.

2. `TacticalGeometryCache`
   - Precomputed buffers, not trainable:
     - `src`: `(E,)` int64
     - `dst`: `(E,)` int64
     - `edge_family`: `(E,)` int64
     - `direction_id`: `(E,)` int64
     - `distance_id`: `(E,)` int64
     - `between_mask`: `(E, 64)` float/bool, nonzero only for sliding rays
   - Candidate edges include:
     - directed king-neighbor relations
     - directed knight relations
     - white pawn attack diagonals
     - black pawn attack diagonals
     - rook-like sliding ray pairs along ranks/files
     - bishop-like sliding ray pairs along diagonals
   - Do not call Stockfish or any engine. This cache is pure board geometry.

3. `OccupancyAdapter`
   - Input: original tensor `(B, C, 8, 8)` and flattened `(B, 64, C)`.
   - Preferred mode: deterministic occupancy from encoder metadata:
     - `occ`: `(B, 64, 1)`
     - optional `white_occ`, `black_occ`: `(B, 64, 1)` each
     - optional `side_to_move`: `(B, 1)` or `(B, 2)`
   - Fallback mode: learned soft occupancy `sigmoid(Linear(C, 1)(x_v))`.
   - The fallback is allowed only as an input-derived soft summary; it must not use labels or engine metadata.
   - Minimal experiment should use deterministic occupancy for `simple_18` if piece-plane mapping is known.

4. `PathGate`
   - For each edge `e`, gather:
     - `z_src`: `(B, E, d)`
     - `z_dst`: `(B, E, d)`
     - `edge_type_emb`: `(E, e_dim)`
     - `direction_emb`: `(E, dir_dim)`
     - `distance_emb`: `(E, dist_dim)`
     - `path_clear`: `(B, E, 1)`, for ray edges computed as `prod(1 - occ_between + eps)` or a stable log-sum equivalent
     - `path_blocked_mean`: `(B, E, 1)`
   - Concatenate endpoint summaries and geometry embeddings.
   - Output `g`: `(B, E, 1)` via small MLP and sigmoid.
   - For jump edges, set `path_clear=1` and `path_blocked_mean=0`.
   - For ray edges, multiply or concatenate path-clearance features; do not hard-delete edges.

5. `SheafDiffusionLayer`
   - Trainable typed restrictions:
     - `A[k]`: `(r, d)`
     - `B[k]`: `(r, d)`
     - key `k = edge_family x direction_group`, optionally tied by color-180 symmetry if enabled.
   - For gathered endpoints:
     - `a = A[k] @ z_src`: `(B, E, r)`
     - `b = B[k] @ z_dst`: `(B, E, r)`
     - `delta = sqrt(g + eps) * (a - b)`: `(B, E, r)`
   - Compute edge energy:
     - `edge_energy = sum(delta ** 2, dim=-1)`: `(B, E)`
   - Scatter gradient-like messages:
     - `m_src += A[k].T @ (g * (a - b))`
     - `m_dst -= B[k].T @ (g * (a - b))`
   - Normalize by gated degree.
   - Update:
     - `z_next = LayerNorm(z - step_size * normalized_message) + squarewise_mlp(z)`
   - `squarewise_mlp` is `Linear(d, 2d) -> GELU -> Dropout -> Linear(2d, d)` and does not mix squares.

6. `TensionReadout`
   - Inputs:
     - final nodes `zL`: `(B, 64, d)`
     - edge energies from each layer: list of `(B, E)`
     - optional node energy via scatter-add: `(B, 64, 1)`
   - Pooling:
     - mean over squares of `zL`: `(B, d)`
     - max over squares of `zL`: `(B, d)`
     - mean node energy: `(B, 1)`
     - max node energy: `(B, 1)`
     - log-sum-exp edge energy with temperature `energy_tau`: `(B, 1)`
     - optional per-family mean energy: `(B, num_edge_families)`
   - Classifier:
     - `Linear(readout_dim, hidden) -> GELU -> Dropout -> Linear(hidden, num_classes)`
   - Output: logits `(B, num_classes)`.

### Forward-pass pseudocode

```text
forward(x):
    # x: (B, C, 8, 8)
    square_x = flatten_board(x)                         # (B, 64, C)
    z = square_encoder(square_x)                        # (B, 64, d)

    occ_info = occupancy_adapter(x, square_x)           # occ: (B, 64, 1)
    geom = geometry_cache.to(x.device)

    all_edge_energies = []
    all_node_energies = []

    for layer in sheaf_layers:
        endpoint_features = gather(z, geom.src, geom.dst)
        path_features = summarize_between_squares(occ_info.occ, geom.between_mask)
        gates = path_gate(endpoint_features, path_features, geom.type_embeddings)

        z, edge_energy, node_energy = layer(z, gates, geom)
        all_edge_energies.append(edge_energy)
        all_node_energies.append(node_energy)

    readout = tension_readout(z, all_edge_energies, all_node_energies)
    logits = classifier(readout)                        # (B, num_classes)
    return logits
```

### Tensor shapes

| Symbol | Shape | Meaning |
|---|---:|---|
| `x` | `(B, C, 8, 8)` | encoded board |
| `square_x` | `(B, 64, C)` | flattened square features |
| `z` | `(B, 64, d)` | node state |
| `src`, `dst` | `(E,)` | edge endpoints |
| `z_src`, `z_dst` | `(B, E, d)` | gathered endpoint states |
| `g` | `(B, E, 1)` | differentiable edge gates |
| `delta` | `(B, E, r)` | sheaf residuals |
| `edge_energy` | `(B, E)` | per-edge tension |
| `node_energy` | `(B, 64, 1)` | incident tension |
| `readout` | `(B, R)` | pooled feature vector |
| `logits` | `(B, num_classes)` | model output |

### Default config fields

```yaml
model:
  name: attack_sheaf_tension_net
  input_channels: null
  num_classes: 2
  encoding: simple_18
  node_dim: 32
  stalk_dim: 16
  sheaf_layers: 3
  edge_type_embedding_dim: 8
  direction_embedding_dim: 8
  distance_embedding_dim: 4
  gate_hidden_dim: 64
  readout_hidden_dim: 64
  dropout: 0.10
  energy_tau: 0.25
  initial_step_size: 0.15
  max_step_size: 0.50
  use_degree_norm: true
  tie_color180: false
  occupancy_mode: deterministic_if_available_else_learned_soft
  include_edge_families:
    king: true
    knight: true
    pawn_white: true
    pawn_black: true
    rook_ray: true
    bishop_ray: true
```

### Encoding support

- `simple_18`: primary minimal experiment. Use deterministic piece occupancy if the encoder's piece planes are known.
- `lc0_static_112`: supported by the same public tensor interface. Use deterministic occupancy if Codex can map LC0 piece planes safely; otherwise use learned soft occupancy.
- `lc0_bt4_112`: supported by the same public tensor interface. Treat history planes carefully: occupancy should default to current-position planes only if the encoder metadata exposes them; otherwise use learned soft occupancy and log that fallback.

Implementation requirement: never infer labels or source class from encoding-specific channels. If an encoding contains non-board metadata whose meaning is unclear, the adapter must ignore it unless existing repository documentation identifies it as legal board-state context.

### Parameter estimate

For `C=112`, `d=32`, `r=16`, `L=3`, roughly:

- square encoder: `112*32 + 32 ≈ 3.6k`
- typed restrictions: approximately `3 layers * 28 keys * 2 maps * 16*32 ≈ 86k`
- gate MLP shared or per-layer: `~7k` if shared, `~21k` if per-layer
- squarewise MLPs: `3 * (32*64 + 64*32) ≈ 12k`
- readout MLP: `~9k`
- embeddings and norms: `<5k`

Expected total: approximately `120k-160k` parameters depending on exact number of typed direction keys and whether gates are shared by layer. This is intentionally not a bigger CNN.

### FLOP and complexity estimate

Let `E` be the number of candidate directed edges; with all king, knight, pawn, rook-ray, and bishop-ray candidate pairs on an `8x8` board, `E` should be on the order of `2k`, depending on exact inclusion conventions.

Per sample per sheaf layer:

- endpoint gather: `O(E*d)`
- restrictions and adjoints: `O(E*r*d)`
- gate MLP: `O(E*gate_hidden_dim)`
- scatter-add: `O(E*d)`

With `E≈2000`, `d=32`, `r=16`, `L=3`, the sheaf part is roughly `5-8M` multiply-add scale operations per position before batching. It should be comparable to a small chess CNN, not to a deep LC0 clone.

### Logits interface

The module must expose exactly:

```text
logits = model(x)
```

where `x` is `(batch, C, 8, 8)` and `logits` is `(batch, num_classes)`. No extra inputs are allowed at inference time.

## 8. Loss, Training, And Regularization

Primary loss:

- Binary cross entropy via `torch.nn.CrossEntropyLoss` on logits `(B, 2)` and binary labels `(B,)`.
- If existing training code expects `num_classes=2`, keep that unchanged.

Class weighting:

- Compute class weights from the training split's binary labels only.
- Do not compute weights from validation/test.
- Do not use fine-label-specific weights for the primary run, because the model's target is binary and fine labels are for reporting.

Optional auxiliary losses, disabled by default for the minimal falsification run:

1. `gate_entropy_regularizer`
   - Purpose: discourage all gates from saturating open or closed.
   - Formula: small penalty toward moderate entropy, e.g. coefficient `1e-4`.
   - Uses no labels.
2. `restriction_norm_penalty`
   - Purpose: keep `A[k]` and `B[k]` bounded.
   - Coefficient: `1e-5`.
3. `energy_readout_dropout`
   - Purpose: prevent the classifier from using only a single max-energy scalar.
   - Implement as ordinary dropout in the readout MLP, not as a separate loss.

Do not add supervised auxiliary targets such as "tactic motif," "engine best move," "centipawn swing," "PV length," or "verification confidence."

Default training config:

```yaml
training:
  batch_size: 256
  epochs: 30
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.0001
  scheduler: cosine_if_existing_baselines_use_it_else_none
  gradient_clip_norm: 1.0
  label_smoothing: 0.0
  seed: 1729
  deterministic: true
  mixed_precision: match_existing_training_default
```

Regularizers:

- dropout `0.10` in squarewise MLP and readout MLP
- weight decay `1e-4`
- clamped positive sheaf step size:
  - parameterize `step = max_step_size * sigmoid(raw_step)`
  - default `max_step_size=0.50`
- degree normalization with epsilon for stable empty-degree cases

What must stay fixed for fair comparison:

- split files
- encoder choice
- target binarization
- epoch budget
- batch size if memory permits; otherwise record effective batch size
- optimizer family and LR schedule unless the repository has baseline-specific defaults that must be matched
- data augmentation policy
- early-stopping policy
- metric computation
- random seeds for all compared current-data runs

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `identity_sheaf_same_edges` | Replace learned `A[k]` and `B[k]` with shared identity/projection maps, keeping the same candidate attack edges and gates. | Nontrivial relation-specific sheaf restrictions add signal beyond ordinary weighted message passing. | If this matches the full model, the central sheaf claim is falsified; future work should not repeat learned-restriction attack sheaves. |
| `no_energy_readout` | Remove edge/node tension-energy features from the readout; pool only final node states. | Explicit tension energy, not just diffusion, helps classify puzzle-likeness. | If performance is unchanged, energy readout is unnecessary and the model may just be a graph mixer. |
| `edge_shuffle_control` | Keep the same number of edges and type counts but randomly permute endpoints with a fixed seed, preserving no chess geometry. | Chess attack geometry matters rather than parameter count or scatter operations. | If shuffled edges perform similarly, the architecture is exploiting generic mixing or dataset artifacts. |
| `ray_edges_only` | Keep rook/bishop sliding ray edges and remove king/knight/pawn jumps. | Long-range line tension is the primary source of useful signal. | Poor performance here but good full-model performance means short-range and pawn/knight relations matter. |
| `jump_edges_only` | Keep king/knight/pawn edges and remove rook/bishop rays. | Non-ray tactical pressure alone is sufficient. | If this matches full model, expensive sliding-ray machinery may be unnecessary. |
| `hard_clearance_gate` | For sliding rays, replace learned path gate with deterministic near-binary path-clear feature. | Learned gates over blocked/latent rays help capture pins, batteries, and blockers. | If hard clearance wins, learned gates are too noisy and should be simplified. |
| `learned_soft_occ_only` | Force occupancy fallback even when deterministic piece planes are known. | Deterministic occupancy extraction is important for rule-aligned geometry. | If learned soft occupancy matches deterministic occupancy, encoding metadata dependency can be relaxed. |
| `color180_tied` | Tie opposite-direction restriction parameters under color-swap plus 180-degree rotation, where channel permutation is safe. | Partial chess symmetry reduces sample complexity without imposing invalid D4 symmetry. | If tying hurts, dataset conventions or encoding details break the assumed partial symmetry. |
| `remove_path_features` | Gates see endpoints and type embeddings but not between-square occupancy summaries. | Blockers and ray clearance are essential to tactical tension. | If unchanged, path features are not being used; inspect gate saturation and edge-family energies. |

Smallest ablation that can falsify the central claim: `identity_sheaf_same_edges`. It preserves edge geometry, gates, parameter scale where practical, and training budget while removing the nontrivial sheaf restrictions. If it matches the full model within noise, the proposed sheaf object is not earning its complexity.

## 10. Benchmark And Falsification Criteria

Baselines:

- Existing best `simple_18` simple CNN.
- Existing best `simple_18` residual CNN.
- Existing best small/medium/deep CNN variant under the same encoder.
- Existing best LC0 BT4-style CNN/residual result if comparing on `lc0_static_112` or `lc0_bt4_112`.
- Required internal ablation: `identity_sheaf_same_edges`.
- Required control: `edge_shuffle_control`.

Primary metrics:

- validation and test accuracy
- balanced accuracy
- macro F1
- AUROC if probability outputs are already supported
- AUPRC if probability outputs are already supported
- false-positive rate on fine label `0`
- positive recall on fine labels `1` and `2` separately
- confusion table: `true fine label 0/1/2 -> predicted binary output 0/1`

Secondary artifacts:

- parameter count
- training time per epoch
- best epoch by validation balanced accuracy
- calibration metrics if already available: Brier score and ECE
- per-edge-family mean tension on validation examples, aggregated by true fine label
- gate saturation statistics by edge family

Success threshold:

- On `simple_18`, the full model should improve test balanced accuracy by at least `+2.0` absolute percentage points or AUROC by at least `+0.015` over the best current non-ensemble `simple_18` baseline, while not increasing fine-label-0 false-positive rate by more than `+1.0` absolute percentage point.
- It must also beat `identity_sheaf_same_edges` by at least `+1.0` absolute balanced-accuracy point or `+0.010` AUROC.
- At least one of fine-label `1` or fine-label `2` positive recall should improve without collapsing the other by more than `2.0` absolute percentage points.

Failure threshold:

- The full model is within `±0.5` balanced-accuracy points of `identity_sheaf_same_edges` and `no_energy_readout` across at least two seeds.
- The full model loses to the best current baseline by more than `1.0` balanced-accuracy point with no compensating gain in fine-label `1` or `2` recall.
- `edge_shuffle_control` is statistically indistinguishable from the full model, indicating geometry is not contributing.

Abandon condition:

- Abandon this idea family if the full model does not beat `identity_sheaf_same_edges` and `edge_shuffle_control` under the same encoder and budget. Do not repeat the next cycle as "bigger attack sheaf," "deeper attack sheaf," or "more edge types" unless a diagnostic artifact shows a concrete bug in the implementation.

Scaling condition:

- Only try `lc0_static_112` or `lc0_bt4_112` after the `simple_18` minimal experiment beats both the best comparable baseline and the central ablation.
- Only increase `node_dim`, `stalk_dim`, or `sheaf_layers` after the structural signal is established; such scaling is not the core idea.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_attack_sheaf_tension/README.md` | Create | Human-readable summary of the idea, leakage constraints, expected tests, and benchmark command snippets. |
| `ideas/20260421_attack_sheaf_tension/research_packet.md` | Create | Copy this handoff packet or a link/reference to the downloaded Markdown artifact. |
| `ideas/20260421_attack_sheaf_tension/ablation_plan.yaml` | Create | Config entries for full model, `identity_sheaf_same_edges`, `no_energy_readout`, and `edge_shuffle_control`. |
| `src/chess_nn_playground/models/attack_sheaf_tension.py` | Create | `AttackSheafTensionNet`, `TacticalGeometryCache`, `OccupancyAdapter`, `PathGate`, `SheafDiffusionLayer`, and readout modules. |
| `src/chess_nn_playground/models/registry.py` | Edit | Register `attack_sheaf_tension_net` without disturbing existing model names. |
| `configs/attack_sheaf_tension_simple18.yaml` | Create | Minimal `simple_18` training config with fixed budget and binary target. |
| `configs/attack_sheaf_tension_lc0_static112.yaml` | Create only after simple_18 signal | Same model with `lc0_static_112` channel count and safe occupancy mode. |
| `configs/attack_sheaf_tension_lc0_bt4_112.yaml` | Create only after simple_18 signal | Same model with `lc0_bt4_112` channel count and safe current-position occupancy handling. |
| `tests/test_attack_sheaf_geometry.py` | Create | Unit tests for edge counts, endpoint ranges, between-square masks, no self-edges unless explicitly intended, and deterministic cache creation. |
| `tests/test_attack_sheaf_forward.py` | Create | Forward-shape test for `(B,C,8,8)->(B,2)`, CPU determinism smoke test, finite logits, and gradient flow through gates/restrictions. |
| `tests/test_attack_sheaf_no_leakage.py` | Create | Assert model forward signature accepts only board tensor; configs do not include engine or verification fields. |
| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Edit after consuming | Preserve hard constraints and add lessons from this pass: attack-sheaf duplicate rules, required central ablation, and clearer anti-leakage language for deterministic chess-rule features. |

Codex must update `ideas/chatgpt_pro_deep_math_research_prompt.md` after consuming this output. The update should preserve all hard constraints while adding reusable lessons, anti-duplicate rules, clearer output requirements, or failure-mode guidance discovered during implementation and benchmarking.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0427_tuesday_local_attack_sheaf_tension.md
  generated_at: "2026-04-21T04:27:51-07:00"
  weekday: tuesday
  timezone_token: local
  timezone_name: America/Los_Angeles
  idea_slug: attack_sheaf_tension
  title: "Codex Handoff Packet: Directed Attack-Sheaf Tension Network"
  file_type: text/markdown
  intended_next_consumer: Codex
```

```yaml
idea_yaml:
  idea_id: "20260421_attack_sheaf_tension"
  idea_name: "Directed Attack-Sheaf Tension Network"
  model_name: "attack_sheaf_tension_net"
  core_claim: "Puzzle-likeness is partially captured by localized tension energy in a nontrivial sheaf Laplacian over directed chess attack relations."
  idea_fingerprint:
    - occupancy_gated_directed_chess_attack_graph
    - typed_relation_restriction_maps
    - sheaf_laplacian_diffusion
    - explicit_tension_energy_readout
    - no_engine_features
    - no_full_dihedral_invariance
  minimum_encoder: simple_18
  supported_encodings:
    - simple_18
    - lc0_static_112
    - lc0_bt4_112
  forbidden_inputs:
    - stockfish_scores
    - principal_variations
    - node_counts
    - verification_metadata
    - source_labels_as_features
    - proposed_labels
    - unresolved_candidate_labels
  required_ablation_to_falsify: identity_sheaf_same_edges
  required_control: edge_shuffle_control
  primary_success_metric: balanced_accuracy
  secondary_success_metrics:
    - auroc
    - macro_f1
    - fine_label_1_recall
    - fine_label_2_recall
    - fine_label_0_false_positive_rate
```

```yaml
config_yaml:
  model:
    name: attack_sheaf_tension_net
    input_channels: null
    num_classes: 2
    encoding: simple_18
    node_dim: 32
    stalk_dim: 16
    sheaf_layers: 3
    edge_type_embedding_dim: 8
    direction_embedding_dim: 8
    distance_embedding_dim: 4
    gate_hidden_dim: 64
    readout_hidden_dim: 64
    dropout: 0.10
    energy_tau: 0.25
    initial_step_size: 0.15
    max_step_size: 0.50
    use_degree_norm: true
    tie_color180: false
    occupancy_mode: deterministic_if_available_else_learned_soft
    include_edge_families:
      king: true
      knight: true
      pawn_white: true
      pawn_black: true
      rook_ray: true
      bishop_ray: true
  data:
    train_split: data/splits/crtk_sample_3class/split_train.parquet
    val_split: data/splits/crtk_sample_3class/split_val.parquet
    test_split: data/splits/crtk_sample_3class/split_test.parquet
    binary_target:
      fine_0: 0
      fine_1: 1
      fine_2: 1
    fine_labels_for_reporting_only: true
  training:
    batch_size: 256
    epochs: 30
    optimizer: adamw
    learning_rate: 0.0003
    weight_decay: 0.0001
    scheduler: cosine_if_existing_baselines_use_it_else_none
    gradient_clip_norm: 1.0
    label_smoothing: 0.0
    seed: 1729
    deterministic: true
    mixed_precision: match_existing_training_default
    class_weighting: train_binary_inverse_frequency
  regularization:
    gate_entropy_coefficient: 0.0
    restriction_norm_coefficient: 0.00001
    readout_dropout: 0.10
  ablations:
    - identity_sheaf_same_edges
    - no_energy_readout
    - edge_shuffle_control
    - ray_edges_only
    - jump_edges_only
    - hard_clearance_gate
    - learned_soft_occ_only
    - color180_tied
    - remove_path_features
```

```yaml
model_spec:
  public_interface:
    input_shape: [batch, C, 8, 8]
    output_shape: [batch, num_classes]
    forward_args:
      - x
    forbidden_forward_args:
      - stockfish_score
      - pv
      - node_count
      - verification_metadata
      - source_label
      - proposed_label
  modules:
    SquareEncoder:
      input: [batch, 64, C]
      output: [batch, 64, node_dim]
      operations:
        - linear_or_1x1_conv
        - layer_norm
        - gelu
    TacticalGeometryCache:
      buffers:
        src: [E]
        dst: [E]
        edge_family: [E]
        direction_id: [E]
        distance_id: [E]
        between_mask: [E, 64]
      trainable: false
    OccupancyAdapter:
      input: [batch, C, 8, 8]
      output:
        occ: [batch, 64, 1]
        optional_white_occ: [batch, 64, 1]
        optional_black_occ: [batch, 64, 1]
    PathGate:
      input:
        z_src: [batch, E, node_dim]
        z_dst: [batch, E, node_dim]
        path_features: [batch, E, path_feature_dim]
        geometry_embeddings: [E, geometry_embedding_dim]
      output:
        gates: [batch, E, 1]
    SheafDiffusionLayer:
      restrictions:
        A: [num_relation_keys, stalk_dim, node_dim]
        B: [num_relation_keys, stalk_dim, node_dim]
      output:
        z_next: [batch, 64, node_dim]
        edge_energy: [batch, E]
        node_energy: [batch, 64, 1]
    TensionReadout:
      input:
        z_final: [batch, 64, node_dim]
        edge_energies: [layers, batch, E]
        node_energies: [layers, batch, 64, 1]
      output:
        readout: [batch, readout_dim]
    Classifier:
      output: [batch, num_classes]
  complexity:
    candidate_edges_order: "~2000"
    parameter_estimate: "120k-160k for C=112,node_dim=32,stalk_dim=16,layers=3"
    flops_per_position_estimate: "5M-8M multiply-add scale operations for sheaf layers"
```

```yaml
research_continuity:
  idea_fingerprint: "occupancy-gated directed chess attack graph with nontrivial sheaf restriction maps and tension-energy readout"
  closest_duplicate_risk: "Plain GNN-on-squares, Neural Sheaf Diffusion without chess-specific attack geometry, or an LC0-style policy/value tower with renamed edge features."
  do_not_repeat_if_this_fails:
    - "Do not propose a deeper or wider attack-sheaf network as the next core idea."
    - "Do not repeat typed attack-graph message passing unless a diagnostic shows the implementation was wrong."
    - "Do not replace the central ablation with generic CNN comparisons; identity_sheaf_same_edges is mandatory."
    - "Do not add Stockfish-derived tactical labels, centipawn swings, or PV supervision to rescue this idea."
  suggested_next_search_directions:
    - "Non-engine differentiable shallow move-search surrogate using legal move generation but no engine scores."
    - "Causal invariance across encodings/source subsets to detect puzzle-source artifacts."
    - "Optimal-transport comparison between attacker and defender mass distributions."
    - "Information bottleneck model that predicts puzzle-likeness from compressed material/king-safety sufficient statistics."
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add an anti-duplicate rule: "If attack-sheaf tension fails its identity-sheaf ablation, do not propose another attack-sheaf, bigger attack graph, or deeper sheaf diffusion as the next idea." | Prevents cycling on the same structural hypothesis after falsification. | `Hard Constraints` or a new `Do Not Repeat Failed Idea Families` subsection. |
| Add a required central-ablation field for every future idea: "Name the smallest ablation that removes the mathematically novel object while preserving most compute." | Forces each research pass to define a real falsification test, not just baseline comparison. | `Required Markdown File Content`, sections `9` and `10`. |
| Clarify that deterministic chess-rule geometry derived from the board tensor is allowed, but engine evaluation/search outputs are not. | Avoids over-refusing legal attack maps while preserving leakage rules. | `Hard Constraints` leakage bullets. |
| Add: "Do not impose full board rotation/reflection invariance unless the idea explicitly handles pawns, castling, side-to-move, and channel permutations." | Prevents invalid symmetry proposals for chess. | `Research Goal` or `Mathematical Thesis` guidance. |
| Add a prompt-memory line recording this fingerprint: `occupancy_gated_directed_chess_attack_graph + nontrivial_sheaf_restrictions + tension_energy_readout`. | Helps future cycles reject near-duplicates. | `Common Approaches Rejected` or a new `Prior Idea Fingerprints` subsection. |
| Require per-fine-label reporting impact in success and failure thresholds, not just binary aggregate metrics. | Protects against a model improving easy positives while hurting near-puzzles or flooding negatives. | `Benchmark And Falsification Criteria`. |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0427_tuesday_local_attack_sheaf_tension.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes, `identity_sheaf_same_edges` plus `edge_shuffle_control`
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
