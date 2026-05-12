# Codex Handoff Packet: File-Mirror Tension Sheaf

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0437_tuesday_los_angeles_mirror_tension_sheaf.md`
- Generated at: 2026-04-21 04:37 PDT, UTC-07:00
- Weekday: Tuesday
- Timezone: los_angeles
- Idea slug: `mirror_tension_sheaf`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: File-Mirror Tension Sheaf
- One-sentence thesis: Puzzle-like chess positions can be detected from board-only inputs by learning a small signed directed sheaf diffusion over pseudo-legal attack, defense, and x-ray relations, then measuring how much the resulting local tactical tension changes under a learnable file-mirror partial-equivariance gate.
- Idea fingerprint: `board_only:pseudo_attack_defense_xray_graph + signed_directed_sheaf_laplacian_energy + file_mirror_partial_equivariance_gate + energy_stats_classifier`
- Why this is not a common CNN/ResNet/Transformer variant: The primary computation is not square-local convolution, residual image smoothing, or all-square attention. It constructs a chess-rule incidence object from the input board, applies typed sheaf coboundary and divergence operators on attack-defense edges, and classifies from sheaf energy statistics plus a learned partial symmetry discrepancy.
- Current-data minimal experiment: Train `MirrorTensionSheafNet` on `simple_18` with the existing `crtk_sample_3class` train/val/test split, binary target `fine_label == 0 -> 0`, `fine_label in {1,2} -> 1`, three fixed seeds, and compare against the strongest already-logged simple/residual CNN available for the same encoding. Report binary metrics plus the required fine-label confusion table.
- Expected information gain if it fails: A clean failure will tell us that explicit attack-defense cochain tension and approximate file-mirror discrepancy do not add measurable signal beyond existing board encoders, so the next cycle should leave sheaf/attack-graph operators and search instead for sequence-free differentiable move-choice or information-bottleneck objectives.

## 3. Problem Restatement And Data Contract

Task: classify a single chess board position as binary non-puzzle versus puzzle-like.

Labels and outputs:

- Network output logits shape: `(batch, num_classes)`.
- For the current benchmark, use `num_classes = 2`.
- Binary class `0`: non-puzzle.
- Binary class `1`: puzzle-like.
- Existing fine labels are evaluation strata and supervised-label source only:
  - fine label `0`: known non-puzzle.
  - fine label `1`: verified near-puzzle.
  - fine label `2`: verified puzzle.
- Binary target mapping for this experiment: `0 -> 0`, `1 -> 1`, `2 -> 1`.
- Do not invent new class `1` or class `2` examples. Do not promote unresolved candidates to positives. Unresolved candidates must remain unresolved or excluded according to the existing dataset contract.

Allowed inputs:

- A tensor `x` of shape `(batch, C, 8, 8)` from one of the available encodings:
  - `simple_18`
  - `lc0_static_112`
  - `lc0_bt4_112`
- Deterministic chess-rule features computed only from the current board tensor, such as current piece occupancy, side-to-move plane when present, pseudo-legal attack rays, defense rays, and geometric file mirror of the current input tensor with the correct plane permutation.

Forbidden neural-network input features:

- Stockfish scores.
- Principal variations.
- Engine node counts.
- Engine depth, search instability, or mate distance.
- Verification metadata.
- Source labels, proposed labels, or unresolved-candidate status.
- Any feature derived from future moves, puzzle solution lines, game result, source database identity, or benchmark split identity.

Tensor and split contract:

- Model target: PyTorch `nn.Module`.
- Input: `(batch, C, 8, 8)`.
- Output: `(batch, num_classes)` logits.
- Train split: `data/splits/crtk_sample_3class/split_train.parquet`.
- Validation split: `data/splits/crtk_sample_3class/split_val.parquet`.
- Test split: `data/splits/crtk_sample_3class/split_test.parquet`.

Leakage checklist for Codex:

- [ ] The graph builder uses only `x` and deterministic chess geometry.
- [ ] No engine columns are read by the model or data transform.
- [ ] Fine labels are used only to derive supervised train targets and evaluation reports.
- [ ] Split identity is not passed to the model.
- [ ] Candidate-resolution metadata is not passed to the model.
- [ ] Mirroring is an input-space transformation only; it does not inspect labels or engine information.
- [ ] Any side-to-move, castling, or repetition-like planes are used only if already present in the selected encoding.

## 4. Research Map

This idea borrows mathematical operators and inductive biases, not implementation code, from the following sources.

1. Hansen and Gebhart, “Sheaf Neural Networks,” arXiv:2012.06333, https://arxiv.org/abs/2012.06333. Borrowed: the idea that a graph convolution can be generalized by a sheaf Laplacian with edge-dependent restriction maps. Not copied: their datasets, code, architecture depth, or task.
2. Bodnar et al., “Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs,” arXiv:2202.04579, https://arxiv.org/abs/2202.04579. Borrowed: diffusion through learned nontrivial sheaf geometry and the proof intuition that nontrivial sheaves can represent heterophilic relations better than scalar graph smoothing. Not copied: their training setup or graph benchmarks.
3. Zhang et al., “MagNet: A Neural Network for Directed Graphs,” arXiv:2102.11391, https://arxiv.org/abs/2102.11391. Borrowed: the lesson that directionality can be represented spectrally rather than discarded by symmetrization. Not copied: complex Hermitian magnetic convolution; this packet uses real typed sheaf maps.
4. He et al., “A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian,” arXiv:2209.00546, https://arxiv.org/abs/2209.00546. Borrowed: signed directionality as a first-class signal. Not copied: the magnetic signed Laplacian formula or node-classification objective.
5. Romero and Lohit, “Learning Partial Equivariances from Data,” NeurIPS 2022 / arXiv:2110.10211, https://arxiv.org/abs/2110.10211 and https://www.merl.com/publications/TR2022-148. Borrowed: do not hard-code a full symmetry when the domain only approximately respects it; learn how much equivariance should matter. Not copied: group convolution layers or their parameterization.
6. Cohen and Welling, “Group Equivariant Convolutional Networks,” ICML 2016, https://proceedings.mlr.press/v48/cohenc16.html. Borrowed: the formal language of equivariance as controlled weight sharing. Not copied: G-CNN layers; full board rotations/reflections are explicitly not imposed.
7. Carroll, “Finite Group Equivariant Neural Networks for Games,” arXiv:2009.05027, https://arxiv.org/abs/2009.05027. Borrowed: board games can benefit from symmetry-aware architectures, but game symmetries must be chosen with care. Not copied: finite group network construction.
8. Duta et al., “Sheaf Hypergraph Networks,” NeurIPS 2023, https://proceedings.neurips.cc/paper_files/paper/2023/hash/27f243af2887d7f248f518d9b967a882-Abstract-Conference.html. Borrowed: higher-order relations can be enriched by sheaf structure. Not copied: hypergraph network implementation; this packet deliberately stays with a compact directed edge sheaf for the minimal experiment.
9. Lichess puzzle themes page, https://lichess.org/training/themes. Borrowed: practical reminder that common puzzle motifs include forks, pins, skewers, discovered attacks, hanging pieces, exposed kings, deflection, and interference. Not copied: puzzle labels, puzzle IDs, puzzle-generation rules, or any Lichess data.
10. FIDE Laws of Chess, https://handbook.fide.com/chapter/e012023. Borrowed: chess has directional pawn movement, castling, and side-to-move structure, so full rotation/reflection invariance is mathematically unsafe. Not copied: no legal-adjudication machinery or move search.

No citation above is required for runtime. The model must remain board-only and must not ingest external annotations.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple square CNN | simple CNN | Already covered. Local image filters are weakly aligned with long-range pins, skewers, and discovered attacks unless depth grows. |
| Standard residual CNN | residual CNN | Residual smoothing improves optimization but does not introduce a new tactical object; it remains an ordinary board-image model. |
| Small/medium/deep CNN scaling | small/medium/deep CNN variants | Depth/width scaling is expressly disallowed as the core idea and provides low research novelty. |
| LC0-style CNN or residual CNN | LC0 BT4-style CNN and residual variants | Already represented by baselines and risks becoming an LC0 clone rather than a task-specific puzzle-likeness operator. |
| Ordinary ViT over 64 square tokens | no exact baseline, closest is deep CNN with global receptive field | Vanilla all-square attention is a standard Transformer idea, parameter-hungry for current data, and not explicitly tied to chess attack-defense structure. |
| Plain GNN-on-squares with adjacency to neighboring squares | no exact baseline, closest is image CNN | Eight-neighbor square adjacency is just a graph rewriting of local convolution and misses rook, bishop, queen, knight, and pawn attack geometry. |
| Plain GNN-on-pseudo-legal moves without sheaf restrictions | no exact baseline | It uses chess geometry but collapses attacks, defenses, x-rays, and pins into generic messages; the central hypothesis needs signed local incompatibility, not generic aggregation. |
| Hyperparameter tuning of learning rate, dropout, batch size, or optimizer | all baselines | Useful for polish but not a research idea; must not be the core proposal. |
| Ensembling several existing models | possible ensemble of CNN/residual/LC0 variants | Disallowed, computationally heavier, and gives little mechanistic information if it works. |
| More data or relabeling unresolved candidates | data pipeline rather than model baseline | Disallowed as the core idea and risks label fabrication or leakage. |
| Full dihedral equivariance over board rotations/reflections | group-equivariant CNN variants | Chess is not fully invariant under these transforms: pawns are directional, side-to-move matters, castling rights are asymmetric in encodings, and king safety is color-relative. |
| Engine-distillation target or search-instability feature | not a valid baseline under constraints | Directly violates the no-Stockfish/PV/node-count/verification-metadata input rule. |

## 6. Mathematical Thesis

### Input space

For a selected encoding with `C` channels, each sample is an input tensor

\[
X \in \mathcal{X}_C = \mathbb{R}^{C \times 8 \times 8}.
\]

A deterministic adapter extracts a current-board canonical occupancy view

\[
B(X) = (O, P, S, R),
\]

where `O` is occupancy on 64 squares, `P` is piece type and color when recoverable from the encoding, `S` is side to move when represented by the encoding, and `R` is optional castling/auxiliary board-state information only when already represented in `X`. The adapter may also pass raw projected channels forward, so the architecture does not depend exclusively on perfect canonical extraction.

### Target definition

The supervised target is

\[
y = \mathbf{1}\{\text{fine label} \in \{1,2\}\}.
\]

Fine label is not an input. It is only a supervised training/evaluation label. The model estimates logits for `P(y=0|X)` and `P(y=1|X)`.

### Distribution assumptions

Let `D_train`, `D_val`, and `D_test` be the fixed CRTK sample split distributions induced by the existing parquet files. The idea assumes:

1. Puzzle-like positions overrepresent localized tactical stress: overloaded defenders, pieces with simultaneous attacking and defensive roles, x-ray pressure, and king-adjacent attack imbalance.
2. Non-puzzles may still have attacks and material contact, but the attack-defense graph is more locally consistent and less concentrated around a small set of critical edges.
3. Near-puzzles may lie between these regimes, so evaluation by fine-label confusion is informative even though training is binary.

These are hypotheses, not proven facts.

### Symmetry and equivariance assumptions

Full chessboard rotation/reflection invariance is rejected. In particular:

- A 180-degree rotation changes pawn direction unless colors and side-to-move are also transformed.
- Horizontal rank reflection breaks pawn movement.
- File mirror `a <-> h`, `b <-> g`, `c <-> f`, `d <-> e` is closer to a chess-law symmetry than other geometric flips, but it still can interact with castling-right encodings, opening priors, and kingside/queenside tactical frequencies.

Therefore the architecture uses partial file-mirror equivariance:

- Compute sheaf-energy features for `X`.
- Compute the same features for `M(X)`, where `M` is a deterministic file mirror with correct plane permutation where needed.
- Learn a scalar or vector gate `rho(X) in [0,1]` deciding how much the mirror discrepancy should influence classification.
- Do not force `f(X) = f(M(X))`.

### Formal operator/object

Define a directed typed multigraph

\[
G_X = (V, E_X, \tau, \sigma)
\]

where `V` is the 64-square set and `E_X` contains board-derived directed relations:

- own piece attacks enemy piece or enemy king zone;
- enemy piece attacks own piece or own king zone;
- own piece defends own piece;
- enemy piece defends enemy piece;
- sliding-piece x-ray pressure through one blocker;
- pawn-control edges with color-relative orientation;
- optional king-zone pressure edges for the eight squares around each king.

Each edge has type `tau(e)` and sign/direction code `sigma(e) in {+1,-1}`. These are deterministic from `X` and chess geometry, not from search.

A sheaf over `G_X` assigns each square a stalk `F(v) = R^d` and each directed edge a stalk `F(e) = R^d`, with learnable type-conditioned restriction maps

\[
R_{e \leftarrow s(e)} = A_{\tau(e)}^{src}, \quad
R_{e \leftarrow t(e)} = A_{\tau(e)}^{dst}.
\]

For node features `h_v in R^d`, the edge coboundary is

\[
(\delta_F h)_e = A_{\tau(e)}^{dst} h_{t(e)} - \sigma(e) A_{\tau(e)}^{src} h_{s(e)}.
\]

The sheaf energy is

\[
\mathcal{E}_F(h; X) = \sum_{e \in E_X} w_e \left\|(\delta_F h)_e\right\|_2^2,
\]

where `w_e` is a bounded positive edge weight from the relation type and optional learned edge gate.

A diffusion layer applies a stable residual step

\[
h^{k+1} = \operatorname{LN}\left(h^k - \eta_k \delta_F^\top W \delta_F h^k + \phi_k(h^k)\right),
\]

with `0 < eta_k <= eta_max`, LayerNorm, and a small node MLP `phi_k`.

The mirror-tension feature is

\[
T_M(X) = \rho(X) \cdot \left|s_F(X) - s_F(MX)\right|,
\]

where `s_F` is a vector of energy statistics after shared sheaf diffusion.

### Core hypothesis

Puzzle-like positions are not merely positions with many attacks. They are positions where directed tactical constraints are locally inconsistent in a small number of strategically important attack-defense fibers. A sheaf coboundary is an appropriate object for this because it measures incompatibility across typed relations rather than averaging neighbors.

### Proposition/objective

Proposition 1, proven architectural property: If the file mirror operator `M` is implemented as a permutation of squares and channel planes, if edge construction is equivariant under `M`, and if sheaf restriction maps are shared by mirrored edge types, then the sheaf energy vector satisfies

\[
s_F(MX) = \Pi_M s_F(X)
\]

for a fixed permutation `Pi_M` of mirrored statistic bins. If statistic bins are mirror-pooled, the pooled energy is invariant.

Proof sketch: `M` induces a graph isomorphism from `G_X` to `G_{MX}`. Shared restriction maps mean the coboundary matrix transforms by conjugation with the node and edge permutation matrices. Norm-squared edge energies are preserved under permutation. Aggregation by mirrored bins either permutes or pools those values. Therefore the stated equivariance/invariance follows.

Proposition 2, proven implementation property: The model does not require engine information. All graph edges are computed from board occupancy, piece movement rules, and current encoding planes. The classifier receives no Stockfish scores, PVs, node counts, verification metadata, source labels, proposed labels, unresolved status, or split ID.

Hypothesis 1, not proven: The learned sheaf energy statistics improve test ROC-AUC or PR-AUC over current CNN/residual baselines on the same encoding.

Hypothesis 2, not proven: The mirror-tension gate is helpful specifically for classifying near-puzzle fine label `1`, because near-puzzles may contain tactical-looking structure whose importance depends on kingside/queenside context rather than a fully invariant pattern.

### What is proven

- The sheaf coboundary/energy is a well-defined differentiable function of node features for a fixed graph.
- With deterministic board-only graph construction, no forbidden engine feature is mathematically required.
- Under exact file-mirror graph isomorphism and shared maps, the energy statistics are equivariant or invariant depending on pooling.

### What is hypothesized

- That puzzle-likeness correlates with localized sheaf tension.
- That learned restriction maps discover useful typed incompatibilities such as pinned defenders, forkable pieces, and x-ray pressure.
- That partial file-mirror discrepancy improves generalization without over-constraining chess asymmetries.

### Counterexamples and failure modes

- Quiet endgame studies may be puzzle-like with low immediate attack-defense tension.
- Positions with many random hanging pieces may have high tension but not be curated puzzles.
- Some puzzles require multi-move zugzwang, opposition, fortress, or underpromotion ideas poorly represented by one-ply attack geometry.
- If the encodings do not expose current piece planes reliably, deterministic graph extraction may become brittle.
- If the dataset’s positive labels are dominated by engine-search phenomena rather than human tactical motifs, board-only sheaf energy may underperform.

## 7. Architecture Specification

### Model name

`MirrorTensionSheafNet`

### Recommended files

- `src/chess_nn_playground/models/mirror_tension_sheaf.py`
- `src/chess_nn_playground/models/components/chess_attack_graph.py`
- `src/chess_nn_playground/models/components/file_mirror.py`

### Modules

1. `EncodingBoardExtractor`
   - Input: `x: FloatTensor[B, C, 8, 8]`, `encoding_name`.
   - Output:
     - `canonical: FloatTensor[B, K, 8, 8]`, where `K` should include current piece/color planes when available.
     - `state: Dict` with optional side-to-move and castling planes if present.
   - Must be deterministic and board-only.
   - For `simple_18`, use the known current-board planes.
   - For `lc0_static_112` and `lc0_bt4_112`, map current piece planes and side-to-move/castling planes according to the repo’s encoding metadata. If Codex cannot verify a plane, fail fast rather than guessing.

2. `AttackDefenseGraphBuilder`
   - Input: canonical occupancy and optional state.
   - Output packed edge tensors:
     - `src: LongTensor[B, E_max]`
     - `dst: LongTensor[B, E_max]`
     - `edge_type: LongTensor[B, E_max]`
     - `edge_sign: FloatTensor[B, E_max]`
     - `edge_mask: BoolTensor[B, E_max]`
   - Suggested edge types:
     - `own_attack_enemy`
     - `enemy_attack_own`
     - `own_defense`
     - `enemy_defense`
     - `own_xray_enemy`
     - `enemy_xray_own`
     - `own_pawn_control`
     - `enemy_pawn_control`
     - `own_king_zone_pressure`
     - `enemy_king_zone_pressure`
     - `own_line_blocker_pressure`
     - `enemy_line_blocker_pressure`
   - Use pseudo-legal attacks and sliding rays only. Do not run engine search. Do not compute best moves or legal-move outcomes.
   - `E_max = 2048` is sufficient for a padded first implementation. If exceeded, keep the highest-priority edges in this priority order: king-zone pressure, direct attacks, x-rays, defenses, pawn controls. Log overflow count as a diagnostic artifact.

3. `RawBoardProjector`
   - `Conv2d(C, d_model, kernel_size=1)` followed by flattening to `[B, 64, d_model]`.
   - Purpose: preserve raw encoding information not represented in the graph extractor.

4. `NodeInitializer`
   - Concatenate raw square projection with small learned square embeddings and optional side-to-move embedding.
   - Linear projection to stalk dimension `d`.
   - Output: `h0: FloatTensor[B, 64, d]`.

5. `TypedSheafDiffusionLayer`
   - Parameters:
     - `A_src: FloatTensor[num_edge_types, d, d]`
     - `A_dst: FloatTensor[num_edge_types, d, d]`
     - `edge_gate_mlp: Linear(edge_type_embedding + src/dst node summary -> 1)` optional; default enabled but bounded by sigmoid.
     - `eta: learned scalar`, constrained by sigmoid to `[0, eta_max]`.
     - `node_mlp: Linear(d,d) -> GELU -> Linear(d,d)`.
   - Forward pseudocode:

```text
for each batch b:
    hs = gather(h[b], src[b])           # [E_max, d]
    ht = gather(h[b], dst[b])           # [E_max, d]
    As = A_src[edge_type[b]]            # [E_max, d, d]
    At = A_dst[edge_type[b]]            # [E_max, d, d]
    r  = matmul(At, ht) - edge_sign * matmul(As, hs)   # sheaf coboundary
    r  = r * edge_mask
    w  = sigmoid(edge_gate(...)) or 1.0
    edge_energy = w * sum(r*r, dim=-1)
    div_src = scatter_add(edge_sign * matmul(transpose(As), r) * w, src)
    div_dst = scatter_add(matmul(transpose(At), r) * w, dst)
    div = div_dst - div_src
h_next = LayerNorm(h - eta * div + node_mlp(h))
return h_next, edge_energy
```

6. `SheafEnergyReadout`
   - Inputs: final node features and per-layer edge energies.
   - Outputs statistic vector `s`:
     - mean energy by edge type: `num_edge_types`.
     - max energy by edge type: `num_edge_types`.
     - top-`k` pooled energies global: `k=8`.
     - king-zone energy own/enemy: `2`.
     - concentration ratio: top-8 energy divided by total energy.
     - node divergence norm mean/max.
     - final node mean/max pooling.

7. `FileMirrorPartialGate`
   - Builds `x_m = file_mirror(x, encoding_name)`.
   - Runs the same `encode_once` path on `x` and `x_m` with shared weights.
   - Computes `delta_s = abs(s - mirror_unpermute(s_m))`.
   - Learns `rho = sigmoid(gate_mlp([s, s_m, delta_s]))`, scalar or vector.
   - Final feature vector: `[s, rho * delta_s, rho, pooled_node_features]`.
   - Important: do not force logits to match under mirror. The gate lets data decide.

8. `ClassifierHead`
   - `LayerNorm -> Linear(F, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, num_classes)`.
   - Output logits `(B, num_classes)`.

### Forward pass summary

```text
def forward(x):
    s, node_pool, diagnostics = encode_once(x)
    xm = file_mirror(x, encoding_name)
    sm, _, _ = encode_once(xm)
    sm_aligned = mirror_unpermute_stats(sm)
    delta = abs(s - sm_aligned)
    rho = sigmoid(gate_mlp(concat(s, sm_aligned, delta)))
    z = concat(s, rho * delta, rho, node_pool)
    logits = classifier(z)
    return logits
```

### Tensor shapes

- Input: `[B, C, 8, 8]`.
- Raw square projection: `[B, 64, d_model]`.
- Node stalk features: `[B, 64, d]`.
- Edge tensors: `[B, E_max]`.
- Coboundary residuals: `[B, E_max, d]`.
- Energy stats `s`: approximately `[B, 2*num_edge_types + k + 6 + 2*d]`.
- Final features: approximately `[B, 2*len(s) + gate_dim + 2*d]`.
- Output logits: `[B, num_classes]`.

### Parameter estimate

Default minimal config:

- `d_model = 48`
- `stalk_dim = 16`
- `num_sheaf_layers = 3`
- `num_edge_types = 12`
- `head_hidden_dim = 96`

Approximate parameters for `C=112`:

- Raw projection: `112*48 + 48 = 5,424`.
- Node initialization and square/type embeddings: about `5,000` to `8,000`.
- Sheaf maps: `num_edge_types * 2 * d * d = 12 * 2 * 16 * 16 = 6,144`.
- Three node MLPs: about `3 * (2*16*16 + biases) = 1,600` if narrow, or about `6,000` with hidden expansion.
- Gate/readout/classifier: about `15,000` to `30,000` depending on statistic vector length.
- Total target: `35k` to `60k` parameters, far smaller than a deep CNN or LC0-style model.

### FLOP/complexity estimate

For padded `E_max=2048`, `d=16`, `L=3`:

- Sheaf map products: `O(B * L * E_max * d^2)` ≈ `B * 3 * 2048 * 256 = B * 1.57M` multiply-add scale.
- Scatter divergence: `O(B * L * E_max * d)` ≈ `B * 98k`.
- Raw projection: `O(B * C * 64 * d_model)` ≤ `B * 344k` for `C=112`, `d_model=48`.
- Mirror branch doubles the encode cost. Total still modest relative to deep image CNNs.

### Config fields

Required:

- `model.name: mirror_tension_sheaf`
- `model.input_channels`
- `model.num_classes`
- `model.encoding_name`
- `model.d_model`
- `model.stalk_dim`
- `model.num_sheaf_layers`
- `model.num_edge_types`
- `model.e_max`
- `model.edge_types`
- `model.use_xray_edges`
- `model.use_king_zone_edges`
- `model.use_file_mirror_gate`
- `model.mirror_axis: file`
- `model.gate_mode: scalar` or `vector`
- `model.dropout`
- `model.return_diagnostics`

Encoding support:

- Must support `simple_18` first for the minimal experiment.
- Add `lc0_static_112` and `lc0_bt4_112` only after tests verify the extractor’s current-board plane mapping.
- The logits interface is identical for all encodings.

## 8. Loss, Training, And Regularization

Primary loss:

- `CrossEntropyLoss` on binary labels.
- Class weights computed from the training split only, e.g. inverse square-root frequency, clipped to `[0.5, 2.0]`.

Optional auxiliary losses, default low weight or disabled for first run:

1. Energy boundedness regularizer:

\[
\lambda_E \cdot \operatorname{mean}(\max(0, \mathcal{E} - E_{cap})^2)
\]

Default: `lambda_E = 1e-4`, `E_cap = 20`. Purpose: prevent unstable sheaf maps, not to encode labels.

2. Restriction-map spectral regularizer:

\[
\lambda_A \sum_t (\|A_t^{src}\|_F^2 + \|A_t^{dst}\|_F^2)
\]

Default: rely on weight decay first; use explicit `lambda_A = 1e-5` only if energies explode.

3. Mirror-gate anti-collapse diagnostic regularizer:

- Do not penalize in the first run.
- Log mean and standard deviation of `rho`.
- Only add a weak entropy regularizer if `rho` saturates to exactly `0` or `1` in the first epoch and gradients vanish.

Training defaults for fair comparison:

- Batch size: use the repo’s baseline batch size if already standardized; otherwise `128` for `simple_18`, lower if memory requires.
- Optimizer: AdamW.
- Learning rate: use the baseline default if available; otherwise `3e-4`.
- Weight decay: `1e-4`.
- Epochs: match the baseline training budget.
- Early stopping: validation ROC-AUC or validation loss, consistent with baseline protocol.
- Determinism: set Python, NumPy, and PyTorch seeds; use deterministic dataloader order where the repo supports it.
- Mixed precision: same setting as baselines.

Regularizers:

- Dropout in classifier head: `0.10`.
- LayerNorm after each sheaf diffusion step.
- Clamp or sigmoid-parameterize `eta` to avoid unstable diffusion.
- Initialize restriction maps near identity plus small noise for defense-like edges and near signed identity for attack-like edges. This is an initialization, not a hand-coded tactic label.

Must stay fixed for fair comparison:

- Same train/val/test parquet split.
- Same binary target mapping.
- Same allowed input encoding.
- Same maximum epochs and early-stopping rule.
- Same data augmentations policy. Do not add label-changing augmentations.
- Same evaluation script and fine-label confusion report.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Zero-energy readout ablation | Keep raw projection, graph construction, node pooling, and classifier, but remove sheaf energy statistics and mirror energy discrepancy. | Smallest falsification of the central claim that sheaf tension carries useful signal. | If unchanged within noise, abandon sheaf-energy readout as the core idea. |
| Untyped scalar graph ablation | Replace typed restriction maps with one scalar message weight shared across all edge types. | Typed attack/defense/x-ray relations matter beyond generic graph connectivity. | If unchanged, the sheaf structure is overengineered; consider simpler board-rule features only. |
| Undirected unsigned ablation | Symmetrize all edges and set all signs to `+1`. | Direction and sign are necessary for tactical pressure. | If unchanged, directed signed structure is not the source of any gain. |
| Random edge-type ablation | Preserve graph topology but randomly permute edge types per batch with fixed seed. | Learned maps depend on chess relation semantics, not just edge count. | If unchanged, relation labels are not being used meaningfully. |
| No x-ray edges | Remove sliding through-blocker pressure edges. | Pins, skewers, and discovered attacks are important to puzzle-likeness. | If unchanged, x-ray feature engineering may be unnecessary or incorrectly implemented. |
| No king-zone edges | Remove king-neighborhood pressure edges. | Puzzle-likeness is partly driven by king tactical exposure. | If unchanged, classifier may rely more on material/contact motifs than mating threats. |
| No file-mirror branch | Use `s(X)` only; remove `M(X)`, `delta_s`, and `rho`. | Partial mirror tension contributes beyond sheaf energy alone. | If unchanged, keep sheaf diffusion but drop mirror gate in future versions. |
| Forced mirror invariance | Average features or logits from `X` and `M(X)` with `rho=1`, no learned gate. | Learned partial equivariance is safer than hard invariance. | If forced invariance wins, the gate may be unnecessary; if it loses, partial symmetry is justified. |
| Occupancy-only graph | Build graph from occupied squares but remove piece-type-specific movement, using generic queen-like rays plus knight jumps. | Chess piece movement details matter. | If unchanged, extractor/edge semantics are not adding expected tactical specificity. |
| Single diffusion layer | Use one sheaf layer instead of three while keeping readout. | Multiple tension propagation steps are necessary to connect attack-defender chains. | If unchanged, retain the simpler one-layer variant for efficiency. |

## 10. Benchmark And Falsification Criteria

Baselines:

- Best existing `simple_18` simple CNN result.
- Best existing `simple_18` residual CNN result.
- Best existing small/medium/deep CNN result on the same split and encoding.
- If available under identical evaluation, compare to LC0 BT4-style CNN/residual results, but the primary minimal experiment is same-encoding `simple_18` to isolate architecture effect.

Metrics:

- Test ROC-AUC.
- Test PR-AUC for puzzle-like class.
- Balanced accuracy.
- F1 for class `1`.
- Calibration: expected calibration error if already supported.
- Confusion table by true fine label `0/1/2 -> predicted binary output 0/1`.
- Parameter count and approximate inference time.
- Diagnostic artifacts: edge overflow count, mean energy by edge type, mirror gate `rho` distribution.

Artifacts to save:

- Config YAML.
- Seed list.
- Training curves.
- Validation selection checkpoint metadata.
- Test metrics JSON.
- Fine-label confusion report.
- Ablation metrics table.
- Diagnostic plots or CSVs for energy by edge type and mirror gate distribution.

Success threshold:

- Primary: improve test ROC-AUC by at least `+0.015` absolute over the best same-encoding non-ensemble baseline, averaged over three seeds, with no worse than `-0.005` PR-AUC.
- Secondary: improve fine label `1` recall at the same validation-selected threshold by at least `+0.03` absolute without increasing fine label `0` false-positive rate by more than `+0.02` absolute.
- Efficiency: parameter count should remain below `250k` for the default model.

Failure threshold:

- If mean test ROC-AUC is within `±0.003` of the zero-energy readout ablation, the central sheaf-tension claim is falsified.
- If random edge-type ablation matches full model within `±0.003` ROC-AUC and `±0.003` PR-AUC, the typed sheaf interpretation is falsified.
- If full model underperforms the best same-encoding baseline by more than `0.010` ROC-AUC across three seeds, do not tune depth/width as a rescue.

Abandon condition:

- Abandon this idea if both the zero-energy readout ablation and the random edge-type ablation match or exceed the full model while diagnostics show nontrivial training convergence. That means the graph/sheaf mechanism is not doing the causal work.

Scaling condition:

- Only scale after passing the smallest ablation. Allowed scaling is constrained to preserving the operator: `stalk_dim 16 -> 24`, `E_max 2048 -> 3072`, or `num_sheaf_layers 3 -> 4`. Do not turn it into a generic larger CNN/ResNet/Transformer.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/2026-04-21_mirror_tension_sheaf/README.md` | Create | Copy this handoff packet or a concise implementation-facing version of it. |
| `ideas/2026-04-21_mirror_tension_sheaf/ablation_plan.md` | Create | Ablation table, seed plan, expected artifacts, falsification thresholds. |
| `ideas/2026-04-21_mirror_tension_sheaf/results_template.md` | Create | Empty template for metrics, fine-label confusion, and diagnostics. |
| `src/chess_nn_playground/models/mirror_tension_sheaf.py` | Create | `MirrorTensionSheafNet`, `TypedSheafDiffusionLayer`, readout, gate, classifier head. |
| `src/chess_nn_playground/models/components/chess_attack_graph.py` | Create | Deterministic board-only pseudo-legal attack/defense/x-ray edge builder with tests. |
| `src/chess_nn_playground/models/components/file_mirror.py` | Create | File mirror transform for each supported encoding, with explicit plane permutation. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `mirror_tension_sheaf` without changing existing model names. |
| `configs/mirror_tension_sheaf_simple18.yaml` | Create | Minimal experiment config for `simple_18`, `num_classes=2`, default dimensions, split paths. |
| `configs/mirror_tension_sheaf_lc0_static112.yaml` | Optional create after extractor tests | Same architecture for `lc0_static_112`; do not run until plane mapping is verified. |
| `configs/mirror_tension_sheaf_lc0_bt4_112.yaml` | Optional create after extractor tests | Same architecture for `lc0_bt4_112`; do not run until plane mapping is verified. |
| `tests/test_mirror_tension_sheaf_shapes.py` | Create | Assert input `(B,C,8,8)` returns logits `(B,2)` for all verified encodings. |
| `tests/test_attack_graph_no_leakage.py` | Create | Assert graph builder consumes only tensor/canonical board info and no label/engine fields. |
| `tests/test_file_mirror_involution.py` | Create | Assert file mirror applied twice recovers original tensor for supported encodings up to exact plane permutation. |
| `tests/test_sheaf_energy_equivariance.py` | Create | On synthetic symmetric boards, assert energy stats permute correctly under file mirror when gate is disabled and maps are shared. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Preserve hard constraints; add anti-duplicate note for this idea, lessons about partial symmetry, and stricter requirement that future ideas include smallest central-claim ablation. |

Codex must update `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` after consuming this packet. It should preserve leakage rules, label rules, falsification requirements, and anti-duplicate requirements while adding reusable lessons and failure-mode guidance discovered during implementation.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0437_tuesday_los_angeles_mirror_tension_sheaf.md
  title: "Codex Handoff Packet: File-Mirror Tension Sheaf"
  generated_at: "2026-04-21 04:37 PDT"
  timezone: los_angeles
  weekday: tuesday
  idea_slug: mirror_tension_sheaf
  file_type: markdown
```

```yaml
idea_yaml:
  idea_id: "2026-04-21_mirror_tension_sheaf"
  idea_name: "File-Mirror Tension Sheaf"
  one_sentence_thesis: "Detect puzzle-likeness from board-only pseudo-legal attack-defense-xray sheaf tension plus learned file-mirror partial equivariance."
  fingerprint: "board_only:pseudo_attack_defense_xray_graph + signed_directed_sheaf_laplacian_energy + file_mirror_partial_equivariance_gate + energy_stats_classifier"
  core_operator: "typed signed directed sheaf coboundary and energy readout"
  symmetry: "learned partial file-mirror equivariance, not full board invariance"
  forbidden_inputs:
    - stockfish_scores
    - principal_variations
    - node_counts
    - verification_metadata
    - source_labels_as_features
    - proposed_labels_as_features
    - unresolved_status_as_feature
  allowed_inputs:
    - board_tensor
    - deterministic_board_only_attack_defense_graph
    - deterministic_file_mirror_transform
  minimal_experiment:
    encoding: simple_18
    train_split: data/splits/crtk_sample_3class/split_train.parquet
    val_split: data/splits/crtk_sample_3class/split_val.parquet
    test_split: data/splits/crtk_sample_3class/split_test.parquet
    seeds: [0, 1, 2]
  success_threshold:
    roc_auc_absolute_gain_over_best_same_encoding_baseline: 0.015
    fine_label_1_recall_gain_at_threshold: 0.03
  abandon_if:
    - "zero-energy readout ablation matches full model within 0.003 ROC-AUC"
    - "random edge-type ablation matches full model within 0.003 ROC-AUC and PR-AUC"
```

```yaml
config_yaml:
  model:
    name: mirror_tension_sheaf
    encoding_name: simple_18
    input_channels: 18
    num_classes: 2
    d_model: 48
    stalk_dim: 16
    num_sheaf_layers: 3
    num_edge_types: 12
    e_max: 2048
    edge_types:
      - own_attack_enemy
      - enemy_attack_own
      - own_defense
      - enemy_defense
      - own_xray_enemy
      - enemy_xray_own
      - own_pawn_control
      - enemy_pawn_control
      - own_king_zone_pressure
      - enemy_king_zone_pressure
      - own_line_blocker_pressure
      - enemy_line_blocker_pressure
    use_xray_edges: true
    use_king_zone_edges: true
    use_file_mirror_gate: true
    mirror_axis: file
    gate_mode: scalar
    topk_energy: 8
    dropout: 0.10
    eta_max: 0.20
    return_diagnostics: true
  data:
    train_split: data/splits/crtk_sample_3class/split_train.parquet
    val_split: data/splits/crtk_sample_3class/split_val.parquet
    test_split: data/splits/crtk_sample_3class/split_test.parquet
    binary_target_map:
      fine_0: 0
      fine_1: 1
      fine_2: 1
  training:
    seeds: [0, 1, 2]
    batch_size: 128
    optimizer: adamw
    learning_rate: 0.0003
    weight_decay: 0.0001
    class_weighting: inverse_sqrt_frequency_clipped
    class_weight_clip: [0.5, 2.0]
    early_stopping_metric: val_roc_auc
    match_baseline_epoch_budget: true
    deterministic: true
  regularization:
    energy_cap: 20.0
    energy_cap_lambda: 0.0001
    restriction_weight_decay_via_optimizer: true
    mirror_gate_entropy_lambda: 0.0
  evaluation:
    metrics:
      - roc_auc
      - pr_auc
      - balanced_accuracy
      - f1_positive
      - fine_label_confusion
      - parameter_count
      - inference_time
    diagnostics:
      - edge_overflow_count
      - energy_by_edge_type
      - mirror_gate_rho_distribution
```

```yaml
model_spec:
  class_name: MirrorTensionSheafNet
  module_path: chess_nn_playground.models.mirror_tension_sheaf
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  components:
    - EncodingBoardExtractor
    - AttackDefenseGraphBuilder
    - RawBoardProjector
    - NodeInitializer
    - TypedSheafDiffusionLayer
    - SheafEnergyReadout
    - FileMirrorPartialGate
    - ClassifierHead
  forward_contract:
    - "extract canonical board-only occupancy from x"
    - "build pseudo-legal attack/defense/xray graph"
    - "project raw board to square node features"
    - "run typed signed sheaf diffusion"
    - "read out sheaf energy statistics"
    - "repeat shared encode path on file-mirrored x"
    - "compute gated mirror tension"
    - "return logits only unless diagnostics requested"
  default_parameter_budget_max: 250000
  expected_default_parameter_count_range: [35000, 60000]
  leakage_safe_by_design: true
```

```yaml
research_continuity:
  idea_fingerprint: "board_only:pseudo_attack_defense_xray_graph + signed_directed_sheaf_laplacian_energy + file_mirror_partial_equivariance_gate + energy_stats_classifier"
  closest_duplicate_risk: "Any future attack-graph sheaf, signed directed Laplacian, or mirror-equivariant energy model over pseudo-legal chess relations should be treated as overlapping unless it changes the core operator and falsification target."
  do_not_repeat_if_this_fails:
    - "Do not retry a signed directed sheaf over pseudo-legal attack/defense/xray edges with only larger stalk dimension or more layers."
    - "Do not retry file-mirror partial equivariance as the main novelty unless diagnostics show the sheaf operator worked and only the gate failed."
    - "Do not replace this with a plain GNN-on-squares and call it a new graph idea."
  suggested_next_search_directions:
    - "Differentiable one-ply candidate-move bottleneck without engine scores or PVs."
    - "Causal invariance between material/contact-preserving board transforms and puzzle labels."
    - "Optimal-transport matching between attacker and defender sets around kings and high-value pieces."
    - "Energy-based model over legal move masks computed from board only, trained with contrastive board corruptions rather than engine targets."
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add an anti-duplicate rule: “Do not propose another signed directed sheaf/Laplacian over pseudo-legal attack-defense-xray edges if `mirror_tension_sheaf` fails its smallest ablation.” | Prevents future cycles from recycling the same idea with superficial renaming. | Hard Constraints or Research Goal |
| Add: “Every proposed architecture must name the smallest ablation that can falsify its central mechanism.” | Forces future ideas to separate architectural novelty from incidental performance. | Required Markdown File Content, Ablation Plan |
| Add: “When using chess symmetries, state exactly which board transform is used and why full dihedral symmetry is unsafe.” | Avoids mathematically false invariance assumptions for pawns, side-to-move, and castling. | Mathematical Thesis |
| Add: “If deterministic chess-rule features are used, state whether they are pseudo-legal, legal, or search-derived; search-derived is forbidden.” | Makes leakage boundaries clearer while allowing board-only geometry. | Problem Restatement And Data Contract |
| Add: “Research continuity must include a closest-duplicate-risk sentence with operator-level detail.” | Helps future research cycles avoid near-duplicates. | Machine-Readable Blocks |
| Add: “If an idea uses an auxiliary loss, explain why it does not create or imply new labels.” | Protects label integrity. | Loss, Training, And Regularization |

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
