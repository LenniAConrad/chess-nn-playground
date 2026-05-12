# Codex Handoff Packet: Threat-Topology Betti Bottleneck Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0814_tuesday_los_angeles_threat_topology.md`
- Generated at: 2026-04-21 08:14 UTC-07:00
- Weekday: Tuesday
- Timezone: America/Los_Angeles (`los_angeles`)
- Idea slug: `threat_topology`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Threat-Topology Betti Bottleneck Network
- One-sentence thesis: Puzzle-like chess positions often contain unusually coherent high-pressure tactical regions, and a rank-based cubical Betti-curve bottleneck can test whether the connectedness and holes of rule-only pressure fields add signal beyond material, attack counts, histograms, and ordinary CNN texture.
- Idea fingerprint: `current-board piece planes -> pseudo-legal rule-only pressure surplus fields -> rank-top-k cubical superlevel sets on the 8x8 board -> Betti-0, Betti-1, boundary, and top-k mean curves -> small MLP fused with a matched CNN stem -> binary logits`.
- Why this is not a common CNN/ResNet/Transformer variant: The central representation is not learned local convolution, residual depth, attention over squares, or an attack graph; it is an explicit topological summary of the geometry of scalar pressure fields under cubical filtrations, with a rank-shuffle ablation that preserves the same scalar values while destroying board adjacency.
- Current-data minimal experiment: Train `ThreatTopologyNet` on `simple_18` using the existing `crtk_sample_3class` train/val/test split for the same epoch, optimizer, class weighting, and reporting settings as the current simple CNN benchmarks.
- Smallest central falsification ablation: Before the Betti encoder, randomly permute the 64 square ranks independently per sample and pressure field, preserving the exact sorted pressure values and top-k counts but destroying chess-board adjacency; if this matches the main model, the topology claim is falsified.
- Expected information gain if it fails: A clean failure says that static pressure-field topology is not a useful bottleneck for puzzle-likeness on this split, and the next pass should avoid Betti/topological pressure-map variants rather than merely changing thresholds, weights, or CNN fusion.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The task is binary chess puzzle-likeness classification from a single board position.

Fine labels:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

Training target for the default coarse benchmark:

- output `0`: non-puzzle, corresponding to fine label `0`
- output `1`: puzzle-like, corresponding to fine labels `1` and `2`

Diagnostics must still report the rectangular `3x2` matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Available input encodings:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant
- `lc0_static_112`
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists

Model contract:

- PyTorch module accepts input tensor `(batch, C, 8, 8)`.
- It returns logits `(batch, num_classes)`, with `num_classes=2`.
- Shared trainer, reports, confusion matrices, predictions, and leaderboards must keep working.

Benchmark split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Do not point the existing non-streaming trainer directly at the roughly 45M-row full Parquet dataset.

Leakage checklist:

- Allowed: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board.
- Allowed in this idea: pseudo-legal attack maps computed from current occupancy and piece movement rules, with blockers for sliders, without asking whether a move is legal under check constraints.
- Forbidden as neural-network inputs: Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, and any label-derived feature.
- Do not fabricate class `1` or class `2`; use only the fine labels already present in the split.
- Treat any unresolved candidate pool as unresolved, never as verified near-puzzle or verified puzzle.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated. This packet does not require them.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry may only be derived from known current-board piece planes. History planes may be consumed by a learned neural adapter but must not be interpreted as rule-derived geometry unless their semantics are explicitly mapped. Adapters must fail closed when channel semantics are unknown.

## 4. Research Map

External ideas used:

| Source | What is borrowed | What is not copied |
|---|---|---|
| Edelsbrunner and Harer, “Persistent Homology - a Survey,” 2008. URL: https://webhomes.maths.ed.ac.uk/~v1ranick/papers/edelhare.pdf | The idea that connected components and holes can be tracked over a filtration of a scalar function. | No off-the-shelf persistence package is required; no full persistence diagram is the core output. |
| Adams et al., “Persistence Images: A Stable Vector Representation of Persistent Homology,” JMLR 2017. URL: https://jmlr.org/papers/v18/16-337.html | The general practice of turning topological summaries into fixed-length vectors for ordinary ML models. | The model does not compute persistence images; it uses rank-Betti curves and simple cubical counts on an 8x8 grid. |
| Hofer et al., “Deep Learning with Topological Signatures,” NeurIPS 2017. URL: https://proceedings.neurips.cc/paper_files/paper/2017/hash/883e881bb4d22a7add958f2d6b052c9f-Abstract.html | The principle that topological signatures can be paired with neural networks for supervised learning. | The architecture here is not their topological signature layer and does not learn task-optimal diagram coordinates. |
| Carriere et al., “PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures,” AISTATS 2020. URL: https://proceedings.mlr.press/v108/carriere20a.html | The broader precedent for neural layers consuming topological descriptors. | The minimal experiment avoids differentiable persistence-diagram learning; topology is a deterministic bottleneck over chess pressure fields. |
| Wagner, Chen, and Vucini, “Efficient Computation of Persistent Homology for Cubical Data,” 2011. URL: https://chaochen.github.io/publications/chen_topoinvis_2011.pdf | The use of cubical complexes for image-like data. | The implementation should use a tiny custom 8x8 cubical-count/connected-component routine rather than a general cubical PH library. |

Candidate search trace:

| Candidate mechanism considered | Why it lost to the selected idea |
|---|---|
| Differentiable Morse critical-point counts on pressure fields | Interesting, but harder to implement robustly and easier to reduce to noisy local extrema counts on an 8x8 board. |
| Topological regularization loss on CNN activation maps | Too close to “regularize a CNN”; the topological object would be learned and less directly falsifiable as chess structure. |
| Conformal selective prediction focused on near-puzzles | Useful for deployment, but not a new board operator and too adjacent to uncertainty/calibration work already represented by credal/evidential packets. |
| Causal invariance across source/date/site environments | Potentially strong, but the current prompt does not guarantee usable environment metadata, and source/provenance features are forbidden as inputs. |
| Source-free information bottleneck with augmentation adversaries | Hard to falsify cleanly; may suppress exactly the tactical asymmetries needed for puzzles. |
| Motif grammar / MDL template model over tactical lines | Too close to ray-language automata and pseudo-likelihood/code-length imported families. |
| Hyperbolic embeddings of king-zone threat geometry | Mostly a metric-learning wrapper unless paired with a new operator; weaker central ablation. |
| Matroid/circuit model over defender coverage | Too close to static attack-defense graph/spectral structure and likely to collapse into another incidence operator. |
| Low-rank tensor nuclear-norm interactions over piece planes | Too close to ANOVA/Mobius constellation packets. |
| Energy-based model of legal-board plausibility | Too close to masked-codec and pseudo-likelihood packets unless a completely new observable is introduced. |
| Pure entropy/free-energy of attack maps | Too hand-crafted and likely count/histogram based; topology gives a stronger geometry-preserving falsifier. |
| Rank-cubical topology of pressure fields | Selected because it has a precise board operator, uses only current-board rules, has a clean histogram-preserving falsifier, and is outside imported sheaf/move/transport/codec families. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Persistent homology | Rank-top-k cubical superlevel sets of rule-only pressure fields; sampled Betti-0 and Betti-1 curves | pressure fields `(B, F, 8, 8)` -> topology vector `(B, F*K*4)` | Per-sample pressure-rank shuffle before cubical counts | No sheaf, Hodge, graph Laplacian, attack incidence cochains, or curvature/tension energy |
| Topological bottleneck | Low-dimensional Betti/perimeter/top-k-mean features fused with a matched CNN stem | topology vector -> MLP embedding `(B, 64)` | Histogram-only and no-topology controls | Not a larger CNN, not a vanilla Transformer, not a learned attention pooling trick |
| Rank invariance | Top-k masks depend on square ordering, not on calibrated pressure scale | scalar field `(B,8,8)` -> masks `(B,K,8,8)` | Monotone recalibration should leave topology branch unchanged | Not pseudo-likelihood, masked code-length, or ordinal/credal evidence |
| Chess pressure geometry | Pseudo-legal current-board attack surplus scalarization | piece planes `(B,12,8,8)` -> fields `(B,4,8,8)` | All-ones weights, no-value-bonus, pressure-CNN controls | Not one-ply move deltas, legal move trees, Sinkhorn transport, or ray automata |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Plain simple CNN | `src/chess_nn_playground/models/trunk/cnn.py` | Already exists and does not test a new structural hypothesis. |
| Plain residual CNN | `src/chess_nn_playground/models/trunk/residual_cnn.py` | Already exists; extra residual depth is ordinary capacity scaling. |
| LC0-style CNN or residual CNN on `lc0_bt4_112` | Existing LC0 BT4-style CNN/residual variants | Already represented and too close to copying LC0-style spatial processing. |
| Ordinary ViT over 64 square tokens | Generic Transformer baseline | Too generic; attention over squares alone gives no chess-specific falsifiable operator. |
| Plain GNN on square adjacency | Common graph neural network | Too standard and likely inferior to a CNN on an 8x8 grid unless given richer, already-imported chess graph structure. |
| Wider/deeper network or optimizer tuning | All existing neural baselines | Disallowed as the core idea and would not explain puzzle-likeness. |
| Ensembling several existing models | Any combination of current baselines | Disallowed as a core idea and hides rather than tests an inductive bias. |
| Training directly on the full 45M-row Parquet | Existing trainer without streaming | Not valid until streaming support exists; also “add more data” is not a research mechanism. |
| Static attack-defense graph/sheaf/Hodge model | Imported sheaf/Hodge packets | Already researched; changing edge labels or pooling would be a near-duplicate. |
| One-ply move-delta DeepSets/attention/spectrum | Imported counterfactual move-delta packets | Already researched and would introduce a move-candidate family this packet intentionally avoids. |
| Piece-target Sinkhorn/transport pressure model | Imported optimal-transport packets | Already researched; this idea uses no coupling, transport plan, or Sinkhorn objective. |
| Ordinal ladder or credal near-puzzle head | Imported ordinal and credal packets | Does not introduce a board operator and is already represented as label-head research. |
| Sparse witness-piece bottleneck | Imported sparse witness packet | The selected model summarizes global pressure-field topology, not selected occupied pieces. |
| Ray-language finite automata | Imported ray-language packet | The pressure generator uses piece rules, but it does not tokenize rays or run automata over ray strings. |
| Mobius/ANOVA constellations | Imported constellation packet | Rank-Betti curves are topological summaries of scalar fields, not high-order piece tuple interactions. |
| Masked-board codec or pseudo-likelihood ratio | Imported masked-codec and pseudo-likelihood packets | The model does not reconstruct, score, mask, or estimate board likelihood. |

## 6. Mathematical Thesis

Input space definition.

Let `S = {0,...,7} x {0,...,7}` be the set of board squares. Let `X` be the set of current-board encodings that can be decoded into piece indicators

```text
O_{c,p}(u) in {0,1}
```

for color `c in {white, black}`, piece type `p in {P,N,B,R,Q,K}`, and square `u in S`, together with side-to-move `m(x)`. For the minimal experiment, `X` is the `simple_18` representation.

Label/target definition.

The fine label is `L in {0,1,2}`. The coarse training target is

```text
Y = 1[L in {1,2}]
```

so known non-puzzles are negative and verified near-puzzles plus verified puzzles are positive. The fine label is used for diagnostics only, not as an input.

Data distribution assumptions.

The training, validation, and test splits are treated as samples from related but not necessarily identical distributions over current-board positions and labels. The model assumes that some puzzle-like positions have static tactical geometry visible in the current board, but it does not assume every puzzle can be identified without search. The fine label `1` is expected to be noisier or more ambiguous than fine label `2`, so class-1 recall at matched class-0 false-positive rate is a required diagnostic.

Allowed symmetry or equivariance assumptions.

Chess is not fully rotation/reflection invariant because pawns, castling, side-to-move, and board orientation matter. This model does not quotient by board symmetries. It uses ordinary board adjacency in the displayed coordinate system and constructs side-relative pressure fields so that “side-to-move pressure” and “opponent pressure” are comparable across colors. No file mirror, orbit quotient, or Reynolds pooling is used.

Core hypothesis.

Let `A_c(v)` be rule-only pseudo-legal attack pressure by color `c` on square `v`, using current occupancy for blockers and no legal-move, check, mate, or search oracle. A puzzle-like position is more likely when the side-to-move pressure surplus forms spatially coherent high-rank regions around valuable enemy targets and when the opponent’s counter-pressure topology differs in a structured way. This coherence is not fully captured by material, total attack counts, pressure histograms, or local CNN texture.

Formal object.

Use fixed piece weights

```text
omega_P=1, omega_N=3, omega_B=3, omega_R=5, omega_Q=9, omega_K=0.5
nu_P=1, nu_N=3, nu_B=3, nu_R=5, nu_Q=9, nu_K=12
```

with both weight lists configurable and ablated.

For a position `x`, define pseudo-legal attack indicators

```text
att_{c,p}^x(u,v) = 1
```

when a piece of color `c` and type `p` on `u` attacks `v` under current-board piece rules. Pawns use color-dependent attack directions. Sliders stop at the first occupied square. This is attack geometry, not legal move generation.

Define attack pressure and material-value fields:

```text
A_c(v) = sum_{u,p} omega_p O_{c,p}(u) att_{c,p}^x(u,v)
M_c(v) = sum_p nu_p O_{c,p}(v)
```

Let `bar(m)` be the opponent of the side to move. Let `G_k(v)` be a fixed Chebyshev-distance kernel centered at king square `k`, for example `G_k(v)=exp(-d_infty(v,k)/2)`, computed only from the current board.

The first experiment uses four scalar fields, each in `(8,8)`:

```text
F_1(v) = A_m(v) - A_bar(m)(v) + alpha M_bar(m)(v)
F_2(v) = A_bar(m)(v) - A_m(v) + alpha M_m(v)
F_3(v) = (A_m(v) - A_bar(m)(v)) * G_{king_bar(m)}(v) + alpha M_bar(m)(v)
F_4(v) = (A_bar(m)(v) - A_m(v)) * G_{king_m}(v) + alpha M_m(v)
```

with default `alpha=0.25`. These are pressure-surplus and king-shell pressure-surplus fields for the moving side and the opponent.

For a scalar field `F:S->R` and a rank budget `k`, let `T_k(F)` be the set of the `k` squares with largest `F` values, with deterministic square-index tie-breaking. Let `K_k(F)` be the cubical complex formed by the union of closed unit cells corresponding to `T_k(F)`, including their faces. Define

```text
B(F;k) = (beta_0(K_k(F)), beta_1(K_k(F)), boundary_edges(K_k(F)), mean_{v in T_k(F)} F(v)).
```

The topology descriptor is

```text
Topo(x) = concat_{j=1..4, k in Klist} B(F_j(x);k)
```

where the default rank list is

```text
Klist = [1,2,4,6,8,12,16,24,32,48].
```

Optimization objective.

Let `z_cnn(x)` be a small matched CNN stem. Let `z_topo(x)=MLP(Topo(x))`. The model predicts

```text
logits = Head(concat(z_cnn(x), z_topo(x))).
```

The training objective is balanced cross-entropy:

```text
min_theta E_{(x,Y)} w_Y CE(Head_theta(z_cnn(x), z_topo(x)), Y).
```

Proposition: rank-Betti topology detects geometry that histograms cannot.

For any strictly increasing function `phi:R->R`, and any scalar field `F` with deterministic tie-breaking unchanged,

```text
T_k(phi o F) = T_k(F)
```

for all `k`, hence

```text
B(phi o F;k) = B(F;k)
```

except for the mean-value coordinate, which can be omitted in the pure topology ablation.

There exist two fields `F` and `G` with the same multiset of 64 scalar values and therefore identical histograms, quantiles, top-k means, and sorted values, but with different rank-Betti descriptors. For example, let the top four values occupy a contiguous `2x2` block in `F` and four mutually non-adjacent corners in `G`. At `k=4`, `beta_0(K_4(F))=1`, while `beta_0(K_4(G))=4`; the sorted scalar values are identical.

Proof sketch.

A strictly increasing `phi` preserves the ordering of all square values, so every top-k set is preserved. The cubical complex built from the top-k set is therefore identical, giving identical Betti numbers and boundary counts. For the histogram counterexample, the scalar multiset is identical by construction, but the induced top-k cubical complexes differ in connected components. Thus any model branch that sees only sorted values cannot distinguish the two boards, while the Betti branch can.

What is actually proven.

The rank-Betti descriptor is invariant to monotone recalibration of pressure scores, up to deterministic tie issues, and it can distinguish spatial adjacency patterns that pressure histograms cannot.

What remains only hypothesized.

It is only a hypothesis that puzzle-like positions in the CRTK split have distinctive pressure-field topology that generalizes beyond material, king distance, attack totals, and source artifacts. It is also only hypothesized that an MLP over these descriptors helps class `1` near-puzzle recall rather than mostly helping obvious class `2` puzzles.

Counterexamples where the idea should fail.

- Quiet endgame studies where the puzzle depends on zugzwang, opposition, or long-horizon legality rather than static pressure topology.
- Tactical puzzles where the key move is a retreat, interference, or underpromotion whose motif is not visible as current-board high-pressure connectedness.
- Positions with many attacks in a messy non-puzzle that produce topology similar to true puzzles.
- Boards where the relevant feature is one exact legal move, pinned-piece legality, or forced line; this packet intentionally avoids legal-move trees and search.
- Dataset artifacts where positives are mostly selected by material or phase; then topology may add little beyond baseline CNNs and material-like pressure weights.

Self-critique.

The strongest objection is that this is still derived from static attack pressure, so it may merely repackage a hand-crafted attack map that a CNN could learn from piece planes. The minimal experiment is still worth running because the central ablation preserves exact pressure values and top-k counts while destroying geometry; if the rank-shuffled model ties the main model, the idea dies cleanly. Another risk is implementation complexity in exact connected-component counts, but the 8x8 board permits a deterministic, vectorized label-propagation or tiny loop implementation with focused tests.

## 7. Architecture Specification

Module names.

- `Simple18PiecePlaneAdapter`
- `Lc0CurrentPiecePlaneAdapter`
- `RulePressureFields`
- `RankCubicalBettiEncoder`
- `ThreatTopologyBranch`
- `MatchedBoardCnnStem`
- `ThreatTopologyNet`

Forward-pass steps and shapes.

1. Input:

```text
x: (B, C, 8, 8)
```

2. Decode current-board piece planes for the rule branch:

```text
pieces: (B, 12, 8, 8)
side_to_move: (B,)
```

For `simple_18`, use the configured 12 piece-plane order and side-to-move channel. For LC0 encodings, only enable this step if the config explicitly maps current-board piece planes and side-to-move semantics; otherwise raise a clear error.

3. Compute pseudo-legal attack pressure:

```text
A_white, A_black: (B, 8, 8)
M_white, M_black: (B, 8, 8)
king kernels: (B, 2, 8, 8)
pressure_fields: (B, F=4, 8, 8)
```

The attack generator must be deterministic and rule-only. It must not call an engine, legal-move generator, checkmate detector, or source-label logic.

4. Encode topology:

For each field and each `k in Klist`, compute the top-k mask:

```text
topk_masks: (B, F, K, 8, 8)
```

For each mask compute:

```text
beta0: connected components under 4-neighbor cell adjacency
beta1: beta0 - V + E - C, where C is active cell count, E unique active cubical edges, V unique active cubical vertices
boundary_edges: number of active-cell sides touching inactive/outside cells
topk_mean: mean field value over active cells
```

Output:

```text
topology_features: (B, F, K, 4)
flattened_topology_features: (B, 4*10*4) = (B, 160)
```

5. Topology branch:

```text
LayerNorm(160)
Linear(160, 128) + SiLU + Dropout(0.10)
Linear(128, 64) + SiLU
z_topo: (B, 64)
```

6. Matched CNN stem over the original encoding:

```text
Conv2d(C, 32, kernel=3, padding=1) + BatchNorm + SiLU
Conv2d(32, 64, kernel=3, padding=1) + BatchNorm + SiLU
Conv2d(64, 64, kernel=3, padding=1) + BatchNorm + SiLU
global_avg_pool: (B,64)
global_max_pool: (B,64)
z_cnn: concat -> (B,128)
```

7. Fusion head:

```text
z = concat(z_cnn, z_topo): (B,192)
Linear(192,128) + SiLU + Dropout(0.10)
Linear(128,2)
logits: (B,2)
```

Parameter-count estimate.

For `simple_18`:

- CNN stem: about 61k parameters.
- Topology MLP: about 29k parameters.
- Fusion head: about 25k parameters.
- BatchNorm and small constants: under 1k.
- Total: roughly 115k-130k trainable parameters.

For `lc0_*` with `C=112`, the first convolution adds about 27k more parameters, so total is roughly 145k-160k.

FLOP or complexity estimate.

- CNN stem: `O(B * 8 * 8 * (C*32*9 + 32*64*9 + 64*64*9))`.
- Pressure generation: constant-size rule loops over at most 32 pieces and 64 target squares, effectively `O(B * 32 * ray_length)` with small constants.
- Top-k masks: `O(B * F * 64 log 64)` or `O(B * F * K * 64)` depending on implementation.
- Connected components: either vectorized min-label propagation for at most 64 iterations on `(B*F*K)` masks, or a tiny deterministic CPU/GPU loop. Complexity `O(B * F * K * 64 * I)` with `I<=64`, small enough for the current sample split.

Generated candidate-set memory.

This model generates top-k masks, not move candidates. With batch size `B`, number of fields `F=4`, number of rank budgets `K=10`, and board size `64`:

- Boolean masks: `B*F*K*64` elements. For `B=512`, this is `512*4*10*64 = 1,310,720` booleans, about 1.3 MB if stored as uint8/bool before framework overhead.
- Float labels or propagated component ids may require about 4-8x more memory during component computation.
- Chunking plan: if memory or Python overhead is high, process fields in chunks of `F_chunk=1` or process rank budgets in two chunks of five, concatenate `(B,F,K,4)` features, and keep the CNN branch unchanged.

Required config fields.

```yaml
model:
  name: threat_topology_net
  input_channels: 18
  num_classes: 2
  piece_plane_order: simple_18_default
  side_to_move_channel: 12
  rank_ks: [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]
  pressure_alpha: 0.25
  pressure_piece_weights: [1, 3, 3, 5, 9, 0.5]
  target_piece_values: [1, 3, 3, 5, 9, 12]
  topology_stats: [beta0, beta1, boundary_edges, topk_mean]
  topology_ablation: none
  topology_dropout: 0.10
```

Encoding-adapter assumptions.

- `simple_18`: first experiment should use this encoding. The adapter must confirm the 12 piece-plane order from the project’s encoding metadata or config. If the order is absent, fail with a message asking Codex to add the mapping inside the repo, not to infer silently.
- `lc0_static_112`: support only if current-board piece planes and side-to-move channel are explicitly mapped. The learned CNN stem may consume all 112 channels. The rule topology branch may consume only mapped current-board channels.
- `lc0_bt4_112`: same as `lc0_static_112`. History channels may go into `MatchedBoardCnnStem`; they must not be used for deterministic pressure geometry unless exporter semantics are explicit. Zero-filled unavailable history must not be interpreted as evidence of prior moves.
- Unknown encoding: fail closed.

How the model returns logits.

`ThreatTopologyNet.forward(x)` returns only the tensor `logits` with shape `(B,2)`, unless an optional debug flag is set outside the shared trainer path. The default path must remain trainer-compatible.

Pseudocode.

```text
forward(x):
    pieces, stm = piece_adapter(x)              # (B,12,8,8), (B,)
    fields = rule_pressure_fields(pieces, stm)  # (B,4,8,8)

    if topology_ablation == "rank_shuffle":
        fields_for_topology = shuffle_square_ranks(fields, preserve_values=True)
    elif topology_ablation == "king_ring_shuffle":
        fields_for_topology = shuffle_within_king_distance_rings(fields, pieces, stm)
    else:
        fields_for_topology = fields

    topo = rank_cubical_betti(fields_for_topology, rank_ks)  # (B,4,10,4)
    z_topo = topology_mlp(flatten(topo))                     # (B,64)

    z_cnn = matched_cnn_stem(x)                              # (B,128)
    logits = fusion_head(concat(z_cnn, z_topo))              # (B,2)
    return logits
```

## 8. Loss, Training, And Regularization

Primary loss.

- Balanced binary/coarse cross-entropy implemented as standard two-class `CrossEntropyLoss(weight=class_weights)`.
- Target is `0` for fine label `0`, `1` for fine labels `1` and `2`.

Auxiliary loss.

- None required for the minimal experiment.
- Optional debug-only branch loss: train a topology-only classifier from `z_topo` with low weight `0.1` to monitor whether the topology branch carries signal. Do not enable this in the first fair benchmark unless all ablations use the same auxiliary setup.

Class weighting.

- Use the existing benchmark’s balanced class weighting.
- Do not tune class weights per ablation.

Batch size expectations.

- Default `batch_size: 512`.
- If topology computation is slow, reduce to `256` only if all central models and ablations use the same batch size, or use topology chunking while keeping the batch size unchanged.

Learning-rate and optimizer defaults.

- Optimizer: AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Epochs: `3` for minimal current-data experiment.
- Early stopping patience: `2`.
- Mixed precision: `false` for the first implementation, because exact top-k and component-count behavior should be debugged deterministically first.

Regularizers.

- Dropout `0.10` in topology MLP and fusion head.
- Weight decay as above.
- Optional topology-branch dropout: with probability `0.05`, zero `z_topo` during training only. Enable only after the core ablation suite works; otherwise it complicates interpretation.

Determinism requirements.

- Use `seed: 42`.
- Enable deterministic PyTorch behavior where the repo already supports it.
- Deterministic top-k tie-breaking: add a tiny fixed square-index epsilon, e.g. `1e-6 * normalized_square_index`, before ranking. The epsilon must be independent of labels and dataset provenance.
- Random ablations must be seeded and logged.

What must stay unchanged for fair comparison.

- Same train/val/test split.
- Same coarse binary target mapping.
- Same number of epochs, optimizer family, batch size, class weighting, device policy, and reporting pipeline as the matched simple CNN baseline.
- No additional data, no full-dataset training, no engine features, and no source/provenance features.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `no_topology_fusion` | Replace `z_topo` with zeros but keep the same CNN stem and fusion head input size via learned zero/constant adapter | Tests whether Betti topology adds anything beyond the matched CNN | If equal to main, topology branch is unnecessary. |
| `rank_shuffle` | Randomly permute the 64 pressure ranks per sample and field before Betti computation; sorted values and top-k means are preserved | Smallest central falsifier: actual board adjacency of high-pressure cells matters | If equal to main, the central topology claim fails. |
| `histogram_only` | Replace Betti and boundary stats with sorted quantiles/top-k means of each pressure field | Tests whether scalar pressure distributions, not topology, explain gains | If equal to main, topology is just a pressure histogram proxy. |
| `degree_class_square_permutation` | Apply a fixed random square permutation within corner/edge/interior degree classes before cubical counts | Tests whether grid adjacency semantics matter while preserving easy boundary-degree marginals | If equal to main, board geometry is not being used meaningfully. |
| `king_ring_preserving_shuffle` | Shuffle pressure ranks within Chebyshev-distance rings around both kings | Tests whether gains are only king-distance profiles rather than local connectedness | If equal to main, topology may be a king-distance shortcut. |
| `pressure_fields_only_cnn` | Feed the four pressure fields through the same small CNN stem, no Betti features | Tests whether attack-map preprocessing alone explains gains | If equal to main, ordinary convolutions over pressure maps are sufficient. |
| `beta0_only` | Keep connected-component counts but remove `beta1` and boundary stats | Tests whether fragmentation of high-pressure islands is enough | If main beats this, holes/boundaries add signal. |
| `beta1_boundary_only` | Remove `beta0`, keep holes and boundary stats | Tests whether enclosure/ring structure matters more than component count | If equal to main, connected-island count is not the key feature. |
| `all_one_attack_weights` | Set all attacking piece weights to `1` | Tests whether material-valued pressure is the true driver | If equal to main, piece-value weighting is not needed; if much worse, gains may be material-heavy. |
| `no_target_value_bonus` | Set `alpha=0`, removing `M_c` terms from fields | Tests whether topology of attacks alone matters without explicit target-value planes | If equal to main, target-value bonus is unnecessary; if main only wins here, watch for material shortcuts. |
| `pure_topology` | Remove CNN branch and train only topology MLP | Tests standalone signal and shortcut risk | If pure topology is strong but ablations are also strong, it may be exploiting material/king-distance artifacts. |
| `random_labels_smoke_test` | Train with shuffled training targets for one debug run | Tests for implementation leakage or report bugs | Non-chance performance indicates a serious pipeline leak. |

The central semantics-destroying randomized ablation is `rank_shuffle`. It preserves candidate count, rank budgets, top-k mean pressure values, side-to-move handling, piece-weighted pressure distributions, and class balance while destroying the proposed square-adjacency semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against.

- Existing `simple_18` simple CNN.
- Existing `simple_18` residual CNN of comparable size.
- Existing small/medium/deep variants if already included in the benchmark table.
- Existing LC0-style CNN/residual only as secondary context; the primary fair comparison is `simple_18`.
- `ThreatTopologyNet` with `no_topology_fusion`.
- `ThreatTopologyNet` with `rank_shuffle`.
- `ThreatTopologyNet` with `histogram_only`.
- `pressure_fields_only_cnn`.

Metrics to inspect.

- Validation and test accuracy.
- Balanced accuracy.
- Macro F1.
- AUROC if the reporting stack already supports probabilities.
- Average precision / PR-AUC if available.
- Cross-entropy / NLL.
- Calibration summary if already available, preferably ECE.
- Required `3x2` fine-label diagnostic matrix for every main and central ablation run.

Required fine-label confusion.

For the main model and every central ablation, report:

```text
true fine label 0 -> predicted 0 / predicted 1
true fine label 1 -> predicted 0 / predicted 1
true fine label 2 -> predicted 0 / predicted 1
```

Near-puzzle diagnostic.

At a matched fine-label-`0` false-positive rate, compare fine-label-`1` recall. Use either:

- the false-positive rate achieved by the best existing simple CNN baseline, or
- a fixed `5%` fine-label-`0` false-positive rate if threshold-sweep code already exists.

Also report fine-label-`2` recall at the same threshold to ensure the model is not helping near-puzzles by sacrificing verified puzzles.

Required artifacts.

- Saved config for main and ablation runs.
- Checkpoint or final model state if the repo normally saves it.
- Validation and test metrics JSON.
- Prediction file with row id, fine label, coarse target, predicted class, and positive probability/logit.
- `3x2` diagnostic matrices for main and central ablations.
- A short report comparing main, no-topology, rank-shuffle, histogram-only, and pressure-CNN controls.
- Leaderboard update if the repo has an existing leaderboard script.

Success threshold.

Treat the idea as successful enough to scale only if all of the following hold on validation and mostly repeat on test:

- Main model improves over `no_topology_fusion` by at least `0.5` absolute percentage points in balanced accuracy or AUROC.
- Main model improves fine-label-`1` recall by at least `2.0` absolute percentage points at matched fine-label-`0` false-positive rate.
- `rank_shuffle` loses at least half of the main model’s gain over `no_topology_fusion`.
- `histogram_only` does not match the main model.
- No severe degradation of fine-label-`2` recall or calibration relative to the matched CNN baseline.

Failure threshold.

Treat the idea as failed if any of the following occur:

- Main model is within `0.2` percentage points of `no_topology_fusion` and the best simple CNN on the main metric.
- `rank_shuffle` or `histogram_only` matches or beats the main model within run noise.
- Fine-label-`1` recall does not improve at matched fine-label-`0` false-positive rate.
- The topology branch causes training instability, nondeterministic reports, or obvious leakage symptoms.

What result would make you abandon the idea.

Abandon this family if `rank_shuffle`, `degree_class_square_permutation`, and `histogram_only` all match the main model while `pressure_fields_only_cnn` performs similarly. That would mean the useful signal is pressure-map preprocessing or scalar distributions, not rank-cubical topology.

What result would justify scaling.

Scale only after a clean current-data win with central ablations. Reasonable next scaling would be longer training on the same split, LC0 current-board adapter support with fail-closed semantics, and then streaming/full-data experiments after the trainer supports streaming.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_threat_topology/idea.yaml` | Create | Machine-readable idea metadata, status, input representation, central ablation, and latest result placeholders. |
| `ideas/20260421_threat_topology/math_thesis.md` | Create | Section 6 math thesis, proposition, proof sketch, hypotheses, and counterexamples. |
| `ideas/20260421_threat_topology/architecture.md` | Create | Module contracts, tensor shapes, topology encoder details, parameter estimates, and pseudocode. |
| `ideas/20260421_threat_topology/implementation_notes.md` | Create | Adapter fail-closed rules, deterministic top-k tie-breaking, pressure-generator rules, and component-count test cases. |
| `ideas/20260421_threat_topology/trainer_notes.md` | Create | Loss, config defaults, fairness constraints, and report requirements. |
| `ideas/20260421_threat_topology/ablations.md` | Create | Ablation table from Section 9, with command/config names. |
| `ideas/20260421_threat_topology/train.py` | Create | Thin wrapper or documented entrypoint invoking the shared trainer with `configs/threat_topology_simple18.yaml`; no custom trainer fork unless necessary. |
| `ideas/20260421_threat_topology/config.yaml` | Create | Local copy of the run config for this idea. |
| `ideas/20260421_threat_topology/report_template.md` | Create | Required metrics, confusion matrices, near-puzzle diagnostic, and success/failure checklist. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Preserve hard constraints; add this packet to imported memory after implementation; add anti-duplicate note for rank-cubical pressure-field Betti topology if it fails or becomes represented. |
| `src/chess_nn_playground/models/threat_topology_net.py` | Create | `ThreatTopologyNet`, adapters, `RulePressureFields`, `RankCubicalBettiEncoder`, and ablation switches. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register builder function, e.g. `build_threat_topology_net`, under model name `threat_topology_net`. |
| `configs/threat_topology_simple18.yaml` | Create | Benchmark config using `simple_18`, `batch_size: 512`, `epochs: 3`, balanced class weighting, and model fields listed above. |
| `tests/test_threat_topology_net.py` | Create | Focused tests for output shape, fail-closed unknown encoding behavior, top-k invariance under monotone transform, and histogram counterexample with different `beta0`. |
| `tests/test_rule_pressure_fields.py` | Create | Focused tests for pawn, knight, king, and slider pressure on small synthetic boards; ensure no legal-move/checkmate oracle is called. |
| `tests/test_rank_cubical_betti_encoder.py` | Create | Test `2x2` block has `beta0=1`, four separated corners have higher `beta0`, ring mask has `beta1=1`, and top-k tie-breaking is deterministic. |

Codex prompt update after consuming this output.

After implementation and benchmarking, Codex should update `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` with the result. If the idea fails, add an anti-duplicate rule such as: “Do not repeat rank-cubical Betti/persistent-topology bottlenecks over static pressure maps unless the new operator is not a pressure-field filtration and has a different central falsifier.” If it succeeds, add it to imported research memory with its exact fingerprint and central ablations so the next research pass can propose a genuinely different mechanism.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0814_tuesday_los_angeles_threat_topology.md
  generated_at: 2026-04-21T08:14:00-07:00
  weekday: Tuesday
  timezone: los_angeles
  idea_slug: threat_topology
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_threat_topology
  name: Threat-Topology Betti Bottleneck Network
  slug: threat_topology
  status: draft
  created_at: 2026-04-21T08:14:00-07:00
  author: ChatGPT Pro
  short_thesis: Rank-cubical Betti curves of rule-only pressure fields may capture puzzle-like tactical geometry beyond pressure histograms and CNN texture.
  novelty_claim: Uses top-k cubical topology of current-board pressure surplus fields, not sheaves, Hodge operators, move deltas, transport, ordinal heads, pseudo-likelihood, or symmetry quotients.
  expected_advantage: Better fine-label-1 recall at matched fine-label-0 false-positive rate if near-puzzles contain coherent pressure topology.
  central_falsification_ablation: rank_shuffle preserving sorted pressure values and top-k counts while destroying square adjacency before Betti computation
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112 and lc0_bt4_112 only with explicit current-piece channel mapping
  output_heads: binary logits [batch, 2]
  compute_notes: About 115k-130k parameters for simple_18; topology masks use B*4*10*64 booleans and can be chunked by field or rank budget.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/threat_topology_simple18.yaml
  model_path: src/chess_nn_playground/models/threat_topology_net.py
  latest_result_path: null
  notes: Fail closed if piece-plane semantics are unknown; no legal move generation, engine analysis, source metadata, or label-derived inputs.
```

```yaml
config_yaml:
  run:
    name: threat_topology_simple18
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
    name: threat_topology_net
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
  model_name: threat_topology_net
  file_path: src/chess_nn_playground/models/threat_topology_net.py
  builder_function: build_threat_topology_net
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18PiecePlaneAdapter
    - Lc0CurrentPiecePlaneAdapter
    - RulePressureFields
    - RankCubicalBettiEncoder
    - ThreatTopologyBranch
    - MatchedBoardCnnStem
    - ThreatTopologyNet
  required_config_fields:
    - input_channels
    - num_classes
    - piece_plane_order
    - side_to_move_channel
    - rank_ks
    - pressure_alpha
    - pressure_piece_weights
    - target_piece_values
    - topology_ablation
  expected_parameter_count: 115k-130k for simple_18; 145k-160k for lc0_* with 112 input channels
  expected_memory_notes: Topology masks require B*4*len(rank_ks)*64 booleans; for B=512 and 10 ranks this is about 1.3M mask entries before temporary labels. Chunk by pressure field or rank budget if needed.
```

```yaml
research_continuity:
  idea_fingerprint: current-board pseudo-legal pressure surplus scalar fields + rank-top-k cubical Betti-0/Betti-1/boundary curves + MLP fusion + rank-shuffle topology falsifier
  already_researched_family_overlap: Uses safe pseudo-legal attack pressure as scalar image construction, but does not build attack-defense graphs, sheaves, Hodge Laplacians, move-delta bags, Sinkhorn transport, pseudo-likelihoods, ordinal/credal heads, sparse witnesses, ray automata, orbit quotients, tempo interventions, or masked codecs.
  closest_duplicate_risk: Static attack-defense graph family, because pressure fields are derived from attacks; distinction is that the learned object is a cubical filtration on square scalar fields with histogram-preserving topology ablations, not a piece/square incidence graph or sheaf operator.
  do_not_repeat_if_this_fails:
    - rank-cubical Betti curves over static attack-pressure fields
    - persistent-topology bottlenecks over side-to-move pressure surplus maps with only different thresholds or weights
    - pressure-field topology fused with a larger CNN as the only change
    - topology variants whose central ablation is still just rank-shuffle or histogram-only on the same fields
  suggested_next_search_directions:
    - label-safe selective prediction that is not credal/ordinal and has a deployment-calibration falsifier
    - source-free information bottleneck with non-provenance environments and strong no-tactics-suppression tests
    - generative motif compression not based on masked-board likelihood or class-conditioned pseudo-likelihood
    - causal invariance only if genuine allowed environment metadata exists and is not used as neural input
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Threat-Topology Betti Bottleneck Network` to imported research memory after implementation, with its fingerprint and central ablations. | Prevents the next research loop from proposing the same rank-cubical pressure topology with renamed thresholds. | `Imported Research Memory` |
| If the result fails, add an anti-duplicate rule against Betti/persistent-topology bottlenecks over static pressure maps unless the formal observable is not a rank filtration of pressure fields. | Avoids wasting cycles on topological variants that only change `k` values, pressure weights, or fusion MLP size. | Anti-duplicate paragraphs after static attack-defense graph restrictions |
| Add a reusable requirement for topology-like ideas: include histogram-preserving, geometry-destroying, and degree/boundary-marginal-preserving randomized controls. | This packet’s main lesson is that topology claims need stronger controls than “remove the branch.” | `Ablation Plan` requirements |
| Add fail-closed adapter language for LC0-style encodings whenever deterministic rule features require channel semantics. | Prevents silent misuse of 112-plane history/current channels as current-board geometry. | `Project Context` or `Problem Restatement And Data Contract` |
| Record whether class-1 recall at matched class-0 FPR was informative. | Near-puzzle behavior is a key diagnostic; preserving this lesson helps future ideas target ambiguity rather than only obvious puzzles. | `Benchmark And Falsification Criteria` |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0814_tuesday_los_angeles_threat_topology.md`
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
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Mobius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
