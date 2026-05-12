# Codex Research Batch: Additional Architecture Ideas 10

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`
- Generated at: 2026-04-24 22:13
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: architecture batch, not implemented

## Purpose

This batch adds seven more neural architecture ideas. The archive already contains many unusual mathematical operators and several practical CNN baselines, so this packet targets a middle region:

- still implementable in PyTorch without external solvers
- not just width/depth tuning
- no engine features
- no move search
- no direct repeat of the recent practical batches

The new computation shapes:

- capsule-style motif binding
- multiple deterministic board scan orders
- cross-stitch fusion between board and piece branches
- differentiable decision forests
- vector-quantized latent codebooks
- hypercolumn square readouts
- multiplicative convolutional conjunctions

## Shared Data Contract

Task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- train on binary labels only unless a candidate explicitly describes an auxiliary supervised fine-label head
- keep fine labels `0`, `1`, and `2` for diagnostics
- always report the fine-label `3 x 2` diagnostic matrix

First implementation:

- input: `simple_18`
- dataset: existing `crtk_sample_3class` splits
- trainer: shared experimental training pipeline

Forbidden model inputs:

- Stockfish scores, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, unresolved candidate status, dataset provenance, or anything derived from them.
- Engine search, legal move search, forced-line search, mate/stalemate oracles, tablebase outcomes, or future game outcomes.

Allowed model inputs:

- Current-board tensor.
- Side-to-move, castling, and en-passant planes already included in `simple_18`.
- Deterministic coordinate planes.
- Safe material/count summaries where explicitly listed.

## Ranked Shortlist

| Rank | Candidate | Main object | Why it is useful |
|---|---|---|---|
| 1 | Capsule Motif BoardNet | Local motif capsules with routing-by-agreement | Tests explicit motif binding without attention or graph construction. |
| 2 | Multi-Order Board Scan Network | Shared sequence model over several fixed square orders | Tests order-sensitive global context without move generation. |
| 3 | Cross-Stitch CNN-Token Fusion Net | CNN and occupied-token branches coupled at intermediate layers | Stronger fusion than late concat, still practical. |
| 4 | Neural Decision Forest BoardNet | CNN features routed through differentiable oblique trees | Tests piecewise decision boundaries and interpretable splits. |
| 5 | Vector-Quantized Motif Codebook Net | Board latents quantized into learned codebook entries | Tests discrete motif inventories and code usage diagnostics. |
| 6 | Hypercolumn Square Readout CNN | Per-square features from all depths pooled through square heads | Tests whether intermediate features carry useful square evidence. |
| 7 | Multiplicative Conjunction ConvNet | Product-gated conv branches for local feature conjunctions | Tests tactical AND-like feature interactions in a plain conv net. |

Best next full packet from this batch:

```text
Capsule Motif BoardNet
```

Reason: it is distinct from the current archive, has interpretable diagnostics, and can be implemented without new data infrastructure.

## Candidate 1: Capsule Motif BoardNet

### Thesis

Local chess motifs are not only scalar activations; they have type, pose, orientation, and part-whole relationships. A capsule-style model can encode local patterns as small vectors and route them into higher-level tactical motif capsules by agreement.

### Fingerprint

```text
simple_18
+ local conv capsule stem
+ primary square capsules
+ motif capsules
+ routing-by-agreement
+ capsule norm/readout head
```

### Why It Is Distinct

- Not attention: routing weights are iterative capsule agreement, not query-key softmax over values.
- Not prototype patch dictionary: capsules carry vector pose and agreement, not only nearest prototype scores.
- Not graph/sheaf: no explicit attack graph or relation complex.
- Not piece-token hybrid: tokens are local feature capsules, not only occupied pieces.

### Architecture Sketch

Stem:

```text
h = ConvStack(simple_18 + coords)
```

Primary capsules:

```text
primary_caps = Conv2d(h, C_primary * D_caps, kernel_size=3, padding=1)
primary_caps -> (B, N_caps, D_caps)
```

Recommended:

```text
N_caps = 8 x 8 x C_primary
C_primary = 8
D_caps = 8 or 16
```

Motif capsules:

```text
M = 16 or 32
D_motif = 16
```

Each primary capsule predicts each motif capsule:

```text
u_hat[i, m] = W_m u_i
```

Run `T=3` routing steps:

```text
c_i_m = softmax_m(b_i_m)
s_m = sum_i c_i_m * u_hat[i, m]
v_m = squash(s_m)
b_i_m += dot(u_hat[i, m], v_m)
```

Readout:

```text
motif_norms = norm(v_m)
motif_vectors = flatten(v_m)
logits = MLP([motif_norms, pooled_board_features])
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `no_routing_mean_pool` | Replace routing with mean pooled capsule predictions | Agreement routing matters | If equal, routing is unnecessary. |
| `scalar_capsules` | Use scalar capsule activations only | Vector pose matters | If equal, capsule vectors are overkill. |
| `random_motif_transforms` | Freeze random `W_m` transforms | Learned motif transforms matter | If equal, capsule head is generic capacity. |
| `cnn_matched_params` | Plain CNN with matched parameter count | Capsules beat regular conv capacity | If equal, use CNN. |
| `one_routing_step` | Use one routing step | Iterative agreement matters | If equal, routing can be simplified. |

### Diagnostics

- Motif capsule norms by fine label.
- Routing entropy by motif.
- Top activating board positions for each motif capsule.
- Whether fine-label `1` has higher routing ambiguity than fine-label `0`.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/capsule_motif_boardnet.py
tests/test_capsule_motif_boardnet.py
configs/bench_capsule_motif_boardnet_simple18.yaml
configs/bench_capsule_motif_boardnet_no_routing.yaml
```

Keep the first version small. Capsule routing can become expensive if every capsule predicts every motif with large matrices.

## Candidate 2: Multi-Order Board Scan Network

### Thesis

A chess board can be read as several short sequences. Different scan orders expose different dependencies: rank-major order, file-major order, diagonal order, spiral-from-king order, and center-out order. A shared sequence model over fixed board orders can provide global context without attention or move generation.

### Fingerprint

```text
square embeddings
+ multiple deterministic square orders
+ shared GRU/SSM/1D conv scanner
+ order-wise pooled states
+ binary head
```

### Why It Is Distinct

- Not ray state-space scans: this scans the full board in several global orders, not only individual chess lines.
- Not attention: no token-to-token softmax.
- Not patch mixer: sequential state carries order-sensitive context.
- Not move-delta: no legal moves or future boards.

### Architecture Sketch

Embed squares:

```text
X: (B, 64, D)
```

Build fixed orders:

```text
rank_major
file_major
diag_sweep
anti_diag_sweep
center_out
king_relative_spiral
```

For each order `o`:

```text
X_o = gather(X, order_o)
H_o = SharedScanner(X_o)
z_o = pool(H_o)
```

Scanner options:

- small bidirectional GRU
- depthwise separable 1D conv stack
- simple gated recurrent unit

First implementation should use a small bidirectional GRU or 1D conv stack, whichever fits existing dependencies best.

Fuse:

```text
z = concat(z_order_1, ..., z_order_K, mean_square_pool)
logits = MLP(z)
```

Default config:

```yaml
model:
  name: multi_order_board_scan_network
  square_dim: 48
  scanner_hidden: 64
  scanner_type: bigru
  orders: [rank_major, file_major, diag_sweep, center_out]
  dropout: 0.1
```

Expected parameter range:

```text
250k-700k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `rank_major_only` | Use one order only | Multi-order views matter | If equal, extra orders unnecessary. |
| `random_orders` | Replace chess-shaped orders with fixed random permutations | Order semantics matter | If equal, ordering is generic sequence capacity. |
| `no_recurrence_1d_mlp` | Replace scanner with per-token MLP and pool | Sequential state matters | If equal, sequence model unnecessary. |
| `untied_order_scanners` | Separate scanner per order | Weight sharing matters | If untied wins, order-specific parameters help. |
| `cnn_matched_params` | Plain CNN matched params | Scans beat conv baseline | If equal, use CNN. |

### Diagnostics

- Per-order pooled norm.
- Ablation delta by order.
- Order agreement and disagreement.
- Whether king-relative order helps king-heavy positions.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/multi_order_board_scan.py
tests/test_multi_order_board_scan.py
configs/bench_multi_order_board_scan_simple18.yaml
configs/bench_multi_order_board_scan_random_orders.yaml
```

All orders must be fixed buffers or deterministic index tensors. Do not generate legal moves.

## Candidate 3: Cross-Stitch CNN-Token Fusion Net

### Thesis

Late fusion between a CNN branch and a piece-token branch may be too weak. A cross-stitch network can let the branches exchange information at multiple depths through learned linear mixing, while still keeping the model practical.

### Fingerprint

```text
CNN board branch
+ occupied-piece token branch
+ cross-stitch units at several depths
+ final fused head
```

### Why It Is Distinct

- Not the Piece-Token CNN Hybrid parent: fusion happens throughout the network, not only late concat.
- Not attention: cross-stitch coefficients are small learned branch-mixing matrices, not token routing.
- Not commutative view consistency: no residual map-defect objective.

### Architecture Sketch

Branches:

```text
h_board_t: (B, C, 8, 8)
h_token_t: (B, Pmax, D)
```

At each cross-stitch stage, summarize each branch:

```text
b = board_pool(h_board_t)
p = token_pool(h_token_t)
```

Exchange information:

```text
[b_new, p_new] = A_t [b, p]
```

where:

```text
A_t in R^{2 x 2}
```

or per-channel grouped:

```text
A_t in R^{G x 2 x 2}
```

Inject back:

```text
h_board_t += board_adapter(b_new)
h_token_t += token_adapter(p_new)
```

Final:

```text
logits = MLP([board_pool, token_pool, cross_stitch_diagnostics])
```

Default config:

```yaml
model:
  name: cross_stitch_cnn_token_fusion
  board_width: 64
  token_width: 64
  stages: 3
  cross_stitch_groups: 8
  dropout: 0.1
```

Expected parameter range:

```text
400k-900k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `late_fusion_only` | Disable intermediate cross-stitch, keep final concat | Cross-depth fusion matters | If equal, parent hybrid is enough. |
| `board_only` | Remove token branch | Tokens matter | If equal, board branch dominates. |
| `token_only` | Remove board branch | CNN maps matter | If equal, token branch dominates. |
| `diagonal_stitch` | Cross-stitch matrix forced diagonal | Cross-branch exchange matters | If equal, exchange unnecessary. |
| `random_token_coords` | Shuffle token square coordinates | Token geometry matters | Should degrade if token branch is real. |

### Diagnostics

- Learned cross-stitch matrices by stage.
- Amount of board-to-token versus token-to-board transfer.
- Branch norm before and after stitching.
- Fine-label diagnostics for late fusion versus cross-stitch.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/cross_stitch_cnn_token_fusion.py
tests/test_cross_stitch_cnn_token_fusion.py
configs/bench_cross_stitch_cnn_token_fusion_simple18.yaml
configs/bench_cross_stitch_cnn_token_fusion_late_only.yaml
```

Reuse any existing occupied-token extraction utilities if implemented. If not, start with a fail-closed simple token adapter for `simple_18`.

## Candidate 4: Neural Decision Forest BoardNet

### Thesis

Chess puzzle-likeness may be piecewise: different board regimes require different cues. A differentiable decision forest on top of a CNN feature vector can model soft oblique splits and leaf predictors without a sparse expert router.

### Fingerprint

```text
CNN trunk
+ differentiable oblique decision trees
+ leaf logits
+ path probability readout
```

### Architecture Sketch

Trunk:

```text
z = pool(CNN(x))
```

Each tree has internal nodes:

```text
p_node = sigmoid(w_node dot z + b_node)
```

Path probability for each leaf:

```text
pi_leaf = product of branch probabilities along path
```

Leaf logits:

```text
logits_tree = sum_leaf pi_leaf * leaf_logit_leaf
```

Forest:

```text
logits = mean_tree logits_tree
```

Default config:

```yaml
model:
  name: neural_decision_forest_boardnet
  trunk_width: 64
  trunk_depth: 4
  num_trees: 8
  tree_depth: 4
  dropout: 0.1
```

Expected parameter range:

```text
250k-800k
```

### Why It Is Distinct

- Not sparse expert routing: all trees contribute softly.
- Not submodular coverage: no diminishing-return set function.
- Not ordinary MLP head: the head is constrained to piecewise tree paths.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `mlp_head_matched` | Replace forest with matched MLP head | Tree structure matters | If equal, use MLP. |
| `single_tree` | Use one tree | Forest diversity matters | If equal, simplify. |
| `depth2_trees` | Shallower trees | Decision depth matters | If equal, shallow splits enough. |
| `hard_leaf_stopgrad` | Use hard routing in eval/training variant | Soft routing matters | If hard wins, soft forest may be too smooth. |
| `random_trunk_forest` | Freeze random trunk, train forest | Learned board features matter | If strong, suspect shortcuts. |

### Diagnostics

- Leaf usage entropy.
- Per-tree disagreement.
- Dominant leaves by fine label.
- Split feature norms.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/neural_decision_forest_boardnet.py
tests/test_neural_decision_forest_boardnet.py
configs/bench_neural_decision_forest_boardnet_simple18.yaml
configs/bench_neural_decision_forest_boardnet_mlp_head.yaml
```

Keep tree depth modest. A depth-4 binary tree has `16` leaves; with `8` trees this is already enough.

## Candidate 5: Vector-Quantized Motif Codebook Net

### Thesis

Force local board features to pass through a learned discrete codebook. The classifier reads code usage, spatial code maps, and quantized features. This tests whether a compact inventory of board motifs is useful for puzzle-likeness.

### Fingerprint

```text
CNN encoder
+ vector quantization codebook
+ code usage histogram
+ quantized spatial map
+ classifier head
```

### Why It Is Distinct

- Not masked codec surprise: no masked pretraining or code-length objective.
- Not prototype patch dictionary: quantization is inside the feature map and trained end-to-end.
- Not TinyChessMicroNet: no hard tiny parameter target.

### Architecture Sketch

Encoder:

```text
h = CNN(x)  # (B, D, 8, 8)
```

For each square feature `h_s`, find nearest code:

```text
k_s = argmin_k ||h_s - e_k||^2
q_s = e_{k_s}
```

Use straight-through estimator:

```text
q_st = h + stopgrad(q - h)
```

Readout:

```text
code_histogram
quantized_map_pool
commitment_loss
classifier_head(q_st, histogram)
```

Default config:

```yaml
model:
  name: vq_motif_codebook_net
  encoder_width: 64
  code_dim: 32
  num_codes: 64
  commitment_weight: 0.1
  dropout: 0.1
```

Expected parameter range:

```text
300k-900k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `no_quantization` | Use continuous encoder features | Codebook bottleneck matters | If equal, VQ unnecessary. |
| `random_codebook` | Freeze random codes | Learned motif inventory matters | If equal, codes are generic bins. |
| `histogram_only` | Classify from code histogram only | Spatial code map matters | If equal, spatial arrangement unnecessary. |
| `map_only_no_hist` | Remove histogram | Global motif counts matter | If equal, map features enough. |
| `small_codebook` | Use 16 codes | Codebook capacity matters | If equal, smaller inventory works. |

### Diagnostics

- Code usage entropy.
- Dead code count.
- Top codes by fine label.
- Code maps for validation examples.
- Commitment loss curve.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/vq_motif_codebook_net.py
tests/test_vq_motif_codebook_net.py
configs/bench_vq_motif_codebook_net_simple18.yaml
configs/bench_vq_motif_codebook_net_no_quantization.yaml
```

Watch for code collapse. If many codes are dead, add entropy regularization or exponential moving average code updates only after the first simple run.

## Candidate 6: Hypercolumn Square Readout CNN

### Thesis

Intermediate CNN layers may detect different chess cues: early local piece contacts, middle motifs, and later global context. A hypercolumn readout gathers per-square features from every depth and classifies from square-level evidence maps plus global pooling.

### Fingerprint

```text
CNN trunk with saved layer outputs
+ per-square hypercolumns
+ square evidence head
+ global evidence aggregation
```

### Architecture Sketch

Run a CNN trunk:

```text
h1, h2, h3, h4 = trunk_layers(x)
```

Project each depth to a common width:

```text
p_t = Conv1x1(h_t -> D)
```

Hypercolumn:

```text
H_square = concat(p1, p2, p3, p4)  # (B, 4D, 8, 8)
```

Square evidence:

```text
e = Conv1x1(H_square -> E)
square_logits = Conv1x1(e -> 2)
```

Global head:

```text
z = concat(mean_pool(e), max_pool(e), topk_pool(square_logits))
logits = MLP(z)
```

Default config:

```yaml
model:
  name: hypercolumn_square_readout_cnn
  trunk_width: 64
  trunk_depth: 4
  hyper_width: 32
  evidence_width: 32
  dropout: 0.1
```

Expected parameter range:

```text
250k-800k
```

### Why It Is Distinct

- Not specialist-head CNN: heads are per-square hypercolumns, not region specialists.
- Not FPN: no top-down pyramid fusion.
- Not attention perturbation: evidence maps are direct readouts, not mask policies.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `last_layer_only` | Use only final trunk layer | Hypercolumns matter | If equal, intermediate features unnecessary. |
| `no_square_logits` | Remove square evidence map, pool hypercolumns directly | Square evidence helps | If equal, evidence map is decorative. |
| `mean_pool_only` | Remove max/top-k pools | Sparse strong squares matter | If equal, mean context enough. |
| `cnn_head_matched` | Ordinary pooled CNN head matched params | Hypercolumn readout matters | If equal, use simpler head. |
| `random_layer_order` | Shuffle depth order before projection | Layer depth semantics matter | If equal, projections just add capacity. |

### Diagnostics

- Square evidence heatmaps.
- Layer projection norms.
- Top-k evidence squares by fine label.
- Whether early or late hypercolumn channels dominate.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/hypercolumn_square_readout_cnn.py
tests/test_hypercolumn_square_readout_cnn.py
configs/bench_hypercolumn_square_readout_cnn_simple18.yaml
configs/bench_hypercolumn_square_readout_cnn_last_only.yaml
```

This should be straightforward if the trunk is written as a list of blocks.

## Candidate 7: Multiplicative Conjunction ConvNet

### Thesis

Many chess motifs are conjunctions: attacker plus target plus blocker absence, king exposure plus line pressure, or material cue plus square pattern. A conv net with explicit multiplicative gates can represent local AND-like interactions more directly than additive conv stacks.

### Fingerprint

```text
plain CNN stem
+ paired conv branches
+ product gates
+ residual fusion
+ pooled classifier
```

### Architecture Sketch

Block:

```text
a = Conv3x3_A(h)
b = Conv3x3_B(h)
g = sigmoid(Conv1x1_G(h))
product = a * b
y = Conv1x1(concat(a, b, product, g * a))
h = h + y
```

Optional low-rank product:

```text
a, b have branch_width < width
product projected back to width
```

Default config:

```yaml
model:
  name: multiplicative_conjunction_convnet
  width: 64
  depth: 5
  branch_width: 32
  dropout: 0.1
  use_coordinate_planes: true
```

Expected parameter range:

```text
300k-900k
```

### Why It Is Distinct

- Not tropical circuit: no explicit min-plus clauses.
- Not Boolean bitboard network: operations happen on learned conv features, not soft bitboard predicates.
- Not ordinary gated CNN: product features are explicit readout channels, not only sigmoid scaling.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `additive_only` | Replace `a * b` with `a + b` | Multiplicative conjunction matters | If equal, product unnecessary. |
| `gate_only_no_product` | Use sigmoid gates but no product feature | Product adds beyond gating | If equal, ordinary gates enough. |
| `single_branch_matched` | One wider conv branch matched params | Paired feature factors matter | If equal, branch split unnecessary. |
| `late_product_only` | Add product only in final head | Local products matter | If equal, products need not be in blocks. |
| `cnn_matched_params` | Plain CNN matched params | Conjunction blocks beat conv capacity | If equal, use CNN. |

### Diagnostics

- Product branch norm by layer.
- Gate saturation statistics.
- Performance delta on `additive_only`.
- Calibration difference versus plain CNN.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/multiplicative_conjunction_convnet.py
tests/test_multiplicative_conjunction_convnet.py
configs/bench_multiplicative_conjunction_convnet_simple18.yaml
configs/bench_multiplicative_conjunction_convnet_additive_only.yaml
```

Use normalization after product fusion, not before every tiny operation. Keep the first model stable and easy to compare.

## Implementation Queue

Recommended order:

1. `Capsule Motif BoardNet`
2. `Hypercolumn Square Readout CNN`
3. `Multiplicative Conjunction ConvNet`
4. `Cross-Stitch CNN-Token Fusion Net`
5. `Multi-Order Board Scan Network`
6. `Neural Decision Forest BoardNet`
7. `Vector-Quantized Motif Codebook Net`

Reasoning:

- Capsules and hypercolumns provide clear diagnostics.
- Multiplicative convs are simple to implement and compare.
- Cross-stitch fusion is useful if token utilities exist.
- Multi-order scans are straightforward but need order-index care.
- Neural forests are easy but may be head-only gains.
- VQ codebooks are interesting but can collapse and need more monitoring.

## Shared Benchmark Rules

For every candidate:

- use the same splits
- use the same coarse binary target
- report the same fine-label diagnostics
- compare against simple CNN, residual CNN, and strongest practical baseline available
- include the central ablation before claiming any gain

Metrics:

- AUROC
- accuracy
- balanced accuracy
- F1
- calibration
- parameter count
- inference latency
- fine-label `3 x 2` diagnostic matrix
- class-1 recall at matched fine-label-0 false-positive rate if available

## Anti-Duplicate Rules

Do not repeat these later with only small parameter changes:

| Family | Avoid Near-Duplicate |
|---|---|
| Capsule Motif BoardNet | Another capsule-routing motif model unless routing or capsule semantics change materially. |
| Multi-Order Board Scan | Another board-scan sequence model unless orders or scanner objective change. |
| Cross-Stitch Fusion | Another CNN-token fusion model unless fusion happens differently than cross-stitch exchange. |
| Neural Decision Forest | Another differentiable forest head with only depth/tree count changes. |
| VQ Motif Codebook | Another vector-quantized board codebook unless the quantization objective changes. |
| Hypercolumn Readout | Another multi-depth square readout unless square evidence aggregation changes. |
| Multiplicative Conjunction ConvNet | Another product-gated conv net with only branch-width changes. |

## Continuity Note

This batch should be treated as an exploration set. The strongest candidates are not necessarily the strangest ones. If one plain-ish architecture beats the exotic packets, future research should compare against it before scaling more complicated operators.
