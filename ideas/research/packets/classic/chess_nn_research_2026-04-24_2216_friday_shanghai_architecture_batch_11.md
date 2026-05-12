# Codex Research Batch: Additional Architecture Ideas 11

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`
- Generated at: 2026-04-24 22:16
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: architecture batch, not implemented

## Purpose

This batch adds eight more neural architecture ideas. It deliberately avoids repeating the most recent packets:

- no capsule routing repeat
- no multi-order scan repeat
- no cross-stitch CNN-token fusion repeat
- no neural forest repeat
- no VQ codebook repeat
- no hypercolumn square readout repeat
- no multiplicative convolution repeat
- no ConvNeXt/FPN/patch-mixer repeat

The new shapes are:

- explicit empty-square opportunity modeling
- global scratchpad memory
- learnable fixed pooling trees
- coordinate-conditioned spatial modulation
- channel-bilinear role interaction heads
- feature sieving stages
- ring/shell recurrent summaries
- rank/file memory grids

These are research candidates, not benchmark results.

## Shared Data Contract

Task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- train on binary labels only unless a candidate explicitly lists an auxiliary loss
- keep fine labels `0`, `1`, and `2` for diagnostics
- report the fine-label `3 x 2` diagnostic matrix

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
- Safe current-board material/count summaries where explicitly listed.
- Deterministic current occupancy and empty-square masks.

## Ranked Shortlist

| Rank | Candidate | Main object | Why it is useful |
|---|---|---|---|
| 1 | Empty-Square Opportunity Network | Separate occupied-square and empty-square branches | Tests whether empty destination/escape/interference squares carry signal. |
| 2 | Global Scratchpad BoardNet | Small recurrent global memory broadcast back to board features | Cheap global context without attention or pair fields. |
| 3 | Learnable Pooling Tree BoardNet | Fixed square hierarchy with learned aggregation nodes | Structured board summarization without FPN upsampling. |
| 4 | Spatial FiLM Coordinate Net | Coordinate-conditioned per-square affine modulation | Tests whether location-specific feature modulation beats plain coord planes. |
| 5 | Channel-Bilinear Role Mixer | Low-rank bilinear interactions between pooled feature groups | Captures feature conjunctions at the head without local product blocks. |
| 6 | Evidence Sieve Network | Sequential feature masks that filter evidence before classification | Tests staged feature filtering rather than staged logits. |
| 7 | Ring-Shell Recurrent BoardNet | GRU over king/center shell summaries | Compact radial context around important anchors. |
| 8 | Rank-File Memory Grid Net | Learned rank and file memory vectors coupled to square features | Global line communication without axial conv or Schur solves. |

Best next full packet from this batch:

```text
Empty-Square Opportunity Network
```

Reason: many chess tactics are about empty squares, escape squares, landing squares, interference squares, and holes. Most simple board models over-focus on occupied pieces. This idea has a sharp ablation: remove empty-square modeling.

## Candidate 1: Empty-Square Opportunity Network

### Thesis

Chess tactics often depend on empty squares: escape squares, mating squares, promotion paths, discovered-attack landing squares, fork squares, and blocking/interference squares. A classifier that separately models occupied-square evidence and empty-square opportunity may capture useful signal missed by ordinary CNN pooling.

### Fingerprint

```text
simple_18
+ occupied mask branch
+ empty mask branch
+ opportunity field head
+ occupied/empty interaction fusion
+ binary logits
```

### Why It Is Distinct

- Not sparse witness: the branch is not selecting a few occupied pieces.
- Not move-delta: no legal destination enumeration or future board state.
- Not relational query algebra: no joins or database executor.
- Not generic CNN: empty-square field is an explicit branch and ablation target.

### Architecture Sketch

Build masks from the input:

```text
occ_mask:   (B, 1, 8, 8)
empty_mask: 1 - occ_mask
```

Shared trunk:

```text
h = CNNStem(x + coords)
```

Occupied branch:

```text
h_occ = ConvStack(h * occ_mask)
z_occ = mean/max/topk_pool(h_occ, mask=occ_mask)
```

Empty branch:

```text
h_empty = ConvStack(h * empty_mask)
opportunity = Conv1x1(h_empty -> K)
z_empty = mean/max/topk_pool(opportunity, mask=empty_mask)
```

Interaction:

```text
z_pair = [
  z_occ,
  z_empty,
  z_occ * z_empty,
  abs(z_occ - z_empty)
]
logits = MLP(z_pair)
```

Optional maps:

```text
opportunity channels:
  escape_like
  landing_like
  blocker_like
  promotion_lane_like
  king_zone_empty_like
```

These are learned names for diagnostics only, not supervised labels.

Default config:

```yaml
model:
  name: empty_square_opportunity_network
  trunk_width: 64
  branch_width: 48
  opportunity_channels: 8
  depth: 4
  dropout: 0.1
  use_coordinate_planes: true
```

Expected parameter range:

```text
300k-900k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `occupied_only` | Remove empty branch | Empty-square opportunity matters | If equal, branch unnecessary. |
| `empty_only` | Remove occupied branch | Occupied pieces still matter | If strong, empty branch may exploit occupancy shortcuts. |
| `random_empty_mask` | Random same-density empty mask | Real empty squares matter | If equal, mask semantics weak. |
| `no_occ_empty_interaction` | Concatenate branches without product/difference | Interaction matters | If equal, simple branch concat is enough. |
| `cnn_matched_params` | Plain CNN matched params | Explicit empty branch beats capacity | If equal, use CNN. |

### Diagnostics

- Empty branch activation norm by fine label.
- Top opportunity squares in validation examples.
- Performance stratified by occupancy count.
- Branch contribution after removing occupied or empty masks.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/empty_square_opportunity_network.py
tests/test_empty_square_opportunity_network.py
configs/bench_empty_square_opportunity_network_simple18.yaml
configs/bench_empty_square_opportunity_network_occupied_only.yaml
```

The mask must be derived only from current-board piece occupancy planes.

## Candidate 2: Global Scratchpad BoardNet

### Thesis

A board CNN can be augmented with a small recurrent global scratchpad: a fixed number of memory vectors that summarize the board, are updated a few times, and broadcast context back to squares through affine modulation. This gives global communication without attention or dense square-pair tensors.

### Fingerprint

```text
CNN board features
+ K global memory slots
+ recurrent board-to-memory updates
+ memory-to-board FiLM broadcast
+ binary head
```

### Why It Is Distinct

- Not attention: no query-key pairwise routing over squares.
- Not Tensor-Core Pair Field: no `64 x 64` pair state.
- Not early-exit cascade: all samples use the same scratchpad depth.
- Not adapter-sandwich residual: memory is dynamic per sample, not static adapters.

### Architecture Sketch

Board features:

```text
h0 = CNNStem(x)
```

Initialize memory:

```text
m0 = learned_memory + MLP(global_pool(h0))
```

For `T=3..5` scratchpad steps:

```text
summary_t = pool([h_t, h_t * coord_features])
m_{t+1} = GRUCell(summary_t, m_t)
film_t = MLP(m_{t+1}) -> gamma_t, beta_t
h_{t+1} = ConvBlock(gamma_t * h_t + beta_t)
```

If using multiple memory slots:

```text
m_t: (B, K, D)
summary_t projected into K slot summaries
```

No square-to-memory attention is required in the first version; use fixed pooled summaries into each slot.

Default config:

```yaml
model:
  name: global_scratchpad_boardnet
  width: 64
  memory_slots: 4
  memory_dim: 64
  scratchpad_steps: 4
  dropout: 0.1
```

Expected parameter range:

```text
350k-900k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `no_scratchpad` | Plain CNN with matched params | Scratchpad global context matters | If equal, memory unnecessary. |
| `one_step` | Single memory update | Recurrent context matters | If equal, use one step. |
| `no_broadcast` | Memory used only in final head | Broadcast back to board matters | If equal, late global vector enough. |
| `random_memory` | Freeze random memory slots | Learned memory semantics matter | If equal, memory acts as noise/capacity. |
| `single_slot` | One global memory vector | Multiple slots matter | If equal, simplify. |

### Diagnostics

- Memory slot norm by step.
- Similarity between memory slots.
- Board activation change after each broadcast.
- Final accuracy by number of scratchpad steps.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/global_scratchpad_boardnet.py
tests/test_global_scratchpad_boardnet.py
configs/bench_global_scratchpad_boardnet_simple18.yaml
configs/bench_global_scratchpad_boardnet_no_scratchpad.yaml
```

Keep memory updates small and stable:

```text
h_{t+1} = h_t + 0.25 * ConvBlock(film(h_t, m_t))
```

## Candidate 3: Learnable Pooling Tree BoardNet

### Thesis

Instead of pooling the whole board at once or using an FPN, build a fixed hierarchy over the `8 x 8` board: squares become `2 x 2` cells, cells become quadrants, quadrants become a board root. Each tree node has a small learned aggregator and passes features upward.

### Fingerprint

```text
square CNN features
+ fixed 2x2 pooling tree
+ learned node aggregators
+ root and intermediate node readout
+ binary head
```

### Why It Is Distinct

- Not FPN: no top-down upsampling or multiresolution feature maps.
- Not patch mixer: aggregation follows a fixed hierarchy, not all-token MLP mixing.
- Not submodular coverage: no diminishing-return set function.
- Not neural decision forest: this is a feature aggregation tree, not a tree classifier.

### Architecture Sketch

Start with square features:

```text
h_square: (B, 64, D)
```

Level 1:

```text
16 nodes, each aggregates 4 squares
```

Level 2:

```text
4 nodes, each aggregates 4 level-1 nodes
```

Level 3:

```text
1 root node, aggregates 4 quadrants
```

Aggregator:

```text
node = MLP([
  mean(children),
  max(children),
  child_1,
  child_2,
  child_3,
  child_4
])
```

Readout:

```text
z = concat(root, mean(level2), max(level1), global_square_pool)
logits = MLP(z)
```

Default config:

```yaml
model:
  name: learnable_pooling_tree_boardnet
  square_width: 64
  node_width: 64
  aggregator_hidden: 128
  dropout: 0.1
```

Expected parameter range:

```text
250k-700k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `global_pool_only` | Remove tree and global pool square features | Tree hierarchy matters | If equal, tree unnecessary. |
| `fixed_mean_tree` | Use mean aggregators only | Learned aggregation matters | If equal, simple pooling enough. |
| `random_tree_groups` | Randomly group squares into same-size tree | Board hierarchy matters | If equal, grouping semantics weak. |
| `quadrant_only` | Use only level-2 quadrants | Full hierarchy matters | If equal, skip level 1. |
| `cnn_matched_params` | Plain CNN matched params | Tree readout beats capacity | If equal, use CNN. |

### Diagnostics

- Node activation norms by level.
- Which quadrant/root features dominate.
- Sensitivity to random tree groups.
- Fine-label metrics for global pool versus tree.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/learnable_pooling_tree_boardnet.py
tests/test_learnable_pooling_tree_boardnet.py
configs/bench_learnable_pooling_tree_boardnet_simple18.yaml
configs/bench_learnable_pooling_tree_boardnet_global_pool.yaml
```

Use fixed index tensors for child groups.

## Candidate 4: Spatial FiLM Coordinate Net

### Thesis

Appending coordinate planes may be too weak. Instead, generate per-square affine modulation parameters from deterministic coordinate features and side-relative coordinates, then modulate CNN features at multiple depths.

### Fingerprint

```text
CNN features
+ coordinate MLP
+ per-square gamma/beta maps
+ spatial FiLM modulation
+ pooled head
```

### Why It Is Distinct

- Not coordinate planes only: coordinates modulate features throughout the network.
- Not material-phase adapter: conditioning is deterministic square coordinate, not material summary.
- Not hypernetwork CNN: it does not generate convolution kernels; it generates per-square affine maps.

### Architecture Sketch

Coordinate tensor:

```text
c_s = [rank, file, center_distance, edge_distance, side_relative_rank, square_color]
```

Coordinate MLP:

```text
gamma_l(s), beta_l(s) = MLP_l(c_s)
```

At layer `l`:

```text
h_l = ConvBlock(h_{l-1})
h_l = gamma_l * h_l + beta_l
```

Use bounded modulation first:

```text
gamma_l = 1 + 0.25 * tanh(raw_gamma_l)
beta_l = 0.25 * tanh(raw_beta_l)
```

Default config:

```yaml
model:
  name: spatial_film_coordinate_net
  width: 64
  depth: 5
  coord_hidden: 32
  dropout: 0.1
```

Expected parameter range:

```text
250k-700k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `coord_planes_only` | Append coordinates, no FiLM | Spatial FiLM helps beyond coords | If equal, use coord planes. |
| `no_side_relative_coord` | Remove side-relative coordinate | Perspective matters | If equal, simpler coords enough. |
| `shared_gamma_only` | Use global channel FiLM, no per-square beta | Spatial modulation matters | If equal, global modulation enough. |
| `random_coord_map` | Randomly permute coordinate assignment to squares | Coordinate semantics matter | If equal, FiLM is generic capacity. |
| `cnn_matched_params` | Plain CNN matched params | Spatial FiLM beats capacity | If equal, use CNN. |

### Diagnostics

- Learned gamma/beta maps by layer.
- Modulation magnitude near center, edge, back rank, and king side.
- Performance with and without side-relative coordinates.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/spatial_film_coordinate_net.py
tests/test_spatial_film_coordinate_net.py
configs/bench_spatial_film_coordinate_net_simple18.yaml
configs/bench_spatial_film_coordinate_net_coord_planes_only.yaml
```

Precompute coordinate features as buffers. Do not allocate them every forward pass.

## Candidate 5: Channel-Bilinear Role Mixer

### Thesis

Ordinary heads pool channels additively. A low-rank bilinear head can explicitly model pairwise interactions between role summaries, such as own-heavy-piece features with opponent-king-zone features, without building square-pair tensors or local product conv blocks.

### Fingerprint

```text
CNN trunk
+ grouped role summaries
+ low-rank bilinear feature products
+ pooled classifier head
```

### Why It Is Distinct

- Not multiplicative conjunction convnet: products happen in the pooled head, not inside local conv blocks.
- Not tensor-ring square interactions: no high-order square tensor.
- Not determinant volume: no Gram/logdet object.
- Not piece-token role contrast: roles are feature groups, not occupied-token sets.

### Architecture Sketch

Build pooled summaries:

```text
z_global = pool(h)
z_king = pool king-zone masks if safe
z_center = pool center mask
z_edge = pool edge mask
z_material = MLP(counts)
```

Project each summary to low-rank factors:

```text
a_r = U_r z_r
b_s = V_s z_s
```

Bilinear interaction:

```text
i_{r,s} = a_r * b_s
```

Fuse:

```text
logits = MLP([z_all, i_all])
```

Default config:

```yaml
model:
  name: channel_bilinear_role_mixer
  trunk_width: 64
  role_dim: 64
  bilinear_rank: 16
  roles: [global, center, edge, king_zone, material]
  dropout: 0.1
```

Expected parameter range:

```text
250k-800k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `concat_only` | Remove bilinear products | Bilinear role interaction matters | If equal, products unnecessary. |
| `random_role_masks` | Replace center/edge/king masks with random masks | Role semantics matter | If equal, role masks are generic. |
| `material_no_bilinear` | Material summary only concatenated, not bilinear | Material interaction matters | If equal, avoid shortcut risk. |
| `rank4_bilinear` | Lower bilinear rank | Interaction capacity matters | If equal, keep small. |
| `mlp_head_matched` | Matched MLP over summaries | Bilinear structure matters | If equal, MLP enough. |

### Diagnostics

- Interaction norm by role pair.
- Role-pair ablation deltas.
- Whether material interactions dominate.
- Fine-label behavior for king-zone interactions.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/channel_bilinear_role_mixer.py
tests/test_channel_bilinear_role_mixer.py
configs/bench_channel_bilinear_role_mixer_simple18.yaml
configs/bench_channel_bilinear_role_mixer_concat_only.yaml
```

Treat king-zone pooling as optional and fail closed if king decoding is unsafe.

## Candidate 6: Evidence Sieve Network

### Thesis

Instead of refining logits, the model can refine features by repeatedly filtering them through learned evidence sieves. Each sieve stage produces a soft mask over channels and squares, passes selected evidence onward, and leaves a diagnostic trail.

### Fingerprint

```text
CNN features
+ repeated sieve masks
+ filtered feature residuals
+ sieve diagnostics
+ binary head
```

### Why It Is Distinct

- Not iterative logit refinement: refinement happens in feature maps, not logits.
- Not attention: masks are local feature filters, not pairwise token routing.
- Not dropout consensus: masks are learned deterministic sieves, not stochastic ablations.
- Not sparse witness: no discrete top-k selection of pieces.

### Architecture Sketch

Initial features:

```text
h0 = CNNStem(x)
```

For `T=3..5` sieves:

```text
mask_t = sigmoid(Conv1x1([h_t, coords]))
filtered_t = h_t * mask_t
residual_t = ConvBlock(filtered_t)
h_{t+1} = h_t + residual_t
```

Collect diagnostics:

```text
mask_mean_t
mask_entropy_t
filtered_norm_t
delta_norm_t
```

Final head:

```text
logits = MLP([pool(h_T), sieve_diagnostics])
```

Default config:

```yaml
model:
  name: evidence_sieve_network
  width: 64
  sieve_steps: 4
  dropout: 0.1
  use_coordinate_planes: true
```

Expected parameter range:

```text
300k-800k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `no_sieves` | Same-depth CNN without masks | Sieving matters | If equal, masks unnecessary. |
| `random_sieves` | Freeze random same-density masks | Learned masks matter | If equal, regularization effect only. |
| `one_sieve` | Use one sieve stage | Repeated filtering matters | If equal, simplify. |
| `no_sieve_diagnostics` | Remove mask stats from head | Diagnostics carry signal | If equal, final features enough. |
| `channel_only_sieve` | Mask channels but not squares | Spatial masks matter | If equal, channel gating enough. |

### Diagnostics

- Mask entropy by stage.
- Fraction of board retained by sieve stage.
- Whether fine-label `1` has higher mask entropy.
- Validation examples with high/low retained evidence.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/evidence_sieve_network.py
tests/test_evidence_sieve_network.py
configs/bench_evidence_sieve_network_simple18.yaml
configs/bench_evidence_sieve_network_no_sieves.yaml
```

Avoid hard top-k in the first version. Keep masks smooth and differentiable.

## Candidate 7: Ring-Shell Recurrent BoardNet

### Thesis

Important chess context often radiates from anchors: kings, center squares, edges, and promotion zones. Summarize the board in fixed rings/shells around these anchors and process the shells with a small recurrent model.

### Fingerprint

```text
CNN feature map
+ fixed ring/shell masks
+ shell sequence summaries
+ GRU over shells
+ binary head
```

### Why It Is Distinct

- Not king-cage path DP: no shortest paths or percolation.
- Not king-shelter microkernel: shells are recurrent global summaries, not local crop filters.
- Not topology: no Euler or Betti curves.
- Not support-function envelope: no convex widths or support directions.

### Architecture Sketch

Feature map:

```text
h = CNN(x)
```

Define shell families:

- own king Chebyshev rings: distance `0,1,2,3,4+`
- opponent king Chebyshev rings
- center rings
- edge-distance rings

For each family:

```text
s_t = pool(h over shell_t)
H_family = GRU(s_0, s_1, ..., s_T)
z_family = final_state + max_state
```

Fuse:

```text
z = concat(global_pool(h), z_own_king, z_opp_king, z_center, z_edge)
logits = MLP(z)
```

Default config:

```yaml
model:
  name: ring_shell_recurrent_boardnet
  trunk_width: 64
  shell_dim: 64
  gru_hidden: 64
  dropout: 0.1
```

Expected parameter range:

```text
250k-700k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `global_pool_only` | Remove shell branches | Shell context matters | If equal, shell recurrence unnecessary. |
| `no_king_shells` | Remove own/opponent king shells | King anchoring matters | If equal, center/edge dominate. |
| `random_shell_masks` | Random same-size shell masks | Shell geometry matters | If equal, mask semantics weak. |
| `shell_mean_no_gru` | Mean over shells, no recurrence | Shell order matters | If equal, recurrent order unnecessary. |
| `center_only_shells` | Keep center rings only | King-specific shells matter | If equal, simplify. |

### Diagnostics

- Shell hidden state norms by radius.
- Own versus opponent king shell contribution.
- Fine-label metrics after removing king shells.
- Shell mask fail-closed rate if king decode is unsafe.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/ring_shell_recurrent_boardnet.py
tests/test_ring_shell_recurrent_boardnet.py
configs/bench_ring_shell_recurrent_boardnet_simple18.yaml
configs/bench_ring_shell_recurrent_boardnet_global_pool.yaml
```

If king planes cannot be safely decoded, disable king shells and keep center/edge shells.

## Candidate 8: Rank-File Memory Grid Net

### Thesis

Maintain learned memory vectors for each rank and each file. Squares write into their rank/file memories, then rank/file memories write back to squares. This gives global rank/file communication without axial convolutions, line solves, or attention.

### Fingerprint

```text
square features
+ 8 rank memories
+ 8 file memories
+ square-to-memory pooling
+ memory-to-square broadcast
+ binary head
```

### Why It Is Distinct

- Not axial rank-file conv: communication is through persistent memories, not 1D convolution kernels.
- Not Schur-Ray: no line incidence solve.
- Not attention: no learned pairwise square weights or softmax routing.
- Not global scratchpad: memories are tied to rank/file identities, not generic global slots.

### Architecture Sketch

Square features:

```text
h: (B, 64, D)
```

Initialize memories:

```text
r_i = learned_rank_embed_i
f_j = learned_file_embed_j
```

For `T=2..4` updates:

Square to rank/file:

```text
r_i = GRU(r_i, pool_{s on rank i}(h_s))
f_j = GRU(f_j, pool_{s on file j}(h_s))
```

Rank/file to square:

```text
h_{i,j} = h_{i,j} + MLP([r_i, f_j, h_{i,j}])
```

Readout:

```text
z = concat(pool_squares(h), pool_ranks(r), pool_files(f))
logits = MLP(z)
```

Default config:

```yaml
model:
  name: rank_file_memory_grid_net
  square_dim: 64
  memory_dim: 64
  update_steps: 3
  dropout: 0.1
```

Expected parameter range:

```text
300k-800k
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `no_memory` | Square CNN/MLP only | Rank/file memory matters | If equal, memory unnecessary. |
| `rank_memory_only` | Remove file memories | File memory matters | If equal, simplify. |
| `file_memory_only` | Remove rank memories | Rank memory matters | If equal, simplify. |
| `global_memory_only` | Replace 8+8 memories with one global memory | Line identity matters | If equal, use scratchpad. |
| `random_square_to_memory` | Randomly assign squares to memory IDs | Rank/file semantics matter | If equal, memories are generic capacity. |

### Diagnostics

- Rank memory norms by rank.
- File memory norms by file.
- Update-step deltas.
- Whether line memories specialize by side-relative rank.

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/rank_file_memory_grid_net.py
tests/test_rank_file_memory_grid_net.py
configs/bench_rank_file_memory_grid_net_simple18.yaml
configs/bench_rank_file_memory_grid_net_no_memory.yaml
```

Use fixed square-to-rank and square-to-file index tensors. Do not compute legal rook/bishop moves.

## Implementation Queue

Recommended order:

1. `Empty-Square Opportunity Network`
2. `Rank-File Memory Grid Net`
3. `Global Scratchpad BoardNet`
4. `Spatial FiLM Coordinate Net`
5. `Evidence Sieve Network`
6. `Channel-Bilinear Role Mixer`
7. `Ring-Shell Recurrent BoardNet`
8. `Learnable Pooling Tree BoardNet`

Reasoning:

- Empty-square modeling is underexplored and sharply ablatable.
- Rank/file memory is chess-shaped without copying Schur-Ray or axial convs.
- Scratchpad memory is a strong general-purpose global context mechanism.
- Spatial FiLM is a clean upgrade over coordinate planes.
- Evidence sieves create useful diagnostics.
- Bilinear role mixing is likely easy but may overlap with strong MLP heads.
- Shell recurrence needs safe king decoding.
- Pooling trees are simple but may be less expressive than CNN/FPN baselines.

## Shared Benchmark Rules

For every candidate:

- use the same train/val/test splits
- use the same coarse binary target
- report fine-label diagnostics
- compare to simple CNN, residual CNN, and strongest available practical baseline
- run the central ablation before promoting the idea

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
| Empty-Square Opportunity | Another occupied/empty dual branch unless the empty-square target or fusion changes. |
| Global Scratchpad | Another global memory-slot CNN unless the update/broadcast mechanism changes. |
| Learnable Pooling Tree | Another fixed board hierarchy unless grouping or node objective changes. |
| Spatial FiLM Coordinate | Another coordinate modulation CNN unless modulation is materially different from per-square affine maps. |
| Channel-Bilinear Role Mixer | Another pooled bilinear role head with only rank changes. |
| Evidence Sieve | Another feature-mask refinement model with only sieve count changes. |
| Ring-Shell Recurrent | Another shell/ring recurrent model unless anchors or shell objective change. |
| Rank-File Memory Grid | Another rank/file memory model unless memories communicate differently. |

## Continuity Note

The archive now has many high-concept ideas. This batch is meant to keep generating implementable architectures that still test distinct chess-shaped inductive biases. The first two candidates, empty-square opportunity and rank/file memory, are the most promising new directions in this packet.
