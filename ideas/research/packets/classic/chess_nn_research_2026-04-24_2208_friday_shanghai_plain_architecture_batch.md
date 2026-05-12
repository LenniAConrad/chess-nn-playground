# Codex Research Batch: Plain Practical Neural Architectures

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`
- Generated at: 2026-04-24 22:08
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: plain practical architecture batch, not implemented

## Purpose

This packet deliberately steps away from the exotic ideas. These are plain neural network architectures for the current chess puzzle-likeness benchmark:

- no new algebraic bottleneck
- no graph/sheaf machinery
- no move generation
- no attention as the central trick
- no engine-derived features
- no theorem-like mechanism claim

The goal is to add strong, normal baselines that would be reasonable to implement before judging the more unusual research packets.

## Shared Data Contract

Task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- train on binary labels only
- keep fine labels `0`, `1`, and `2` for diagnostics
- report the fine-label `3 x 2` diagnostic matrix

First implementation:

- input: `simple_18`
- dataset: existing `crtk_sample_3class` splits
- trainer: shared experimental training pipeline

Forbidden model inputs:

- Stockfish scores, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, unresolved candidate status, dataset provenance, or anything derived from them.
- Engine search, legal-move search, forced-line search, mate/stalemate oracles, tablebase outcomes, or future game outcomes.

Allowed model inputs:

- Current-board tensor.
- Side-to-move, castling, and en-passant planes already in `simple_18`.
- Deterministic coordinate planes generated inside the model.
- Safe material/count summaries if used by a model branch.

## Ranked Shortlist

| Rank | Candidate | Main object | Why it is useful |
|---|---|---|---|
| 1 | ConvNeXt BoardNet | Modern plain conv blocks on an `8 x 8` board | Strong regular CNN comparator. |
| 2 | Board FPN CNN | Multi-resolution feature pyramid over `8 x 8`, `4 x 4`, and `2 x 2` maps | Tests whether coarse board context helps ordinary CNNs. |
| 3 | Piece-Plane Gated CNN | Separate piece-plane groups plus learned color/type gates | Plain way to respect input channel semantics. |
| 4 | Patch Mixer BoardNet | MLP-Mixer over `2 x 2` board patches | Simple non-attention patch model. |
| 5 | Specialist-Head CNN | Shared trunk with plain region/material specialist heads | Tests multi-head specialization without MoE complexity. |
| 6 | Shallow Wide Residual BoardNet | Fewer layers, wider channels, strong pooling head | Direct counterpoint to deep residual stacks. |

Best next implementation from this batch:

```text
ConvNeXt BoardNet
```

Reason: it is the cleanest stronger regular CNN baseline and should be easy to implement, test, and ablate.

## Candidate 1: ConvNeXt BoardNet

### Thesis

Use a small ConvNeXt-style architecture adapted to `8 x 8` chess boards: depthwise spatial mixing, inverted channel MLPs, residual scaling, coordinate planes, and a strong global pooling head.

This is plain architecture engineering, not a new research mechanism.

### Fingerprint

```text
simple_18
+ coordinate planes
+ ConvNeXt-style depthwise blocks
+ squeeze/excitation or global response gate
+ mean/max/std pooling head
+ binary logits
```

### Architecture Sketch

Input:

```text
x: (B, 18, 8, 8)
```

Append coordinate planes:

```text
rank
file
center_distance
side_relative_rank
```

Stem:

```text
Conv2d(22, width, kernel_size=3, padding=1)
```

Block:

```text
h = h + gamma * ConvNeXtBlock(h)
```

where:

```text
depthwise 5x5 conv
channels_last LayerNorm or BatchNorm2d
1x1 expand: width -> 4 * width
GELU or ReLU
1x1 contract: 4 * width -> width
optional squeeze-excitation
layer scale gamma
```

Head:

```text
mean_pool(h)
max_pool(h)
std_pool(h)
small MLP
logits
```

Default config:

```yaml
model:
  name: convnext_boardnet
  input_channels: 18
  width: 64
  depth: 5
  expansion: 4
  kernel_size: 5
  dropout: 0.1
  use_coordinate_planes: true
  use_se: true
  num_classes: 2
```

Expected parameter range:

```text
250k-700k
```

### Why It Is Plain But Useful

The existing simple CNN and residual CNN are conventional, but they may not be strong enough comparators for the more unusual packets. A ConvNeXt-style block is still normal CNN engineering, but it gives:

- larger effective local window
- better channel mixing
- explicit residual scaling
- stronger global pooling
- easy ablations

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `residual_cnn_matched` | Match params with ordinary residual CNN | ConvNeXt block adds value | If equal, existing residual blocks are enough. |
| `kernel3_only` | Use depthwise `3 x 3` instead of `5 x 5` | Larger board-local window helps | If equal, use smaller kernel. |
| `no_coordinate_planes` | Remove coordinate planes | Coordinates help | If equal, input planes already encode enough. |
| `mean_pool_only` | Remove max/std pooling | Rich pooling helps | If equal, simplify head. |
| `no_se` | Remove squeeze/excitation | Global channel gating helps | If equal, omit SE. |

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/convnext_boardnet.py
tests/test_convnext_boardnet.py
configs/bench_convnext_boardnet_simple18.yaml
configs/bench_convnext_boardnet_kernel3.yaml
```

Use the repo's existing normalization preference if there is one. If `LayerNorm` over channels-last is annoying in this codebase, use `BatchNorm2d` first and keep the model simple.

## Candidate 2: Board FPN CNN

### Thesis

Chess positions often need both exact square detail and coarse whole-board phase. A plain feature-pyramid network can process the board at `8 x 8`, `4 x 4`, and `2 x 2` resolutions, then fuse the maps back into a single classifier.

### Fingerprint

```text
simple_18
+ 8x8 conv stem
+ 4x4 and 2x2 pooled branches
+ top-down feature fusion
+ pooled classification head
```

### Architecture Sketch

Stem:

```text
x8 = ConvStack(x)                    # (B, W, 8, 8)
x4 = ConvStack(avg_pool2d(x8, 2))     # (B, 2W, 4, 4)
x2 = ConvStack(avg_pool2d(x4, 2))     # (B, 4W, 2, 2)
```

Top-down fusion:

```text
y4 = x4 + upsample(project(x2), size=4)
y8 = x8 + upsample(project(y4), size=8)
```

Head:

```text
pool(y8) + pool(y4) + pool(x2)
MLP
logits
```

Default config:

```yaml
model:
  name: board_fpn_cnn
  input_channels: 18
  width: 48
  blocks_per_level: 2
  dropout: 0.1
  use_coordinate_planes: true
  num_classes: 2
```

Expected parameter range:

```text
300k-800k
```

### Why It Is Plain But Useful

This is a normal CNN pyramid. It is useful because an `8 x 8` board has very little spatial depth, so coarse pooled maps may give whole-board context quickly without resorting to attention, line solvers, or graph construction.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `single_resolution_matched` | Use only `8 x 8` stack with matched params | Pyramid context matters | If equal, FPN is unnecessary. |
| `bottom_up_only` | Remove top-down fusion | Fusion matters | If equal, multiscale pooled head is enough. |
| `no_2x2_level` | Remove coarsest level | Whole-board coarse context matters | If equal, `4 x 4` is enough. |
| `late_pool_only` | Pool each level without fusing maps | Spatial top-down fusion matters | If equal, simpler branch pooling wins. |
| `no_coordinate_planes` | Remove coordinates | Explicit board location helps | If equal, skip coords. |

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/board_fpn_cnn.py
tests/test_board_fpn_cnn.py
configs/bench_board_fpn_cnn_simple18.yaml
configs/bench_board_fpn_cnn_single_resolution.yaml
```

Keep interpolation deterministic and simple:

```text
nearest
```

The board is tiny; do not overcomplicate upsampling.

## Candidate 3: Piece-Plane Gated CNN

### Thesis

The `simple_18` channels are not arbitrary image channels. A plain CNN can respect this by first processing semantically related channel groups, then using learned gates to mix piece types and colors.

### Fingerprint

```text
piece-plane groups
+ group-specific stems
+ side/color/type gates
+ ordinary CNN trunk
+ pooled head
```

### Architecture Sketch

Split input channels into safe groups according to known `simple_18` semantics:

```text
white pieces
black pieces
side/state planes
```

If the exact channel mapping is not known, fail closed by using a generic grouped split and logging it.

Group stems:

```text
white_h = ConvStack(white_piece_planes)
black_h = ConvStack(black_piece_planes)
state_h = ConvStack(state_planes)
```

Gate summary:

```text
counts = safe material/count summary
gates = sigmoid(MLP(counts))
```

Fusion:

```text
h = concat(gate_w * white_h, gate_b * black_h, gate_s * state_h)
h = Conv1x1(h)
h = ordinary_residual_trunk(h)
```

Head:

```text
mean/max pool
MLP
logits
```

Default config:

```yaml
model:
  name: piece_plane_gated_cnn
  group_width: 24
  trunk_width: 72
  trunk_depth: 4
  gate_hidden: 32
  dropout: 0.1
```

Expected parameter range:

```text
250k-700k
```

### Why It Is Plain But Useful

This is still just grouped convs plus gates. It gives a normal way to test whether respecting piece-plane semantics helps over treating all input channels as interchangeable.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `ungrouped_stem_matched` | Single stem with matched params | Semantic groups help | If equal, group stems are unnecessary. |
| `no_gates` | Concatenate group features without gates | Material/state gating helps | If equal, gates are unnecessary. |
| `random_channel_groups` | Randomly assign channels to groups | Real channel semantics matter | If equal, grouping is generic capacity. |
| `counts_only_gates` | Use gates as classifier with no trunk | Gate counts are not the whole model | If strong, watch material shortcut. |
| `shared_group_stem` | Same stem weights for white/black piece groups | Color-specific stems matter | If equal, share weights. |

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/piece_plane_gated_cnn.py
tests/test_piece_plane_gated_cnn.py
configs/bench_piece_plane_gated_cnn_simple18.yaml
configs/bench_piece_plane_gated_cnn_ungrouped.yaml
```

The biggest practical issue is `simple_18` channel semantics. Add a small adapter and a unit test that verifies expected group sizes.

## Candidate 4: Patch Mixer BoardNet

### Thesis

Use a plain MLP-Mixer-style model over `2 x 2` chess patches. This is a simple non-attention alternative to square-token models: mix information across board patches with MLPs, then mix channels with MLPs.

### Fingerprint

```text
simple_18
+ 2x2 patch embedding
+ token-mixing MLP
+ channel-mixing MLP
+ pooled head
```

### Architecture Sketch

Patchify:

```text
8 x 8 board -> 16 patches of size 2 x 2
patch_dim = 18 * 2 * 2 = 72
```

Patch embedding:

```text
patch_tokens: (B, 16, D)
```

Mixer block:

```text
tokens = tokens + token_mlp(norm(tokens).transpose(1, 2)).transpose(1, 2)
tokens = tokens + channel_mlp(norm(tokens))
```

Head:

```text
mean(tokens)
max(tokens)
MLP
logits
```

Default config:

```yaml
model:
  name: patch_mixer_boardnet
  patch_size: 2
  token_count: 16
  embed_dim: 96
  depth: 4
  token_mlp_dim: 64
  channel_mlp_dim: 192
  dropout: 0.1
```

Expected parameter range:

```text
300k-900k
```

### Why It Is Plain But Useful

This is not attention and not a Transformer. It is a plain patch MLP architecture. It tests whether chess puzzle-likeness benefits from direct cross-patch mixing without convolutional locality.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `patch1_square_mixer` | Use `1 x 1` square tokens instead of `2 x 2` patches | Patch grouping matters | If better, square tokens are preferable. |
| `patch4_coarse_mixer` | Use `4 x 4` patches | Spatial resolution matters | If equal, model can be simpler. |
| `no_token_mixing` | Remove token MLP | Cross-patch mixing matters | If equal, channel MLP dominates. |
| `no_channel_mixing` | Remove channel MLP | Channel interactions matter | If equal, token mixing dominates. |
| `cnn_matched_params` | Plain CNN with matched params | Mixer beats normal conv capacity | If equal, use CNN. |

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/patch_mixer_boardnet.py
tests/test_patch_mixer_boardnet.py
configs/bench_patch_mixer_boardnet_simple18.yaml
configs/bench_patch_mixer_boardnet_no_token_mixing.yaml
```

Use `einops` only if it is already in dependencies. Otherwise implement patchify with `torch.nn.Unfold`.

## Candidate 5: Specialist-Head CNN

### Thesis

A plain shared CNN trunk can feed several small specialist heads: king-zone head, center-control head, material/phase head, and global board head. A learned fusion layer combines their logits/features. This tests specialization without a full mixture-of-experts router.

### Fingerprint

```text
shared CNN trunk
+ fixed region pools
+ several small specialist heads
+ simple learned fusion
+ binary logits
```

### Architecture Sketch

Shared trunk:

```text
h = CNN(x)  # (B, W, 8, 8)
```

Specialist features:

```text
global_feat = mean/max_pool(h)
center_feat = pool fixed center mask
edge_feat = pool fixed edge mask
king_feat = pool own/opponent king zones if safely decoded
material_feat = MLP(safe counts)
```

Each head:

```text
head_i(feat_i) -> small feature or logits_i
```

Fusion:

```text
fusion_input = concat(head_features, logits_i)
logits = MLP(fusion_input)
```

Default config:

```yaml
model:
  name: specialist_head_cnn
  trunk_width: 64
  trunk_depth: 4
  head_hidden: 32
  fusion_hidden: 64
  dropout: 0.1
```

Expected parameter range:

```text
250k-700k
```

### Why It Is Plain But Useful

This architecture uses ordinary modules, but it gives useful diagnostics: which specialist heads are actually contributing. It may be stronger than a single pooled head without becoming a sparse expert system.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `single_global_head` | Use only global pooling head | Specialist pools help | If equal, heads are unnecessary. |
| `no_king_head` | Remove king-zone head | King-zone specialization helps | If equal, omit it. |
| `no_material_head` | Remove material/count head | Count summary helps | If equal, trunk learns it. |
| `uniform_logit_average` | Average specialist logits without fusion MLP | Learned fusion helps | If equal, simplify. |
| `same_region_random_masks` | Replace center/edge masks with random same-size masks | Region semantics matter | If equal, masks are generic. |

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/specialist_head_cnn.py
tests/test_specialist_head_cnn.py
configs/bench_specialist_head_cnn_simple18.yaml
configs/bench_specialist_head_cnn_single_global.yaml
```

The king-zone head must fail closed if king planes are not safely decoded.

## Candidate 6: Shallow Wide Residual BoardNet

### Thesis

On an `8 x 8` board, depth may be less useful than width and a good head. A shallow wide residual CNN can test whether the benchmark wants broad feature extraction rather than long convolutional stacks.

### Fingerprint

```text
simple_18
+ wide stem
+ 2-3 residual blocks
+ channel attention
+ strong pooled head
```

### Architecture Sketch

Stem:

```text
Conv2d(18 + coord_planes, width, 3, padding=1)
```

Residual block:

```text
Conv3x3(width, width)
BatchNorm/ReLU
Conv3x3(width, width)
SE gate
skip connection
```

Use only:

```text
depth = 2 or 3
width = 96 or 128
```

Head:

```text
mean_pool
max_pool
std_pool
material/count MLP optional
MLP
logits
```

Default config:

```yaml
model:
  name: shallow_wide_residual_boardnet
  width: 96
  depth: 3
  dropout: 0.15
  use_coordinate_planes: true
  use_se: true
  use_count_head: true
```

Expected parameter range:

```text
500k-1.2M
```

### Why It Is Plain But Useful

This is almost intentionally boring. It answers a practical question:

```text
Before inventing unusual modules, does a wider shallow residual CNN already capture most of the signal?
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `deep_narrow_matched` | More layers, fewer channels, matched params | Width beats depth | If equal, depth/width does not matter much. |
| `no_se` | Remove channel attention | Channel gating helps | If equal, simplify. |
| `no_count_head` | Remove material/count head | Counts help the plain baseline | If equal, trunk learns enough. |
| `mean_pool_only` | Remove max/std pooling | Head richness helps | If equal, simplify. |
| `small_width_control` | Width 48 or 64 | Extra width matters | If equal, use smaller model. |

### Implementation Notes

Suggested files:

```text
src/chess_nn_playground/models/trunk/shallow_wide_residual_boardnet.py
tests/test_shallow_wide_residual_boardnet.py
configs/bench_shallow_wide_residual_boardnet_simple18.yaml
configs/bench_shallow_wide_residual_boardnet_deep_narrow.yaml
```

This should be treated as a baseline, not a novel family.

## Plain Batch Implementation Priority

Recommended order:

1. `ConvNeXt BoardNet`
2. `Board FPN CNN`
3. `Patch Mixer BoardNet`
4. `Piece-Plane Gated CNN`
5. `Specialist-Head CNN`
6. `Shallow Wide Residual BoardNet`

Reason:

- `ConvNeXt BoardNet` gives the cleanest strong CNN comparator.
- `Board FPN CNN` tests multiresolution context without unusual machinery.
- `Patch Mixer BoardNet` gives a plain non-convolutional baseline.
- `Piece-Plane Gated CNN` depends on reliable channel semantics.
- `Specialist-Head CNN` is useful but slightly more diagnostic-heavy.
- `Shallow Wide Residual BoardNet` is mostly a capacity sanity check.

## Shared Benchmark Rules

For every model in this packet, compare against:

- existing simple CNN baselines
- existing residual CNN baselines
- `Multi-Scale Dilated Board Mixer CNN` if implemented
- `Piece-Token CNN Hybrid` if implemented

Report:

- parameter count
- AUROC
- accuracy
- balanced accuracy
- F1
- calibration
- fine-label `3 x 2` diagnostic matrix
- class-1 recall at matched fine-label-0 false-positive rate if tooling supports it

## Anti-Duplicate Rules

Do not later re-propose the same plain ideas with only small name changes:

| Family | Avoid Near-Duplicate |
|---|---|
| ConvNeXt BoardNet | Another depthwise-inverted-conv board CNN with only width/depth changes. |
| Board FPN CNN | Another pyramid CNN with only different level widths. |
| Piece-Plane Gated CNN | Another grouped-channel gated CNN unless channel semantics change. |
| Patch Mixer BoardNet | Another MLP-Mixer board model with only token count changes. |
| Specialist-Head CNN | Another region-head CNN unless the heads or falsifiers are meaningfully different. |
| Shallow Wide Residual BoardNet | Another shallow/wide residual capacity check with only parameter changes. |

## Continuity Note

These models are useful because they are plain. If one of them wins, it raises the bar for the more speculative packets. If none of them improves on the current baselines, that is also useful: it suggests the existing regular CNN suite is already a reasonable comparator.
