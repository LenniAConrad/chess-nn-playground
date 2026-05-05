# Codex Research Packet: Tensor-Core Square-Pair Field Network

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2148_friday_shanghai_tensorcore_pairfield.md`
- Generated at: 2026-04-24 21:48
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: full architecture packet, not implemented

## One-Sentence Thesis

Build a deliberately GPU-saturating chess model: represent the board as a dense `64 x 64` square-pair field, update it through tensor-core-friendly batched matrix multiplications, and classify puzzle-likeness from pair-field energy, relation structure, and square updates.

## Why This One Is For A Big GPU

Most previous ideas were designed to be elegant, sparse, or CPU-tolerable. This one makes the opposite tradeoff:

- fixed shapes
- dense tensors
- large batch sizes
- BF16/FP16-friendly dimensions
- no ragged piece lists
- no Python loops over positions
- no sparse graph construction
- no top-k branches in the forward pass
- no dynamic move generation

The board is small, so a normal CNN underuses a large GPU. This architecture intentionally creates enough dense work to use tensor cores:

```text
batch x heads x 64 x 64 x width
```

It is a research architecture, not a lightweight baseline.

## Data Contract

Task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- `0`, `1`, and `2` remain diagnostics only
- train on binary labels
- report fine-label `3 x 2` diagnostic matrix

First implementation:

- input: `simple_18`
- dataset: existing `crtk_sample_3class` splits
- trainer: shared experimental training pipeline

Forbidden model inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, tablebase outcomes, or future game outcomes.

Allowed model inputs:

- Current board tensor.
- Side-to-move, castling, and en-passant planes already in `simple_18`.
- Deterministic square coordinates.
- Fixed relation planes over square pairs, such as same rank, same file, same diagonal, square color parity, distance bins, and knight/king offsets.

## Core Abstraction

Instead of choosing a sparse chess object, use the full square-pair universe:

```text
N = 64
pairs = N x N = 4096
```

Every ordered pair of squares gets a learned feature:

```text
F[i, j]
```

The model asks:

```text
Which dense square-pair interaction fields separate puzzle-like positions from non-puzzles?
```

This is not meant to be minimal. It is meant to be highly parallel.

## Why It Is Not Plain Attention

The architecture resembles attention only at the tensor shape level. The key differences:

- no token-to-token softmax bottleneck
- no causal or sequence attention
- no learned query-only selector
- no attention dropout or attention-map interpretation as probability
- pair fields are retained as features, not just used to route values
- relation planes are mixed directly into pair states
- readout uses pair-field energy and pair diagnostics

The central object is a dense pair field:

```text
F in R^{B x H x 64 x 64 x Dp}
```

not an attention distribution.

## Relation Bank

Precompute fixed square-pair relation planes:

```text
R in {0,1}^{K x 64 x 64}
```

Include:

```text
same_square
same_rank
same_file
same_diag
same_anti_diag
same_square_color
opposite_square_color
knight_offset
king_offset
manhattan_distance_1
manhattan_distance_2
manhattan_distance_3
chebyshev_distance_1
chebyshev_distance_2
same_center_ring
same_edge_class
rank_order_forward
file_order_forward
```

These are fixed current-board-independent geometry relations. They do not encode attacks, legal moves, checks, mates, or engine pressure.

## Architecture Sketch

### Step 1: Square Token Projection

Input:

```text
x: (B, 18, 8, 8)
```

Flatten to:

```text
X: (B, 64, Cin)
```

Add fixed coordinate features:

```text
rank
file
side_relative_rank
center_distance
edge_distance
square_color
```

Project to tensor-core-friendly width:

```text
X0 = Linear(Cin + coord_dim, D)
```

Recommended:

```text
D = 128
```

### Step 2: Dense Bilinear Pair Field

For each head `h`, compute:

```text
Q_h = X Wq_h
K_h = X Wk_h
V_h = X Wv_h
```

Then build a bilinear pair score:

```text
S_h[i,j] = dot(Q_h[i], K_h[j]) / sqrt(d)
```

Unlike attention, do not softmax over `j`.

Instead, construct pair channels:

```text
F_h[i,j] = MLP_pair([
  S_h[i,j],
  X_i,
  X_j,
  R[:, i, j]
])
```

For GPU efficiency, avoid a large per-pair Python MLP. Use factorized dense operations:

```text
pair_base = Q @ K^T
rel_bias = relation_mix_h @ R
F_scalar = gelu(pair_base + rel_bias)
```

Then lift to channels with batched matmul:

```text
F = F_scalar[..., None] * W_pair_head
```

or use several bilinear ranks:

```text
F_hk[i,j] = (Q_hk[i] * K_hk[j]).sum()
```

### Step 3: Pair-Field Update To Squares

Use pair field to update square tokens:

```text
message_i = sum_j norm(F[i,j]) * V[j]
```

But use a stable dense normalization, not softmax:

```text
W_ij = tanh(F_scalar_ij) / sqrt(N)
message = W @ V
```

Then:

```text
X_next = LayerNorm(X + Linear(message))
```

This is a dense `B*H` batched matrix multiply of shape:

```text
(64 x 64) @ (64 x d)
```

repeated across heads and layers.

### Step 4: Pair-Field Blocks

Stack `L` blocks:

```text
for layer in 1..L:
    build dense pair field
    update square tokens
    update pair diagnostics
```

Recommended first large-GPU config:

```text
layers = 6
heads = 16
head_dim = 64
model_dim = 256
pair_rank = 32
```

This is intentionally larger than the other research packets.

### Step 5: Readout

Collect square summaries:

```text
mean(X)
max(X)
occupied_mean(X)
king_zone_mean(X)
```

Collect pair summaries:

```text
mean(abs(F))
mean(F^2)
max(abs(F))
row_energy = mean_j F[i,j]^2
col_energy = mean_i F[i,j]^2
same_rank_energy
same_file_energy
same_diag_energy
knight_offset_energy
king_zone_pair_energy
occupied_to_empty_energy
occupied_to_occupied_energy
```

Classify from:

```text
MLP([square_summary, pair_summary])
```

## Tensor Contract

```text
input:           (B, 18, 8, 8)
tokens:          (B, 64, D)
relation_bank:   (K, 64, 64)
Q,K,V:           (B, H, 64, Dh)
pair_scores:     (B, H, 64, 64)
pair_features:   optional (B, H, 64, 64, R)
messages:        (B, H, 64, Dh)
tokens_next:     (B, 64, D)
pair_summary:    (B, S)
logits:          (B, 2)
```

## Big-GPU Optimization Contract

The implementation should be designed around:

```text
BF16 autocast
torch.compile
channels-last input where useful
static tensor shapes
batch size 1024 or 2048 if memory allows
model widths divisible by 8 or 16
head dimensions divisible by 8 or 16
no Python loops over squares or heads
einsum or bmm lowered to matmul
```

Preferred operations:

```text
linear projections
bmm
einsum over dense fixed shapes
LayerNorm or RMSNorm
GELU/SwiGLU MLP
```

Avoid:

```text
ragged token extraction
topk in the main forward pass
scatter over variable edge lists
python-chess in the model forward
sparse COO kernels
branchy per-position logic
small unbatched matmuls
```

## Suggested GPU Configs

### Medium GPU Smoke Config

```yaml
model:
  name: tensorcore_pairfield
  input_channels: 18
  model_dim: 128
  heads: 8
  head_dim: 32
  layers: 3
  pair_rank: 16
  relation_count: 18
  norm: rmsnorm
  activation: gelu
training:
  precision: bf16
  batch_size: 512
```

### Big GPU Main Config

```yaml
model:
  name: tensorcore_pairfield
  input_channels: 18
  model_dim: 256
  heads: 16
  head_dim: 64
  layers: 6
  pair_rank: 32
  relation_count: 18
  norm: rmsnorm
  activation: swiglu
  use_pair_energy_readout: true
  use_relation_energy_readout: true
training:
  precision: bf16
  batch_size: 2048
  compile: true
```

### Heavy GPU Stress Config

```yaml
model:
  name: tensorcore_pairfield
  input_channels: 18
  model_dim: 384
  heads: 24
  head_dim: 64
  layers: 8
  pair_rank: 48
  relation_count: 18
training:
  precision: bf16
  batch_size: 4096
  compile: true
  gradient_checkpointing: optional
```

The heavy config is not a fair first comparison. It is a throughput stress test for large GPUs.

## Estimated Compute Shape

Let:

```text
B = batch size
N = 64
H = heads
Dh = head dim
L = layers
```

Per layer, dominant dense operations are:

```text
QK^T:       B * H * N * N * Dh
WV:         B * H * N * N * Dh
MLP:        B * N * D * hidden
```

For:

```text
B = 2048
H = 16
N = 64
Dh = 64
L = 6
```

the pair matmul scale is roughly:

```text
2 * B * H * N * N * Dh * L
~= 103 billion multiply-add scale operations
```

This is the point: the model is intentionally dense enough to keep a large GPU busy.

## Chess Meaning

The dense pair field can represent:

- every occupied-to-occupied relation
- every occupied-to-empty square relation
- king-zone pair tension
- same-line heavy-piece alignment
- knight-fork geometry
- square-color pressure
- local and long-range interactions in the same tensor

Unlike a relation-query model, it does not choose a small number of joins. It lets the GPU compute all pair interactions, then learns which pair-field energies matter.

## Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `cnn_only_matched` | Replace pair-field blocks with parameter-matched CNN/MLP | Tests pair-field value | Full should beat if dense interactions matter. |
| `no_pair_update` | Compute pair summaries but do not update square tokens through pair field | Tests pair-to-square message path | Full should improve if pair propagation matters. |
| `no_pair_readout` | Use pair field only for messages, remove pair-energy summaries | Tests pair diagnostics | Drop means pair summaries carry signal. |
| `relation_bank_shuffle` | Replace relation planes with density-preserving random pair masks | Tests chess pair geometry | Should degrade if relation semantics matter. |
| `softmax_attention_control` | Replace tanh pair weights with ordinary softmax attention | Tests pair-field versus attention | Pair-field should be competitive or better. |
| `low_head_count` | Reduce heads from 16 to 2 with similar params elsewhere | Tests parallel pair diversity | Should degrade if many pair subspaces matter. |
| `pair_energy_only` | Remove square token final state and classify from pair energies | Tests standalone pair field | Useful diagnostic even if weaker. |

## Hardware Ablations

Because this packet is hardware-aware, report engineering metrics:

| Run | Why |
|---|---|
| `bf16_vs_fp32` | Confirms tensor-core path is numerically acceptable. |
| `compiled_vs_eager` | Measures whether `torch.compile` helps fixed-shape pair blocks. |
| `batch_scaling_512_1024_2048` | Shows whether throughput improves with larger batches. |
| `heads_scaling_4_8_16` | Separates model gain from GPU saturation. |
| `pair_rank_scaling_8_16_32` | Tests pair-field capacity. |

These are not central scientific ablations, but they are required to justify the "big GPU" premise.

## Diagnostics

Required shared diagnostics:

- binary accuracy
- AUROC
- PR-AUC
- Brier
- ECE
- fine-label `3 x 2` matrix

Architecture-specific diagnostics:

- same-rank pair energy by label
- same-file pair energy by label
- diagonal pair energy by label
- knight-offset pair energy by label
- occupied-to-occupied pair energy
- occupied-to-empty pair energy
- king-zone pair energy
- pair-field entropy proxy
- per-head energy specialization
- relation-shuffle ablation gap
- GPU throughput in boards/sec
- peak memory
- achieved mixed-precision speedup

## Expected Positive Result

The idea is promising if:

```text
full > cnn_only_matched
full > softmax_attention_control or matches it with better throughput
relation_bank_shuffle drops
no_pair_update drops
no_pair_readout drops
large batch throughput scales well
```

The strongest case is:

```text
pair-field model beats practical baselines and is much faster per effective parameter on a large GPU
```

## Expected Negative Result

Treat the idea as weak if:

- CNN-only matched equals the full model.
- Relation-bank shuffle matches full model.
- Softmax attention control beats it cleanly.
- Big-GPU configs overfit without improving validation/test.
- Throughput does not improve with larger batch sizes.
- Pair energies correlate mostly with material count.

## Failure Modes

- It may become a brute-force attention-like model with weak inductive bias.
- Large batch training may hurt generalization if optimization is not adjusted.
- Pair fields can overfit square priors.
- Pair summaries may collapse to material/occupancy density.
- The model is intentionally expensive; it is not suitable as a default baseline.

## Implementation Plan

### Files

```text
src/chess_nn_playground/models/tensorcore_pairfield.py
tests/test_tensorcore_pairfield.py
configs/model/tensorcore_pairfield_medium.yaml
configs/model/tensorcore_pairfield_big.yaml
```

### Modules

```text
SquareTokenProjector
SquarePairRelationBank
PairFieldBlock
PairEnergyReadout
TensorCorePairFieldNet
```

### Forward Pseudocode

```text
def forward(x):
    X = square_projector(x)                  # (B, 64, D)
    relation_bank = self.relation_bank       # (K, 64, 64)
    pair_summaries = []

    for block in blocks:
        Q, K, V = block.qkv(X)               # (B, H, 64, Dh)
        pair = matmul(Q, K.transpose(-1,-2)) # (B, H, 64, 64)
        pair = pair * scale
        pair = pair + block.relation_bias(relation_bank)
        weights = tanh(pair) * inv_sqrt_64
        msg = matmul(weights, V)             # (B, H, 64, Dh)
        X = block.update(X, msg)
        pair_summaries.append(block.summarize(pair, relation_bank, x))

    z_square = pool_square_tokens(X, x)
    z_pair = concat(pair_summaries)
    return classifier(concat(z_square, z_pair))
```

Implementation note:

- The loop over layers is acceptable.
- Do not loop over heads or squares.
- Relation bias should be precomputed or broadcasted as `(1, H, 64, 64)`.

## Unit Tests

Required:

- forward returns finite logits `(B, 2)`
- relation bank shape is fixed and deterministic
- relation bank contains no wraparound bugs for rank/file/diagonal relations
- `relation_bank_shuffle` preserves relation densities
- ablations run with same input shape
- BF16 autocast forward works on CUDA if available
- CPU smoke test works with tiny config

## Anti-Shortcut Controls

### Material Bucket Metrics

Report metrics inside coarse material buckets. Pair fields can easily become material-density detectors.

### Relation Bank Shuffle

Shuffle relation planes while preserving:

```text
density
symmetry
diagonal status
```

This is the central semantics-destroying ablation.

### Occupancy-Density Control

Train a control using only:

```text
occupied square count
piece counts
same-rank occupied pair counts
same-file occupied pair counts
same-diag occupied pair counts
```

If this count-only control matches, pair field semantics are not justified.

### Square Prior Control

Apply a fixed square permutation to input board planes and relation bank consistently. If performance stays high, model may rely on square priors rather than chess geometry. Apply a mismatched permutation as a stronger falsifier.

## Best Immediate Experiment

Do not begin with the heavy config. Start with:

```text
medium config
batch_size = 512
bf16 if CUDA supports it
```

Run:

```text
main
cnn_only_matched
relation_bank_shuffle
no_pair_update
softmax_attention_control
```

Then, only if the scientific ablations pass, run the big-GPU scaling study:

```text
batch 512 -> 1024 -> 2048
heads 4 -> 8 -> 16
pair_rank 8 -> 16 -> 32
```

The central question is:

```text
Can dense square-pair tensor-core computation turn a large GPU into better chess puzzle-likeness signal, not just more FLOPs?
```

If not, archive it as a clean negative result and keep future work focused on smaller structured models.

