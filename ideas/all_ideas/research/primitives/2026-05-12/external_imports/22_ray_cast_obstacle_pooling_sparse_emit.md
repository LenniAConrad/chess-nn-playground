# Research Advisory: Proposed Neural Primitives for `chess-nn-playground`

This document outlines five novel neural primitives designed to advance chess engine evaluation beyond standard architectural compositions. These proposals focus on **computational topology**, **stateful inference**, and **domain-specific conservation laws**.

---

### 1. primitive_raypool
**Name:** Ray-Cast Obstacle Pooling (`RayPool`)

**One-line claim:** Dynamically aggregates features along 8 cardinal directions, strictly terminating computation at data-defined obstacle tokens via a fused parallel scan.

**Mathematical signature:**
$f: \mathbb{R}^{B \times 64 \times d} \times [0,1]^{B \times 64} \rightarrow \mathbb{R}^{B \times 64 \times d}$
$Y_i = \sum_{dir \in \mathcal{D}} \sum_{s=1}^{7} \gamma^s X_{i + s \cdot dir} \prod_{k=1}^{s-1} (1 - O_{i + k \cdot dir})$

**Why this does not decompose into existing PyTorch ops:**
Standard PyTorch requires generating a 64x64 path-integral mask via cumulative products to simulate "blocking." This is an O(N^3) operation. A true primitive executes this natively with 1D prefix sums in a custom CUDA kernel, bypassing the dense interaction matrix entirely.

**Chess-specific motivation:**
Sliding pieces (Rooks, Bishops, Queens) exert influence that is strictly bounded by the first piece they encounter. `RayPool` bakes "Line of Sight" directly into the operator.

**Generalisation beyond chess:**
Applicable to 2D grid pathfinding and robotic line-of-sight algorithms.

**Complexity:**
- Forward: O(N * d)
- Backward: O(N * d)
- Incremental update: O(1) along unchanged rays.

**Status:** proposed

---

### 2. primitive_deltagelu
**Name:** State-Cached Sparse Delta Activation (`DeltaGELU`)

**One-line claim:** Evaluates non-linearities only on changed input indices, maintaining an internal graph cache to enable deep O(1) updates across activation boundaries.

**Mathematical signature:**
$f: \mathbb{R}^{B \times d}_{cache} \times \mathbb{R}^{B \times \Delta k} \times \mathbb{N}^{B \times \Delta k} \rightarrow \mathbb{R}^{B \times \Delta k}$
$C_t[idx] = C_{t-1}[idx] + \Delta X$
$\Delta Y = \text{GELU}(C_t[idx]) - \text{GELU}(C_{t-1}[idx])$

**Why this does not decompose into existing PyTorch ops:**
PyTorch operations are stateless. This primitive fuses the state cache and the derivative expansion into a single stateful autograd node, which is impossible via composition.

**Chess-specific motivation:**
NNUE achieves speed through O(1) updates. Standard deep networks lose this at the first non-linearity. `DeltaGELU` allows O(1) speedups to propagate deep into the network.

**Status:** proposed

---

### 3. primitive_legal_attn
**Name:** Data-Dependent Sparse Adjacency Attention (`LegalMoveAttn`)

**One-line claim:** Computes attention strictly over a variable-length list of engine-provided adjacency indices, bypassing dense NxN logit calculations.

**Mathematical signature:**
$f: \mathbb{R}^{B \times 64 \times d} \times \mathbb{N}^{B \times 64 \times M_{max}} \rightarrow \mathbb{R}^{B \times 64 \times d}$
$Y_i = \sum_{j \in E_i} \text{softmax}_j \left( \frac{Q_i \cdot K_j}{\sqrt{d}} \right) V_j$

**Why this does not decompose into existing PyTorch ops:**
Standard attention (SDPA) physically instantiates the dense 64x64 matrix even when masked. This primitive scatters/gathers only along the exact topological edges provided.

**Chess-specific motivation:**
The chess graph is sparse (avg 35 edges). A piece should structurally only "attend" to squares it can actually move to or defend.

**Status:** proposed

---

### 4. primitive_zerosum_exchange
**Name:** Mass-Conserving Feature Routing (`ZeroSumExchange`)

**One-line claim:** A routing operator where feature magnitude sent to node j is strictly deducted from node i, ensuring global feature conservation.

**Mathematical signature:**
$Y_i = X_i + \sum_{j} A_{j,i} (W X_j) - \sum_{j} A_{i,j} (W X_i)$

**Why this does not decompose into existing PyTorch ops:**
A fused primitive ensures that the gradient of total feature mass sums to exactly zero across the graph, preventing the numerical drift inherent in multi-op compositions.

**Chess-specific motivation:**
Chess is zero-sum. A pinned piece cannot "spend" its defensive capacity on an offensive sortie. This operator forces the network to allocate finite piece-utility.

**Status:** proposed

---

### 5. primitive_sparse_emit
**Name:** Sparsity-Preserving Linear Operator (`SparseEmitLinear`)

**One-line claim:** A linear layer that aborts dot-product calculations falling below a learned threshold to output a compressed sparse tensor directly.

**Mathematical signature:**
$Y_{i,j} = \text{threshold}(\sum X_{i,k} W_{k,j}, \tau)$

**Why this does not decompose into existing PyTorch ops:**
Standard masking calculates the full dense matrix before zeroing. This primitive prunes the dot product in-flight, never allocating dense intermediate memory.

**Chess-specific motivation:**
In quiet positions, most "tactical danger" features should evaluate to exact zero. If a feature is below threshold, it should cost zero FLOPs to propagate.

**Status:** proposed

---
## Reasoning & Self-Audit (Internal Review)

**Design Philosophy:**
The transition from the i242/i243 scout phase to **Primitive Engineering** is driven by the realization that current architectures are limited by their reliance on dense, sequence-agnostic operators. 

**What I Cut:**
- **Color-Symmetry Enforcer:** Rejected as it is a weight-sharing composition, not a primitive.
- **Bipartite Piece Gate:** Rejected for being a simple sigmoid-mask composition.
- **Static Grid Convolutions:** Rejected because 3x3 kernels cannot capture long-range influence as efficiently as a dedicated ray-casting primitive.
