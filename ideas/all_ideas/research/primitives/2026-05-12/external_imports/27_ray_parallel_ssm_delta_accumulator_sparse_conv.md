# Research Proposal: Novel Neural Primitives for Chess Evaluation

This document outlines five proposed neural-network primitives for the `chess-nn-playground` project. These proposals focus on mathematical operators that exploit the specific structural properties of chess (sparsity, ray-based movement, and incremental state updates) while remaining generalizable to broader deep-learning contexts.

---

### primitive_ray_ssm

**Name:** Ray-Parallel Selective State Space Model (Ray-SSM)

**One-line claim:** A selective recurrence operator that propagates information along eight cardinal chess rays simultaneously with $O(N)$ complexity, replacing $O(N^2)$ global attention for sliding pieces.

**Mathematical signature:**
Let $x \in \mathbb{R}^{64 	imes d}$ be the board state. Let $D \in \{1 \dots 8\}$ be ray directions.
For each direction $d$, define a sequence order $\pi_d$ (e.g., A1 $ightarrow$ A8).
$h_{i,d} = A_{i,d} h_{i-1,d} + B_{i,d} x_i$
$y_i = \sum_{d=1}^8 C_{i,d} h_{i,d}$
Where $A, B, C$ are functions of $x_i$ (selective), and the recurrence follows board geometry.

**Why this does not decompose into existing PyTorch ops:**
Existing SSMs (like Mamba) are 1D. While 2D-SSMs exist, they typically use causal rasters (top-left to bottom-right). Ray-SSM uses a *multi-directional topological sort* of the graph defined by the board's rank, file, and diagonals. To implement this in standard PyTorch, one would need 8 separate scans with complex indexing that breaks the fused-kernel efficiency of selective SSMs.

**Chess-specific motivation:**
Chess is defined by "rays" (Rooks, Bishops, Queens). Standard Convolutions have a limited receptive field; Attention is too expensive. Ray-SSM allows a piece at A1 to "see" a piece at H8 through a stateful recurrence that mimics the legal path of a Bishop, capturing long-range blockers/interactions in a single forward pass.

**Generalisation beyond chess:**
Applicable to any 2D/3D grid where physics or logic follows specific axes (e.g., fluid dynamics on a grid, or routing in urban street networks).

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot d)$ vs Attention $O(N^2 \cdot d)$
- Backward: $O(N \cdot d)$
- Incremental update: $O(	ext{ray\_length})$ — significantly faster than re-computing global attention.

**Scout-scale falsification test:**
Replace the global attention layer in the `i242` architecture with a `RaySSM` layer. Target: Achieve equal or lower "matched-recall near-puzzle FP rate" on CRTK Class 1 with at least a 30% increase in inference nodes/second on an RTX 3070.

**Failure mode catalogue:**
- Could be seen as a hidden rebrand of a "4-way S6" (Mamba-2D) if the directionality isn't strictly ray-bound.
- Numerical instability in the $A$ matrix during the scan if piece-conditioning is too aggressive.
- The overhead of 8 parallel scans might negate the theoretical $O(N)$ gains on small 8x8 grids.

**Status:** proposed

---

### primitive_delta_accel

**Name:** Differentiable Delta-Accumulator (DDA)

**One-line claim:** A primitive that maintains a latent global state by processing only the *change* in input tokens, generalising NNUE's $O(1)$ update to arbitrary high-dimensional embeddings.

**Mathematical signature:**
$S_t = S_{t-1} + \sum_{i \in 	ext{Changed}} (f(x_{i, 	ext{new}}) - f(x_{i, 	ext{old}}))$
$y = 	ext{LayerNorm}(\phi(S_t))$
Where $f$ is a learned embedding and $\phi$ is a non-linear projection.

**Why this does not decompose into existing PyTorch ops:**
PyTorch is built on the "tensor-in, tensor-out" paradigm for a single board. This op requires a persistent buffer $S$ that persists across `forward()` calls and a gradient path that correctly handles the "additive identity" of the state. Standard RNNs ($h = \sigma(Wh + Ux)$) do not support the $O(1)$ update property because $h$ is a non-linear function of its previous self.

**Chess-specific motivation:**
A chess move changes exactly two squares (usually). NNUE's success is due to its $O(1)$ accumulator. DDA brings this efficiency to deep layers, allowing the network to "update" its understanding of the board rather than re-calculating it from scratch at every MCTS node.

**Generalisation beyond chess:**
Sparse event streams (high-frequency trading, real-time sensor monitoring) where the input vector is mostly static between time steps.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(d)$ (independent of board size $N$)
- Backward: $O(d)$ per modified token
- Incremental update: $O(1)$ relative to $N$.

**Scout-scale falsification test:**
Insert DDA as the first hidden layer in a 12-block MLP. Compare wall-clock time for a search of depth 10 against a standard MLP. A success is defined as <10% loss in accuracy with >5x speedup in "per-move" forward passes.

**Failure mode catalogue:**
- If not careful, this is just a Linear layer with a stateful wrapper (rebrand).
- Floating point drift: over thousands of moves, the additive $S$ might accumulate errors compared to a fresh forward pass.
- Requires custom CUDA kernels to be faster than standard `torch.add`.

**Status:** proposed

---

### primitive_move_gated_conv

**Name:** Topology-Conditional Sparse Convolution

**One-line claim:** A convolution where the kernel weights are dynamic and determined by the *legal move graph* of the input board.

**Mathematical signature:**
$y_i = \sum_{j \in 	ext{LegalMoves}(i)} W_{piece(i), piece(j)} \cdot x_j$
The connectivity $Adj(i, j)$ is a bitmask derived from the input $x$.

**Why this does not decompose into existing PyTorch ops:**
Standard `Conv2d` uses a fixed spatial grid (e.g., 3x3). `GraphConv` uses a static or pre-calculated adjacency. This op requires the *adjacency matrix to be a function of the input feature map* in a way that dictates the flow of the gradient only through "legal" paths.

**Chess-specific motivation:**
In chess, a Knight at D4 is "near" F5 but "far" from D5. Standard CNNs treat them as equally distant or use many layers to learn the L-shape. This primitive enforces the "rules of the game" directly into the connectivity, forcing the network to process information according to piece dynamics.

**Generalisation beyond chess:**
Pathfinding networks, circuit design (where connectivity depends on switch states), and logistics optimization.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot 	ext{avg\_degree})$
- Backward: $O(N \cdot 	ext{avg\_degree})$
- Incremental update: $O(	ext{changed\_moves})$

**Scout-scale falsification test:**
Benchmark against a standard 3x3 ResNet. Use the "legal-move mask" as the adjacency. Success: Higher accuracy on "Matched-recall near-puzzle FP rate" (tactical sensitivity) with fewer parameters.

**Failure mode catalogue:**
- Might decompose into a "Masked Attention" where $Q, K$ are fixed and only the mask changes (hard to prove novelty).
- Computationally expensive to re-generate the legal-move mask for every sample in a batch.
- Gradient might not flow well to piece-type embeddings if the move-graph is too sparse.

**Status:** proposed

---

### primitive_involution_sym

**Name:** Bilateral Involution Operator (BIO)

**One-line claim:** A weight-sharing primitive that enforces exact color-flip and board-mirror symmetry within the layer's internal logic, rather than via data augmentation.

**Mathematical signature:**
Let $P$ be the board-reversal permutation matrix. The operator $F$ must satisfy:
$F(P x) = P F(x)$
The weights $W$ are constrained such that $W = P W P^{-1}$.

**Why this does not decompose into existing PyTorch ops:**
While Group Equivariant CNNs (G-CNNs) exist for rotation/translation, the "color-flip + board-reversal" is a specific involution. Implementing this typically involves running two streams and averaging (architecture) or flipping data. A primitive implementation would define the *weight initialization and update* on the manifold of symmetric matrices, ensuring 100% symmetry by construction.

**Chess-specific motivation:**
A position and its color-flipped mirror are identical in value (negated). Standard networks often "learn" that a White King on G1 is safe but need more data to learn the same for a Black King on G8. BIO forces this knowledge from step zero.

**Generalisation beyond chess:**
Digital twins (where left/right symmetry is physical), parity-check codes, and synthetic chemistry.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot d)$ (same as linear/conv)
- Backward: $O(N \cdot d)$
- Incremental update: N/A

**Scout-scale falsification test:**
Train a model with 50% of the usual data. If the BIO-primitive model achieves the same accuracy as a standard model trained on 100% data (due to inherent symmetry), the primitive works.

**Failure mode catalogue:**
- Too restrictive: the network might need slight asymmetries to model specific engine biases.
- Implementing the constraint $W = P W P^{-1}$ as a soft-penalty is a "trick"; a real primitive needs a hard-coded symmetric basis.
- Might be redundant if HalfKA encoding already handles symmetry at the input level.

**Status:** proposed

---

### primitive_soft_logic_gate

**Name:** Differentiable Bit-Logic Aggregator (DBLA)

**One-line claim:** A soft-logic primitive that performs learned AND/OR/XOR operations over bitboard-like features, replacing sum-product arithmetic with Boolean-style logic.

**Mathematical signature:**
$z = 1 - \prod_i (1 - w_i x_i)$ [Soft-OR]
$z = \prod_i (w_i x_i + (1 - w_i))$ [Soft-AND]
Where $w_i \in [0, 1]$ are learned weights.

**Why this does not decompose into existing PyTorch ops:**
Standard neurons use $f(\sum wx + b)$. While an MLP *can* approximate an AND gate, it is a poor fit for the sharp transitions of Boolean logic. DBLA changes the fundamental accumulation from "Sum" to "Product" (or its log-space equivalent), creating a different gradient flow that is more sensitive to "missing" conditions (e.g., "Is the square empty AND is it my turn?").

**Chess-specific motivation:**
Chess rules are discrete and Boolean (bitboards). "Is the king in check" is a complex logical reduction of multiple conditions. Using logical primitives allows the network to find "bottleneck" conditions more efficiently than additive weights.

**Generalisation beyond chess:**
Formal verification, rule-based expert systems, and circuit discovery.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot d)$
- Backward: $O(N \cdot d)$
- Incremental update: N/A

**Scout-scale falsification test:**
Apply DBLA to the raw bitboard inputs before any other layers. Compare it to a single-layer MLP of equal width. Success: significantly better "hard-negative" discrimination on tactical puzzles (where one Boolean condition, like a pinned piece, changes everything).

**Failure mode catalogue:**
- Numerical vanishing/exploding gradients due to repeated products (requires log-space implementation).
- The "Soft" logic might just collapse into a standard Sigmoid-activated MLP in practice.
- Hard to optimize on current hardware compared to highly-tuned MatMuls.

**Status:** proposed

---

## What I cut (Self-Audit)

1. **"Global King-Centric Attention"**: Rejected because it's just standard attention with a specific positional encoding. Decomposes into `Softmax(QK^T)V`.
2. **"Monte-Carlo Tree Search Layer"**: Rejected because it's an architecture/algorithm integration, not a primitive. It doesn't have a simple differentiable signature.
3. **"Piece-Type Mixture of Experts"**: Rejected because MoE is an existing primitive. Using "Piece-Type" as the router is an application, not a new operator.
4. **"Move-Graph Laplacian Smoothing"**: Rejected because it's a standard Graph Convolution technique. Not novel enough to be a new primitive in the `torch.nn` sense.
