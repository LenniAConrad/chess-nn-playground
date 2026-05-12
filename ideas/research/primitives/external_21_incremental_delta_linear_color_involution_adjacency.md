# Research Proposal: New Neural Primitives for Chess Evaluation

**Project:** `chess-nn-playground`  
**Advisor:** Deep Research Scout  
**Target:** Tsinghua University, Dept. of Automation

---

## 1. primitive_incremental_delta_linear (IDL)

**Name:** Incremental Delta-Linear Operator

**One-line claim:** A stateful linear operator that computes forward passes in O(K) time for K sparse input changes, maintaining a differentiable global state.

**Mathematical signature:**
Let $x \in \{0, 1\}^{N}$ be a sparse input. The operator maintains internal state $S \in \mathbb{R}^{M}$.
- **Forward:** $y_t = S_{t-1} + \sum_{i \in \text{indices}(\Delta x_t)} W_i$, where $S_t = y_t$.
- **Backward:** $\frac{\partial \mathcal{L}}{\partial W} = \sum_t \frac{\partial \mathcal{L}}{\partial y_t}$.

**Why this does not decompose:** Existing PyTorch `nn.Linear` is stateless and recomputes the full product $Wx$ every call. This primitive introduces persistent computational state into the graph. Decomposing it breaks the autograd graph for state $S$ across independent MCTS node calls unless wrapped in an $O(N)$ RNN structure.

**Chess-specific motivation:** Generalizes the NNUE accumulator property into a differentiable layer. Since chess moves change only ~2-4 squares, this allows $O(1)$ evaluation relative to board size.

**Complexity:**
- Forward: $O(K)$ incremental vs $O(N \cdot M)$
- Backward: $O(N \cdot M)$ (accumulated)
- Incremental update: $O(K)$

**Scout-scale falsification test:** Replace input Linear layer of i193 with IDL. Measure NPS. Success: >5x speedup with $\le$ 0.005 loss in PR AUC.

---

## 2. primitive_involutive_equivariant_linear (IEL)

**Name:** Color-Involutive Equivariant Linear

**One-line claim:** A linear operator with weight-tying constraints that natively enforces color-swap and spatial-mirroring symmetry as a single atomic contraction.

**Mathematical signature:**
$f(x): \mathbb{R}^{C \times 8 \times 8} \to \mathbb{R}^{C' \times 8 \times 8}$. Weight $W$ is constrained: $W = \mathcal{T}(W)$, where $\mathcal{T}$ involves a $180^\circ$ rotation and channel-wise involution.

**Why this does not decompose:** Standard `nn.Conv2d` cannot tie weights across spatial rotations and channel-permutation indices simultaneously in the CUDA kernel.

**Chess-specific motivation:** Chess is symmetric under (color swap + vertical flip). IEL enforces this as a hard prior, making the 173k scout dataset act like 346k.

**Scout-scale falsification test:** Compare IEL i242 vs baseline with 2x data augmentation. Success: Equal PR AUC with 0.5x training time.

---

## 3. primitive_adjacency_gated_reduction (AGR)

**Name:** Legal-Move Adjacency Gated Reduction

**One-line claim:** A sparse message-passing operator where the receptive field is dynamically defined by the legal move mask of the input position.

**Mathematical signature:**
$y_i = \text{Pool}(\{ x_j \cdot w_{type(i)} \mid j \in \text{LegalMoves}(i) \})$

**Why this does not decompose:** Standard Attention is $O(N^2)$. AGR uses a content-dependent sparse topology that changes per pass. Decomposing into masked attention is inefficient; AGR computes only legal edges.

**Chess-specific motivation:** Exploits legal-move sparsity. A piece's influence is restricted to its legal moves, focusing attention on tactically relevant squares.

**Scout-scale falsification test:** Replace attention in i242 with AGR. Success: Reduction in "blind spot" tactical blunders in CRTK Class 1.

---

## 4. primitive_piece_pair_interaction (PPI)

**Name:** Piece-Pair Interaction Kernel

**One-line claim:** A non-linear operator that computes a learned interaction term for every pair of pieces, indexed by piece type and relative distance.

**Mathematical signature:**
$y = \sum_{i,j} \phi(\text{type}_i, \text{type}_j, \text{dist}(i,j))$

**Why this does not decompose:** This is a higher-order interaction. In PyTorch, constructing a $64 \times 64$ pair-wise matrix is memory intensive; PPI implements this as a fused kernel.

**Chess-specific motivation:** Directly models "Batteries" or attacker counts near the King without relying on deep convolution stacks to discover distance relations.

**Scout-scale falsification test:** Add PPI as a side-car feature extractor to i193. Success: Improved PR AUC on material imbalances.

---

## 5. primitive_gated_spatial_lookup (GSL)

**Name:** Gated Spatial-Temporal Lookup

**One-line claim:** An activation-like primitive that gates feature $x_i$ based on the content of square $j$, where $j$ is the target of a specific piece (e.g., the King).

**Mathematical signature:**
$y_i = x_i \cdot \sigma(W \cdot x_{\text{index}(piece)})$

**Why this does not decompose:** Standard ops are position-invariant. GSL uses dynamic indexing to find specific landmarks. Slicing/broadcasting in PyTorch breaks the static graph; GSL is a differentiable gate.

**Chess-specific motivation:** Evaluation is King-centric. GSL allows features to be "King-aware" in one step, shortening the tactical integration path.

**Scout-scale falsification test:** Insert King-gated GSL into i243. Success: Lower score variance between consecutive moves in PV.
