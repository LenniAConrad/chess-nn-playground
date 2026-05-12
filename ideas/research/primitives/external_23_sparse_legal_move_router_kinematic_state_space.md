# Research Proposal: Novel Neural Primitives for Chess Evaluation
**Project:** `chess-nn-playground`
**Advisor:** Gemini (Deep Research)

## Overview
This proposal introduces five new neural primitives designed to encode the "physics" of chess directly into the gradient flow. These operators are structurally distinct from standard PyTorch modules and are optimized for the constraints of engine-scale inference on hardware like the RTX 3070.

---

### 1. primitive_slmr: Sparse Legal-Move Router
**One-line claim:** A sparse graph-interaction operator where connectivity is dynamically redefined by the legal moves in the current board state.

**Mathematical Signature:**
$$f(X, M) \rightarrow Y$$
*   $X \in \mathbb{R}^{B, 64, D}$ (Square features)
*   $M \in \{0, 1\}^{B, 64, 64}$ (Legal-move adjacency matrix)
*   $Y_i = \text{Agg}(\{ \phi(X_i, X_j, W) \mid M_{ij} = 1 \})$

**Why it is unique:**
Standard Attention materializes a dense $N \times N$ matrix. SLMR is a true sparse primitive using CSR/COO formats for fused scatter-add, where the topology is a hard, data-dependent constraint that changes every forward pass.

**Complexity:**
*   **Forward:** $O(L)$ (where $L$ is the number of legal moves) vs Attention $O(N^2)$
*   **Incremental Update:** $O(1)$ on a bounded-change input

---

### 2. primitive_kds: Kinematic Deformable Sampling
**One-line claim:** A 2D convolution variant where sampling offsets are looked up from a learned "kinematic table" based on the piece-type at the origin.

**Mathematical Signature:**
$$Y(p) = \sum_{k=1}^K W_k \cdot X(p + \Delta p_{\text{piece}(p), k})$$

**Why it is unique:**
Standard Conv uses fixed grids. KDS uses **index-based offset selection**, meaning the kernel "shape" is a discrete property of the specific piece sitting at square $p$.

**Chess-specific motivation:**
Automatically adjusts receptive fields (e.g., long radial lines for Rooks, local boxes for Kings) without manual feature engineering.

---

### 3. primitive_issc: Incremental State-Space Cell
**One-line claim:** A selective state-space operator that updates a global board representation using only the sparse "delta" between consecutive positions.

**Mathematical Signature:**
$$h_t = A(m_t)h_{t-1} + B(m_t)x(m_t)$$

**Why it is unique:**
Most RNNs/SSMs process full tokens. ISSC is **delta-driven**, treating "the change" as the primary input and maintaining internal state across moves, effectively acting as a neural Stockfish NNUE Accumulator.

---

### 4. primitive_cif: Color-Invariant Folding
**One-line claim:** A non-linear activation primitive that enforces perfect color-symmetry by mapping board states to a canonical "active-player" manifold.

**Mathematical Signature:**
$$f(x) = \sigma(x) \cdot \text{sgn}(\text{side\_to\_move})$$

**Why it is unique:**
It moves color symmetry from the input encoding (flipping planes) into the **weight space**, ensuring equivariance by construction through tied gradient flows.

---

### 5. primitive_bpip: Bilinear Piece-Interaction Pooling
**One-line claim:** A second-order interaction primitive computing a weighted outer product between piece-type feature planes to capture complex material imbalances.

**Mathematical Signature:**
$$Y = \sum_{i,j} W_{i,j} (P_i \otimes P_j)$$

**Why it is unique:**
Unlike linear layers, BPIP explicitly calculates the interaction *between* planes (e.g., Rook & Knight vs Queen) as a first-class multiplicative feature. This allows the model to "count" non-linear piece combinations like the "Bishop pair" bonus.

---

### Evaluation Criteria (Scout-Scale)
To validate these primitives on your RTX 3070, use the **matched-recall near-puzzle FP rate** as the primary metric rather than aggregate PR AUC. A successful primitive must demonstrate at least a **5x speedup** or a significant reduction in false positives for verified-near-puzzle positions.
