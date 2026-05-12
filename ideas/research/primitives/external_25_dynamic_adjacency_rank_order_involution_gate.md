# Deep Research: Neural Primitives for Chess Evaluation
**Project:** chess-nn-playground
**Advisor:** Gemini
**Date:** May 2026

---

## Reasoning Mode: Selection Rationale
To move beyond the "architectural soup" of the i242/i243 scouts, we must identify mathematical operations where the **computation graph itself** changes. 

The primary filter used for these five primitives was the **"Fixed-Grid Infidelity"** problem. Most current chess NNs treat the board as a static image (CNNs) or a fully connected set of tokens (Transformers). However, chess is a **dynamic graph** where connectivity is restricted by rules. These proposals focus on:
1.  **Topology:** Forcing the network to respect legal move graphs (DAG).
2.  **Order Statistics:** Learning from relative piece strength rather than just spatial presence (ROP).
3.  **Symmetry Groups:** Hard-coding the $Z_2$ involution of color-swaps (IIG).
4.  **Differential Updates:** Optimizing for the fact that only 2/64 squares change per move (SAD).
5.  **Metric Space:** Correcting the Euclidean bias of standard convolutions (GPI).

---

### 1. primitive_dynamic_adjacency_gate (DAG)

**Name:** Dynamic Adjacency-Conditioned Gating

**One-line claim:** A gating primitive where the connectivity graph is a learnable, non-differentiable function of input state, bypassable only via straight-through estimation.

**Mathematical signature:**
$$y = (G(x) \odot Wx) + b$$
Where $G: \mathbb{R}^{B 	imes N 	imes D} 	o \{0, 1\}^{B 	imes N 	imes N}$ is a discrete adjacency matrix generator. 
$$G(x)_{i,j} = \mathbb{I}[	ext{Move}(i 	o j) \in 	ext{LegalMoves}(x)]$$

**Why this does not decompose into existing PyTorch ops:**
Standard `torch.nn` ops use fixed masks or soft-attention weights. This introduces a **hard, discrete topological constraint** directly into the kernel. Unlike a Softmax mask, the gradient flow is zeroed for non-legal connections at the hardware level.

**Chess-specific motivation:**
Chess info-flow is constrained by the move graph. Standard Convs assume spatial proximity. DAG enforces that a Piece at d2 can only update features of d4 if that move is legal in state $x$.

**Complexity:**
- Forward: $O(B \cdot 	ext{ActiveEdges} \cdot D)$
- Backward: $O(B \cdot 	ext{ActiveEdges} \cdot D)$
- Incremental update: $O(1)$

**Scout-scale falsification test:**
Replace global self-attention in i242 with a DAG layer.
- **Metric:** Matched-recall near-puzzle FP rate.
- **Success:** $>15\%$ reduction in FPs on CRTK Class 1 puzzles.

**Status:** proposed

---

### 2. primitive_rank_order_pool (ROP)

**Name:** Permutation-Invariant Rank-Order Pooling

**One-line claim:** A pooling operator that sorts feature activations and applies learned weights to sorted values, capturing relative strength rather than spatial location.

**Mathematical signature:**
$$y = W \cdot 	ext{sort}(x, 	ext{dim}=-1)$$
$$f: \mathbb{R}^{B 	imes N 	imes D} 	o \mathbb{R}^{B 	imes N 	imes K}$$

**Why this does not decompose into existing PyTorch ops:**
Learns a linear transformation in the **sorted value domain**. Standard layers are sensitive to index; ROP is sensitive to the *distribution of magnitudes*.

**Chess-specific motivation:**
Evaluation is often about the 1st/2nd strongest attackers. ROP learns "if my top 2 attackers are stronger than their top 2 defenders" regardless of piece identity.

**Complexity:**
- Forward: $O(D \log D)$
- Backward: $O(D)$
- Incremental update: $O(\log D)$

**Scout-scale falsification test:**
Insert after initial feature extraction in 234-scout.
- **Metric:** PR AUC on material-balanced imbalanced positions.
- **Success:** Outperforming Global Average Pooling by $>2\%$.

**Status:** proposed

---

### 3. primitive_involutional_inv_gate (IIG)

**Name:** Bit-Flip Involutional Symmetry Gating

**One-line claim:** A weight-sharing operator enforcing color-swap symmetry by design, where the White-to-move gradient is the negative-conjugate of Black-to-move.

**Mathematical signature:**
$$W_{eff} = W \cdot (-1)^{s}$$ where $s \in \{0, 1\}$ is side-to-move.
$$y = \sigma(x st W_{eff} + b \cdot (-1)^s)$$

**Why this does not decompose into existing PyTorch ops:**
A **kernel-level primitive** where weights flip sign based on a binary scalar $s$ during the *same* forward pass, forcing the network to reside in the symmetry space.

**Chess-specific motivation:**
A +1.0 eval for White must be -1.0 for Black in a mirrored position. IIG makes this a mathematical necessity, doubling training data efficiency.

**Complexity:**
- Forward: $O(1)$ overhead.
- Backward: $O(1)$.

**Scout-scale falsification test:**
Apply to i193 conv-only parent.
- **Metric:** Symmetry Error.
- **Success:** Zero symmetry error and target PR AUC in 6 epochs instead of 12.

**Status:** proposed

---

### 4. primitive_sparse_accumulator_diff (SAD)

**Name:** Difference-Encoded Sparse Accumulator

**One-line claim:** A linear-to-nonlinear primitive computing outputs based solely on the XOR-diff of the input bitmask, generalizing NNUE updates.

**Mathematical signature:**
$$H_t = H_{t-1} + W(\Delta x^+ - \Delta x^-)$$
$$y = \phi(H_t)$$

**Why this does not decompose into existing PyTorch ops:**
It is a **"Change-Forwarding"** primitive. It requires persistent hidden state $H$ across calls, unlike stateless standard modules.

**Chess-specific motivation:**
In MCTS, board changes by 2 squares. SAD turns the efficiency of Stockfish’s HalfKA into a generic neural primitive.

**Complexity:**
- Forward (Incremental): $O(k \cdot D)$ where $k \ll N$.
- Incremental update: $O(k \cdot D)$.

**Scout-scale falsification test:**
Drop into i243 proposal.
- **Metric:** Inference NPS.
- **Success:** $5	imes$ to $10	imes$ increase in NPS without loss in PR AUC.

**Status:** proposed

---

### 5. primitive_geometric_piece_interaction (GPI)

**Name:** Non-Euclidean Geometric Interaction Kernel

**One-line claim:** A kernel computing interactions using the "Chess Metric" (Chebyshev distance) instead of L2/Euclidean.

**Mathematical signature:**
$$y_{i,j} = \sum_{m,n} W_{m,n} \cdot x_{i+m, j+n}$$
Where weights $W$ are sampled from $g(d_{Chebyshev}((i,j), (m,n)))$.

**Why this does not decompose into existing PyTorch ops:**
Uses **metric-aware sampling** where the "receptive field" is a perfect square, reflecting King/Queen movement logic.

**Chess-specific motivation:**
Diagonal moves are not "further" in chess logic. Standard kernels treat diagonals as $\sqrt{2}$, which is strategically false.

**Complexity:**
- Forward: $O(N \cdot K^2)$ (Identical to Conv).
- Incremental update: $O(K^2)$.

**Scout-scale falsification test:**
Swap 3x3 convs in 234-scout with GPI kernels.
- **Metric:** Accuracy on "Long-range piece coordination" set.
- **Success:** Significant improvement in detecting discovered attacks.

**Status:** proposed

---

### Candidates Rejected During Self-Audit
1. **Piece-Type Attention:** Rebrand of Multi-Head Attention.
2. **Squeeze-and-Excitation (Chess):** Standard composition.
3. **Bitboard-Convolution:** Encoding, not a primitive.
