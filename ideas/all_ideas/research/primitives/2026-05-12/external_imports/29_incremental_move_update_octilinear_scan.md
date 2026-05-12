# Research Proposal: Novel Neural Primitives for Chess Evaluation
**Project:** chess-nn-playground  
**Target Hardware:** RTX 3070 (8 GiB)  
**Baseline:** i193 (Conv-only) / i242 (Decomposed Attention)

---

### 1. primitive_imua

**Name:** Incremental Move-Update Accumulator (IMUA)

**One-line claim:** A differentiable state-update operator that maps a parent latent to a child latent using sparse move-delta indices, generalizing NNUE to deep latents.

**Mathematical signature:**
$$f: (H_{t}, \mathcal{I}_{add}, \mathcal{I}_{sub}) \to H_{t+1}$$
$$H_{t+1} = \text{LayerNorm}(H_t + \phi(\mathcal{I}_{add}) - \phi(\mathcal{I}_{sub})) \odot \text{Gate}(H_t)$$
Where $H \in \mathbb{R}^d$, and $\mathcal{I}$ are sparse index sets for piece-square entries.

**Why this does not decompose into existing PyTorch ops:**
Standard RNNs process sequences linearly. IMUA is a **tree-structured update**. While one could hack this with `EmbeddingBag`, IMUA requires a custom kernel to maintain a non-linear latent state that survives branching MCTS paths without re-computing the full board, something standard linear layers cannot do efficiently.

**Chess-specific motivation:**
It exploits the **incremental nature of chess**. In MCTS, you only change 2 squares per node. IMUA allows the network to carry over "concepts" from the parent node and only update the delta, drastically increasing search throughput.

**Generalisation beyond chess:**
Dynamic graph environments where edges/nodes change incrementally (e.g., chemical reaction modeling).

**Complexity (forward, backward, incremental-update):**
- Forward: $O(d)$
- Backward: $O(d)$ per move-path
- Incremental update: $O(1)$ relative to board size.

**Scout-scale falsification test:**
Replace the ResNet backbone with an IMUA chain. 
- **Baseline:** i193 (ResNet). 
- **Metric:** Nodes per second during search. 
- **Success:** $\geq 2\times$ NPS with < 1% loss in PR AUC.

**Failure mode catalogue:**
- Hidden rebrand of a Recursive Neural Network.
- Numerical drift over deep plies.
- Gradient vanishing in the "gate" mechanism.

---

### 2. primitive_oss

**Name:** Octilinear Selective Scan (OSS)

**One-line claim:** A 2D spatial operator performing parallel selective scans along the 8 chess rays to model piece-blockage and long-range influence.

**Mathematical signature:**
$$f: X \in \mathbb{R}^{B \times 64 \times d} \to Y \in \mathbb{R}^{B \times 64 \times d}$$
For each direction $k \in \{1 \dots 8\}$:
$h_t = A_k(x_t)h_{t-1} + B_k(x_t)x_t$
Where $A_k$ is a data-dependent transition matrix (selectivity).

**Why this does not decompose into existing PyTorch ops:**
Unlike `Conv2d` (local) or `Attention` (global/isotropic), OSS is **directional and recursive**. It uses the "Selective Scan" (S6) logic but mapped to the 8 directions of the chess board. There is no standard PyTorch op that performs a 1D recurrence across a 2D grid in 8 specific directions simultaneously.

**Chess-specific motivation:**
Chess is a game of **blocked rays**. A Bishop’s power depends on whether squares are occupied. OSS treats the board as a sequence of squares where the "state" (influence) is propagated or blocked based on piece occupancy.

**Generalisation beyond chess:**
Medical imaging (X-ray/CT ray-tracing simulations) and Lidar processing.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot d)$ (where $N=64$)
- Backward: $O(N \cdot d)$
- Incremental update: $O(N)$ (must re-scan affected rays).

**Scout-scale falsification test:**
Replace attention layers in i242 with OSS. 
- **Baseline:** i242 (Chess-decomposed attention). 
- **Metric:** Matched-recall near-puzzle FP rate. 
- **Success:** Lower FP rate than i242 with significantly lower VRAM usage.

**Failure mode catalogue:**
- High implementation complexity (needs Triton/CUDA).
- Redundant if the model is deep enough to "simulate" rays via Convs.
- Sensitive to piece-ordering within the scan.

---

### 3. primitive_eptp

**Name:** Equivariant Piece-Type Permutator (EPTP)

**One-line claim:** A layer that enforces invariance to piece-identity swaps for identical piece types, effectively augmenting data within the architecture.

**Mathematical signature:**
$$f: \mathcal{S} \to \mathbb{R}^d, \text{ where } \mathcal{S} = \{ (type_i, pos_i) \}_{i=1}^n$$
$f$ must satisfy $f(\pi(\mathcal{S})) = f(\mathcal{S})$ for any permutation $\pi \in S_{k}$ where $S_k$ is the group of identical pieces (e.g., the two White Knights).

**Why this does not decompose into existing PyTorch ops:**
Standard CNNs are position-sensitive but identity-agnostic. `DeepSets` are identity-sensitive but ignore the 2D spatial grid. EPTP combines these by enforcing **joint-equivariance** between the piece-type set and the board's spatial symmetry.

**Chess-specific motivation:**
Neural networks often overfit to specific piece "slots." EPTP forces the model to realize that Knight A and Knight B are functionally the same entity, improving generalization on the limited 173k position dataset.

**Generalisation beyond chess:**
Multi-agent robotics and particle physics.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(n \cdot d)$
- Backward: $O(n \cdot d)$
- Incremental update: $O(1)$.

**Scout-scale falsification test:**
Use EPTP in the initial embedding layer. 
- **Baseline:** Standard 12-plane input. 
- **Metric:** Generalization gap (Train vs. Val accuracy). 
- **Success:** $>10\%$ reduction in gap without adding parameters.

**Failure mode catalogue:**
- Hidden rebrand of a specific Graph Isomorphism Network (GIN).
- May lose "piece-square" nuance if too rigid.
- Computationally slower than simple tensor lookups.

---

### 4. primitive_lmhi

**Name:** Legal-Masked Hyper-Interaction (LMHI)

**One-line claim:** A primitive where the neural connectivity graph is dynamically defined and restricted by the legal moves available in the current position.

**Mathematical signature:**
$$Y = \text{Softmax}\left(\frac{(Q \cdot K^T) \odot \mathcal{M}_{legal}}{\sqrt{d}}\right)V$$
Where $\mathcal{M}_{legal}$ is the binary adjacency matrix of the **legal move graph**.

**Why this does not decompose into existing PyTorch ops:**
Standard attention uses fixed masks (causal) or learned masks. LMHI uses an **external, non-differentiable structural mask** as an input feature. Efficiently backpropagating through a sparse, content-dependent mask requires a custom sparse-attention kernel.

**Chess-specific motivation:**
A position is defined by its legal outcomes. LMHI prevents the network from attending to "noise" (illegal moves) and focuses high-dimensional interaction only where a piece can actually travel.

**Generalisation beyond chess:**
Legal/Contract analysis (like Wudao) where clauses only interact with legally relevant precedents.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(E \cdot d)$ where $E \approx 35$ (avg legal moves).
- Backward: $O(E \cdot d)$
- Incremental update: $O(E)$.

**Scout-scale falsification test:**
Integrate into the i243 dual-stream model. 
- **Baseline:** i243 (Standard masking). 
- **Metric:** Verified-near-puzzle recall. 
- **Success:** Significant lift in recall for tactical "only-move" puzzles.

**Failure mode catalogue:**
- Rebranded "Masked Attention" if the kernel isn't specialized for sparsity.
- Latency issues if move-gen is on CPU.
- Missing "threat" detection (defending against an illegal-but-proximal move).

---

### 5. primitive_dblf

**Name:** Differentiable Bit-Logical Filter (DBLF)

**One-line claim:** A soft-logic primitive that approximates bitwise logical operations (AND, OR, XOR) on latent vectors to simulate classical board-control logic.

**Mathematical signature:**
$$f(x, y) = \text{Sigmoid}\left(\frac{x \odot y - \theta}{\tau}\right)$$
Where $\theta$ is a learned threshold and $\tau$ is a temperature parameter.

**Why this does not decompose into existing PyTorch ops:**
Standard activations are unary. Standard multi-input layers are additive. DBLF is a **non-linear conjunction** operator. It models "A AND B" logic in high-dimensional space, which is structurally different from a linear sum followed by a ReLU.

**Chess-specific motivation:**
Engine evaluation is built on bitwise logic (e.g., `Occupancy AND Attack_Mask`). DBLF allows the network to "learn" these logical filters explicitly, making it more efficient at understanding piece coordination than a standard MLP.

**Generalisation beyond chess:**
Hardware emulation and symbolic AI.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(d)$
- Backward: $O(d)$
- Incremental update: $O(1)$.

**Scout-scale falsification test:**
Replace the final valuation MLP layers with DBLF gates. 
- **Baseline:** Standard 3-layer MLP. 
- **Metric:** Accuracy on "material exchange" puzzles. 
- **Success:** Higher accuracy on tactical counting with fewer parameters.

**Failure mode catalogue:**
- Numerical instability as $\tau \to 0$.
- Redundant if the network is wide enough to simulate logic.
- Difficulty in finding the "sweet spot" for threshold initialization.

---

## Final Ranking
1. **OSS:** Highest potential for "Mamba-level" impact on engine speed.
2. **IMUA:** Most immediate benefit for MCTS throughput.
3. **LMHI:** Best for tactical accuracy/puzzle discrimination.
4. **EPTP:** Essential for small-data (173k) generalization.
5. **DBLF:** High-risk, high-reward experimental primitive.

---
**What I cut during self-audit:**
*   *Chess-GCN:* Too close to standard Message Passing Neural Networks.
*   *Dihedral-Equivariant Pooling:* Rebrand of existing Group-Conv work.
*   *MCTS-Lookahead-Buffer:* An architecture, not a primitive.