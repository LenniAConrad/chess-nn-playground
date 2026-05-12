# Research Brief: New Neural Primitives for Chess Evaluation
**Project:** `chess-nn-playground`
**Advisor:** AI Advisor
**Recipient:** Lennart (Tsinghua University, Dept. of Automation)
**Date:** May 12, 2026

---

## Executive Summary

To move beyond the limitations of standard Transformer and CNN architectures discovered in the `i193` through `i243` scouts, we propose five novel neural primitives. These operators are designed to be "first-class citizens" in a deep learning framework, prioritizing **incremental updates**, **topological sparsity**, and **chess-specific group symmetries**.

---

### primitive_ila

**Name:** Incremental Latent Accumulator (ILA)

**One-line claim:** A differentiable state-update primitive that computes the next latent state $h_{t+1}$ using only the sparse board-delta (the move) in $O(1)$ time.

**Mathematical signature:**
$f: \mathbb{R}^d \times \mathcal{M} \times \Theta \to \mathbb{R}^d$  
$h_{t+1} = h_t + \phi(m_t, \theta)$  
Where $h \in \mathbb{R}^d$ is the latent vector, $m \in \mathcal{M}$ is a discrete move (from-to square), and $\phi$ is a sparse embedding lookup and transformation.

**Why this does not decompose into existing PyTorch ops:**
While it resembles an RNN, standard PyTorch RNNs (LSTM/GRU) require the full input vector $x_t \in \mathbb{R}^N$ at every step to compute gates. ILA operates on the *difference* in the computation graph. To implement this in standard PyTorch without a custom kernel, one would have to materialize the full board state to feed into a layer, making the cost $O(N)$ (board size) rather than the $O(1)$ move-based update.

**Chess-specific motivation:**
Chess is inherently incremental. NNUE’s success is rooted in the HalfKA accumulator's $O(1)$ update. ILA generalizes this property into a deep-learning primitive, allowing a deep network to maintain a "running evaluation" that updates instantly as MCTS explores the tree.

**Generalisation beyond chess:**
Applicable to any sparse-event sequence where state changes are highly localized, such as discrete event simulations, high-frequency financial tick data, or real-time editing in large-scale code repositories.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(d)$
- Backward: $O(d)$ (via BPTT or sparse gradient routing)
- Incremental update: $O(d)$ (Independent of board size $N$)

**Scout-scale falsification test:**
Replace the first 3 convolutional layers of a standard ResNet baseline with an ILA-based feature extractor.  
**Baseline:** `i193` (Conv-only).  
**Metric:** Time-to-accuracy ratio.  
**Success:** Achieving equal PR AUC to the baseline with a $\geq 5\times$ speedup in "per-move" forward passes during a simulated search.

**Failure mode catalogue:**
- Floating-point drift over deep search paths (numerical instability).
- Vanishing gradients across long move sequences if not carefully gated.
- Risk of being a "hidden rebrand" of a simple Sparse Embedding Lookup + Sum if non-linearities aren't handled correctly.

**Status:** proposed

---

### primitive_lmtg

**Name:** Legal-Move Topology Gating (LMTG)

**One-line claim:** A dynamic gating primitive where the connectivity of the feature map is strictly constrained by the legal move graph of the current board.

**Mathematical signature:**
$f: \mathbb{R}^{B \times 64 \times d} \times \{0,1\}^{B \times 64 \times 64} \to \mathbb{R}^{B \times 64 \times d}$  
$Y_i = \sigma(\text{Gather}(X, \text{LegalMoves}_i)) \cdot W$  
Where the "mask" is a dynamic adjacency matrix $A \in \{0, 1\}^{64 \times 64}$ provided as a second input tensor.

**Why this does not decompose into existing PyTorch ops:**
Standard `SparseConv` or `GCN` assumes a static or learned adjacency. LMTG uses a **hard-coded, input-dependent** adjacency that changes every forward pass. In current PyTorch, this requires `einsum` with a massive sparse mask or custom loops, which are significantly slower than a native kernel optimized for "Rule-Based Sparsity."

**Chess-specific motivation:**
Information in chess flows along legal move lines (e.g., a pinned piece only "sees" its king and the pinner). While Attention sees everything and Convs see a local $3 \times 3$ grid, LMTG sees only what is legally reachable, forcing the network to respect the game's fundamental topology.

**Generalisation beyond chess:**
Molecular modeling (bonds as legal paths) and circuit design where signal propagation is constrained by physical traces.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot K \cdot d)$ where $K$ is avg. legal moves (~35)
- Backward: $O(N \cdot K \cdot d)$
- Incremental update: $O(K \cdot d)$ (Only update squares affected by the move)

**Scout-scale falsification test:**
Insert LMTG into a Chess-Decomposed Attention (`i242`) variant. Replace the dense attention matrix with the LMTG mask.  
**Metric:** Matched-recall near-puzzle FP rate.  
**Success:** A reduction in FP rate for "Class 1" puzzles, as the network can no longer "hallucinate" interactions between squares not legally connected.

**Failure mode catalogue:**
- GPU underutilization due to irregular sparse patterns.
- Gradient "dead ends" where pieces have very few legal moves.
- Most likely a "hidden rebrand" of Sparse-Attention if the mask isn't treated as a primitive-level constraint.

**Status:** proposed

---

### primitive_bpdo

**Name:** Bit-Population Differentiable Operator (BPDO)

**One-line claim:** A differentiable primitive that learns to count and correlate specific bit-patterns (bitboards) without explicit linear projections.

**Mathematical signature:**
$f: \mathbb{R}^{B \times 64} \to \mathbb{R}^{B \times k}$  
$y_k = \sum_{i=1}^{64} \text{sigmoid}(\tau \cdot (x_i - p_{i,k}))$  
Where $p \in [0,1]$ is a learned "target bitboard" and $\tau$ is a temperature parameter that allows the sum to approximate `popcount` as $\tau \to \infty$.

**Why this does not decompose into existing PyTorch ops:**
While `(X * P).sum()` is similar, BPDO uses a non-linear "closeness" metric per bit *before* the summation. This creates a different gradient profile that specifically rewards "matching the mask" rather than just high activations. It is the differentiable equivalent of a bitwise `AND` followed by `POPCNT`.

**Chess-specific motivation:**
Evaluation relies heavily on "counting" (e.g., number of attackers vs. defenders). BPDO allows the network to learn to count specific patterns (like "pawn islands" or "king safety holes") natively, rather than approximating counting through deep stacking of ReLU layers.

**Generalisation beyond chess:**
Error-correcting codes, DNA sequence matching, and any domain using bit-vector representations.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot k)$
- Backward: $O(N \cdot k)$
- Incremental update: $O(k)$ (Update only the changed bits)

**Scout-scale falsification test:**
Add a BPDO head in parallel to the policy head of a 234-scout architecture.  
**Metric:** Accuracy on a "Material Count" auxiliary task.  
**Success:** Achieving $>99\%$ material accuracy in 2 epochs, outperforming a standard Linear layer.

**Failure mode catalogue:**
- Saturation of the sigmoid leading to vanishing gradients.
- Redundancy with 1x1 convolutions if $\tau$ is set too low.
- Numerical instability at extremely high $\tau$.

**Status:** proposed

---

### primitive_sei

**Name:** Symmetry-Equivariant Involution (SEI)

**One-line claim:** A weight-locked operator that enforces exact equivariance under the "Chess Group" (color swap + horizontal mirror) without data augmentation.

**Mathematical signature:**
$f: X \to Y$ s.t. $f(g \cdot X) = g \cdot f(X)$ for $g \in \{I, \text{ColorSwap}, \text{Mirror}, \text{ColorSwap} \circ \text{Mirror}\}$.  
Implemented via a kernel $K$ where weights are tied: $W_{i,j} = \omega(i,j)$, and $\omega$ is a symmetric basis function across both spatial and channel dimensions.

**Why this does not decompose into existing PyTorch ops:**
Standard Group-Equivariant CNNs (G-CNNs) typically handle rotations ($C_4$). Chess symmetry is unique because it involves a **feature-channel swap** (White pieces $\leftrightarrow$ Black pieces) coupled with a **spatial flip**. SEI is a native op that performs this dual-coupling in a single pass without doubling the feature map size.

**Chess-specific motivation:**
Engines often train on both sides to learn symmetry. SEI bakes the "Laws of Physics" of chess into the weights, effectively doubling the training data density (173k positions act like 346k "virtual" positions).

**Generalisation beyond chess:**
Bipartite graph matching and signed social networks where "flipping the sign" of nodes requires a corresponding flip in edge features.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot d^2)$ (Equivalent to standard Conv)
- Backward: $O(N \cdot d^2)$
- Incremental update: Not applicable.

**Scout-scale falsification test:**
Train on a subset of data containing *only* White-to-move positions. Test on Black-to-move positions.  
**Success:** Zero loss in accuracy on unseen "flipped" data compared to a baseline that requires both during training.

**Failure mode catalogue:**
- Reduced model capacity (fewer free parameters).
- High implementation complexity for the custom autograd function.
- Incompatibility with "asymmetric" input encodings (e.g., King-relative planes).

**Status:** proposed

---

### primitive_dss

**Name:** Directional Stopping Scan (DSS)

**One-line claim:** A prefix-sum based operator that propagates information along 8 chess directions but "stops" or "attenuates" when encountering a non-zero feature (a piece).

**Mathematical signature:**
$Y_{i, \vec{d}} = X_i + \alpha_i \cdot Y_{i-\vec{d}, \vec{d}}$  
Where $\vec{d} \in \{N, S, E, W, NE, NW, SE, SW\}$ and $\alpha_i$ is a learned "transparency" gate derived from square $i$'s occupancy.

**Why this does not decompose into existing PyTorch ops:**
This is a **Selective Scan** (akin to Mamba’s S6) generalized to a 2D grid with 8 specific directional flows. While multiple Conv layers can approximate this, a single DSS op captures "sliding piece logic" (the entire ray) in one pass. It does not decompose into Attention because the "stop" is content-dependent and sequential along the ray.

**Chess-specific motivation:**
Captures "X-ray" attacks and "Line of Sight" which are $O(N)$ phenomena. A Conv layer only sees $3 \times 3$; it takes 4 layers for a Rook's influence to reach the other side. DSS does it in one step.

**Generalisation beyond chess:**
Ray-tracing in computer graphics, path-finding in grid worlds, and medical imaging (CT scan reconstruction).

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot 8)$
- Backward: $O(N \cdot 8)$
- Incremental update: $O(\sqrt{N})$ (Update along the affected rays)

**Scout-scale falsification test:**
Replace the middle 4 layers of a 234-scout ResNet with a single DSS block.  
**Metric:** Detection of "Long-range pins."  
**Success:** Significant improvement in near-puzzle FP rate for positions where the tactic involves pieces 5+ squares away.

**Failure mode catalogue:**
- Sequential dependency makes it harder to parallelize than Convs.
- Vanishing/exploding values along long rays.
- Most likely a "hidden rebrand" of a 1D-S6 if the 2D directional coupling isn't implemented as a single kernel.

**Status:** proposed

---

## What I Cut (Self-Audit Results)

1.  **Piece-Symmetry Attention:** Rejected because it is a "Self-Attention" block with a specific weight-tying scheme. It decomposes entirely into `Linear`, `Softmax`, and `BMM`. It's a configuration, not a primitive.
2.  **Differentiable MCTS Layer:** Rejected as an architecture-level composition ("Meta-Layer") rather than an atomic operator. It fails the "reusable as `torch.nn.Op`" test.
3.  **Static Positional Bias:** Rejected because it is functionally identical to adding a learned constant to the input planes—an encoding trick.

---

## Advisor's Reasoning & Self-Audit (Thinking Mode)

### The 2026 Context
As of May 2026, the `i243` results indicate that simply stacking attention (Transformers) or convolution (ResNets) is hitting diminishing returns. The ICLR 2026 paper "ReCouPLe" showed that causal grounding (which we address with `LMTG`) is more important than raw parameter count. Stockfish 18’s experimentation with sparse MoE-NNUE confirms that **speed + sparsity** is the winning combo.

### Logic Behind Proposals
1.  **Incrementalism (`ILA`, `DSS`):** The primary bottleneck for Tsinghua’s scout-scale hardware (RTX 3070) is the wall-clock time for forward passes during MCTS. These primitives move the complexity from $O(\text{Board})$ to $O(\text{Delta})$. This is structurally different from anything in `torch.nn` today.
2.  **Topological Fidelity (`LMTG`, `DSS`):** Neural networks often struggle with "chess blindness" because they don't know the rules. By making the rules a **constraint on the gradient flow** (the primitive), we bake the physics of the game into the engine without needing $10^9$ positions to learn it.
3.  **Group Equivariance (`SEI`):** Data augmentation is a hack. True equivariance in the operator doubles effective data scale.

### The "Hidden Rebrand" Audit
I specifically audited `primitive_ila` against "Recurrent Neural Networks." The differentiator is the **state-delta**. A standard RNN $h_t = f(x_t, h_{t-1})$ requires $x_t$ (the whole board). ILA $h_t = f(m_t, h_{t-1})$ requires only the move. This requires a different C++/CUDA kernel that accesses weights differently.
