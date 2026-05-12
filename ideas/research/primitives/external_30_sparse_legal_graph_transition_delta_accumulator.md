# Deep Research: Novel Neural Primitives for Chess Evaluation

## Research Briefing: Project chess-nn-playground
**Advisor:** Gemini (AI Research Collaborator)
**Target:** Tsinghua University, Dept. of Automation (Prof. Han Jungong)
**Hardware Target:** Single RTX 3070 (8 GiB VRAM)

---

### 1. primitive_sparse_transition_flow

**Name:** Sparse Legal-Move Graph Transition (SLMGT)

**One-line claim:** A graph-structured operator where the adjacency matrix is a non-differentiable, state-dependent mask of legal transitions, applied to features via a specialized sparse-scatter-reduce kernel.

**Deep Reasoning (Self-Audit):**
*   *Initial thought:* Use Ray-Tracing Attention. *Audit:* Rejected; that's just a mask on existing attention.
*   *Pivot:* Focus on the "legal move" graph as a hard constraint.
*   *Literature Grounding:* Inspired by Sparse Graph Neural Networks (GNNs) but differs because the graph is a direct function of the board rules (physics), not learned or sampled.
*   *Non-decomposability:* Standard `torch.nn.Linear` or `Conv2d` can't handle a dynamic topology without densifying to $O(N^2)$. This primitive requires a custom CUDA kernel that iterates over the `int64` legal-move bitmask.

**Mathematical signature:**
$$f: X \in \mathbb{R}^{B 	imes 64 	imes d}, \mathcal{M} \in \{0, 1\}^{B 	imes 64 	imes 64} 	o Y \in \mathbb{R}^{B 	imes 64 	imes d}$$
$$Y_i = 	ext{Agg}_{j \in \{j | \mathcal{M}_{ij}=1\}} \left( \phi(X_i, X_j) ight)$$
Where $\mathcal{M}$ is the legal move bitmask and $\phi$ is a learned edge function.

**Why this does not decompose into existing PyTorch ops:**
Existing ops assume static or data-driven connectivity. SLMGT uses a **rule-defined sparse topology**. To implement this in PyTorch currently, you would either use a dense mask (wasteful $O(N^2)$) or a generic GNN library (overhead for dynamic graphs). A native primitive would utilize the bitmask to avoid materializing the adjacency matrix entirely.

**Chess-specific motivation:**
A Knight on d4 "sees" its 8 destinations instantly. Standard Convs must layer 3+ times to connect d4 to f5. This primitive allows features to flow along legal move paths in a single step, mimicking look-ahead logic.

**Generalisation beyond chess:**
Constrained pathfinding in robotics or logic-gate simulation where valid state-transitions are governed by a rigid, non-learned rulebook.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(B \cdot 	ext{avg\_legal\_moves} \cdot d)$ vs Attention $O(B \cdot 64^2 \cdot d)$
- Backward: $O(B \cdot 	ext{avg\_legal\_moves} \cdot d)$
- Incremental update: $O(d \cdot 	ext{moved\_piece\_neighbors})$

**Scout-scale falsification test:**
Swap the first Dense layer of the `i193` baseline with SLMGT. **Success:** Reducing False Positives on "hanging piece" puzzles by >5% with <10% latency increase.

**Failure mode catalogue:**
- Memory-bound rather than compute-bound.
- Numerical instability if the aggregator handles zero-degree nodes poorly.
- Complexity of writing the custom CUDA kernel for the bitmask.

**Status:** proposed

---

### 2. primitive_differential_accumulator

**Name:** Gated Delta-Accumulator (GDA)

**One-line claim:** A recurrent state-update primitive that computes the output as a gated integration of input *changes* (deltas) rather than absolute values.

**Deep Reasoning (Self-Audit):**
*   *Initial thought:* Is this just a Delta-RNN? *Audit:* Delta-RNNs still process the full state.
*   *Pivot:* Generalize the NNUE accumulator update. In NNUE, we add/subtract features of the moved piece.
*   *Novelty:* This primitive makes that "delta-update" differentiable and learnable, allowing the network to "forget" or "emphasize" certain changes.
*   *Non-decomposability:* Standard RNNs compute $h_t = f(h_{t-1}, x_t)$. GDA computes $h_t = h_{t-1} + \Delta h$.

**Mathematical signature:**
$$h_t = h_{t-1} + \sigma(g(\Delta x_t)) \cdot \psi(\Delta x_t)$$
$$f: \Delta X \in \mathbb{R}^{B 	imes d_{in}} 	o \Delta Y \in \mathbb{R}^{B 	imes d_{out}}$$
Where $\Delta x_t = x_t - x_{t-1}$.

**Why this does not decompose into existing PyTorch ops:**
It changes the gradient flow. In a standard network, the gradient of the output w.r.t a distant input square is often diluted. In GDA, the gradient flows directly through the "accumulator" line, specifically focusing on the sparse changes.

**Chess-specific motivation:**
Chess moves are sparse (2-3 squares change). GDA allows the network to maintain a "running evaluation" that only recalculates the parts of the feature map affected by the move, mimicking Stockfish's incremental update speed.

**Generalisation beyond chess:**
High-frequency sensor data (IMU, LIDAR) where the "delta" between frames is sparse.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(B \cdot d)$
- Backward: $O(B \cdot d)$
- Incremental update: $O(	ext{sparsity}(\Delta x) \cdot d)$ — Constant $O(1)$ regarding total board size.

**Scout-scale falsification test:**
Train on the 173k dataset with sequential move sequences. **Success:** Maintaining <0.01 MSE in the value head while using 1/10th the FLOPs during a PV-walk search.

**Failure mode catalogue:**
- Floating point drift over long sequences.
- Implementation difficulty for varied batch sparsity.
- Vanishing gradients in the gate $\sigma$ if the delta is too small.

**Status:** proposed

---

### 3. primitive_group_involution_norm

**Name:** Symm-Involution Normalization (SIN)

**One-line claim:** A normalization layer that enforces feature-space invariance under the $C_2$ (color-flip) and $D_4$ (board-rotation) groups by projecting into a symmetry-invariant subspace.

**Deep Reasoning (Self-Audit):**
*   *Initial thought:* Just use Data Augmentation. *Audit:* Augmentation doesn't guarantee invariance, only "learns" it.
*   *Pivot:* Create a normalization primitive that *forces* invariance at the distribution level.
*   *Literature Grounding:* Similar to Group Equivariant CNNs (Cohen et al.), but implemented as a **normalization** op rather than a convolution op.

**Mathematical signature:**
$$\hat{X} = rac{1}{|G|} \sum_{g \in G} g(X)$$
$$	ext{SIN}(X) = \gamma \odot rac{X - 	ext{mean}(\hat{X})}{	ext{std}(\hat{X})} + eta$$
Where $G$ is the Chess Symmetry Group (color-flip, horizontal flip, etc.).

**Why this does not decompose into existing PyTorch ops:**
`LayerNorm` is agnostic to the 2D spatial arrangement. SIN treats the feature tensor as a set of group-related views and normalizes across them. It guarantees that $	ext{SIN}(x) = 	ext{SIN}(g(x))$ for any $g \in G$.

**Chess-specific motivation:**
Prevents "asymmetry bias" (where the engine evaluates White slightly differently than Black in identical positions). It ensures the value head is a true mathematical invariant of the position's group orbit.

**Generalisation beyond chess:**
Medical imaging (MRI) or satellite photography where the orientation of the sensor is arbitrary.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(B \cdot |G| \cdot N \cdot d)$
- Backward: $O(B \cdot |G| \cdot N \cdot d)$
- Incremental update: N/A

**Scout-scale falsification test:**
Insert SIN after the feature extractor in `i243`. **Success:** Identical loss/accuracy on a test set and its mirrored counterpart with zero epochs of "mirroring" data augmentation.

**Failure mode catalogue:**
- Too restrictive (over-smoothing).
- Slow forward pass (8x compute if naive).
- Redundant if the base conv kernels already learned the symmetry.

**Status:** proposed

---

### 4. primitive_tensor_rank_gate

**Name:** Low-Rank Mixture Gating (LRMG)

**One-line claim:** A conditional computation primitive that selects between weight-tensors of different CP-ranks based on the input's "tactical complexity" (entropy).

**Deep Reasoning (Self-Audit):**
*   *Initial thought:* Use MoE (Mixture of Experts). *Audit:* MoE routes to different *functions*.
*   *Pivot:* Route to different *precisions* of the same function.
*   *Novelty:* This is an "Adaptive Precision" primitive. It treats weight-rank as a dynamic variable.

**Mathematical signature:**
$$Y = 	ext{Softmax}(W_{gate}X) \cdot [W_{	ext{rank-1}}X, W_{	ext{rank-4}}X, W_{	ext{full}}X]$$
Where each $W$ is a different rank-approximation of the same base weight tensor.

**Why this does not decompose into existing PyTorch ops:**
Standard `Linear` has a fixed rank. LRMG uses a gating mechanism to control the **rank-decomposition** of the operation dynamically.

**Chess-specific motivation:**
Endgames and quiet positions have low "information density." A Rank-1 approximation of the board is often sufficient to tell that White is winning a 3-pawn vs 1-pawn endgame. Save the full-rank compute for the "chaotic" middlegames.

**Generalisation beyond chess:**
Real-time video streaming or edge-device inference where compute must scale with content complexity.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(B \cdot d \cdot k)$ (variable $k$)
- Backward: $O(B \cdot d \cdot k)$
- Incremental update: $O(k)$

**Scout-scale falsification test:**
Measure the correlation between chosen Rank $k$ and the presence of "Blunder" labels in CRTK. **Success:** High rank selection for tactical puzzles, low rank for quiet positions, with 40% lower average FLOPs.

**Failure mode catalogue:**
- Gating collapse (always picking full-rank).
- Overhead of the gate being larger than the savings.
- Difficulty in training the low-rank weights to be consistent with the full-rank ones.

**Status:** proposed

---

### 5. primitive_perm_relational_pooling

**Name:** Piece-Type Relational Pooler (PTRP)

**One-line claim:** A pooling operator that aggregates information based on the categorical identity (Piece-Type) rather than spatial location.

**Deep Reasoning (Self-Audit):**
*   *Initial thought:* Just use a mask + global pool. *Audit:* That's a composition of two ops.
*   *Pivot:* Create a unified Set-Pooling primitive that extracts "Piece-Pair" relationship features.
*   *Non-decomposability:* Standard pooling is spatial (2x2). PTRP is categorical (Pool all 'Knights'). It produces a fixed-size representation regardless of how many pieces are on the board.

**Mathematical signature:**
$$P = \{p_1, \dots, p_6\}$$
$$Y_p = 	ext{MLP}(	ext{Max}_{i \in 	ext{Type}(p)} \{ \phi(X_i) \})$$
$$f: \mathbb{R}^{64 	imes d} 	o \mathbb{R}^{6 	imes d_{out}}$$

**Why this does not decompose into existing PyTorch ops:**
Unlike `GlobalAveragePool`, this creates a **multi-channel relational summary**. It is invariant to the *location* of the pieces but variant to the *count and type* of pieces. It's a primitive for "Set Summarization" with fixed categories.

**Chess-specific motivation:**
Instantly captures "material balance" (the most important feature in chess) and specific synergies (e.g., Bishop pair) without needing to "see" where they are.

**Generalisation beyond chess:**
Multi-agent systems where agents belong to specific classes (e.g., Healers vs. Tanks).

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot d)$
- Backward: $O(N \cdot d)$
- Incremental update: $O(d)$

**Scout-scale falsification test:**
Measure performance on "Material Imbalance" puzzles (Class 3). **Success:** Outperforming spatial-only baselines on material-critical evaluations by >8%.

**Failure mode catalogue:**
- Losing too much spatial context.
- Redundant if the main network already counts pieces (though PTRP makes it explicit and efficient).
- "Bag-of-pieces" limitation: cannot see coordination between different piece types easily.

**Status:** proposed

---

### What I cut during self-audit
1.  **Bitboard-Convolution:** Rejected. Decomposes into `Conv2d` with a binary mask. No new mathematical signature.
2.  **Elo-Conditioned Batch-Norm:** Rejected. Violates rule against using metadata as features.
3.  **Alpha-Beta Recurrence:** Rejected. This was an architecture proposal (unrolling a search tree) rather than a primitive.
4.  **Ray-Tracing Attention:** Rejected. It's just a masked attention variant; it doesn't create a new computation graph profile.
