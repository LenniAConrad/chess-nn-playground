# Research Proposal: Novel Neural Primitives for Chess Evaluation
**Project:** chess-nn-playground  
**Scope:** New Mathematical Operators for PyTorch  

---

### primitive_dua

**Name:** Delta-Update Accumulator (DUA)

**One-line claim:** A deep-layer generalization of the HalfKA accumulator that maintains a persistent state and updates only via sparse input differences (deltas).

**Mathematical signature:**
$f: \Delta X_{t} \in \mathbb{R}^{[B, k, d]}, S_{t-1} \in \mathbb{R}^{[B, N, d]} \to S_{t} \in \mathbb{R}^{[B, N, d]}$
Where $S_t = S_{t-1} + \sum_{i=1}^k \text{Linear}(\Delta x_i)$, and $k$ is the number of changed squares (usually 2).

**Why this does not decompose into existing PyTorch ops:**
Standard `nn.Linear` or `nn.Conv2d` layers must process the full input $X$ (size $N$) every pass. DUA implements a persistent GPU buffer that bypasses the $O(N)$ memory bandwidth bottleneck by performing scattered atomic additions. This is a stateful primitive, unlike standard stateless layers.

**Chess-specific motivation:**
Chess moves are sparse. In NNUE, the first layer uses this for speed; DUA allows this "incrementalism" to exist in deeper, latent layers, potentially allowing massive hidden dimensions that are only updated, never fully re-computed during MCTS.

**Generalisation beyond chess:**
Applicable to any sparse-event sequence where a global state is modified by local perturbations, such as event-based camera processing (DVS) or real-time packet inspection.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N)$ (initial) / $O(k)$ (subsequent) vs Linear $O(N)$
- Backward: $O(k)$ (sparse gradients on indices)
- Incremental update on a bounded-change input: $O(k)$

**Scout-scale falsification test:**
Replace the second layer of the i243 dual-stream model with a DUA. The model should maintain identical accuracy to a full re-compute baseline but achieve a $>3\times$ increase in NPS (nodes per second) on an RTX 3070 during inference.

**Failure mode catalogue:**
*   **Floating point drift:** Cumulative sums of deltas may diverge from a full re-compute over long games (requires periodic re-sync).
*   **Memory management:** Managing persistent state buffers per search branch in MCTS is complex.
*   **Autograd:** PyTorch’s graph manager typically resists in-place state updates (requires custom `torch.autograd.Function`).

**Status:** proposed

---

### primitive_oars

**Name:** Occlusion-Aware Ray Scan (OARS)

**One-line claim:** A parallel prefix-scan operator that aggregates features along the 8 chess directions, "stopping" based on learned piece-blocker density.

**Mathematical signature:**
$f: X \in \mathbb{R}^{[B, 8, 8, d]} \to Y \in \mathbb{R}^{[B, 8, 8, d]}$
$Y_{p, \vec{d}} = \text{Scan}(X, \vec{d}, \otimes)$ where $\otimes$ is a selective associative operator: $a \otimes b = a + \sigma(W_{block} \cdot a) \cdot b$.

**Why this does not decompose into existing PyTorch ops:**
Standard convolutions have fixed kernels and limited receptive fields. OARS is an $O(N)$ selective scan (similar to Mamba/S6 logic) but executed on a 2D grid along 8 specific spatial rays simultaneously, where the "occlusion" is a learned function of the intermediate state.

**Chess-specific motivation:**
Chess is defined by "lines of sight." Current Convs must "stack" many layers to see across the board; OARS sees the entire ray in a single pass while respecting the physical intuition of blockers (pieces).

**Generalisation beyond chess:**
Path-tracing in graphics, urban visibility analysis, and grid-based RL agents with line-of-sight constraints.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N)$ vs Attention $O(N^2)$
- Backward: $O(N)$
- Incremental update on a bounded-change input: $O(\sqrt{N})$ (only rays passing through changed squares).

**Scout-scale falsification test:**
Compare a 2-layer OARS stack against a 12-layer ResNet. OARS should achieve better matched-recall on long-range tactics (e.g., back-rank mates, long skewers) due to its native global ray awareness.

**Failure mode catalogue:**
*   **Numerical Stability:** Parallel scans can be sensitive to the order of operations in floating-point.
*   **Kernel Overhead:** Writing an efficient CUDA scan for 8 directions is significantly harder than standard `cuDNN` calls.
*   **Over-smoothing:** Information might wash out over long rays without high-precision gating.

**Status:** proposed

---

### primitive_epik

**Name:** Equivariant Piece-Identity Kernels (EPIK)

**One-line claim:** A convolutional kernel whose weights are dynamically generated from a basis that is invariant to piece-type relabeling and color-symmetry involutions.

**Mathematical signature:**
$K = \sum_{i} \alpha_i B_i$, where $\{B_i\}$ is a basis of kernels satisfying $G \cdot B = B$ for the group $G$ of chess symmetries (D4 + color swap).
$f: X, G_{state} \to Y$ using $W = \rho(G_{state})$.

**Why this does not decompose into existing PyTorch ops:**
Standard `nn.Conv2d` learns independent weights for every channel. EPIK forces the weights to live in a restricted subspace defined by the group representations of chess. It is a "Weight-Tying" primitive where tying is defined by group-theoretic constraints.

**Chess-specific motivation:**
A "Knight" is functionally identical whether white or black, on e4 or d5. EPIK bakes this symmetry into the operator, drastically reducing the parameter count and improving sample efficiency on scout-scale data (173k positions).

**Generalisation beyond chess:**
Molecular modeling (isomorphism), crystallography, and permutation-group symmetries in physics simulations.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot K^2)$
- Backward: $O(N \cdot K^2)$ (Gradients are averaged across symmetric kernel components)
- Incremental update on a bounded-change input: N/A

**Scout-scale falsification test:**
Train on only 50k positions with and without EPIK. EPIK should achieve significantly lower validation loss and better "near-puzzle" FP rates because it is constrained from learning "hallucinated" asymmetries.

**Failure mode catalogue:**
*   **Expressivity Bottleneck:** If the symmetry group is too restrictive, the model may fail to learn legitimate asymmetries (e.g., first-move advantage).
*   **Implementation:** Requires pre-calculating the irreducible representations of the chess group.
*   **Initialization:** Standard Kaiming/Xavier initialization fails on constrained basis sets.

**Status:** proposed

---

### primitive_lmmp

**Name:** Legal-Move Manifold Projection (LMMP)

**One-line claim:** An attention operator where the connectivity is strictly defined by the current state's legal move graph, acting as a hard topological constraint.

**Mathematical signature:**
$f: X \in \mathbb{R}^{[N, d]}, \mathcal{M} \in \{0, 1\}^{[N, N]} \to Y \in \mathbb{R}^{[N, d]}$
$Y = \text{Softmax}( (Q K^T) \odot \mathcal{M} ) V$, where $\mathcal{M}_{ij} = 1$ iff a piece can legally move from $i$ to $j$.

**Why this does not decompose into existing PyTorch ops:**
While similar to "Masked Attention," the mask $\mathcal{M}$ is **data-dependent and non-learned**. A dedicated LMMP primitive uses a sparse-matrix-multiply (SpMM) backend optimized for chess move-graph density (~5%), avoiding the $O(N^2)$ dense overhead of standard attention.

**Chess-specific motivation:**
Information in chess should flow where pieces can go. A Knight on g1 should process f3 and h3, not a7. LMMP forces the "neural focus" to align with the tactical manifold of the board.

**Generalisation beyond chess:**
Traffic flow networks, supply chain logistics, and any "dynamic graph" where edges are governed by physical constraints.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(|\mathcal{M}|)$ (number of legal moves) vs $O(N^2)$
- Backward: $O(|\mathcal{M}|)$
- Incremental update on a bounded-change input: $O(1)$ (move graph changes locally)

**Scout-scale falsification test:**
Replace the decomposed attention in i242 with LMMP. The model should require $\approx 40\%$ less VRAM and show significantly improved performance on "Illegal Move" discrimination tasks.

**Failure mode catalogue:**
*   **Sparsity Overhead:** On modern GPUs, dense ops are often faster than sparse ones unless sparsity is $>95\%$.
*   **Gradient Isolation:** If a square has no legal moves, it may receive no gradient, leading to "dead neurons."
*   **CPU-GPU Bottleneck:** Requires a GPU-side move generator to avoid PCI-e latency.

**Status:** proposed

---

### primitive_dbi

**Name:** Differentiable Bitwise Interaction (DBI)

**One-line claim:** A primitive that performs learned bitwise-like operations (AND, OR, XOR, NOT) in a continuous latent space to mimic bitboard heuristics.

**Mathematical signature:**
$f: A, B \in [0, 1]^{[B, N, d]} \to Y \in [0, 1]^{[B, N, d]}$
$Y = \text{DBI}(A, B; \Theta)$, where $\text{DBI}_{AND}(a, b) = a \cdot b$ and $\text{DBI}_{XOR}(a, b) = a+b - 2ab$. The primitive learns which logic gate to apply per channel.

**Why this does not decompose into existing PyTorch ops:**
DBI is a single fused kernel that explores the space of all 16 binary logic gates via a learned interpolation. It treats "feature interaction" as "logical composition" rather than standard additive or multiplicative mixing.

**Chess-specific motivation:**
Classical engines are built on bitboard logic (`pawns & attackers`). DBI allows the network to learn these "hard" filters (e.g., "is this square occupied AND not defended?") in a clean, differentiable way.

**Generalisation beyond chess:**
Program synthesis, circuit design verification, and tabular data with categorical dependencies.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot d)$
- Backward: $O(N \cdot d)$
- Incremental update on a bounded-change input: $O(k \cdot d)$

**Scout-scale falsification test:**
Insert DBI layers between piece planes and the first Conv layer. It should outperform manual feature engineering (like `simple_18`) by automatically discovering tactical bitboard interactions.

**Failure mode catalogue:**
*   **Saturation:** Latent logic gates often suffer from vanishing gradients as they approach 0 or 1.
*   **Redundancy:** The network might collapse to standard MLP behavior if the DBI constraint isn't "hard" enough.
*   **Interpretability:** Learned gates may become uninterpretable high-dimensional noise.

**Status:** proposed

---

### What I cut during self-audit

1.  **S6-Chess (Selective State Spaces):** Rejected because it is an architecture (composition of Selective Scan + Linear). The "Selective Scan" itself is the primitive, which I re-contextualized as **OARS**.
2.  **GCN-Chess:** Rejected as a "hidden rebrand." Standard GCNs are just `Linear(Agg(X))`. I pivoted to **LMMP** for its dynamic, non-learned manifold property.
3.  **Bilinear Piece-Interaction:** Rejected because `(Wx) * (Vy)` is a standard composition. I replaced this with **DBI**, targeting the specific bitwise-logic structure of chess.
4.  **Rotation-Equivariant Convs:** Rejected because libraries like `e2nn` already provide these. I focused on **EPIK** for piece-type identity symmetry, which is domain-specific.