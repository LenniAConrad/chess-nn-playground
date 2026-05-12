# Research Proposal: Novel Neural Primitives for Chess Evaluation

**Project:** `chess-nn-playground`  
**Advisor:** AI Research Assistant  
**Target:** Tsinghua University, Dept. of Automation (Prof. Han Jungong)

---

## 1. Research Strategy & Reasoning (Thinking Mode)

### The "Primitive" Threshold
The goal of this research is to step beyond the "Architecture Era" (i193 to i243) and enter the "Operator Era." In the previous scout phases, we combined standard blocks (Convs, Attention, Linear). Here, we identify structural invariants in chess that are *poorly served* by these general-purpose operators. 

**Logic of Invention:**
1. **Symmetry-First:** Chess is a zero-sum game with perfect involutional symmetry. Normalizing for this rather than just training it leads to better sample efficiency.
2. **Sparsity-Native:** A chess move changes <3% of the board state. Standard CNNs re-read the entire 8x8 grid. We need an operator that treats the "move" as a first-class update to a latent manifold.
3. **Geometry-Aware:** Pieces move on rays and jumps. Standard convolution uses 3x3 or 5x5 squares. We need a kernel that "sees" the chess manifold (ranks, files, diagonals, and knight jumps) as its fundamental neighborhood.

### Self-Audit & Calibration
Before proposing these, I audited them against the "Composition Trap."
* *Is SRA just masked attention?* No. Standard masking computes $N^2$ and zeros it. SRA is a block-sparse kernel that only visits ray-aligned memory addresses, potentially $O(N\sqrt{N})$ or $O(N)$.
* *Is MPC just a gated linear layer?* No. It is a dynamic kernel generator where the weights themselves are a function of the input's discrete categorical state (Piece ID), allowing the convolution to "morph" its logic per square.

---

## 2. Proposed Neural Primitives

### primitive_sda

**Name:** Sparse Differential Accumulator (SDA)

**One-line claim:** A stateful operator that updates high-dimensional latent features using sparse input deltas rather than dense re-computation.

**Mathematical signature:**
$f: (\Delta X_t, S_t) ightarrow S_{t+1}$
Input: $\Delta X_t \in \{(i, p)_{new}, (j, p)_{old}\}^{k 	imes 2}$ (sparse indices and piece IDs).
State: $S \in \mathbb{R}^{d}$.
Operation: $S_{t+1} = S_t - 	ext{Emb}(p_{old}, j) + 	ext{Emb}(p_{new}, i)$.
Gradient: $rac{\partial \mathcal{L}}{\partial 	ext{Emb}}$ is defined via the differential chain.

**Why this does not decompose into existing PyTorch ops:**
PyTorch is stateless at the graph level. Standard NNUE implementations use hand-written C++ accumulators. A primitive SDA integrates this persistence into the `torch.autograd` graph, allowing the network to maintain a high-dimensional "latent board" that is updated incrementally. It cannot be decomposed into a single `forward(X)` call because it depends on the hidden state $S$ of the *previous* move in the variation.

**Chess-specific motivation:**
Chess moves are sparse. NNUE (Stockfish) exploits this for the first layer, but deep networks lose this efficiency in deeper layers. SDA generalizes the "Accumulator" property to any depth, allowing for massive NPS (nodes per second) increases during MCTS.

**Generalisation beyond chess:**
Event-based vision (DVS cameras), real-time physics simulations, and dynamic graph updates where only a few nodes change per step.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(d)$ (constant relative to board size $N$).
- Backward: $O(d)$ per delta.
- Incremental update: $O(1)$ relative to $N$.

**Scout-scale falsification test:**
Replace the primary linear layer of a HalfKA architecture with SDA. 
* **Baseline:** i193 (standard Conv). 
* **Metric:** Throughput (NPS) vs. Near-puzzle FP rate. 
* **Success:** Maintain FP rate while achieving >5x throughput on a single variation search.

**Failure mode catalogue:**
- Numerical drift in the accumulator over deep search trees.
- Memory overhead of storing states $S$ for every node in MCTS.
- Difficulty in parallelizing batch updates of different variation lengths.

**Status:** proposed

---

### primitive_mko

**Name:** Move-Kernel Operator (MKO)

**One-line claim:** A rule-informed convolution where weights are indexed by the legal-move manifold (rays and jumps) rather than spatial offsets.

**Mathematical signature:**
$Y_i = \sum_{j \in \mathcal{M}(i)} W_{	ext{type}(i, j)} \cdot X_j$
Where $\mathcal{M}(i)$ is the set of squares reachable by *any* piece type from square $i$, and $	ext{type}(i, j)$ is the move relationship (e.g., "diagonal step", "knight jump").

**Why this does not decompose into existing PyTorch ops:**
Unlike `Conv2d` (fixed grid) or `Attention` (data-dependent), MKO uses a **rule-dependent fixed-sparse topology**. Implementing this in PyTorch requires $64 	imes 64$ sparse indexing or masking, which is inefficient. A primitive MKO would be a custom CUDA kernel that maps the $64 	imes 64$ board to an 8-neighbor ray + knight-jump manifold.

**Chess-specific motivation:**
A "Knight at f3" and "Knight at d4" exert the same geometric influence. Standard Convs must learn this 64 times. MKO enforces **move-type weight sharing**, baking the rules of chess directly into the "receptive field" of the network.

**Generalisation beyond chess:**
Logic-gate graphs, chemical bond networks, or any system with rigid but non-grid-like connectivity rules.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(B \cdot N \cdot |	ext{move\_types}|)$.
- Backward: $O(B \cdot N \cdot |	ext{move\_types}|)$.
- Incremental update: Not applicable.

**Scout-scale falsification test:**
Replace the 3x3 Conv layers in the i242 scout with MKO layers.
* **Baseline:** i242 (Decomposed Attention).
* **Metric:** Matched-recall near-puzzle FP rate on "long-range tactic" puzzles.
* **Success:** FP rate reduction >15% with no increase in parameter count.

**Failure mode catalogue:**
- "Knight-blindness" if move types aren't exhaustive.
- Over-specialization to chess geometry, making it brittle to non-move features (e.g., square color).
- Kernel launching overhead on small batches.

**Status:** proposed

---

### primitive_ipn

**Name:** Involutional Parity Normalization (IPN)

**One-line claim:** A normalization layer that strictly enforces color-swap antisymmetry through a differential projection mechanism.

**Mathematical signature:**
$X_{norm} = 	ext{Norm}\left(rac{X - \mathcal{T}(X)}{2}ight)$
Where $\mathcal{T}$ is the involution operator (color swap + board flip).

**Why this does not decompose into existing PyTorch ops:**
Standard `LayerNorm` is agnostic to board symmetry. While one can "data augment" by swapping colors, IPN makes the **symmetry structural**. It forces the gradient to only update the "asymmetric component" of the representation. It creates a computation graph where $f(X) = -f(\mathcal{T}(X))$ is a mathematical guarantee, not a learned approximation.

**Chess-specific motivation:**
Deep networks often have "perspective bias" (evaluating White as slightly better in identical positions). IPN bakes the zero-sum nature into the latent features, effectively doubling the training signal per position.

**Generalisation beyond chess:**
Any zero-sum competitive game or physical systems with parity (P) or charge (C) symmetry.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N)$ (same as LayerNorm).
- Backward: $O(N)$.
- Incremental update: Not applicable.

**Scout-scale falsification test:**
Apply IPN to the final three layers of the 234-scout.
* **Baseline:** i234 with standard LayerNorm.
* **Metric:** Symmetry gap $|Eval(P) + Eval(\mathcal{T}(P))|$.
* **Success:** Gap drops to $<10^{-7}$ while maintaining the same PR AUC as the baseline.

**Failure mode catalogue:**
- Information loss if the "sum" component $X + \mathcal{T}(X)$ contained useful context.
- Numerical cancellation if the network is already nearly symmetric.
- Incompatibility with non-zero-sum labels (e.g., "node count" or "time to mate").

**Status:** proposed

---

### primitive_sra

**Name:** Selective Ray Attention (SRA)

**One-line claim:** An attention mechanism where the visibility mask is hard-coded to the cardinal and intercardinal rays of the chess grid.

**Mathematical signature:**
$A = 	ext{Softmax}\left(rac{(Q \cdot K^T) \odot \mathcal{R}}{\sqrt{d}}ight) \cdot V$
Where $\mathcal{R}$ is the "Ray Matrix": $\mathcal{R}_{i,j} = 1$ if square $i$ and $j$ are on the same rank, file, or diagonal.

**Why this does not decompose into existing PyTorch ops:**
This is not "Attention + Mask." In a primitive implementation (e.g., using Triton), the query never looks at the $N^2 - 	ext{ray\_count}$ non-ray squares. This allows for $O(N \sqrt{N})$ complexity rather than $O(N^2)$, which is critical for the 8x8 grid where rays are sparse.

**Chess-specific motivation:**
Tactics travel on lines. Standard attention spends compute power evaluating if a Knight on g1 "attends" to a Pawn on a7. SRA restricts the attention manifold to the "paths of influence" defined by pieces like Rooks, Bishops, and Queens.

**Generalisation beyond chess:**
Medical imaging (X-ray/CT scan paths) and urban planning (street-view line-of-sight).

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \sqrt{N})$ vs $O(N^2)$.
- Backward: $O(N \sqrt{N})$.
- Incremental update: Not applicable.

**Scout-scale falsification test:**
Integrate SRA into the i243 dual-stream proposal.
* **Baseline:** i243 (Standard global attention).
* **Metric:** Speed (Inference latency) vs. matched-recall puzzle accuracy.
* **Success:** $>2	imes$ reduction in attention-block latency with $<0.01$ drop in PR AUC.

**Failure mode catalogue:**
- Knight jumps are "blind" to this attention (requiring an additional jump-head).
- Difficulty capturing "king safety" if the king is far from the rays of the attacking pieces.
- Sparse-kernel performance on modern GPUs often lags behind dense kernels due to lack of optimization.

**Status:** proposed

---

### primitive_mpc

**Name:** Metamorphic Piece Convolution (MPC)

**One-line claim:** A convolution where kernel weights are generated on-the-fly by a hyper-network conditioned on the piece type occupying the central square.

**Mathematical signature:**
$Y = 	ext{Conv}(X, W)$ where $W = \mathcal{G}(	ext{Piece\_ID}(i))$
$\mathcal{G}$ is a small MLP that maps 12 piece IDs to $k 	imes k 	imes C_{in} 	imes C_{out}$ weights.

**Why this does not decompose into existing PyTorch ops:**
Standard convolutions use static filters. "Grouped" or "Depthwise" convolutions use fixed subsets. MPC is **Dynamic Weight Generation**. It is non-decomposable because the kernel itself is a variable in the compute graph, changing its values based on the categorical input state at each spatial location.

**Chess-specific motivation:**
A square controlled by a Pawn vs. a Queen has different tactical meaning. MPC allows the network to "morph" its logic. Instead of learning a general filter, it learns 12 specific "interaction behaviors" (e.g., the "Bishop interaction kernel").

**Generalisation beyond chess:**
Agent-based modeling, where different agent types (e.g., predator vs prey) interact with the same grid in fundamentally different ways.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(N \cdot k^2 \cdot C^2 + 	ext{GenCost})$.
- Backward: $O(N \cdot k^2 \cdot C^2)$.
- Incremental update: $O(k^2 \cdot C^2)$ (re-compute only for the piece that moved).

**Scout-scale falsification test:**
Replace the standard $3 	imes 3$ Convs in the 234-scout with MPC.
* **Baseline:** i193/i234 (Static Convs).
* **Metric:** Training steps to reach target validation loss.
* **Success:** Reach target loss in 40% fewer epochs than the static baseline.

**Failure mode catalogue:**
- Parameter explosion in the generator $\mathcal{G}$.
- Overfitting to piece-specific patterns at the expense of general positional play.
- Extreme slow-down in forward-pass if weight generation is not optimized via weight-caching.

**Status:** proposed

---

## 3. Rejected Candidates

| Primitive | Reason for Rejection |
| :--- | :--- |
| **Bitboard-Gated MLP** | Decomposes into `x * mask`. Not a new primitive; just a gating operation. |
| **MCTS-Aware Softmax** | Architectural component. It depends on labels/search data (node counts), violating Rule 5. |
| **Dihedral-Equivariant Conv** | Already exists (Group Equivariant CNNs). Not "new" novelty. |
| **Piece-Relabeling Attention** | Decomposes into standard attention with a specialized positional encoding. |

---

## 4. References & Audit Log

- **Trend Analysis:** The "incremental update" property matches findings in *Rapfi: Distilling Efficient Neural Network for the Game of Gomoku* (arXiv:2503.13178v1).
- **Symmetry Constraints:** Aligns with research on *BRONet: Block Reflector Orthogonal Layers* (ICML 2026), focusing on structural constraints for expressivity.
- **Search Query:** `gmail search query: "chess-nn-playground scout findings 2024-2026"`

---
**End of Proposal**