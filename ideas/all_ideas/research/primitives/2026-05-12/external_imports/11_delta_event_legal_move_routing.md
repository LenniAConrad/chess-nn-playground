# Inventing Neural Primitives for Chess Evaluation

## Scope note

This document proposes primitive-level operators for chess evaluation. It does **not** propose new model architectures, input encodings, losses, curricula, schedulers, or hyperparameter variants. A necessary caveat: almost any operator can be emulated with sufficiently many low-level tensor operations. Here, "does not decompose into existing PyTorch ops" means the proposed unit would have a distinct native operator boundary, cached autograd state, connectivity rule, or asymptotic update behavior that is not represented by a standard `torch.nn` module such as `Conv2d`, `MultiheadAttention`, `EmbeddingBag`, `LayerNorm`, or a PyG message-passing layer.

## Literature calibration

The target primitive should be judged against genuinely new operator families, not against architecture recipes. Recent examples include selective state-space recurrence, recurrent/linear attention variants, new memory cells, edge-function networks, and structured graph/path operators:

- **Selective state-space operators.** Mamba introduced selective state-space sequence modeling, where SSM parameters are input-conditioned rather than fixed; Mamba-2 later framed attention and SSMs through structured state-space duality.
- **Recurrent / linear attention operators.** RetNet exposes parallel, recurrent, and chunkwise recurrent forms with constant-time recurrent inference. Gated Linear Attention adds learned gates to linear attention and emphasizes hardware-efficient scan kernels.
- **New memory and transformation cells.** xLSTM adds exponential gates plus scalar and matrix memory variants. KAN moves learnable nonlinear functions onto edges rather than using fixed node activations.
- **Structured sparsity and algebraic operators.** Differential Transformer subtracts two attention maps. Group-equivariant CNNs and EGNNs show how exact symmetry actions can define reusable operators. GAT, PyG message passing, dynamic GNN surveys, and Neural Bellman-Ford Networks cover nearby territory for graph attention, dynamic graph learning, and differentiable path reasoning.

The chess-specific bar is stricter than "apply a recent primitive to chess." The useful primitive should exploit at least one of: bounded-change updates, legal-move graph sparsity, ray occlusion, chess-specific group structure, or hard-negative tactical discrimination.

## Ranking at a glance

| Rank | Primitive | Novelty plausibility | RTX 3070 demonstrability | Inference-speed upside | Generalisation beyond chess | Verdict |
|---:|---|---:|---:|---:|---:|---|
| 1 | `primitive_delta_event_accumulator` | High | High | Very high | High | Best first implementation target |
| 2 | `primitive_legal_move_routing` | Medium-high | High | High | High | Best chess-structure target |
| 3 | `primitive_occlusion_ray_scan` | Medium-high | High | High | Medium-high | Strong tactical inductive bias |
| 4 | `primitive_chess_orbit_linear` | Medium | High | Medium | Medium | More "underexplored equivariant primitive" than clean novelty |
| 5 | `primitive_soft_tactical_distance` | Medium | Medium | Medium-low | High | Most conceptually rich, riskiest speed profile |

## Self-audit of the top two

### `primitive_delta_event_accumulator`

The strongest objection is that this is just `EmbeddingBag(mode="sum")` plus a cache. That objection is valid **if** the operator is only "sum embeddings for active features." PyTorch `EmbeddingBag` already computes sums, means, or maxes of bags of embeddings without materializing intermediate embeddings.

The retained version is narrower and stronger: the primitive's formal input is a signed **change stream** plus a persistent differentiable accumulator state, and its backward pass saves the delta trace rather than the dense active set. That gives it a different update complexity and autograd state from `EmbeddingBag`. If implemented as "call `EmbeddingBag` on the whole board every node," it should be rejected.

### `primitive_legal_move_routing`

The strongest objection is that this is just GAT or masked attention over a legal-move graph. GAT already performs masked self-attention over graph neighborhoods, and `MultiheadAttention` accepts dense or batched attention masks.

The retained version makes **edge construction part of the primitive**: legal edges are generated inside the operator from piece occupancy and side-to-move, producing a ragged edge set and segment-wise routing without constructing an `n × n` attention mask. If the legal edge list is precomputed outside the op and passed to a standard PyG `MessagePassing` layer, the novelty claim collapses to "underexplored graph attention for chess."

---

## 1. Proposal

### primitive_delta_event_accumulator

**Name:** Delta-Event Accumulator

**One-line claim:** Maintains a differentiable feature state whose update cost depends on changed events, not input size.

**Mathematical signature:**
For event vocabulary size `M`, width `d`, batch `B`, and per-position change count `k ≪ M`:

\[
f_\Theta:\mathbb{R}^{B\times d}\times \mathbb{Z}^{B\times k}\times \{-1,+1\}^{B\times k}\rightarrow \mathbb{R}^{B\times d}
\]

\[
A_t[b]=A_{t-1}[b]+\sum_{r=1}^{k}s_{b,r}\Theta[e_{b,r}],\qquad Y_t=\phi(A_t)
\]

where `Θ ∈ R^{M×d}`, event ids `e` are integer indices, signs `s` denote insertion/removal, and gradients flow to `Θ`, `A_{t-1}`, and downstream `φ`, not to integer event ids.

**Why this does not decompose into existing PyTorch ops:**
The stateless special case is dangerously close to `EmbeddingBag(mode="sum")`, which is an efficient embedding reduction. The proposed primitive is a stateful signed-delta autograd operator: its saved backward state is the event-change log, not the full active feature bag. That gives a different computation graph and complexity profile from `EmbeddingBag(active_indices)` recomputed each forward pass.

**Chess-specific motivation:**
This directly generalises the useful property behind NNUE-style accumulators: a move changes only a bounded number of piece-square events. For evaluation inside search, the primitive can update the network state after a move without reprocessing all board features. It is especially attractive because the prior scout evidence values speed at least as much as raw accuracy.

**Generalisation beyond chess:**
Useful for sparse-event streams, recommender-session updates, dynamic knowledge graphs, robotics scene updates, and any setting where the input changes by small edits.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(Bkd)` vs closest full sparse embedding reduction `O(Bmd)`, where `m` is active-event count.
- Backward: `O(Bkd)` for event-table gradients plus downstream `φ`.
- Incremental update on a bounded-change input: `O(kd)`.

**Scout-scale falsification test:**
Drop it into the i193-style conv parent only as the first feature aggregation primitive, replacing full recomputation of the initial accumulator. Baseline: same model with stateless sparse/dense aggregation. Train on the 173k × 12-epoch scout setting. Measure matched-recall near-puzzle false-positive rate and cached move-update wall-clock latency. Works if near-puzzle FP is no worse than baseline while cached inference is at least 1.7× faster; fails if speed gain vanishes or gains appear only in aggregate PR AUC.

**Failure mode catalogue:**
- Hidden rebrand: reviewer says it is merely `EmbeddingBag`; reject any implementation that recomputes full active bags.
- Numerical instability: repeated fp16 signed updates drift; require periodic exact refresh or fp32 accumulator.
- Too slow: Python-level event bookkeeping can erase the theoretical `O(k)` gain; needs fused CUDA/Triton path.

**Status:** proposed

---

## 2. Proposal

### primitive_legal_move_routing

**Name:** Input-Induced Legal-Move Routing

**One-line claim:** Routes features over a ragged graph whose edges are generated inside the operator from the current state.

**Mathematical signature:**
For square tokens `X ∈ R^{B×64×d}`, discrete board state `S`, generated legal edge multiset

\[
E_b=\mathcal{L}(S_b)\subseteq \{1,\ldots,64\}^2\times T
\]

and edge type `τ ∈ T`:

\[
f_\Theta(X,S)[b,v]=\sum_{(u,v,\tau)\in E_b}
\alpha_{b,u,v,\tau}\, W_\tau X[b,u]
\]

\[
\alpha_{b,u,v,\tau}=\operatorname{segsoftmax}_{(u,\tau):(u,v,\tau)\in E_b}
\left(q_\tau^\top[X[b,u],X[b,v]]\right)
\]

Gradients flow through `X`, `W`, and `q`; edge generation from `S` is discrete, like routing in conditional-compute operators.

**Why this does not decompose into existing PyTorch ops:**
It is not `MultiheadAttention` with a different mask: standard attention consumes a supplied dense or batched mask, while this operator constructs a ragged, typed edge set internally and performs segment-softmax only over generated legal edges. It is also not plain GAT if the edge list is produced outside the op; the primitive boundary includes state-to-edge construction, typed routing, and saved ragged topology. GAT and PyG MessagePassing are important overlap risks, but they assume a graph/neighborhood input rather than generating chess-legality connectivity as part of the op.

**Chess-specific motivation:**
Legal-move connectivity changes by board. Knights, sliders, pawn captures, checks, pins, and blockers define sparse relationships that a fixed convolution cannot see and dense attention learns wastefully. This primitive puts the changing legal graph directly into the operator without introducing Stockfish labels or handcrafted evaluation metadata.

**Generalisation beyond chess:**
Applies to dynamic scene graphs, collision graphs, packet-routing graphs, molecular conformer graphs with changing bonds, and sparse-event simulations.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(B|E|d)` vs dense attention `O(Bn²d)` with `n=64`.
- Backward: `O(B|E|d)` plus segment-softmax gradient.
- Incremental update on a bounded-change input: `O(ΔE·d)` after updating the affected ragged edge set.

**Scout-scale falsification test:**
Drop the primitive into i242's attention harness by replacing only the attention operator, not the streams or encodings. Baseline: same harness with standard masked/dense attention and the conv-only i193 parent. Measure matched-recall near-puzzle FP rate and evaluations/sec. Works if it beats dense attention on latency and reduces near-puzzle FP versus i242 at matched recall; fails if it only improves easy-negative PR AUC.

**Failure mode catalogue:**
- Hidden rebrand: if implemented as precomputed `edge_index + GATConv`, novelty drops to "graph-attention layer."
- Numerical instability: segment-softmax over tiny neighborhoods may produce high-variance gradients.
- Too slow: legal-edge construction on CPU can dominate; needs fused or bitboard-backed GPU batching.

**Status:** proposed

---

## 3. Proposal

### primitive_occlusion_ray_scan

**Name:** Occlusion-Censored Ray Scan

**One-line claim:** Aggregates along directional rays with blocker-aware visibility inside the primitive.

**Mathematical signature:**
For board tokens `X ∈ R^{B×64×d}`, binary occupancy `O ∈ {0,1}^{B×64}`, directions `R=8`, and ray `ρ(i,r,ℓ)`:

\[
f_\Theta(X,O)[b,i]=
\sum_{r=1}^{8}\sum_{\ell=1}^{L_{i,r}}
\left[
\prod_{m=1}^{\ell-1}(1-O[b,\rho(i,r,m)])
\right]
\gamma_{r,\ell}\, W_r X[b,\rho(i,r,\ell)]
\]

The product is an occlusion gate. Gradients flow through `X`, `γ`, and `W`; occupancy is a discrete routing input.

**Why this does not decompose into existing PyTorch ops:**
A normal `Conv2d` applies fixed local cross-correlation over a rectangular kernel. This primitive performs a variable-length directional scan with multiplicative blocker censoring and saves first-blocker/ray-prefix state in backward. It can be emulated inefficiently with gathers and cumulative products, but its natural computation graph is a fused segmented ray-scan, not convolution or attention.

**Chess-specific motivation:**
Sliding-piece tactics are ray-occlusion problems. Rooks, bishops, queens, pins, skewers, discovered attacks, and king safety depend less on local 3×3 texture than on "what is visible until the first blocker." This operator gives a small model a direct line-of-sight primitive without adding new board planes.

**Generalisation beyond chess:**
Useful for grid worlds, robotic visibility, radiance / line-of-sight reasoning, traffic lanes, circuit layouts, and any occluded directional field.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(BR n d)` for `R=8,n=64` vs dense attention `O(Bn²d)`; vs local conv `O(Bn k²d)` but with fixed receptive field.
- Backward: `O(BR n d)`.
- Incremental update on a bounded-change input: `O(R√n d)` for changed ranks/files/diagonals on square boards; constant-bounded on 8×8 chess.

**Scout-scale falsification test:**
Drop it into the i193 conv parent by replacing one mid-level spatial convolution with one ray-scan operator of matched output width. Baseline: same parameter-count i193 block with a normal conv. Train under the 173k × 12 scout setup. Works if near-puzzle matched-recall FP decreases while eval/sec is at least parity; fails if gains only appear on easy negatives or if the scan is slower than dense attention.

**Failure mode catalogue:**
- Hidden rebrand: reviewer says it is "just cumulative product plus gather"; the primitive claim requires fused segmented ray-scan semantics.
- Numerical instability: long products can underflow if occupancy is relaxed; keep occupancy discrete or use log-domain gates.
- Too slow: irregular ray indexing may underutilize GPU unless rays are prepacked.

**Status:** proposed

---

## 4. Proposal

### primitive_chess_orbit_linear

**Name:** Chess-Group Orbit Linear

**One-line claim:** Applies a linear map tied by chess-specific group orbits over squares, colors, and piece roles.

**Mathematical signature:**
Let index set `I = squares × colors × piece_roles`, `|I|=N`, and finite group `G` act on `I`. For `X ∈ R^{B×N×d_in}`:

\[
f_\Theta(X)[b,i,o]=\sum_{j\in I}\sum_{c=1}^{d_{in}}
\Theta_{\operatorname{orb}_G(i,j),c,o}X[b,j,c]
\]

where

\[
\operatorname{orb}_G(i,j)=\{(g i,g j):g\in G\}
\]

and the same parameter is used for all pairs in the same orbit. This guarantees `f(g·X)=g·f(X)` for the supplied group action.

**Why this does not decompose into existing PyTorch ops:**
For ordinary spatial symmetries, this overlaps with group-equivariant convolution, which is established prior work. The proposed primitive is not a `Conv2d`: its group action simultaneously permutes board squares, color channels, side-relative roles, and selected piece-role orbits. A dense `Linear` with manually tied weights can emulate the forward pass, but the primitive's backward is an orbit-reduced gradient accumulation over group-pair orbits, not independent parameter gradients.

**Chess-specific motivation:**
Chess has more structure than D4 board symmetry. Color swap, side-to-move relativity, king-centered perspective, and partial piece-role relabellings create equivalences that standard convs do not encode. This primitive is meant to reduce sample complexity in the scout-scale regime where attention was data-hungry.

**Generalisation beyond chess:**
Applies to finite-group-structured board games, chemistry with typed symmetries, program graphs with variable renaming, and multi-agent systems with role permutations.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(Bρ d_in d_out)`, where `ρ` is nonzero orbit-pair count, vs dense `Linear` `O(BN²d_in d_out)`.
- Backward: `O(Bρ d_in d_out)` plus orbit-gradient reduction.
- Incremental update on a bounded-change input: `O(ρ_changed d_in d_out)`.

**Scout-scale falsification test:**
Drop it into the i193 parent as a replacement for one same-width 1×1 channel-mixing layer, keeping all inputs unchanged. Baseline: ordinary untied 1×1 linear/conv with similar parameter budget. Measure near-puzzle matched-recall FP and seed-stability. Works if it improves near-puzzle FP without reducing eval/sec by more than 10%; fails if it only reduces parameter count but worsens hard negatives.

**Failure mode catalogue:**
- Hidden rebrand: may be dismissed as parameter tying or known group convolution; novelty is only credible for the chess-specific finite action.
- Numerical instability: orbit sizes can create uneven gradient magnitudes; normalize by orbit cardinality.
- Too slow: orbit indexing can become scatter-heavy unless orbit tables are compacted.

**Status:** proposed

---

## 5. Proposal

### primitive_soft_tactical_distance

**Name:** Soft Tactical Distance Transform

**One-line claim:** Computes differentiable multi-step reachability over a dynamic move graph using a soft min-plus recurrence.

**Mathematical signature:**
For dynamic typed edge set `E_b`, node seeds `S ∈ {0,1}^{B×64}`, edge costs `C_e = ψ_Θ(X_u,X_v,τ)`, temperature `τ_s>0`, and horizon `K`:

\[
D_0[b,v]=
\begin{cases}
0,& S[b,v]=1\\
+\infty,& \text{otherwise}
\end{cases}
\]

\[
D_{t+1}[b,v]=-\tau_s\log\left(
e^{-D_t[b,v]/\tau_s}
+
\sum_{(u,v,\tau)\in E_b}
e^{-(D_t[b,u]+C_{b,u,v,\tau})/\tau_s}
\right)
\]

Return `D_K ∈ R^{B×64}` or all `D_{1:K}`. Gradients flow through edge costs and token features.

**Why this does not decompose into existing PyTorch ops:**
This is not attention because it is a soft min-plus dynamic program, not a convex weighted average of values. Neural Bellman-Ford Networks already show that generalized Bellman-Ford recurrences can be neuralized, so the broad idea is not brand-new. The primitive claim is the chess/dynamic-graph version: a fused differentiable distance transform over an internally supplied sparse legal/attack graph with semiring-style backward.

**Chess-specific motivation:**
Many hard false positives are near-puzzles: positions that look tactically sharp but fail under one or two forcing replies. A soft tactical distance primitive can expose "how many legal/attack steps away" a king, loose piece, or defended square is, without using PVs, engine scores, or node counts. It targets hard-negative discrimination rather than easy material cues.

**Generalisation beyond chess:**
Useful for routing, robot planning, differentiable program analysis, molecule reaction paths, and dynamic graphs where path cost matters more than neighbor averaging.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(BK|E|d_c)` vs dense K-step attention `O(BK n² d)`.
- Backward: `O(BK|E|d_c)` with stored soft predecessor weights.
- Incremental update on a bounded-change input: `O(K·ΔE·d_c)` plus affected-frontier propagation; worst-case `O(K|E|d_c)`.

**Scout-scale falsification test:**
Drop it into the i242 hard-negative classifier harness as a replacement for one attention-like relation operator, with small `K=2` or `K=3`. Baseline: identical harness with legal-move routing or dense attention. Works if matched-recall near-puzzle FP decreases and latency remains within 1.25× of baseline; fails if it helps only full PR AUC or becomes the slowest module.

**Failure mode catalogue:**
- Hidden rebrand: reviewer says it is NBFNet/Bellman-Ford; adjust claim to "underexplored chess primitive" unless fused dynamic legal graph is included.
- Numerical instability: small `τ_s` can create hard argmin gradients; use log-sum-exp stabilization.
- Too slow: K-step recurrence may be unacceptable for engine inference unless `K≤3`.

**Status:** proposed

---

## What I cut

1. **Differential Legal Attention.** Rejected because Differential Transformer already subtracts two softmax attention maps, and adding a legal mask would be an attention-mask variant rather than a new primitive.

2. **Piece-Type MoE Gate.** Rejected because routing piece tokens to expert functions is too close to standard conditional computation / MoE gating; it would be a composition of routing plus MLP experts, not a new operator.

3. **KAN Piece-Square Evaluator.** Rejected because KAN's core novelty is already learnable edge functions / spline-like transformations; using it on chess features would be an application, not a primitive invention.

4. **Hard-Negative Calibration Loss.** Rejected because it is a training objective, not a primitive. It may help CRTK class-1 discrimination, but it violates the "no training tricks" constraint.

5. **Bigger King-Zone Convolution.** Rejected because changing kernel size, dilation, or channel grouping is a hyperparameter / architecture choice over `Conv2d`, not a new computation graph.

## Bibliography

1. Albert Gu and Tri Dao, **"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"**. https://arxiv.org/abs/2312.00752
2. Tri Dao and Albert Gu, **"Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" / Mamba-2**. https://proceedings.mlr.press/v235/dao24a.html
3. Yutao Sun et al., **"Retentive Network: A Successor to Transformer for Large Language Models"**. https://arxiv.org/abs/2307.08621
4. Songlin Yang et al., **"Gated Linear Attention Transformers with Hardware-Efficient Training"**. https://proceedings.mlr.press/v235/yang24ab.html
5. Maximilian Beck et al., **"xLSTM: Extended Long Short-Term Memory"**. https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html
6. Ziming Liu et al., **"KAN: Kolmogorov-Arnold Networks"**. https://arxiv.org/abs/2404.19756
7. Tianzhu Ye et al., **"Differential Transformer"**. https://arxiv.org/abs/2410.05258
8. Taco Cohen and Max Welling, **"Group Equivariant Convolutional Networks"**. https://proceedings.mlr.press/v48/cohenc16.html
9. Víctor Garcia Satorras, Emiel Hoogeboom, and Max Welling, **"E(n) Equivariant Graph Neural Networks"**. https://proceedings.mlr.press/v139/satorras21a.html
10. Petar Veličković et al., **"Graph Attention Networks"**. https://arxiv.org/abs/1710.10903
11. PyTorch Geometric documentation, **"Creating Message Passing Networks"**. https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html
12. Yanping Zheng, Lu Yi, and Zhewei Wei, **"A survey of dynamic graph neural networks"**. https://link.springer.com/article/10.1007/s11704-024-3853-2
13. Zhaocheng Zhu et al., **"Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction"**. https://proceedings.neurips.cc/paper_files/paper/2021/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html
14. PyTorch documentation, **`torch.nn.MultiheadAttention`**. https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
15. PyTorch documentation, **`torch.nn.EmbeddingBag`**. https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
16. PyTorch documentation, **`torch.nn.Conv2d`**. https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
