# Invent New Neural Primitives for Chess Evaluation

I interpreted “non-decomposable” as: no graph-equivalent composition of existing `torch.nn` primitives with the same signature, adjoint, and incremental-update state. A prototype may still use low-level tensor ops, just as LayerNorm can be prototyped from reductions.

Recent primitive-style literature sets the bar: Mamba changed the recurrence itself through selective state-space updates, xLSTM changed gated memory with exponential gates and scalar/matrix memory, KAN moved learnable functions onto edges, and differentiable algorithmic layers such as DataSP/OptNet treat solvers as trainable operators rather than ordinary blocks. NNUE’s lesson is also structural: sparse inputs and small board deltas enable efficient accumulator updates, not merely better feature engineering.

Overall ranking used for ordering: **#1 best overall**, **#2 best speed/generalisation tradeoff**, **#3 fastest scout-scale falsifier**, **#4 strongest global-board operator but slower**, **#5 useful but highest prior-work overlap**.

Axis order:

- Novelty: **1 > 3 > 2 > 4 > 5**
- RTX-3070 demonstrability: **3 > 1 > 2 > 5 > 4**
- Inference-speed advantage: **3 > 1 > 2 > 5 > 4**
- Generalisation beyond chess: **2 > 4 > 1 > 5 > 3**

---

## 1. primitive_symmetric_coalition_pool

**Name:** Elementary-Symmetric Coalition Pool

**One-line claim:** Computes unordered low-order piece coalitions through a generating-polynomial primitive with exact add/remove updates.

**Mathematical signature:**

For piece/event latents \(X\in\mathbb{R}^{B\times m\times d}\), mask \(g\in\{0,1\}^{B\times m}\), parameters \(W\in\mathbb{R}^{d\times q}\), degree \(K\):

\[
u_{bi}=\tanh(X_{bi}W)\in\mathbb{R}^{q},\quad
P_b(z)=\prod_{i=1}^{m}(1+g_{bi}z u_{bi})
\]

with elementwise products. Output \(E\in\mathbb{R}^{B\times K\times q}\):

\[
E_{b,k}=\sum_{|S|=k}\bigodot_{i\in S}u_{bi},\quad k=1,\dots,K.
\]

Forward recurrence:

\[
E_0=\mathbf{1},\quad E_k\leftarrow E_k+g_i u_i\odot E_{k-1}.
\]

Gradients are polynomial and well-defined.

**Why this does not decompose into existing PyTorch ops:**

DeepSets-style sum pooling represents \(\rho(\sum_i\phi(x_i))\), while this operator exposes elementary-symmetric coefficients that preserve low-order multiplicative coalitions without enumerating tuples. Self-attention would materialise pairwise or higher interactions; this primitive maintains a polynomial state with a custom adjoint and delete-update path. A slow Python loop can emulate values, but not the same computation graph, gradient sparsity, or \(O(Kq)\) incremental state.

**Duplicate audit against existing primitive memory:**

Closest blocklist item 1: Signed Piece-Existence Hessian / pair-resonance Hessian. This is not a derivative over piece-existence bits; it computes forward elementary-symmetric moments over learned latents and supports \(k>2\) without curvature terms.

Closest blocklist item 2: delta pair selective bispectra / bilinear hyperedges. This has no legal/ray edge set, no semiring exchange, and no pairwise graph connectivity; all interaction is through unordered polynomial coefficients.

**Chess-specific motivation:**

Chess has exchangeable same-type pieces but non-additive coalitions: bishop pair, doubled rooks, queen+knight mating nets, attacker+defender combinations. A conv or sum pool can blur “two weak pieces jointly decisive” into average evidence. This primitive directly tests whether low-order coalitions reduce verified-near-puzzle false positives.

**Generalisation beyond chess:**

Useful for sparse-event sequences, molecules, particle sets, recommendation baskets, and scene graphs where unordered small coalitions matter.

**Complexity (forward, backward, incremental-update):**

- Forward: \(O(BmKq)\) vs attention \(O(Bm^2q)\) or explicit \(K\)-tuple enumeration \(O(B{m\choose K}q)\)
- Backward: \(O(BmKq)\)
- Incremental update on a bounded-change input: add/delete \(O(BKq)\) with cached coefficients; delete via reverse polynomial division

**Scout-scale falsification test:**

Drop after the final feature map of i193 conv-only: gather existing piece-present square embeddings, apply degree \(K=3,q=64\), concatenate flattened \(E\) to the current head. Baseline: same i193 with mean+max piece pooling and matched parameter count. Metric: matched-recall CRTK class-1 near-puzzle false-positive rate. Works if FP rate falls by ≥5% relative at the same recall with ≤10% eval-speed loss; fails if only aggregate PR AUC improves.

**Failure mode catalogue:**

- Reviewer objection: “This is just DeepSets with products.” Rebuttal depends on showing elementary coefficients cannot be reduced to one additive statistic without losing bounded-degree coalition identity.
- Products may underflow/overflow for \(K>4\); use \(K\le3\), tanh latents, or log-domain coefficients.
- If \(m\le32\) but \(q,K\) are oversized, the head becomes slower than its tactical benefit.

**Status:** proposed

---

## 2. primitive_determinantal_coverage_pool

**Name:** Rank-One Determinantal Coverage Pool

**One-line claim:** Measures whether pieces span complementary latent directions using log-volume, with Sherman–Morrison updates.

**Mathematical signature:**

For \(X\in\mathbb{R}^{B\times m\times d}\), heads \(h=1,\dots,H\), rank \(r\):

\[
v_{bih}=W_hX_{bi}\in\mathbb{R}^{r},\quad
a_{bih}=\operatorname{softplus}(p_h^\top X_{bi})
\]

\[
M_{bh}=\lambda I_r+\sum_{i=1}^{m}a_{bih}v_{bih}v_{bih}^{\top},\quad
y_{bh}=\log\det M_{bh}.
\]

Output \(y\in\mathbb{R}^{B\times H}\), optionally followed by affine mixing. Gradients use:

\[
\partial\log\det M=\operatorname{tr}(M^{-1}\partial M).
\]

**Why this does not decompose into existing PyTorch ops:**

Attention forms weighted sums of values; this returns a determinant of a rank-one Gram accumulation, so the gradient of one piece is mediated through \(M^{-1}\) and all other span directions. DPP/log-det ideas exist for diversity, but not as a reusable chess-evaluation primitive with cached inverse-state updates. A `matmul+slogdet` prototype lacks the primitive’s rank-one add/delete computation graph.

**Duplicate audit against existing primitive memory:**

Closest blocklist item 1: pair-resonance Hessian operators. This is not a signed pair curvature map; pair effects appear only through determinant volume of a PSD matrix.

Closest blocklist item 2: delta pair selective bispectra / bilinear ray-blocked segment attention. There are no ray segments, legal edges, or sparse graph messages; all pieces contribute to a single low-rank coverage matrix.

**Chess-specific motivation:**

A side’s pieces can be redundant or complementary. Three defenders covering the same square are not equivalent to three defenders covering independent escape squares; two rooks stacked behind the same blocker differ from rooks spanning files/ranks. Log-volume gives a primitive-level “coverage diversity” signal that mean pooling and attention often wash out.

**Generalisation beyond chess:**

Applies to sensor placement, active learning, multi-camera coverage, object-set diversity, and retrieval where independent evidence matters.

**Complexity (forward, backward, incremental-update):**

- Forward: \(O(BH(mr^2+r^3))\) vs attention \(O(Bm^2d)\); for \(r\le16,m\le32\), this is small
- Backward: \(O(BH(mr^2+r^3))\)
- Incremental update on a bounded-change input: \(O(Hr^2)\) per add/delete using matrix determinant lemma and Sherman–Morrison inverse update

**Scout-scale falsification test:**

Attach \(H=8,r=12\) determinant heads at i193’s global pooling point; compare against a parameter-matched mean+max+second-moment pool. Measure matched-recall CRTK class-1 FP rate and engine eval nodes/sec. Works if FP drops ≥5% relative and eval speed drops ≤15%; fails if gains appear only on easy negatives.

**Failure mode catalogue:**

- Reviewer objection: “This is just logdet regularisation/DPP.” The distinction must be the forward primitive used as representation, not a diversity loss or sampler.
- Near-singular \(M\) can destabilise gradients; enforce \(\lambda\ge10^{-3}\) and cap \(a_i\).
- \(r^3\) dominates if rank is inflated; keep \(r\) deliberately tiny.

**Status:** proposed

---

## 3. primitive_character_orbit_norm

**Name:** Character-Orbit Normalization

**One-line claim:** Normalizes activations by finite chess-group orbits and irreducible character energies instead of feature dimensions.

**Mathematical signature:**

Let \(X\in\mathbb{R}^{B\times C\times 8\times 8}\). Let finite group \(G\) act by spatial/channel signed permutations \(T_g\). For irreducible character \(\chi_\rho\):

\[
m_{\rho}(X)=\frac{d_\rho}{|G|}\sum_{g\in G}\chi_\rho(g)^{*}T_gX
\]

\[
Y=\sum_{\rho\in\hat G}\gamma_\rho
\frac{m_\rho(X)}{\sqrt{\epsilon+\operatorname{mean}_{\Omega,\rho}(m_\rho(X)^2)}}+\beta_{\text{triv}}.
\]

Output \(Y\in\mathbb{R}^{B\times C\times8\times8}\). Gradients are those of finite linear projections plus normalization.

**Why this does not decompose into existing PyTorch ops:**

LayerNorm normalizes over chosen feature dimensions, not over a finite-group orbit with character projections. Group-equivariant networks usually build equivariance into convolutions; this is a normalization primitive whose Jacobian couples all symmetry-related activations by irrep energy. Equivariant neural networks broadly rely on preserving group actions, but this operator targets normalization rather than message passing or convolution.

**Duplicate audit against existing primitive memory:**

Closest blocklist item 1: CRELU/color-involution graph messages. This has no graph edges, no messages, and no activation gate; it is a representation-aware normalization map.

Closest blocklist item 2: color-involution adjacency updates / dynamic adjacency rank-order gates. There is no learned adjacency or rank-order routing; \(G\) is fixed and the only learned terms are scale/bias per irrep.

**Chess-specific motivation:**

Chess evaluation should transform predictably under board reflections and color swap. Small-data scouts waste samples relearning mirrored/color-swapped evidence. This primitive can enforce the useful part of symmetry while leaving nontrivial irrep channels available for side-to-move and asymmetric tactical information.

**Generalisation beyond chess:**

Best for finite-symmetry domains: board games, cellular automata, grid robotics, puzzle states, and symbolic vision. Less useful for domains without known finite actions.

**Complexity (forward, backward, incremental-update):**

- Forward: \(O(B|G|C64)\) vs LayerNorm \(O(BC64)\); \(|G|\) is small
- Backward: \(O(B|G|C64)\)
- Incremental update on a bounded-change input: \(O(|G|C)\) per changed square with cached orbit moments

**Scout-scale falsification test:**

Replace the first normalization after i193’s opening conv with Character-OrbitNorm using \(G=\) board reflections/rotations allowed by the existing orientation plus color swap where labels are sign-adjusted. Baseline: same model with LayerNorm/GroupNorm and equal affine parameters. Works if CRTK class-1 FP drops ≥3% relative or same FP is reached one epoch earlier; fails if symmetry hurts side-to-move discrimination.

**Failure mode catalogue:**

- Reviewer objection: “This is just group equivariant CNN machinery.” The claim survives only if evaluated as a drop-in normalization primitive, not as a group-conv architecture.
- Wrong group actions can enforce false symmetry, especially around side-to-move and castling-like asymmetries.
- \(|G|\)-fold gathers may be memory-bandwidth bound unless fused.

**Status:** proposed

---

## 4. primitive_kirchhoff_mobility_solve

**Name:** Differentiable Kirchhoff Mobility Solve

**One-line claim:** Solves a board-scale conductance equilibrium to expose bottlenecks, cages, and mobility corridors in one primitive.

**Mathematical signature:**

For square latents \(X\in\mathbb{R}^{B\times N\times d}\), \(N=64\), fixed grid edge set \(E\), incidence \(D\):

\[
c_{be}=\operatorname{softplus}(w_c^\top[X_{bu};X_{bv}])+\epsilon,
\quad
L_b=D^\top\operatorname{diag}(c_b)D+\lambda I
\]

\[
S_b=X_bW_s\in\mathbb{R}^{N\times p},\quad
U_b=L_b^{-1}S_b,\quad
Y_b=U_bW_o.
\]

Output \(Y\in\mathbb{R}^{B\times N\times d'}\). Backward uses implicit differentiation through the SPD solve.

**Why this does not decompose into existing PyTorch ops:**

A convolution or diffusion layer applies a fixed number of local steps; this primitive returns the exact equilibrium of a learned conductance system. Differentiable optimization and graph-learning layers are accepted precedents for solver-valued neural operators, but this proposed primitive is a board-mobility SPD solve, not an unrolled GNN. A naive `torch.linalg.solve` call gives values but not the intended sparse Cholesky/CG adjoint and cached rank-update state.

**Duplicate audit against existing primitive memory:**

Closest blocklist item 1: SLG diffusion / factor-graph legal-state primitives. This is not legal-move diffusion and does not propagate over legal edges; it solves a fixed grid conductance equation.

Closest blocklist item 2: ray-occlusion dispatch / obstacle-pooling sparse emitters. No rays, blockers, or segment reducers are present; bottlenecks arise from Laplacian conductance, not chess-ray scans.

**Chess-specific motivation:**

King safety and fortress evaluation are often about connected safe regions and narrow bottlenecks, not merely attacked-square counts. A one-square blocker can change the effective conductance between king, escape squares, and attacking zones. That global connectivity is hard for shallow convs and expensive for attention.

**Generalisation beyond chess:**

Useful for occupancy grids, robot navigation, circuit-like reasoning, PDE surrogates, scene connectivity, and physical simulation. Effective-resistance-style graph quantities are already used in graph analysis and GNN theory.

**Complexity (forward, backward, incremental-update):**

- Forward: \(O(B((N+|E|)t p))\) with CG or \(O(BN^3)\) dense; closest conv is \(O(B|E|d)\) but only local
- Backward: one or two additional SPD solves plus edge-gradient accumulation
- Incremental update on a bounded-change input: \(O(N^2p)\) with cached inverse/Cholesky rank updates; not O(1)

**Scout-scale falsification test:**

Insert one \(p=8\) Kirchhoff layer after i193’s first 8×8 feature tensor; concatenate \(Y\) before the existing head. Baseline: one parameter-matched 3×3 conv. Works if near-puzzle FP drops ≥5% relative with ≤25% eval-speed loss; fails if it improves only quiet-position regression.

**Failure mode catalogue:**

- Reviewer objection: “This is just a graph diffusion layer.” The primitive must solve the implicit equilibrium exactly, not run \(T\) message-passing iterations.
- Ill-conditioned \(L\) can explode gradients; require \(\epsilon,\lambda>0\) and cap conductances.
- Too slow for engine inference unless \(p\) is tiny and the solver is fused.

**Status:** proposed

---

## 5. primitive_choquet_coalition_integral

**Name:** K-Additive Choquet Coalition Integral

**One-line claim:** Aggregates piece evidence by learned substitute/complement coalitions instead of sums, maxes, or attention weights.

**Mathematical signature:**

For \(X\in\mathbb{R}^{B\times m\times d}\), per channel \(q\), sort \(x_{b\pi_1q}\ge\dots\ge x_{b\pi_mq}\), set \(x_{b\pi_{m+1}q}=0\). Let monotone \(K\)-additive capacity \(\mu_q(S)\) be parameterized by nonnegative unary and low-order coalition terms. Output:

\[
y_{bq}=\sum_{r=1}^{m}(x_{b\pi_rq}-x_{b\pi_{r+1}q})\,
\mu_q(\{\pi_1,\dots,\pi_r\}).
\]

Output \(y\in\mathbb{R}^{B\times d}\). Gradient is piecewise linear; ties use averaged subgradients.

**Why this does not decompose into existing PyTorch ops:**

Mean/max pooling is additive or idempotent; attention computes row-wise normalized weighted sums. Choquet aggregation uses sorted coalition prefixes and a learned non-additive capacity, so the Jacobian changes with order statistics and coalition membership. Choquet/fuzzy-integral neural modules exist, so the global novelty claim should be “underexplored primitive for chess,” not “invented from zero.”

**Duplicate audit against existing primitive memory:**

Closest blocklist item 1: Pareto Antichain Frontier / tail copula concordance. This emits scalar/vector integrals over ordered coalitions, not a frontier, copula, or concordance statistic.

Closest blocklist item 2: dynamic adjacency rank-order gates. Sorting is used for aggregation only; there is no adjacency, message routing, legal edge, or gate over graph neighbours.

**Chess-specific motivation:**

Many chess patterns are non-additive: two attackers may be decisive, a third redundant; bishop pair synergy differs from two isolated bishops; one defender can substitute for several weak defenders. Choquet capacity terms directly model complementarity and substitutability without creating an attention graph.

**Generalisation beyond chess:**

Sensor fusion, risk scoring, medical feature fusion, ensemble aggregation, and multi-criteria decision systems use this kind of non-additive evidence fusion. Recent work continues to explore Choquet-style neural aggregation.

**Complexity (forward, backward, incremental-update):**

- Forward: \(O(Bdm\log m + Bd\,C_K)\) vs mean pooling \(O(Bdm)\), attention \(O(Bm^2d)\)
- Backward: \(O(Bdm\log m + Bd\,C_K)\)
- Incremental update on a bounded-change input: worst \(O(dm)\); expected \(O(d\log m)\) if order changes are local and prefix capacities are cached

**Scout-scale falsification test:**

Replace i193’s global mean/max pool with a 2-additive Choquet pool over piece-present square embeddings; parameter-match with a small MLP pool baseline. Measure matched-recall CRTK class-1 near-puzzle FP rate. Works if FP drops ≥4% relative and speed loss ≤20%; fails if the learned capacity collapses to unary weights.

**Failure mode catalogue:**

- Reviewer objection: “This is known fuzzy-integral pooling.” Correct; the defensible claim is chess-specific primitive adaptation plus incremental capacity caching.
- Sorting causes nondifferentiability at ties; use deterministic tie averaging and monitor gradient noise.
- Full capacities are exponential; restrict to 2-additive or 3-additive terms.

**Status:** proposed

---

# What I cut

- **Legal-move Sinkhorn coupler:** rejected as mostly entropic OT plus legal-edge routing; Sinkhorn/ranking operators are already well studied, and the legal connectivity would collide with move-graph routers.
- **Fenwick sparse accumulator primitive:** rejected as a direct sparse-delta / incremental latent accumulator duplicate.
- **Ray-Hodge scan:** rejected because the useful part was still ray-scan plus ray-occlusion state.
- **Soft alpha-beta backup layer:** rejected as too close to terminal-state, witness/counterwitness, and regret-saddlepoint primitives.
- **Tropical Pareto threat frontier:** rejected as a Pareto antichain/frontier rebrand.
- **Complex spinor interference unit:** rejected as a near-duplicate of complex-amplitude interference.
- **Promotion fanout normalizer:** rejected because promotion-fanout counterfactual tensors are explicitly blocklisted.
- **Differentiable shortest path on legal graph:** rejected because DataSP-style differentiable shortest paths are real, but using legal-move edges made it a move-graph/diffusion duplicate.
- **Cubical persistent-homology cage pool:** rejected from the final 5 because differentiable topology layers already exist and scout-scale latency looked worse than Kirchhoff/Choquet for the same king-cage motivation.
- **Top-k threat sort gate:** rejected as too close to dynamic rank-order gates and ordinary top-k pooling.

---

# References

- Gu, A. and Dao, T. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.” arXiv, 2023. https://arxiv.org/abs/2312.00752
- Beck, M. et al. “xLSTM: Extended Long Short-Term Memory.” arXiv, 2024. https://arxiv.org/abs/2405.04517
- Liu, Z. et al. “KAN: Kolmogorov-Arnold Networks.” arXiv, 2024. https://arxiv.org/abs/2404.19756
- Amos, B. and Kolter, J. Z. “OptNet: Differentiable Optimization as a Layer in Neural Networks.” arXiv, 2017. https://arxiv.org/abs/1703.00443
- Vlastelica, M. et al. “Differentiation of Blackbox Combinatorial Solvers.” ICLR, 2020. https://openreview.net/forum?id=BkevoJSYPB
- Mensch, A. et al. “Differentiable Dynamic Programming for Structured Prediction and Attention.” ICML, 2018. https://proceedings.mlr.press/v80/mensch18a.html
- Stockfish NNUE PyTorch Wiki. “NNUE.” https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html
- Zaheer, M. et al. “Deep Sets.” NeurIPS, 2017. https://arxiv.org/abs/1703.06114
- Kulesza, A. and Taskar, B. “Determinantal Point Processes for Machine Learning.” Foundations and Trends in Machine Learning, 2012. https://arxiv.org/abs/1207.6083
- Bronstein, M. et al. “Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.” arXiv, 2021. https://arxiv.org/abs/2104.13478
- Cohen, T. and Welling, M. “Group Equivariant Convolutional Networks.” ICML, 2016. https://arxiv.org/abs/1602.07576
- Marichal, J.-L. “An axiomatic approach of the discrete Choquet integral as a tool to aggregate interacting criteria.” IEEE Transactions on Fuzzy Systems, 2000. https://doi.org/10.1109/91.890332
- Grabisch, M. “k-order additive discrete fuzzy measures and their representation.” Fuzzy Sets and Systems, 1997. https://doi.org/10.1016/S0165-0114(97)00021-7
- Cuturi, M. et al. “Differentiable Ranking and Sorting using Optimal Transport.” NeurIPS, 2019. https://papers.neurips.cc/paper/8910-differentiable-ranking-and-sorting-using-optimal-transport
