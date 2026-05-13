# New Neural-Network Primitives for Chess Evaluation

Recent primitive-level work used as calibration: Mamba made SSM parameters input-dependent and used a hardware-aware recurrent/parallel scan ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)); xLSTM changed LSTM gating and memory structure ([Beck et al., 2024](https://arxiv.org/abs/2405.04517)); DeltaNet revisited matrix-valued recurrent memory with a delta-rule update and parallel training ([Yang et al., 2024](https://arxiv.org/abs/2406.06484)); Titans introduced a test-time neural memory module ([Behrouz et al., 2024/2025](https://arxiv.org/abs/2501.00663)); KAN moved learnable functions onto edges rather than node activations ([Liu et al., 2024](https://arxiv.org/abs/2404.19756)). That is the bar used here: operator-level changes, not masks, encodings, or module compositions.

Ranking snapshot within the shortlist: novelty plausibility: Hodge > Kirchhoff > Cubical Persistence > Orbit-IrrepNorm > Complementarity. RTX 3070 demonstrability: Orbit-IrrepNorm > Cubical Persistence > Kirchhoff > Hodge > Complementarity. Inference-speed advantage: Orbit-IrrepNorm > Kirchhoff > Hodge > Cubical Persistence > Complementarity. Generalisation beyond chess: Hodge > Cubical Persistence > Complementarity > Kirchhoff > Orbit-IrrepNorm.

1. ### primitive_orbit_irrep_norm

**Name:** Chess-Orbit IrrepNorm

**One-line claim:** Normalise chess features by irreducible symmetry orbits, making color/piece/board symmetries affect gradients directly.

**Mathematical signature:**
$f:\mathbb{R}^{B\times 64\times C}\to\mathbb{R}^{B\times 64\times C}$.
Let a finite chess symmetry group $G_\chi$ act by signed permutation matrices $U_g\in\{0,\pm1\}^{64C\times64C}$. For irreducible representation block $\lambda$,

$$
P_\lambda=\frac{d_\lambda}{|G_\chi|}\sum_{g\in G_\chi}\chi_\lambda(g)^*U_g.
$$

For $x_b=\mathrm{vec}(X_b)$,

$$
z_{b,\lambda}=P_\lambda x_b,\qquad
y_b=\sum_\lambda \gamma_\lambda\frac{z_{b,\lambda}}{\sqrt{\|z_{b,\lambda}\|_2^2/\mathrm{rank}(P_\lambda)+\epsilon}}+\beta P_{\mathrm{triv}}\mathbf{1}.
$$

Use the subgroup that preserves the target convention: e.g. file mirror, color-swap plus board rotation, and optional latent piece-channel relabellings.

**Why this does not decompose into existing PyTorch ops:**
This is not `LayerNorm` or `GroupNorm`: the normalisation axes are representation-theoretic projectors that mix squares, colors, and piece-type channels, not contiguous tensor dimensions. It is also not group convolution; group-equivariant CNNs share weights over transformed filters, while this operator changes the denominator and gradient coupling by irrep block. Group convolution is established prior work ([Cohen & Welling, 2016](https://arxiv.org/abs/1602.07576)); the proposed primitive is an orbit/irrep normalisation map over a finite chess action.

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: CRELU/color-involution graph messages and piece-relabelling/involution gates. Those are message or gate operators; here no edge, message, legal graph, or learned route exists. Closest second family: dynamic adjacency rank-order gates. This has no adjacency and no rank ordering; the only learned parameters are affine irrep scales.

**Chess-specific motivation:**
Small chess datasets waste samples relearning that mirrored/color-swapped positions should share evaluation geometry. This primitive forces the statistics and gradients of symmetric latent directions to be coupled without adding attention. It is most attractive at scout scale because it is cheap and may reduce near-puzzle false positives caused by color/perspective artifacts.

**Generalisation beyond chess:**
Reusable for any finite-group tensor domain: board games, molecules with discrete automorphisms, robotics grids with signed sensor channels.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(B|G_\chi|64C)$ vs LayerNorm $O(B64C)$
- Backward: $O(B|G_\chi|64C)$
- Incremental update on a bounded-change input: $O(|G_\chi|C)$ for cached irrep norms; $O(64C)$ if all output tokens must be materialised

**Scout-scale falsification test:**
Drop this after the first and last feature block of i193 conv-only parent. Baseline: same model with standard LayerNorm or no norm, parameter-count matched. Metric: CRTK class-1 matched-recall near-puzzle false-positive rate at fixed recall. “Works” means ≥5% relative FP reduction with <5% wall-clock slowdown. “Fails” means only aggregate PR AUC improves or latency rises enough to erase search value.

**Failure mode catalogue:**
- Reviewer objection: this may collapse to GroupNorm after a fixed basis change.
- Numerical risk: tiny irrep blocks can produce unstable denominators; use $\epsilon$ and minimum-rank block merging.
- Speed risk: materialising every $U_gX$ naively will be slower than the math suggests; implement as signed index views.

**Status:** proposed

2. ### primitive_weighted_hodge_projector

**Name:** Weighted Board Hodge Projection

**One-line claim:** Decompose latent board pressure into gradient, curl, and harmonic flow with a differentiable weighted Hodge solve.

**Mathematical signature:**
$f:\mathbb{R}^{B\times E\times C}\times\mathbb{R}_{+}^{B\times E}\to\mathbb{R}^{B\times E\times 3C}$, where $E=112$ oriented nearest-neighbour board edges. Let $D_0\in\{-1,0,1\}^{64\times E}$ be vertex-edge incidence and $D_1\in\{-1,0,1\}^{E\times49}$ edge-square incidence. For batch $b$, $W_b=\mathrm{diag}(w_b)$:

$$
G=D_0^\top(D_0W_bD_0^\top+\epsilon I)^{-1}D_0W_bF_b,
$$

$$
R=F_b-G,\qquad C=D_1(D_1^\top W_bD_1+\epsilon I)^{-1}D_1^\top W_bR,
$$

$$
H=R-C,\qquad Y_b=\mathrm{concat}(G,C,H).
$$

**Why this does not decompose into existing PyTorch ops:**
This is not `Conv2d`, graph convolution, attention, or a ray scan. Its connectivity is the fixed cell complex of the board, but the projection metric $W_b$ is input-dependent and changes the global linear solve and its implicit gradient. Hodge-aware simplicial learning exists, so the honest novelty claim is not “Hodge theory is new”; it is a cacheable chess-board Hodge primitive with weighted projection and incremental update as the operator ([Yang et al., OpenReview](https://openreview.net/forum?id=Nm5sp09Q25)).

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: attack-ray sparse attention/ray scans and legal-move graph accumulators. This operator does not send messages along legal moves or rays; it solves global orthogonal projections on a fixed cubical complex. Closest second family: occlusion semiring/ray semiring exchange. No semiring reduction appears; the algebra is weighted least-squares projection with implicit differentiation.

**Chess-specific motivation:**
Tactical pressure is not just local occupancy. A mating net can look like circulation around the king; a pin/skewer can look like a gradient flow into a bottleneck; fortress-like positions can leave harmonic residuals. This gives a conv net one global pressure decomposition without using Stockfish PVs or legal-move graph routing.

**Generalisation beyond chess:**
Useful for any lattice or mesh edge-flow data: traffic, fluid surrogates, biological transport, scene motion, power grids.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(B(64^3+49^3+(64^2+49^2)C))$ vs Conv2d $O(B64Ck^2)$
- Backward: same order via implicit linear-solve backward
- Incremental update on a bounded-change input: $O(r64^2+64C)$ with rank-$r$ Cholesky/Woodbury update; worst-case refactor $O(64^3)$

**Scout-scale falsification test:**
Insert one projection over a learned $C=8$ edge-flow head before the i193 value head. Baseline: same parameters using fixed 3×3 conv and global average pooling. Metric: matched-recall CRTK class-1 near-puzzle FP rate. “Works” means ≥5% relative FP reduction and no more than 15% inference slowdown. “Fails” if it only helps easy negatives or becomes numerically noisy.

**Failure mode catalogue:**
- Reviewer objection: it is just a fixed graph spectral layer plus solves.
- Numerical risk: near-zero $w_e$ makes Laplacians ill-conditioned; clamp $w_e$ and add $\epsilon I$.
- Speed risk: per-position Cholesky can dominate a tiny chess net unless $C$ is small and factors are reused.

**Status:** proposed

3. ### primitive_kirchhoff_forest_pool

**Name:** Differentiable Kirchhoff Forest Pool

**One-line claim:** Pool coordination by differentiating a Matrix-Tree log-determinant over all latent support forests.

**Mathematical signature:**
$f:\mathbb{R}^{B\times N\times N}\times\mathbb{R}_{+}^{B\times K\times N}\to\mathbb{R}^{B\times K\times(1+N^2)}$, $N=64$.
For symmetric edge logits $A_b$, define $w_{ij}=\mathrm{softplus}(A_{ij})$, Laplacian $L=D-W$, and root-strength diagonal $R_{bk}=\mathrm{diag}(\rho_{bk})$. Then

$$
s_{bk}=\log\det(L_b+R_{bk}+\epsilon I),
$$

$$
m_{bk,ij}=\frac{\partial s_{bk}}{\partial A_{bij}}
=\mathrm{softplus}'(A_{bij})\left[M^{-1}_{ii}+M^{-1}_{jj}-2M^{-1}_{ij}\right],
$$

where $M=L_b+R_{bk}+\epsilon I$. Output $Y_{bk}=(s_{bk},m_{bk,:,:})$.

**Why this does not decompose into existing PyTorch ops:**
This is not graph attention: it sums an exponential family of forests through a determinant and returns edge marginals as first-class outputs. The Matrix-Tree theorem connects Laplacian cofactors to spanning trees, and structured neural models have used matrix-tree partition functions before; the adjusted claim is a reusable neural pooling primitive for latent coordination graphs, not a newly discovered theorem ([Kim et al., 2017](https://arxiv.org/abs/1702.00887)).

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: move-graph routers/legal graph accumulators. Here the graph is a learned latent support graph, and no message is routed along legal moves. Closest second family: Pareto antichain or witness-counterwitness primitives. This is not a min/max proof operator; it is a smooth log-partition over forests with determinant gradients.

**Chess-specific motivation:**
Chess evaluation often depends on whether pieces form a connected attacking or defending support structure around a king. A near-puzzle false positive can have similar material and local attacks but lack a connected support forest. This operator gives one global coordination statistic without running search.

**Generalisation beyond chess:**
Scene-graph grounding, molecule stability, road-network reliability, biological interaction graphs, and any task where connectivity matters more than local messages.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(BKN^3)$ vs dense attention $O(BN^2d)$
- Backward: $O(BKN^3)$, or $O(BKN^2)$ after cached inverse/Cholesky for marginal outputs
- Incremental update on a bounded-change input: $O(rN^2)$ for rank-$r$ edge changes using determinant/inverse updates

**Scout-scale falsification test:**
Attach a $K=2$ forest pool to the i193 head using learned 64-node edge logits from the last board feature map. Baseline: same edge logits passed through mean/max pooling and a small MLP. Metric: CRTK class-1 matched-recall FP rate plus node/s throughput. “Works” means ≥5% FP reduction with <15% slowdown. “Fails” if logdet improves only calibration/PR AUC.

**Failure mode catalogue:**
- Reviewer objection: this is “just logdet pooling.”
- Numerical risk: edge logits can make $M$ nearly singular; use root diagonal, jitter, and Cholesky failure checks.
- Speed risk: $N^3$ is acceptable at $N=64$, but too slow if expanded to move nodes.

**Status:** proposed

4. ### primitive_cubical_persistence_pool

**Name:** Incremental Cubical Persistence Pool

**One-line claim:** Turn latent board maps into differentiable component-and-hole curves, not local pooled statistics.

**Mathematical signature:**
$f:\mathbb{R}^{B\times C\times 8\times8}\to\mathbb{R}^{B\times C\times2\times T}$.
For each scalar map $U_{bc}$, build the lower-star cubical complex on the $8\times8$ grid. Let $\mathrm{PH}_d(U_{bc})=\{(\alpha_p,\beta_p)\}$ be persistence pairs for $d\in\{0,1\}$, with birth/death cells $\alpha_p,\beta_p$. For thresholds $t_1,\dots,t_T$,

$$
Y_{bcd\ell}=\sum_{p\in \mathrm{PH}_d(U_{bc})}
\sigma((U_{\alpha_p}-t_\ell)/\tau)\,
\sigma((t_\ell-U_{\beta_p})/\tau).
$$

Gradients route to paired birth/death cells; ties use deterministic $\epsilon$-jitter or soft elder-rule relaxation.

**Why this does not decompose into existing PyTorch ops:**
This is not max pooling, average pooling, convolution, or attention. The computation graph includes a value-order-dependent union-find pairing step whose output connectivity changes when the latent scalar field changes. Topological deep learning and PyTorch-compatible topology libraries exist, so the honest claim is “underexplored primitive for chess,” not “persistent homology is newly invented” ([Hajij et al., 2024](https://link.springer.com/article/10.1007/s10462-024-10710-9)).

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: terminal-state detection and obstacle-pooling sparse emitters. This operator detects neither terminal labels nor ray obstacles; it computes connected components and holes of latent fields. Closest second family: Pareto antichain/frontier primitives. Persistence pairing is filtration topology, not dominance ordering.

**Chess-specific motivation:**
Pawn chains, king cages, open-file tunnels, and sealed fortresses are topological patterns on the board. A conv-only net can miss the distinction between “near mate” and “looks active but has an escape hole.” Persistence gives a cheap global summary of holes and connected regions.

**Generalisation beyond chess:**
Medical segmentation, materials microstructure, occupancy grids, point-cloud summaries, robotics maps.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(BCN\log N)$, $N=64$, vs pooling $O(BCN)$
- Backward: $O(BCN\log N)$ with sparse gradients to paired cells
- Incremental update on a bounded-change input: recompute $O(N\log N)$ on chess boards; dynamic persistence can be $O(\log N+\#\text{changed pairs})$

**Scout-scale falsification test:**
Append $T=8$ persistence curves from the last i193 feature map to the value head. Baseline: global mean/max pooling with equal output dimension. Metric: matched-recall CRTK class-1 near-puzzle FP rate; track inference ms separately. “Works” means ≥3% relative FP reduction with <10% slowdown. “Fails” if it mostly boosts easy-negative PR AUC.

**Failure mode catalogue:**
- Reviewer objection: topological layers already exist; the novelty is chess-specific incremental cubical pooling, not PH itself.
- Numerical risk: many equal board activations create unstable pairings; add deterministic tie-breaking.
- Speed risk: Python union-find will be too slow; needs fused C++/CUDA or tiny-board CPU batching.

**Status:** proposed

5. ### primitive_monotone_complementarity_exchange

**Name:** Monotone Complementarity Exchange

**One-line claim:** Resolve competing latent threats by a monotone complementarity solve with active-set gradients.

**Mathematical signature:**
$f:\mathbb{R}^{B\times p}\times\mathbb{R}^{B\times p\times p}\to\mathbb{R}^{B\times2p}$.
Given $h_b$ and $A_b$, form

$$
M_b=A_b^\top A_b+\mathrm{diag}(\mathrm{softplus}(m_b))+\eta I.
$$

Return $z_b,q_b$ satisfying the monotone LCP

$$
z_b\ge0,\qquad q_b=M_bz_b+h_b\ge0,\qquad z_b\odot q_b=0.
$$

Equivalently,

$$
z_b=\arg\min_{u\ge0}\frac12u^\top M_bu+h_b^\top u,\qquad q_b=M_bz_b+h_b.
$$

On a fixed active set $\mathcal{A}$, backward uses

$$
dz_{\mathcal{A}}=-M_{\mathcal{A}\mathcal{A}}^{-1}(dh_{\mathcal{A}}+dM_{\mathcal{A}\mathcal{A}}z_{\mathcal{A}}),\quad dz_{\bar{\mathcal{A}}}=0.
$$

**Why this does not decompose into existing PyTorch ops:**
This is not ReLU: ReLU is coordinatewise, while this operator chooses a coupled active set through $M$. It overlaps with differentiable optimisation layers such as OptNet and CVXPYlayers; the adjusted claim is a narrow `torch.nn`-style monotone LCP primitive with cached active-set forward/backward, not a generic convex-program layer ([Amos & Kolter, 2017](https://arxiv.org/abs/1703.00443); [Agrawal et al., 2019](https://arxiv.org/abs/1910.12430)).

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: Regret Saddlepoint and Witness-Counterwitness Quantifier. Those are game/logic-style selection operators; this is a continuous complementarity map with KKT gradients. Closest second family: factor-graph/tensor-product legal-state primitives. No legal-state factor graph is compiled; $M,h$ are latent continuous contest parameters.

**Chess-specific motivation:**
Overloaded defenders are complementarity phenomena: one resource cannot fully answer two threats. Tactical near-misses often fail because a defender has exactly one active response; easy negatives do not test this. This primitive lets a small net represent mutually exclusive latent obligations without adding search.

**Generalisation beyond chess:**
Contact physics, market clearing, traffic assignment, resource allocation, differentiable games with inequality constraints.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(Bp_a^3)$ for active set size $p_a$, vs MLP $O(Bp^2)$
- Backward: $O(Bp_a^3)$, usually one triangular solve after factorisation
- Incremental update on a bounded-change input: $O(rp_a^2)$ if active set is unchanged; worst-case $O(p^3)$ after active-set flips

**Scout-scale falsification test:**
Use $p=16$ latent contest variables before the i193 value head. Baseline: same $p$-dimensional bottleneck with two-layer MLP and GELU. Metric: CRTK class-1 matched-recall near-puzzle FP rate and inference latency. “Works” means ≥5% FP reduction with <20% slowdown. “Fails” if active-set flips dominate training or the MLP matches it.

**Failure mode catalogue:**
- Reviewer objection: this is merely OptNet/QP with a chess name.
- Numerical risk: active-set boundaries create nondifferentiable kinks; add $\eta I$, warm starts, and subgradient tests.
- Speed risk: batched active-set solves may be slower than the whole scout network.

**Status:** proposed

## What I cut

- **Legal-move dynamic attention:** duplicate of legal-move graph routers and sparse legal graph transitions; still masked/sparse attention.
- **Ray-occlusion selective scan:** duplicate of ray scans, ray-piece kernels, ray-parallel SSMs, and obstacle-pooling emitters.
- **Piece-existence Möbius/Hessian spectrum:** too close to signed piece-existence Hessian and pair-resonance operators.
- **Promotion counterfactual JVP:** duplicate of promotion-fanout counterfactual tensors and sparse delta/counterfactual families.
- **Chess-group equivariant convolution:** mostly a group-conv rebrand, and too close to color-involution/piece-relabelling gates.
- **Soft minimax proof-number layer:** overlapped with regret saddlepoint and witness-counterwitness quantifier primitives.
- **Optimal-transport attacker/defender assignment:** too close to Sinkhorn/OT layers; weaker novelty than complementarity exchange.
- **Exterior-wedge token interaction:** interesting alternating multilinear operator, but likely judged a tensor-product/hyperedge variant and too slow for scout-scale.
- **Delta-updated NNUE cache variants:** most became “maintain an accumulator under bounded move delta,” which the duplicate blocklist already warns against.
