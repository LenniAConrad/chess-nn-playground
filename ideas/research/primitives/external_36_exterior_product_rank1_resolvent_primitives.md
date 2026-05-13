# New Neural Primitives for Chess Evaluation

Scope: primitive-level operators only. No new architecture, input encoding, loss, data trick, or Stockfish/PV/node-count feature is proposed. The novelty bar was calibrated against recent primitive-level work including [Mamba/S6](https://arxiv.org/abs/2312.00752), [xLSTM](https://arxiv.org/abs/2405.04517), [KAN](https://arxiv.org/abs/2404.19756), and [DeltaNet](https://arxiv.org/abs/2406.06484), plus older but relevant operator precedents such as [G-CNNs](https://proceedings.mlr.press/v48/cohenc16.html), [OptNet](https://arxiv.org/abs/1703.00443), [Clifford Group Equivariant Neural Networks](https://arxiv.org/abs/2305.11141), and differentiable [topology layers](https://proceedings.mlr.press/v108/gabrielsson20a.html). I do not claim any experimental result below; each test is a falsification plan.

Combined ranking used below: exterior-product pool > rank-1 resolvent pool > orbit-stabilized canonicalizer > tropical distance transform > capacitated entropic assignment. Axis ranks: novelty plausibility: 1, 2, 3, 4, 5; RTX-3070 demonstrability: 1, 2, 4, 3, 5; inference-speed advantage: 2, 1, 4, 3, 5; generalisation beyond chess: 2, 1, 4, 5, 3.

1. ### primitive_exterior_product_pool

**Name:** Truncated Exterior Product Pool

**One-line claim:** Pool active pieces as a multivector whose wedge grades encode non-redundant high-order co-presence.

**Mathematical signature:**
For latent piece tokens $X\in\mathbb{R}^{B\times n\times d}$, activity $a\in[0,1]^{B\times n}$, projection $W\in\mathbb{R}^{d\times r}$, and max grade $R$:
$$
z_{bi}=a_{bi}X_{bi}W\in\mathbb{R}^{r},\qquad
M_b=\prod_{i=1}^{n}(1+z_{bi})_{\wedge,\le R}.
$$
Grade $k$ is
$$
M_b^{(k)}=\sum_{|I|=k}\bigwedge_{i\in I}z_{bi}\in\Lambda^k\mathbb{R}^{r}.
$$
Output $Y_b=\operatorname{concat}_{k=0}^{R}\operatorname{vec}(M_b^{(k)})\in\mathbb{R}^{D_R}$, where $D_R=\sum_{k=0}^{R}{r\choose k}$. Gradients are polynomial and well-defined away from optional clipping.

**Why this does not decompose into existing PyTorch ops:**
This is not sum pooling, max pooling, attention, or a bilinear pair layer. The primitive computation graph is grade-projected antisymmetric multiplication in an exterior algebra, with nilpotent deletion $(1+z)^{-1}=1-z$ for incremental updates. Clifford/geometric-algebra networks already use multivectors and grade projections, so the surrounding algebra is not invented here; the new primitive claim is the multiset exterior-product pool with bounded-change deletion semantics, not a standard `torch.nn` operator.

**Duplicate audit against existing primitive memory:**
Closest blocklisted family 1: **Signed Piece-Existence Hessian / pair-resonance Hessian**. This is not a second derivative over piece bits; grade $k$ is an alternating $k$-body volume, not pair sensitivity. Closest family 2: **sparse delta accumulators / incremental latent accumulators**. It has bounded-change updates, but the state algebra is multiplicative exterior product with exact inverse deletion, not additive scatter or accumulator maintenance.

**Chess-specific motivation:**
Many tactical positions are about non-redundant resources: two defenders on the same functional line are weaker than two independent defenders. Wedge products suppress linearly dependent latent directions and highlight independent attacking or defending sets without legal-move graph routing. That targets near-puzzle hard negatives where ordinary pooling overcounts resources.

**Generalisation beyond chess:**
Useful for variable-size sets where independence, redundancy, or high-order co-presence matters: molecules, recommender bundles, scene-object sets, sparse event logs.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(BnrD_R)$ vs explicit $R$-body enumeration $O(Bn^R)$ and attention $O(Bn^2d)$
- Backward: $O(BnrD_R)$
- Incremental update on a bounded-change input: $O(B\Delta rD_R)$, with $\Delta\le4$ piece add/remove events for a normal chess move

**Scout-scale falsification test:**
Drop the primitive after the penultimate piece-token projection in the i193 conv-only parent, with $r=8,R=3$. Baseline: same parameter budget using sum+max pooling over active piece tokens. Train one seed for 6 epochs on the 173k scout set. Works if CRTK class-1 matched-recall near-puzzle FP rate drops by at least 10% with less than 15% nodes-per-second slowdown. Fails if improvement is only aggregate PR AUC or if slowdown exceeds 25%.

**Failure mode catalogue:**
- Hidden rebrand: reviewer may call it polynomial pooling with antisymmetry; the defense is weak unless grade outputs and inverse deletion are actually used.
- Numerical instability: high grades can explode or vanish; grade-wise RMS normalisation may be needed.
- Speed risk: naïve `einsum` wedge tables are too slow; it needs precomputed sparse grade-index kernels.

**Status:** proposed

2. ### primitive_rank1_resolvent_pool

**Name:** Rank-1 Resolvent Pool

**One-line claim:** Pool pieces through an inverse precision matrix so redundant resources are downweighted by linear dependence.

**Mathematical signature:**
Given latent tokens $Z\in\mathbb{R}^{B\times n\times r}$, gates $a\in[0,1]^{B\times n}$, queries $Q\in\mathbb{R}^{B\times m\times r}$, and $\lambda>0$:
$$
S_b=\lambda I_r+\sum_{i=1}^{n}a_{bi}z_{bi}z_{bi}^{\top},\qquad
P_b=S_b^{-1}.
$$
Outputs:
$$
Y_b=Q_bP_b\in\mathbb{R}^{m\times r},\qquad
\ell_{bi}=z_{bi}^{\top}P_bz_{bi},\qquad
s_b=\log\det S_b.
$$
The primitive returns $(Y,\ell,s)$. Gradients follow from $dS^{-1}=-S^{-1}(dS)S^{-1}$ and $d\log\det S=\operatorname{tr}(S^{-1}dS)$.

**Why this does not decompose into existing PyTorch ops:**
A slow prototype can be written with `matmul` and `torch.linalg.solve`, but that graph recomputes a dense solve and loses the primitive’s defining operation: exact rank-1 add/delete by Sherman-Morrison with a custom backward over the maintained inverse. This differs from linear attention and xLSTM-style matrix memory: those store or update a matrix-valued memory, while this exposes the inverse precision/resolvent and leverage scores as the operator output. It is closer in spirit to a neural RLS primitive than to attention.

**Duplicate audit against existing primitive memory:**
Closest blocklisted family 1: **signed edit bilinear memory / delta pair selective bispectra**. Those accumulate bilinear memories or spectra; this returns $(\lambda I+\sum zz^\top)^{-1}$, so every update changes all directions through the inverse. Closest family 2: **sparse delta accumulators / reversible delta kernels**. It does support bounded updates, but the algebra, gradient, and update rule are Woodbury inverse updates, not additive latent deltas.

**Chess-specific motivation:**
Chess resources are often overloaded or collinear. If several defenders encode the same latent direction, the precision inverse and leverage scores expose redundancy instead of counting all defenders independently. This is a direct hard-negative mechanism for “looks defended but is tactically overloaded.”

**Generalisation beyond chess:**
Dynamic sensor fusion, online least squares, streaming recommender sets, active learning, memory compression, and any sparse-event domain where leverage or redundancy matters.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(Bnr^2+Br^3+Bmr^2)$ vs attention $O(Bn^2d)$; for $r\le16$, the cubic term is small
- Backward: $O(Bnr^2+Br^3+Bmr^2)$ with dense solve; $O(Bnr^2)$ with cached Cholesky/inverse factors
- Incremental update on a bounded-change input: $O(B\Delta r^2+Bmr^2)$ by Sherman-Morrison add/delete; worst-case refactor $O(Br^3)$

**Scout-scale falsification test:**
Attach Rank-1 Resolvent Pool to i193’s active piece tokens with $r=12,m=4$. Baseline: second-moment pool $\sum zz^\top$ plus equal-size MLP head. Train 6 epochs on 173k. Works if CRTK class-1 matched-recall FP drops by at least 10% with less than 10% inference slowdown. Fails if it only improves calibration or if Cholesky jitter dominates.

**Failure mode catalogue:**
- Hidden rebrand: if implemented as a plain dense `torch.linalg.solve`, it is only a layer composition; the primitive claim needs rank-1 update/delete semantics.
- Numerical instability: $S$ can be ill-conditioned; require $\lambda$, Cholesky jitter, or eigenvalue clipping.
- Speed risk: too-large $r$ makes $r^3$ and backward solves dominate, killing engine throughput.

**Status:** proposed

3. ### primitive_orbit_stabilized_canonicalizer

**Name:** Orbit-Stabilized Canonicalizer

**One-line claim:** Canonicalise a tensor under an exact finite symmetry group using input-selected orbit representatives and stabilizer averaging.

**Mathematical signature:**
Let finite group $G$ act by signed/permutation-linear maps $T_g$ on $X\in\mathbb{R}^{B\times n\times d}$. Let $\kappa:\mathbb{R}^{n\times d}\to\mathbb{R}^{m}$ be a deterministic additive key. For each batch item:
$$
A(X_b)=\arg\min_{g\in G}^{\operatorname{lex}}\kappa(T_gX_b),
$$
$$
\operatorname{OSCanon}(X_b)=\frac{1}{|A(X_b)|}\sum_{g\in A(X_b)}T_gX_b.
$$
At non-ties, the Jacobian is $T_{g^\*}$; at ties, the subgradient is the stabilizer average.

**Why this does not decompose into existing PyTorch ops:**
Group convolution performs fixed weight sharing or averaging over $G$; this primitive performs data-dependent orbit selection. G-CNNs established group convolution as a reusable symmetry primitive, but they do not canonicalise by argmin-orbit selection with stabilizer-aware gradients. This is also not MoE routing: the selected branches are exact group actions, not learned experts.

**Duplicate audit against existing primitive memory:**
Closest blocklisted family 1: **CRELU/color-involution graph messages and color-involution adjacency updates**. This sends no messages and builds no adjacency; it maps the whole latent tensor to a canonical orbit representative. Closest family 2: **piece-relabelling/involution gates**. There is no learned gate over piece labels; selection is deterministic finite-group quotienting with explicit tie handling.

**Chess-specific motivation:**
Chess has exact symmetries beyond D4: file mirror, color-swap plus board rotation, side-to-move canonicalisation, and same-type piece-index permutations in piece-token space. A scout-scale model should not spend 173k examples relearning symmetries that can be quotiented exactly. This is an operator-level alternative to data augmentation.

**Generalisation beyond chess:**
Applies to board games, molecules with automorphisms, symbolic states, and scene graphs with finite object-permutation symmetries. Less useful where symmetries are approximate rather than exact.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(B|G|(m+\Delta_\kappa d))$ with additive keys, vs $O(|G|)$ full data augmentation through the whole network
- Backward: $O(Bnd)$ for hard canonicalisation; $O(B|G|nd)$ for a soft training variant
- Incremental update on a bounded-change input: $O(B|G|\Delta d)$ if keys are maintained incrementally; otherwise $O(B|G|nd)$

**Scout-scale falsification test:**
Insert OSCanon immediately after the first latent projection in i193. Baseline: same network with no symmetry augmentation and a second baseline with random mirror/color augmentation outside the compute graph. Train 6 epochs on 173k. Works if it matches or beats augmentation on CRTK class-1 matched-recall FP while being faster at inference. Fails if canonical-boundary discontinuities increase false positives.

**Failure mode catalogue:**
- Hidden rebrand: reviewer may call it group pooling; that objection is valid if hard orbit selection and stabilizer averaging are removed.
- Numerical instability: near-ties can cause abrupt representation jumps; a soft warm-up may be needed.
- Speed risk: if $G$ is expanded beyond exact small chess symmetries, key evaluation dominates.

**Status:** proposed

4. ### primitive_tropical_distance_transform

**Name:** Learnable Tropical Distance Transform

**One-line claim:** Replace local convolution with a global min-plus influence field over board geometry.

**Mathematical signature:**
For lattice sites $S$, $N=|S|$, source costs $X\in\mathbb{R}^{B\times C\times N}$, and learnable nonnegative separable metric $D_\theta(i,j)$:
$$
Y_{bcj}=\min_{i\in S}\left[X_{bci}+D_\theta(i,j)\right].
$$
Smooth variant:
$$
Y^\tau_{bcj}=-\tau\log\sum_i\exp\left(-(X_{bci}+D_\theta(i,j))/\tau\right).
$$
Hard gradients route to argmin source cells; soft gradients are Gibbs weights. Signature: $f:\mathbb{R}^{B\times C\times N}\to\mathbb{R}^{B\times C\times N}$.

**Why this does not decompose into existing PyTorch ops:**
This is min-plus convolution over a learned metric, not multiply-add convolution or attention. A dense soft prototype resembles fixed log-sum-exp attention, but the primitive claim is the exact tropical distance-transform graph with argmin/semiring backward and $O(N)$ separable transform. Morphological and PDE-inspired neural layers exist, so the novelty is not “min-plus math”; the underexplored primitive is a reusable all-source neural distance transform with learned board metric.

**Duplicate audit against existing primitive memory:**
Closest blocklisted family 1: **ray-scan / directional scan / ray-parallel SSM operators**. This computes isotropic or learned-metric all-source distance fields, not directional rays, blockers, or scans. Closest family 2: **attack-ray sparse attention / ray-occlusion dispatch**. No QK scores, no ray mask, no legal-move connectivity, and no message passing.

**Chess-specific motivation:**
King danger, pawn shields, loose-piece pressure, and escape corridors are often distance-field phenomena, not local $3\times3$ texture phenomena. The primitive gives board-wide geometry bias without legal-move graph edges or expensive attention. It is especially plausible for near-puzzle FPs involving one critical escape square.

**Generalisation beyond chess:**
Useful in segmentation, robotics cost maps, game AI grids, image morphology, and sparse-event influence fields.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(BCN)$ for separable board metrics vs dense global attention/log-sum-exp $O(BCN^2)$
- Backward: $O(BCN)$, storing argmin indices or soft weights
- Incremental update on a bounded-change input: $O(BC(A\log N))$ for affected region size $A$; worst-case $O(BCN)$

**Scout-scale falsification test:**
Replace one $7\times7$ conv-style spatial mixer in i193 with TropDist over $8\times8$ latent maps, matched parameter count. Baseline: the original conv block. Train 6 epochs on 173k. Works if CRTK class-1 matched-recall FP drops by at least 8% and inference speed is no worse than baseline. Fails if gains appear only on easy negatives.

**Failure mode catalogue:**
- Hidden rebrand: the soft form can be dismissed as fixed attention; the hard $O(N)$ semiring kernel must be tested.
- Numerical instability: $\tau\to0$ can create sparse, brittle gradients.
- Speed risk: a Python dense $N^2$ prototype proves nothing; it needs a fused distance-transform kernel.

**Status:** proposed

5. ### primitive_capacitated_entropic_assignment

**Name:** Capacitated Entropic Assignment

**One-line claim:** Allocate limited latent resources between pieces and threats with differentiable capacity constraints.

**Mathematical signature:**
Given costs $C\in\mathbb{R}^{B\times m\times n}$, values $V\in\mathbb{R}^{B\times n\times d}$, row capacities $r\in\mathbb{R}_+^m$, column capacities $c\in\mathbb{R}_+^n$, and upper bounds $U\in\mathbb{R}_+^{m\times n}$:
$$
P^*=\arg\min_{0\le P\le U}\langle C,P\rangle+\tau\sum_{ij}(P_{ij}\log P_{ij}-P_{ij})
$$
subject to
$$
P\mathbf{1}\le r,\qquad P^\top\mathbf{1}\le c.
$$
Output:
$$
O=P^*V\in\mathbb{R}^{B\times m\times d}.
$$
For $\tau>0$, gradients follow from implicit differentiation of the KKT system.

**Why this does not decompose into existing PyTorch ops:**
Softmax attention normalises rows independently; it cannot enforce column budgets or upper bounds without becoming a constrained optimisation layer. OptNet already showed that differentiable optimisation layers can encode constraints beyond ordinary dense/conv layers, so the broad idea is not new. The primitive claim here is the specific reusable capacity-assignment operator for neural resource contention, not a full architecture.

**Duplicate audit against existing primitive memory:**
Closest blocklisted family 1: **Regret Saddlepoint / Witness-Counterwitness Quantifier primitives**. This is not a max-min proof or counterwitness operator; it is a primal feasible allocation with entropic smoothing. Closest family 2: **legal-move graph accumulators / legal-edge compilers**. Edges are dense learned costs between latent slots, not precomputed legal moves, and the computation is a constrained solve, not message accumulation.

**Chess-specific motivation:**
A single defender cannot answer every tactic. Near-puzzle false positives often come from counting every defender independently instead of modelling capacity and overloading. This primitive directly represents resource contention without PVs, node counts, Stockfish scores, or legal-move graph routing.

**Generalisation beyond chess:**
Resource allocation, tracking, multi-object assignment, routing with capacities, recommender slate construction, and scheduling.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(BTmn+Bmnd)$ for $T$ Sinkhorn/Newton iterations vs attention $O(Bmnd)$ without capacity constraints
- Backward: $O(BTmn)$ with an unrolled solver or $O(B(m+n)^3)$ for exact KKT solve
- Incremental update on a bounded-change input: warm-start $O(BT'(m+n))$ if one row/column changes; worst-case $O(BTmn)$

**Scout-scale falsification test:**
Use a tiny version on top of i193 piece tokens: $m,n\le16$, $\tau=0.05$, 5 solver iterations. Baseline: row-softmax cross-attention with the same value projection. Train 6 epochs on 173k. Works if class-1 matched-recall FP drops by at least 12% with less than 30% slowdown. Fails if it helps only after engine-scale data.

**Failure mode catalogue:**
- Hidden rebrand: if column capacities or upper bounds are disabled, it collapses toward attention/Sinkhorn-like normalisation.
- Numerical instability: small $\tau$ can make KKT solves ill-conditioned.
- Speed risk: exact solvers may be too slow for engine inference even if accuracy improves.

**Status:** proposed

## What I cut

- **Legal-move attention with a smarter mask** — still sparse/masked attention and directly duplicates legal-move graph routing.
- **Ray-occlusion distance scans** — duplicate of ray-scan, ray-blocked reducers, obstacle-pooling, and directional scan families.
- **Move-delta fastweight cache** — too close to sparse delta accumulators, reversible delta kernels, and sparse differential move kernels.
- **Piece-existence ANOVA/Hessian layer** — duplicate of signed piece-existence Hessian and pair-resonance operators.
- **Complex phase threat interference** — duplicate of complex-amplitude interference.
- **Differentiable alpha-beta/minimax backup** — overlaps with regret saddlepoint, witness-counterwitness quantifiers, and terminal-state primitives.
- **Promotion race fanout tensor** — duplicate of promotion-fanout counterfactual tensor operators.
- **Dynamic legal adjacency rank gate** — duplicate of dynamic adjacency rank-order gates and legal-edge compilers.
- **Persistence-critical pooling** — interesting for king-corridor topology, but differentiable topology layers already exist; the remaining chess version felt more like an application than a new primitive.
- **Choquet/submodular threat pooling** — useful but too easy to implement as sort-plus-weighted-sum, and too close to antichain/frontier-style set-function primitives.

## References used for novelty audit

- Albert Gu and Tri Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” 2023. <https://arxiv.org/abs/2312.00752>
- Maximilian Beck et al., “xLSTM: Extended Long Short-Term Memory,” 2024. <https://arxiv.org/abs/2405.04517>
- Ziming Liu et al., “KAN: Kolmogorov-Arnold Networks,” 2024. <https://arxiv.org/abs/2404.19756>
- Songlin Yang et al., “Parallelizing Linear Transformers with the Delta Rule over Sequence Length,” 2024. <https://arxiv.org/abs/2406.06484>
- Taco Cohen and Max Welling, “Group Equivariant Convolutional Networks,” ICML 2016. <https://proceedings.mlr.press/v48/cohenc16.html>
- David Ruhe, Johannes Brandstetter, and Patrick Forré, “Clifford Group Equivariant Neural Networks,” 2023. <https://arxiv.org/abs/2305.11141>
- Brandon Amos and J. Zico Kolter, “OptNet: Differentiable Optimization as a Layer in Neural Networks,” 2017. <https://arxiv.org/abs/1703.00443>
- Rickard Brüel-Gabrielsson et al., “A Topology Layer for Machine Learning,” AISTATS 2020. <https://proceedings.mlr.press/v108/gabrielsson20a.html>
