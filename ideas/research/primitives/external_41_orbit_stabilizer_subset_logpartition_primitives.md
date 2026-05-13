# New Neural-Network Primitives for Chess Evaluation

Ranked order is the aggregate ranking across plausibility of novelty, RTX-3070 demonstrability, inference-speed advantage, and generalisation beyond chess.

Calibration prior work checked while setting the novelty bar: selective SSM / Mamba ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)), Mamba-2 / SSD ([Dao & Gu, 2024](https://arxiv.org/abs/2405.21060)), xLSTM ([Beck et al., 2024](https://arxiv.org/abs/2405.04517)), Test-Time Training layers ([Sun et al., 2024](https://arxiv.org/abs/2407.04620)), KAN ([Liu et al., 2024](https://arxiv.org/abs/2404.19756)), group-equivariant CNNs ([Cohen & Welling, 2016](https://arxiv.org/abs/1602.07576)), differentiable sorting ([Blondel et al., 2020](https://arxiv.org/abs/2002.08871)), SparseMAP ([Niculae et al., 2018](https://arxiv.org/abs/1802.04223)), and OptNet ([Amos & Kolter, 2017](https://arxiv.org/abs/1703.00443)).

| rank | primitive | novelty confidence | scout-scale demonstrability | speed upside | generalisation |
|---:|---|---:|---:|---:|---:|
| 1 | orbit_stabilizer_canonical | high | high | high | high |
| 2 | subset_logpartition | medium-high | high | high | high |
| 3 | exterior_wedge_pool | medium | high | medium-high | high |
| 4 | lovasz_chain_pool | medium | high | medium | high |
| 5 | pfaffian_conflict_pool | high | low-medium | low | medium |

1. ### primitive_orbit_stabilizer_canonical

**Name:** Orbit-Stabilizer Canonicalization Operator

**One-line claim:** Canonicalises latent features under chess symmetries with a stabilizer-aware backward pass.

**Mathematical signature:**
\(f_G:\mathbb{R}^{B\times n\times d}\rightarrow\mathbb{R}^{B\times n\times d}\). Let finite group \(G\) act by token permutations \(P_g\) and channel representations \(R_g\). With fixed canonical key \(h\),
\[
g_b^*=\arg\min_{g\in G}h(P_gX_bR_g^\top),\qquad Y_b=P_{g_b^*}X_bR_{g_b^*}^\top.
\]
For tie/stabilizer set \(S_b=\{g:h(P_gX_bR_g^\top)=h(P_{g_b^*}X_bR_{g_b^*}^\top)\}\),
\[
\frac{\partial L}{\partial X_b}=\frac{1}{|S_b|}\sum_{g\in S_b}P_g^\top\frac{\partial L}{\partial Y_b}R_g.
\]

**Why this does not decompose into existing PyTorch ops:**
Group convolutions average or convolve over all group elements; this chooses one canonical chart and defines a custom quotient gradient. `argmin`, `gather`, and `flip` can imitate the forward pass, but their default backward is not the stabilizer-averaged Jacobian above. The closest prior family is group-equivariant convolution, which reduces sample complexity by weight sharing but does not implement orbit quotienting ([Cohen & Welling, 2016](https://arxiv.org/abs/1602.07576)).

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: CRELU/color-involution graph messages and piece-relabelling/involution gates. This is not a learned gate, graph message, or channel involution; it is a hard finite-group quotient with specified backward on stabilizers. It also differs from dynamic adjacency rank-order gates because no input-dependent graph or message path is constructed.

**Chess-specific motivation:**
Chess has more symmetry than dihedral board symmetry: side-to-move, color swap, and piece-role involutions create repeated latent cases. At 173k positions, learning these redundancies wastes capacity. Canonicalising the latent board should reduce sample demand without introducing attention.

**Generalisation beyond chess:**
Applies to molecules, symbolic grids, CAD parts, finite-automorphism graphs, and board games with small exact symmetry groups.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|G|nd)\) vs test-time augmentation \(O(|G|\cdot\text{network})\)
- Backward: \(O(B|S|nd)\), usually \(O(Bnd)\)
- Incremental update on a bounded-change input: \(O(|G|d)\) to refresh affected canonical keys, or \(O(\log |G|)\) with cached ordered keys

**Scout-scale falsification test:**
Insert after the first 8×8 latent map in i193. Baseline: i193 trained with the same augmentation budget. Measure CRTK class-1 matched-recall false-positive rate and evals/sec. Works if equal-recall FP falls and inference is faster than test-time augmentation. Fails if canonical-key churn makes training noisy or gives no CRTK gain.

**Failure mode catalogue:**
- Hidden rebrand: if it is only flip/color augmentation, reject it; the stabilizer-aware quotient backward is the primitive.
- Numerical instability: canonical representatives may flip under tiny latent changes; use fixed low-dimensional keys and tie averaging.
- Speed risk: large \(|G|\) kills the benefit; chess use must keep \(|G|\) small.

**Status:** proposed

2. ### primitive_subset_logpartition

**Name:** Bounded Subset Log-Partition Transform

**One-line claim:** Computes exact low-order subset evidence without enumerating pairs, attention edges, or legal-move graphs.

**Mathematical signature:**
\(f_K:\mathbb{R}^{B\times n\times r}\rightarrow\mathbb{R}^{B\times (K+1)\times r}\). For log-weights \(A_{b,i,c}\), set \(C^{(0)}_{0,c}=0\), \(C^{(0)}_{k>0,c}=-\infty\), and
\[
C^{(i)}_{k,c}=\operatorname{logaddexp}\left(C^{(i-1)}_{k,c},\ C^{(i-1)}_{k-1,c}+A_{b,i,c}\right).
\]
Output
\[
Y_{b,k,c}=C^{(n)}_{k,c}=\log\sum_{|S|=k}\exp\sum_{i\in S}A_{b,i,c}.
\]
Gradient: \(\partial Y_{k,c}/\partial A_{i,c}=\Pr(i\in S\mid |S|=k,c)\), computed from forward/backward quotient polynomials.

**Why this does not decompose into existing PyTorch ops:**
This is a fused log-semiring elementary-symmetric-polynomial operator with a subset-marginal backward. A loop of `logaddexp` calls can emulate it, just as `conv2d` can be lowered to matrix operations, but that unrolled graph does not expose the quotient-polynomial backward or \(O(Kr)\) event update. It is not softmax, top-k, sparsemax, or attention.

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: sparse delta accumulators and factor/tensor-product legal-state primitives. It is not a bounded move-delta additive accumulator; its state is a truncated generating polynomial with multiplication/division updates. It is not tensor-product legal-state reasoning because it sums over all size-\(k\) subsets without legal graphs, rays, or explicit pair tensors.

**Chess-specific motivation:**
Tactical hard negatives often differ by whether two or three independent resources exist, not by the maximum single feature. This primitive represents exact “there are \(k\) supporting pieces” evidence while avoiding quadratic attention over all piece pairs. It directly targets near-puzzle false positives where additive evidence overcounts decoration.

**Generalisation beyond chess:**
Useful for sparse-event sequences, recommender slates, molecular fragments, scene-object sets, and any low-order set evidence problem.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BnKr)\) vs attention \(O(Bn^2d)\)
- Backward: \(O(BnKr)\)
- Incremental update on a bounded-change input: \(O(Kr)\) using polynomial factor removal/addition

**Scout-scale falsification test:**
Drop after the first feature projection in i193 with \(K=3,r=32\). Baseline: same i193 plus equal-parameter MLP. Measure matched-recall CRTK class-1 FP rate and evals/sec. Works if FP rate drops by at least 5% relative without more than 10% speed loss. Fails if only aggregate PR AUC improves.

**Failure mode catalogue:**
- Hidden rebrand: reviewer may call it dynamic programming; the primitive claim only holds with fused semiring scan plus subset-marginal backward.
- Numerical instability: log-domain recurrence can still underflow for very negative channels unless centered.
- Speed risk: if \(K>4\) or \(r\) is large, it becomes slower than a small MLP.

**Status:** proposed

3. ### primitive_exterior_wedge_pool

**Name:** Exterior Wedge Evidence Pool

**One-line claim:** Pools ordered evidence as antisymmetric \(k\)-blades, making redundant collinear motifs cancel instead of accumulate.

**Mathematical signature:**
For \(X\in\mathbb{R}^{B\times n\times d}\), choose degree \(k\le 3\). Output
\[
Y_b=\sum_{1\le i_1<\cdots<i_k\le n}X_{b,i_1}\wedge X_{b,i_2}\wedge\cdots\wedge X_{b,i_k}\in \Lambda^k\mathbb{R}^d,
\]
represented by \(\binom{d}{k}\) antisymmetric coordinates. For \(k=2\),
\[
Y_{b,pq}=\sum_{i<j}(X_{b,i,p}X_{b,j,q}-X_{b,i,q}X_{b,j,p}).
\]
Gradients are the corresponding contracted exterior products.

**Why this does not decompose into existing PyTorch ops:**
`einsum` plus antisymmetrisation can emulate small cases, but PyTorch has no exterior-algebra primitive with antisymmetric storage, contraction backward, and prefix-update semantics. The computation graph is not bilinear pooling: symmetric tensor features preserve redundancy, while wedge coordinates are alternating and vanish under linear dependence. Geometric-algebra neural work exists, so the claim is “new primitive for this stack,” not new algebra.

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: complex-amplitude interference and delta pair selective bispectra/bilinear hyperedges. This is real exterior algebra, not complex phase propagation, and it has no ray, blocker, or legal-edge semantics. It is also not a Hessian-over-piece-existence operator: no second derivative or counterfactual piece toggle is computed.

**Chess-specific motivation:**
Two attackers that express the same latent direction should not count like two independent threats. Wedge pooling makes linearly dependent tactical motifs cancel, while independent motifs create oriented area/volume evidence. That is a direct match to CRTK hard negatives where models overcount redundant attackers or defenders.

**Generalisation beyond chess:**
Useful for point clouds, molecules, multi-sensor fusion, retrieval diversity, and scene graphs where independence is more important than raw count.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(Bn\binom{d}{k}k)\) with prefix exterior sums vs attention \(O(Bn^2d)\)
- Backward: \(O(Bn\binom{d}{k}k)\)
- Incremental update on a bounded-change input: \(O(\binom{d}{k}k)\) for append/delete with cached prefix/suffix; \(O(n\binom{d}{k}k)\) for arbitrary reorder

**Scout-scale falsification test:**
Add a \(k=2,d=32\) wedge pool over piece/square latents in i193, then project back to 64 channels. Baseline: same parameter count bilinear pooling. Measure CRTK class-1 matched-recall FP and evals/sec. Works if wedge beats bilinear pooling on FP at equal recall without worse speed. Fails if antisymmetric channels collapse near zero.

**Failure mode catalogue:**
- Hidden rebrand: if implemented as ordinary outer-product pooling without antisymmetric storage/backward, reject it.
- Numerical instability: high-degree blades can explode or vanish; keep \(k\le3\) and normalise blade coordinates.
- Speed risk: \(\binom{d}{k}\) grows fast; scout test must keep \(d\) small.

**Status:** proposed

4. ### primitive_lovasz_chain_pool

**Name:** Submodular Lovász Chain Pool

**One-line claim:** Pools latent evidence through learned diminishing-returns set utility instead of additive, max, or attention pooling.

**Mathematical signature:**
\(f_F:\mathbb{R}^{B\times n\times c}\rightarrow\mathbb{R}^{B\times c}\). For each \((b,c)\), sort \(s_{\pi_1}\ge\cdots\ge s_{\pi_n}\), define \(A_r=\{\pi_1,\ldots,\pi_r\}\), and
\[
y_{b,c}=\sum_{r=1}^n s_{b,\pi_r,c}\left(F_c(A_r)-F_c(A_{r-1})\right).
\]
This is the Lovász/Choquet extension of set function \(F_c\). Subgradient:
\[
\partial y_{b,c}/\partial s_{b,\pi_r,c}=F_c(A_r)-F_c(A_{r-1}),
\]
with averaged subgradients for ties.

**Why this does not decompose into existing PyTorch ops:**
`max`, `mean`, `logsumexp`, and attention pool additive or convex-combination evidence. This operator’s gradient is the marginal gain along a sorted set chain, so one high-scoring item changes the gradient of later items. Lovász-extension losses and differentiable sorting are known prior work ([Berman et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Berman_The_Lovasz-Softmax_Loss_CVPR_2018_paper.pdf), [Blondel et al., 2020](https://arxiv.org/abs/2002.08871)); the honest claim is “underexplored primitive for chess,” not new mathematics.

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: Pareto antichain frontier and witness-counterwitness quantifier primitives. It is not nondomination-frontier enumeration and not existential witness logic; it computes one continuous extension of one submodular utility. It also differs from pair-resonance Hessian operators because interactions enter through marginal-gain ordering, not derivatives over piece existence.

**Chess-specific motivation:**
Extra attackers, defenders, space, and material often have diminishing returns. Near-puzzle false positives often arise when a model linearly over-adds redundant evidence. Lovász pooling hard-codes saturation while keeping gradients dense enough for scout-scale training.

**Generalisation beyond chess:**
Applies to summarisation, recommender slates, sensor fusion, active set selection, and any domain with substitutable evidence.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(Bcn\log n+Bcnq_F)\) vs attention pooling \(O(Bn^2d)\)
- Backward: \(O(Bcnq_F)\) after sorted chains are cached
- Incremental update on a bounded-change input: \(O(\log n)\) for cardinality/partition \(F\); \(O(n)\) worst-case for arbitrary \(F\)

**Scout-scale falsification test:**
Replace global average/max pooling in i193 with Lovász chain pooling over 64 square latents, using a small cardinality-plus-piece-class \(F\). Baseline: same model with logsumexp pooling. Works if CRTK class-1 matched-recall FP improves and calibration does not degrade. Fails if learned \(F\) becomes linear.

**Failure mode catalogue:**
- Hidden rebrand: if \(F(S)=\sum_iw_i\), it is just linear pooling; measure nonzero curvature.
- Numerical instability: sort ties create subgradient ambiguity; use tie-averaged backward.
- Speed risk: arbitrary learned \(F\) can dominate runtime; use small parametric submodular families first.

**Status:** proposed

5. ### primitive_pfaffian_conflict_pool

**Name:** Pfaffian Conflict Pool

**One-line claim:** Summarises signed perfect-pairing conflicts through a Pfaffian, not pairwise attention or bilinear pooling.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times c\times 2m\times 2m}_{\mathrm{skew}}\rightarrow\mathbb{R}^{B\times c}\). For skew-symmetric \(A=-A^\top\),
\[
y_{b,c}=\operatorname{pf}(A_{b,c})=\frac{1}{2^m m!}\sum_{\sigma\in S_{2m}}\operatorname{sgn}(\sigma)\prod_{j=1}^m A_{\sigma(2j-1),\sigma(2j)}.
\]
For nonsingular \(A\),
\[
d\operatorname{pf}(A)=\frac12\operatorname{pf}(A)\operatorname{tr}(A^{-1}dA),\qquad
\frac{\partial L}{\partial A}=\frac12\frac{\partial L}{\partial y}\operatorname{pf}(A)A^{-\top}.
\]

**Why this does not decompose into existing PyTorch ops:**
`torch.linalg.det` gives \(\det(A)=\operatorname{pf}(A)^2\), losing Pfaffian sign and branch-sensitive gradients. A determinant/logdet layer cannot recover the oriented perfect-matching sum without an additional sign convention. Pfaffians are standard mathematics and used in physics, but there is no general `torch.nn.PfaffianPool` primitive.

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: complex-amplitude interference and pair-resonance Hessian/bispectrum families. It is not complex amplitude propagation and has no phase channels. It is not a Hessian, bispectrum, ray-blocked operator, or derivative-over-piece-existence operator; it computes a signed perfect-matching partition function over an antisymmetric relation matrix.

**Chess-specific motivation:**
Tactical conflicts are often mutually exclusive pairings: one defender can answer one threat, but not two; one exchange pairing blocks another. A Pfaffian encodes parity-sensitive global pairing structure in one scalar or channel vector. It is likely too expensive as a default engine primitive, but useful as a high-risk ablation.

**Generalisation beyond chess:**
Relevant to matching problems, fermionic physics, chemistry, structured conflict modelling, and graph pairing summaries.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(Bcm^3)\) via skew elimination vs bilinear pair pooling \(O(Bcm^2)\)
- Backward: \(O(Bcm^3)\)
- Incremental update on a bounded-change input: \(O(cm^2)\) for rank-2 row/column updates; otherwise recompute

**Scout-scale falsification test:**
Use only over the active piece set, capped at \(2m\le32\), in a small auxiliary branch of i243. Baseline: same branch with logdet/DPP-style pooling and bilinear pooling. Works if CRTK class-1 matched-recall FP improves despite lower evals/sec. Fails if speed hit exceeds 25% or Pfaffian values collapse near zero.

**Failure mode catalogue:**
- Hidden rebrand: if sign is unused, it degenerates into determinant/logdet diversity pooling.
- Numerical instability: near-singular skew matrices cause exploding \(A^{-1}\); regularise with \(A+\epsilon J\) and monitor condition number.
- Speed risk: cubic cost is ugly for engine inference; cap piece count and channels.

**Status:** proposed

## What I cut

1. **Legal-move attention with a better mask** — rejected as standard masked attention and duplicate of legal-move graph routers/sparse legal graph transitions.
2. **Ray-prefix SSM / directional scan** — rejected as duplicate of ray-parallel SSMs, directional scans, and ray-scan operators.
3. **Move-delta latent cache** — rejected as duplicate of sparse delta accumulators and incremental latent accumulators.
4. **Chess-group convolution** — rejected because group convolution is established prior work; the proposed survivor is quotient canonicalisation, not group convolution.
5. **Matroid sparsemax projection** — rejected after self-audit as too close to SparseMAP / differentiable structured inference ([Niculae et al., 2018](https://arxiv.org/abs/1802.04223)).
6. **Soft threat-refutation minimax pooling** — rejected as too close to regret saddlepoint and witness-counterwitness quantifier primitives.
7. **Promotion counterfactual fanout** — rejected as directly blocklisted.
8. **DPP/logdet diversity pooling** — rejected because determinant/logdet pooling is standard linear algebra and weaker than Pfaffian sign-sensitive pairing.
9. **Sinkhorn role assignment** — rejected as existing optimal-transport matching, closer to alignment than a new primitive.
10. **KAN-style spline edge activation for squares/pieces** — rejected as imported existing primitive, not chess-invented ([Liu et al., 2024](https://arxiv.org/abs/2404.19756)).
11. **TTT-style fastweight evaluator state** — rejected as existing test-time-training primitive and likely too slow/unstable for per-node chess-engine inference ([Sun et al., 2024](https://arxiv.org/abs/2407.04620)).
