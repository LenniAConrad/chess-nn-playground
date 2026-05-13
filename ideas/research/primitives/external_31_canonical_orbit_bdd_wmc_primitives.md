# New Neural Primitives for Chess Evaluation

Calibration used: I treated recent primitive-level work such as selective SSMs/Mamba, Mamba-2/SSD, xLSTM, TTT layers, KAN, GLinSAT, and dQP as the novelty bar: each changes an operator, state update, constraint layer, or gradient path rather than merely composing existing layers.[^mamba][^mamba2][^xlstm][^ttt][^kan][^glinsat][^dqp]

1. ### primitive_canonical_orbit_st

**Name:** Canonical-Orbit Straight-Through Operator

**One-line claim:** Canonicalise a tensor under a finite chess group, backpropagating only through the selected orbit representative.

**Mathematical signature:**
\(f_G:\mathbb{R}^{B\times n\times d}\rightarrow \mathbb{R}^{B\times n\times d}\times G^B\). For finite group \(G\) with permutation actions \(P_g\), define
\[
g_b^\*=\arg\min_{g\in G}\kappa(P_gX_b),\qquad Y_b=P_{g_b^\*}X_b .
\]
\(\kappa\) is a fixed lexicographic or hash key, not learned. Backward:
\[
\frac{\partial L}{\partial X_b}=P_{g_b^\*}^{-1}\frac{\partial L}{\partial Y_b},
\]
with uniform subgradient over exact ties.

**Why this does not decompose into existing PyTorch ops:**
This is not group convolution: group-equivariant CNNs convolve or average over transformed copies, whereas this takes a hard quotient representative and exposes the inverse action in backward.[^gcnn] A Python emulation would materialise all \(|G|\) copies, key them, select one, and rely on ad-hoc tie handling. The nearest literature is learned canonicalisation, but this proposal uses a fixed chess group action and a straight-through representative selector rather than a learned pose network.[^canonical]

**Duplicate audit against existing primitive memory:**
Closest blocklist family 1: color-involution adjacency updates. This primitive has no adjacency matrix, no message passing, and no learned color gate; it applies a fixed group quotient \(X\mapsto P_{g^\*}X\). Closest blocklist family 2: piece-relabelling/involution gates. Those mix or route channels; this selects one canonical orbit element and preserves tensor shape.

**Chess-specific motivation:**
Chess evaluation is antisymmetric under color swap and partly symmetric under board flips once side-to-move and king perspective are handled. Small-data scouts waste capacity relearning equivalent positions. This primitive removes symmetry multiplicity without an \(|G|\)-way test-time ensemble.

**Generalisation beyond chess:**
Useful for molecules, program states, games, and dynamic graphs where finite symmetries exist but full group convolution is too expensive.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|G|nd)\) keying vs group convolution \(O(B|G|ndk)\)
- Backward: \(O(Bnd)\)
- Incremental update on a bounded-change input: \(O(|G|\Delta d)\) to update keys; \(O(nd)\) only if the canonical representative changes

**Scout-scale falsification test:**
Insert before the first hidden tensor in i193, using only legal board symmetries plus color swap. Baseline: same i193 with ordinary augmentation. Train 173k positions for 6 epochs or fine-tune 2 epochs from an existing scout checkpoint. Primitive works if CRTK class-1 matched-recall near-puzzle FP rate drops by at least 5% with ≤5% eval latency penalty. It fails if gains appear only in aggregate PR AUC.

**Failure mode catalogue:**
- Hidden rebrand: reviewer says this is just data canonicalisation, not a primitive.
- Numerical instability: continuous hidden tensors may create unstable hash ties unless \(\kappa\) is restricted to discrete or quantised channels.
- Speed failure: if \(g^\*\) changes often inside hidden layers, full tensor permutation can dominate.

**Status:** proposed

2. ### primitive_bdd_wmc

**Name:** BDD Weighted-Model-Count Layer

**One-line claim:** Evaluate exact soft Boolean constraint circuits and return log-counts plus marginals as neural features.

**Mathematical signature:**
Given logits \(\ell\in\mathbb{R}^{B\times m}\), \(p=\sigma(\ell)\), and a fixed reduced ordered BDD \(D=(V,E_0,E_1)\), compute
\[
Z_\bot=0,\quad Z_\top=1,\quad Z_v=(1-p_{j(v)})Z_{E_0(v)}+p_{j(v)}Z_{E_1(v)} .
\]
Output:
\[
f_D(\ell)=\left[\log(Z_{\mathrm{root}}+\epsilon),\ \nabla_{\ell}\log(Z_{\mathrm{root}}+\epsilon)\right]\in\mathbb{R}^{B\times(1+m)} .
\]

**Why this does not decompose into existing PyTorch ops:**
This is not an MLP over Boolean features and not soft logic gates stacked as layers. The primitive is a canonical shared-subproblem circuit evaluator whose backward pass returns exact weighted-model-count marginals. BDDs are known to make weighted model counting linear in BDD size; PyTorch has no reusable neural operator for “evaluate this reduced Boolean circuit and expose marginals.”[^wmc]

**Duplicate audit against existing primitive memory:**
Closest blocklist family 1: terminal-state detection primitives. This does not detect mate/draw/terminal states; it evaluates arbitrary differentiable Boolean templates. Closest blocklist family 2: factor-graph/tensor-product legal-state primitives. A BDD is a Shannon-decomposition circuit, not a legal-move graph, sparse transition, or message-passing factor graph.

**Chess-specific motivation:**
Near-puzzle false positives often come from “looks like a tactic, but one Boolean precondition is false”: an escape square exists, a defender is not overloaded, or a pinned piece can still recapture. A BDD-WMC primitive can score conjunctions/disjunctions of learned board predicates exactly, without Stockfish scores, PVs, node counts, or verification metadata as inputs.

**Generalisation beyond chess:**
Neuro-symbolic perception, program verification, constrained recommendation, SAT-guided control.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|V|)\) vs dense MLP over all clauses \(O(Bmd)\)
- Backward: \(O(B|V|)\)
- Incremental update on a bounded-change input: \(O(|\mathrm{Anc}(\Delta)|)\) with cached reverse edges

**Scout-scale falsification test:**
In i193, replace one 64-unit tactical MLP head with BDD-WMC over ≤96 learned Bernoulli predicates and ≤2k BDD nodes. Baseline: same parameter-count MLP. Train frozen trunk plus head for under 2 GPU-hours. Works if CRTK class-1 matched-recall FP rate drops ≥3% and node eval latency stays ≤1.1× baseline.

**Failure mode catalogue:**
- Hidden rebrand: reviewer says it is just a hand-coded feature template.
- Numerical instability: \(Z_{\mathrm{root}}\) can underflow; use log-domain semiring for deeper BDDs.
- Speed failure: badly ordered BDD variables can explode \(|V|\).

**Status:** proposed

3. ### primitive_matroid_rank_envelope

**Name:** Matroid-Rank Envelope Pooling

**One-line claim:** Pool tactical candidates by the best independent set, not by top-k or soft attention.

**Mathematical signature:**
Input candidate features \(X\in\mathbb{R}^{B\times m\times d}\), scores \(s\in\mathbb{R}^{B\times m}\), and a fixed matroid \(\mathcal{M}=(E,\mathcal{I})\) with rank \(r\). For each batch, sort \(s_{\pi_1}\ge\dots\ge s_{\pi_m}\), define \(S_j=\{\pi_1,\dots,\pi_j\}\), and
\[
\alpha_j=r(S_j)-r(S_{j-1})\in\{0,1\}.
\]
Output
\[
y=\sum_{j=1}^m \alpha_j X_{\pi_j}\in\mathbb{R}^{B\times d},\qquad F(s)=\sum_{j=1}^m \alpha_j s_{\pi_j}.
\]
Gradients are piecewise constant away from score ties.

**Why this does not decompose into existing PyTorch ops:**
Top-k pooling is the uniform-matroid special case. This primitive exposes a rank-oracle-dependent greedy basis and the Lovász-style piecewise-linear envelope of a submodular rank function. Lovász extensions have been used in neural losses, but this proposal uses matroid rank as an inference-time pooling operator.[^lovasz]

**Duplicate audit against existing primitive memory:**
Closest blocklist family 1: Pareto antichain frontier. Pareto filtering keeps nondominated points under a partial order; matroid pooling enforces exchange-axiom independence and returns a greedy basis. Closest blocklist family 2: legal-move graph accumulators. Items need not be moves, and there is no graph message flow; only an independence oracle \(r(S)\).

**Chess-specific motivation:**
Tactical evidence is often redundant: three attackers on the same pinned defender should not count like three independent threats. Matroid constraints can encode “at most one per target square,” “at most one per line,” or “one forcing motif per king escape square,” then pool the strongest independent evidence.

**Generalisation beyond chess:**
Diverse retrieval, sensor selection, scheduling, sparse-event summarisation.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B(m\log m+mT_r+d|\mathrm{basis}|))\) vs attention pooling \(O(Bm^2d)\)
- Backward: \(O(B(m+d|\mathrm{basis}|))\)
- Incremental update on a bounded-change input: \(O(\Delta\log m+\Delta T_r)\) if sorted order and greedy certificate are cached

**Scout-scale falsification test:**
Use i193 to emit 64 square-level tactical candidates; replace max/top-k pooling with this primitive. Baseline: top-k pooling with the same downstream head. Works if class-1 matched-recall FP rate drops ≥4% and latency improves or stays flat. Fails if only easy-negative PR AUC improves.

**Failure mode catalogue:**
- Hidden rebrand: if the chosen matroid is uniform, it collapses to top-k.
- Numerical instability: score ties create nondeterministic bases unless tie-breaking is fixed.
- Speed failure: a slow custom independence oracle erases the \(m\log m\) advantage.

**Status:** proposed

4. ### primitive_tactical_lcp_projector

**Name:** Tactical Complementarity Projector

**One-line claim:** Turn mutually exclusive tactical claims into a coupled ReLU-like projection with active-set gradients.

**Mathematical signature:**
Input \(q\in\mathbb{R}^{B\times m}\), \(M\in\mathbb{S}_{++}^{B\times m\times m}\). Output \(z\in\mathbb{R}_+^{B\times m}\) solves
\[
z=\arg\min_{u\ge0}\ \frac12u^\top Mu+q^\top u .
\]
KKT form:
\[
z\ge0,\quad w=Mz+q\ge0,\quad z\odot w=0 .
\]
For active set \(A=\{i:z_i>0\}\), local gradient uses
\[
dz_A=-M_{AA}^{-1}dq_A,\qquad dz_{\bar A}=0
\]
plus the standard \(dM\) term.

**Why this does not decompose into existing PyTorch ops:**
ReLU is the diagonal case \(M=\mathrm{diag}(m_i)\); this primitive couples coordinates through an active-set inverse. Softmax gates normalise competing scores but do not enforce complementarity \(z_iw_i=0\). Differentiable QP/LCP-style layers exist in optimisation literature, so the honest novelty claim is “underexplored primitive for chess evaluation,” not first-ever differentiable optimisation.[^diffopt][^dqp]

**Duplicate audit against existing primitive memory:**
Closest blocklist family 1: regret saddlepoint and witness-counterwitness quantifier primitives. LCP is neither minimax regret nor existential quantification; it is a convex KKT projection. Closest blocklist family 2: legal-move graph transitions. \(M\) is a coupling matrix over claims, not a legal edge set or sparse move router.

**Chess-specific motivation:**
Many chess facts are complementary: a square is usable or refuted; a defender is free or overloaded; a line is open or blocked. The primitive can suppress mutually inconsistent tactical evidence rather than letting an MLP learn that exclusion from data.

**Generalisation beyond chess:**
Contact physics, traffic equilibrium, resource allocation, constrained control.

**Complexity (forward, backward, incremental-update):**
- Forward: worst \(O(Bm^3)\), sparse active-set \(O(Ba^3)\), vs ReLU \(O(Bm)\)
- Backward: \(O(Ba^3)\) for active-set inverse reuse
- Incremental update on a bounded-change input: \(O(a^2)\) if active set unchanged; refactor \(O(a^3)\) otherwise

**Scout-scale falsification test:**
Replace a 32-dimensional ReLU/softplus tactical gate in i193 with \(m=32\) LCP projector, \(M=LL^\top+\epsilon I\). Baseline: same head with ReLU and equal parameters. Works if class-1 matched-recall FP rate drops ≥5% with latency ≤1.2×. Fails if active-set churn dominates runtime.

**Failure mode catalogue:**
- Hidden rebrand: if \(M\) learns near-diagonal, it is just ReLU.
- Numerical instability: nearly singular \(M_{AA}\) causes gradient spikes.
- Speed failure: active-set changes every node, preventing cached solves.

**Status:** proposed

5. ### primitive_delta_cholesky_whiten

**Name:** Bounded-Delta Cholesky Whitening Accumulator

**One-line claim:** Maintain a whitened sparse-event accumulator by rank-k Cholesky updates instead of dense recomputation.

**Mathematical signature:**
Sparse active events \(U,V\in\mathbb{R}^{B\times M\times d}\), weights \(a\in\mathbb{R}_+^{B\times M}\). Define
\[
C=\lambda I+\sum_{i=1}^M a_iU_iU_i^\top,\qquad t=\sum_{i=1}^M a_iV_i,\qquad L=\operatorname{chol}(C).
\]
Output
\[
r=L^{-1}t\in\mathbb{R}^{B\times d}.
\]
On event changes, update \(C\) by signed rank-one Cholesky update/downdate and \(t\) additively.

**Why this does not decompose into existing PyTorch ops:**
Dense whitening can be written with covariance plus Cholesky solve, and whitening layers such as DBN/IterNorm already exist.[^iternorm] The new primitive-level part is the state transition: a bounded sparse edit changes the SPD factor by rank-\(k\) update/downdate, with backward through the triangular update path rather than through a dense recomputed covariance.

**Duplicate audit against existing primitive memory:**
Closest blocklist family 1: sparse delta accumulators / incremental latent accumulators. Those maintain additive vectors; this maintains an SPD factor and outputs an inverse-whitened accumulator \(L^{-1}t\). Closest blocklist family 2: signed edit bilinear memory. This does not emit pairwise bilinear interactions; second-order statistics only condition the whitening geometry.

**Chess-specific motivation:**
A legal move changes a bounded number of sparse events: source, destination, capture, promotion. HalfKA’s power is O(1)-style update; this primitive chases the same property for correlated piece-square features, where raw additive sums overcount motifs such as overloaded defenders or doubled attackers.

**Generalisation beyond chess:**
Streaming recommendation, online anomaly detection, dynamic graphs with sparse edits.

**Complexity (forward, backward, incremental-update):**
- Forward: dense \(O(B(Md^2+d^3))\) vs LayerNorm \(O(BMd)\)
- Backward: dense \(O(B(Md^2+d^3))\), rank-update adjoint \(O(Bkd^2)\)
- Incremental update on a bounded-change input: \(O(kd^2+d^2)\) for \(k\) changed events

**Scout-scale falsification test:**
Drop into i243 HalfKA+dual-stream as the accumulator normaliser, \(d\le32\). Baseline: same model with LayerNorm or unwhitened HalfKA sum. Freeze trunk and train the head for 2–4 epochs. Works if class-1 matched-recall FP rate drops ≥3% and eval latency is not worse than 1.15×. Fails if covariance conditioning gives only training-loss gains.

**Failure mode catalogue:**
- Hidden rebrand: reviewer says it is just DBN/IterNorm on sparse features.
- Numerical instability: Cholesky downdates can fail unless \(\lambda I\) and jitter are conservative.
- Speed failure: full-output refresh destroys bounded-delta advantage if downstream cannot consume \(r\) incrementally.

**Status:** proposed

## What I cut

1. **Attack-ray min-cut flow layer** — too close to legal-move graph routing, attack-ray sparse attention, and graph diffusion.
2. **Persistent cage homology pooling** — attractive for king-net topology, but differentiable persistent-homology layers/extensions already exist, so this is an imported primitive rather than a new one.[^torchph]
3. **Sinkhorn capture-assignment layer** — mostly optimal transport/Sinkhorn with chess vocabulary; not a new primitive.
4. **Piece-existence Shapley marginaliser** — collapses into the existing Hessian/counterfactual piece-existence family.
5. **Ray non-backtracking SSM** — a renamed legal/ray state-space scan; duplicate of ray-parallel SSM and legal graph transition families.
6. **KAN-style piece-square spline layer** — KAN is real prior work, but using spline edge functions for chess is a composition/import, not a new primitive.[^kan]
7. **TTT-per-position inner learner** — TTT layers are real primitive-level work, but for 173k-position scout scale this likely becomes data-hungry and training-trick-adjacent.[^ttt]
8. **Group Reynolds averaging layer** — too close to standard group-equivariant convolution; hard canonical orbit selection has a different computation graph.

## References

[^mamba]: Albert Gu and Tri Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” arXiv:2312.00752, 2023. https://arxiv.org/abs/2312.00752
[^mamba2]: Tri Dao and Albert Gu, “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality,” arXiv:2405.21060, 2024. https://arxiv.org/abs/2405.21060
[^xlstm]: Maximilian Beck et al., “xLSTM: Extended Long Short-Term Memory,” arXiv:2405.04517, 2024. https://arxiv.org/abs/2405.04517
[^ttt]: Yu Sun et al., “Learning to (Learn at Test Time): RNNs with Expressive Hidden States,” arXiv:2407.04620, 2024. https://arxiv.org/abs/2407.04620
[^kan]: Ziming Liu et al., “KAN: Kolmogorov-Arnold Networks,” arXiv:2404.19756, 2024. https://arxiv.org/abs/2404.19756
[^glinsat]: Hongtai Zeng et al., “GLinSAT: The General Linear Satisfiability Neural Network Layer By Accelerated Gradient Descent,” arXiv:2409.17500, 2024. https://arxiv.org/abs/2409.17500
[^dqp]: Connor W. Magoon et al., “Differentiation Through Black-Box Quadratic Programming Solvers,” arXiv:2410.06324, 2024. https://arxiv.org/abs/2410.06324
[^gcnn]: Taco Cohen and Max Welling, “Group Equivariant Convolutional Networks,” arXiv:1602.07576, 2016. https://arxiv.org/abs/1602.07576
[^canonical]: S. O. Kaba et al., “Equivariance with Learned Canonicalization Functions,” ICML 2023 / arXiv:2211.06489. https://arxiv.org/abs/2211.06489
[^wmc]: Adnan Darwiche, “A Differential Approach to Inference in Bayesian Networks,” Journal of the ACM, 2003; see also weighted model counting / arithmetic circuit literature. https://dl.acm.org/doi/10.1145/775152.775154
[^lovasz]: Maxim Berman, Amal Rannen Triki, and Matthew B. Blaschko, “The Lovász-Softmax Loss,” CVPR 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html
[^diffopt]: Akshay Agrawal et al., “Differentiable Convex Optimization Layers,” NeurIPS 2019. https://papers.neurips.cc/paper_files/paper/2019/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html
[^iternorm]: Lei Huang et al., “Iterative Normalization: Beyond Standardization towards Efficient Whitening,” CVPR 2019. https://arxiv.org/abs/1904.03441
[^torchph]: Christoph Hofer et al., `torchph`, a PyTorch package for differentiable persistent homology. https://github.com/c-hofer/torchph
