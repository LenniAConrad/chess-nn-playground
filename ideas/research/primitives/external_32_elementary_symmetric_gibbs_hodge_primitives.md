# New Neural-Network Primitives for Chess Evaluation

This file is a ranked proposal set for `chess-nn-playground`. The order below is the combined ranking. The axis ranking is: 1 = strongest.

Recent primitive calibration used: Mamba introduced input-conditioned selective state-space dynamics plus a hardware-aware recurrent/parallel algorithm ([Gu & Dao, 2023/2024](https://arxiv.org/abs/2312.00752)); xLSTM changed LSTM gating and memory structure ([Beck et al., 2024](https://arxiv.org/abs/2405.04517)); KAN replaced scalar weights with learnable edge functions ([Liu et al., 2024/2025](https://openreview.net/forum?id=Ozo7qJ5vZi)); DeltaNet work treats the delta-rule linear transformer as a hardware-aware sequence primitive ([Yang et al., 2024](https://arxiv.org/abs/2406.06484)).

| # | primitive | novelty plausibility | RTX 3070 demonstrability | inference-speed advantage | generalisation |
|---:|---|---:|---:|---:|---:|
| 1 | `primitive_elem_sym_event` | 2 | 1 | 1 | 2 |
| 2 | `primitive_gibbs_cut_partition` | 1 | 2 | 3 | 1 |
| 3 | `primitive_complementarity_contact` | 3 | 4 | 4 | 1 |
| 4 | `primitive_hodge_cochain_projector` | 4 | 2 | 2 | 1 |
| 5 | `primitive_signed_persistence` | 5 | 3 | 5 | 3 |

1. ### primitive_elem_sym_event

**Name:** Incremental Elementary-Symmetric Event Mixer

**One-line claim:** Computes exact k-way piece co-occurrence coefficients with O(k) updates when a bounded number of tokens changes.

**Mathematical signature:**
\(f_K:\mathbb{R}^{B\times n\times d}\rightarrow\mathbb{R}^{B\times K\times r}\). Let \(u_{b,i}=\phi(x_{b,i})\in\mathbb{R}^r\). Define \(E_{b,0}=\mathbf{1}\) and
\[
E_{b,k}=\sum_{1\le i_1<\cdots<i_k\le n}\prod_{t=1}^{k}u_{b,i_t},\quad k=1,\ldots,K
\]
with elementwise products. Forward recurrence:
\[
E_k^{(t)}=E_k^{(t-1)}+u_t\odot E_{k-1}^{(t-1)}.
\]
Gradient: \(\partial E_k/\partial u_i=E_{k-1}^{(-i)}\), the coefficient excluding token \(i\).

**Why this does not decompose into existing PyTorch ops:**
This is not `sum`, `mean`, attention, or a bilinear pair layer; it is coefficient extraction from a product-generating function with a deletion-based VJP. A reference can be unrolled from low-level ops, but the primitive computation graph is a scan over polynomial coefficients, not \(QK^\top\), convolution, or pooling. Exact \(k\)-way terms otherwise require enumerating \(O(n^k)\) subsets or accepting an approximation.

**Duplicate audit against existing primitive memory:**
Closest blocklisted family 1: Signed Piece-Existence Hessian / pair-resonance Hessian. This operator is a forward symmetric-polynomial transform, not a derivative over piece-existence variables. Closest family 2: delta pair selective bispectra. This has no ray, pair channel, or bispectral frequency coupling; the algebra is \(\prod_i(1+zu_i)\), not pairwise bilinear scoring.

**Chess-specific motivation:**
Many evaluation errors are co-presence errors: bishop pair, rook battery plus open file, defender plus escape square, pawn-chain plus king position. A bounded move changes only a few factors, so the update rule mirrors NNUE's accumulator advantage without being a standard accumulator.

**Generalisation beyond chess:**
Useful for set-valued sparse-event data: recommender baskets, molecule fragments, program-token feature sets, and dynamic scene objects.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BnKr)\) vs explicit k-tuple interaction \(O(Bn^Kr)\) or attention \(O(Bn^2d)\)
- Backward: \(O(BnKr)\)
- Incremental update on a bounded-change input: \(O(\Delta Kr)\) using remove/add polynomial-factor updates

**Scout-scale falsification test:**
Drop into i193 conv-only parent just before the scalar head. Use 64 square tokens, \(K=3,r=32\), append \(E_1,E_2,E_3\) to the normal pooled head. Baseline: same parameter-count MLP head. Works if CRTK class-1 matched-recall near-puzzle FP rate drops by at least 10% with ≤1.10× latency and no PR-AUC regression. Fails if only easy negatives improve.

**Failure mode catalogue:**
- Hidden rebrand risk: reviewer says it is just DeepSets plus products; answer depends on showing exact coefficient/VJP/incremental kernel.
- Numerical risk: products can explode; use log-domain or clamp \(u\) with centered `tanh`.
- Speed risk: \(K>3\) or large \(r\) becomes pointless; keep \(K\le3\).

**Status:** proposed

2. ### primitive_gibbs_cut_partition

**Name:** Gibbs Cut Log-Partition Operator

**One-line claim:** Turns latent edge costs into differentiable bottleneck-cut values and cut-edge marginals on the board grid.

**Mathematical signature:**
For fixed grid \(G=(V,E)\), \(c\in\mathbb{R}_{+}^{B\times E\times d}\), source penalties \(s\in\mathbb{R}^{B\times V\times d}\), and sink penalties \(t\in\mathbb{R}^{B\times V\times d}\):
\[
Z_{b,h}=\sum_{S\subseteq V}\exp\left(-\frac{
\sum_{(i,j)\in\delta S}c_{b,ij,h}+\sum_i s_{b,i,h}\mathbf{1}_{i\notin S}+\sum_i t_{b,i,h}\mathbf{1}_{i\in S}}{\tau}\right)
\]
\[
y_{b,h}=-\tau\log Z_{b,h},\quad m_{b,e,h}=\frac{\partial y_{b,h}}{\partial c_{b,e,h}}.
\]
\(m_e\) is the Gibbs probability that edge \(e\) crosses the cut.

**Why this does not decompose into existing PyTorch ops:**
This is a log-partition over \(2^{|V|}\) cuts with a custom transfer-matrix dynamic program and VJP, not softmax over tokens. Classical min-cut/max-flow returns one optimum cut; this returns differentiable cut marginals and entropy-smoothed bottleneck values. Cut-style differentiable learning exists, so the novelty claim is the neural log-partition/marginal primitive, not the graph-cut concept itself ([Xie et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12945480/)).

**Duplicate audit against existing primitive memory:**
Closest family 1: legal-move graph accumulators / SLG diffusion. This uses a fixed board grid and cut partition function; no legal move graph, no message passing, no diffusion recurrence. Closest family 2: Pareto antichain frontier. A cut subset is not a reply frontier or non-dominated witness set; the gradient is edge-crossing probability, not antichain selection.

**Chess-specific motivation:**
King safety and fortresses are bottleneck problems: one open corridor can invalidate a safe-looking shell. Near-puzzle false positives often differ by one square that opens or closes access; cut marginals expose those chokepoints directly.

**Generalisation beyond chess:**
Image segmentation, network reliability, robot navigation bottlenecks, circuit robustness, and traffic interdiction.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BdH4^W)\) on an \(H\times W\) grid via row transfer, vs attention \(O(Bd(HW)^2)\)
- Backward: \(O(BdH4^W)\)
- Incremental update on a bounded-change input: \(O(d4^W)\) for an affected row using cached prefix/suffix transfer vectors; for \(W=8\), this is board-constant

**Scout-scale falsification test:**
In i193, project latent square features to four edge-capacity channels, run this primitive on the 8×8 grid, append \(y\) and pooled \(m_e\) to the head. Baseline: same projection plus ordinary global pooling. Works if class-1 matched-recall FP rate drops ≥10% at ≤1.20× latency. Fails if latency exceeds 1.5× or gains vanish on near-puzzles.

**Failure mode catalogue:**
- Hidden rebrand risk: reviewer calls it CRF/Ising inference; distinction must be the cut-specific marginal primitive and incremental transfer kernel.
- Numerical risk: low \(\tau\) causes underflow; use log-space transfer.
- Speed risk: \(4^W\) is only attractive because chess width is 8; not engine-scale for wide grids.

**Status:** proposed

3. ### primitive_complementarity_contact

**Name:** Smooth Complementarity Contact Operator

**One-line claim:** Solves a smooth complementarity system so mutually exclusive tactical resources saturate instead of adding linearly.

**Mathematical signature:**
\(f:\mathbb{S}_{+}^{B\times m\times m}\times\mathbb{R}^{B\times m}\rightarrow\mathbb{R}^{B\times m}\times\mathbb{R}^{B\times m}\). Given \(M\succeq0\), \(q\), solve for \(z,w\):
\[
w=Mz+q,\quad z\ge0,\quad w\ge0,\quad z\odot w=0.
\]
Use smooth Fischer-Burmeister equations:
\[
F_\mu(z)=\sqrt{z^2+w^2+\mu^2}-z-w=0.
\]
Output \((z^\*,w^\*)\). Gradient:
\[
dz=-(\partial F_\mu/\partial z)^{-1}\left[(\partial F_\mu/\partial M)dM+(\partial F_\mu/\partial q)dq\right].
\]

**Why this does not decompose into existing PyTorch ops:**
This is not `ReLU`, gating, or MoE routing. The active set is chosen by a global complementarity solve, and the backward pass is an implicit KKT/Fischer-Burmeister solve, not backprop through fixed elementwise operations. Differentiable optimization layers already exist, so the claim is underexplored primitive-for-evaluation rather than first differentiable solver ([Amos & Kolter, 2017](https://arxiv.org/abs/1703.00443)).

**Duplicate audit against existing primitive memory:**
Closest family 1: Regret Saddlepoint / Witness-Counterwitness Quantifier. This is not a min-max search over replies; it is a complementarity feasibility relation \(z_iw_i=0\). Closest family 2: factor-graph/tensor-product legal-state primitives. No legal-state factors or move legality appear; \(M,q\) are latent resource contacts.

**Chess-specific motivation:**
Tactical resources saturate: one defender cannot fully answer two independent threats, and an overloaded piece creates a nonlinear cliff. Additive pooling often overcounts defenders; complementarity forces “either active pressure or slack, not both.”

**Generalisation beyond chess:**
Contact mechanics, market clearing, constrained resource allocation, robotic grasping, and traffic conflict resolution.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(Bm^3)\) Newton/linear solve vs MLP gate \(O(Bm^2)\)
- Backward: \(O(Bm^3)\)
- Incremental update on a bounded-change input: \(O(m^2\Delta)\) with warm-started factor updates if the active set is stable; otherwise \(O(m^3)\)

**Scout-scale falsification test:**
Use i193 latent head features to produce \(m=16\) contact variables and PSD \(M=LL^\top+\epsilon I\). Append \(z^\*,w^\*\) to the scalar head. Baseline: equal-parameter two-layer MLP. Works if class-1 near-puzzle FP rate drops ≥8% with ≤1.25× latency. Fails if the learned \(M\) collapses diagonal and behaves like ReLU.

**Failure mode catalogue:**
- Hidden rebrand risk: if \(M\) becomes diagonal, it degenerates to smoothed ReLU.
- Numerical risk: ill-conditioned \(M\) or tiny \(\mu\) causes unstable implicit gradients.
- Speed risk: \(m>32\) is unlikely to be engine-useful.

**Status:** proposed

4. ### primitive_hodge_cochain_projector

**Name:** Weighted Hodge Cochain Projector

**One-line claim:** Decomposes latent edge pressure into gradient, curl, and harmonic components on a chessboard cell complex.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times E\times d}\times\mathbb{R}_{+}^{E}\rightarrow\mathbb{R}^{B\times E\times 3d}\). Let \(B_1\in\{-1,0,1\}^{V\times E}\) be vertex-edge incidence and \(B_2\in\{-1,0,1\}^{E\times F}\) edge-face incidence. For edge cochain \(u\):
\[
\alpha=(B_1W^{-1}B_1^\top+\lambda I)^{-1}B_1u,\quad g=W^{-1}B_1^\top\alpha
\]
\[
\beta=(B_2^\top WB_2+\lambda I)^{-1}B_2^\top W(u-g),\quad c=B_2\beta,\quad h=u-g-c.
\]
Output \([g,c,h]\). Hodge/Laplacian methods are established in topological signal processing; the proposed primitive is the fixed-complex projector with cached VJP ([Isufi et al., 2024/2025](https://arxiv.org/html/2412.01576v1)).

**Why this does not decompose into existing PyTorch ops:**
Finite-step message passing or convolution cannot equal this exact orthogonal decomposition unless it represents the full Laplacian pseudoinverse spectrum. The primitive has edge-cochain input/output and an adjoint solve over incidence structure, not node-token attention or grid convolution.

**Duplicate audit against existing primitive memory:**
Closest family 1: SLG diffusion. This is not diffusion on a legal-state graph; it is a one-shot cochain decomposition into three orthogonal subspaces. Closest family 2: ray-blocked reducers / directional scans. No rays, blockers, or move lines are enumerated; the algebra is incidence-boundary projection.

**Chess-specific motivation:**
Pressure is not just scalar control. Direct attacks look gradient-like; blockade loops and fortress structures look curl-like; long corridors and global imbalance appear harmonic. These are exactly the distinctions conv pooling tends to blur.

**Generalisation beyond chess:**
Traffic-flow prediction, fluid fields, power grids, cellular-complex learning, and scene-flow decomposition.

**Complexity (forward, backward, incremental-update):**
- Forward: fixed \(W\) with cached factors \(O(BEd)\), vs \(T\)-step graph diffusion \(O(BTEd)\); input-dependent \(W\): \(O(Bd(V^3+F^3))\)
- Backward: same order using adjoint solves
- Incremental update on a bounded-change input: fixed \(W\), \(O(\Delta d)\); input-dependent \(W\), \(O(\Delta^2(V+F)d)\) via low-rank updates

**Scout-scale falsification test:**
In i193, form edge cochains from adjacent latent square differences, run the projector, and pool \([g,c,h]\) into the existing head. Baseline: equal-parameter depthwise 3×3 conv over the same latent maps. Works if near-puzzle FP rate drops ≥8% with ≤1.15× latency. Fails if only harmonic/global channels move and class-1 FP is unchanged.

**Failure mode catalogue:**
- Hidden rebrand risk: with only \(g\), it looks like a Laplacian smoothing layer.
- Numerical risk: \(\lambda\) too small makes harmonic components unstable.
- Speed risk: input-dependent weights destroy the cached-factor advantage.

**Status:** proposed

5. ### primitive_signed_persistence

**Name:** Signed Cubical Persistence Pool

**One-line claim:** Extracts differentiable connected-component and hole lifetimes from signed board-control fields.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times H\times W\times d}\rightarrow\mathbb{R}^{B\times d\times 2\times R\times K}\). For each scalar field \(u\), build the cubical complex on the \(H\times W\) grid. Compute sublevel persistence diagrams:
\[
D_r^+(u)=PH_r(u),\quad D_r^-(u)=PH_r(-u),\quad r\in\{0,1\}.
\]
Return top-\(K\) lifetimes:
\[
Y_{r,k}^{+}=d_{r,k}^{+}-b_{r,k}^{+},\quad Y_{r,k}^{-}=d_{r,k}^{-}-b_{r,k}^{-}.
\]
For unique cell values, \(\partial(d-b)/\partial u_{death}=1\), \(\partial(d-b)/\partial u_{birth}=-1\); ties use averaged subgradients.

**Why this does not decompose into existing PyTorch ops:**
This is not max-pooling, top-k pooling, or connected-component labeling. The forward pass performs boundary/union-find pairing between birth and death cells; the backward pass routes gradients through those paired cells. Differentiable persistent-homology layers already exist, so the novelty is the signed two-filtration chess-control primitive, not first PH differentiability ([Brüel-Gabrielsson et al., 2019/2020](https://arxiv.org/abs/1905.12200); [Papamarkou et al., 2024](https://arxiv.org/html/2402.08871v3)).

**Duplicate audit against existing primitive memory:**
Closest family 1: Terminal-State Detection primitives. This detects topology of latent fields, not legal terminal states. Closest family 2: Pareto frontier / witness-counterwitness quantifiers. Persistence pairs are birth-death events in a filtration, not game-theoretic witnesses or reply antichains.

**Chess-specific motivation:**
Pawn shields and king safety are topological: holes, connected safe regions, and broken corridors matter more than average control. A near-puzzle false positive can be caused by one tiny opening that changes a connected component or creates a hole.

**Generalisation beyond chess:**
Medical images, segmentation masks, occupancy maps, robotics traversability fields, and any grid-valued risk surface.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BdN\log N)\), \(N=HW\), vs pooling \(O(BdN)\)
- Backward: \(O(BdK)\) after persistence pairing
- Incremental update on a bounded-change input: robust dynamic PH is not reliably \(O(1)\); recompute \(O(N\log N)\), board-constant for 8×8

**Scout-scale falsification test:**
Add two latent control planes from i193, run signed persistence with \(R=2,K=4\), append lifetimes to the head. Baseline: same planes with max/top-k pooling. Works if class-1 matched-recall FP rate drops ≥8% with ≤1.25× latency. Fails if improvements appear only on easy negatives or latency exceeds 1.5×.

**Failure mode catalogue:**
- Hidden rebrand risk: if only \(H_0\) is used, reviewer may call it softened connected components.
- Numerical risk: many tied board values produce unstable birth-death assignments.
- Speed risk: PH kernels are awkward on GPU; small 8×8 size helps, but batching must be fused.

**Status:** proposed

## What I cut

1. **Piece-conditioned legal-move attention** — rejected as a direct duplicate of move-graph routers, legal graph accumulators, and sparse legal attention.

2. **Soft shortest-path / geodesic attacker field** — useful, but too close to differentiable shortest-path work and too easy to drift into legal/ray graph routing; see recent differentiable shortest-path work such as DataSP ([2024](https://arxiv.org/html/2405.04923v2)).

3. **Chess-group color-swap convolution** — rejected because it is mostly standard group equivariance plus the project’s existing color-involution gates; equivariant tensor operations are already a mature family ([recent survey context](https://arxiv.org/abs/2207.09453)).

4. **B-matching occupancy softmax** — rejected as a near-duplicate of factor-graph / tensor-product legal-state primitives.

5. **Weighted automaton over files/ranks** — rejected because it becomes a directional scan or ray-parallel SSM once applied to board lines; differentiable finite-state machinery is also already active research ([2023](https://openreview.net/pdf?id=k2hIQYqHTh)).

6. **Boolean Fourier / Reed-Muller piece-existence transform** — rejected as too close to the existing Hessian-over-piece-existence and pair-resonance families.

7. **DPP determinant pooling** — rejected because the core computation is largely `logdet(I+KKᵀ)` plus determinant marginals; interesting, but not chess-structural enough.

8. **Pure harmonic Green’s-function diffusion** — rejected because it looked like SLG diffusion under new vocabulary; the Hodge projector survived because it returns gradient/curl/harmonic subspaces, not a diffused field.

9. **Soft static-exchange stack** — rejected as a hidden minimax / witness-counterwitness primitive and likely too chess-only.
