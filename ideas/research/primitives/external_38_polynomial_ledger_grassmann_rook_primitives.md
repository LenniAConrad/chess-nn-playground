# New Neural Primitives for Chess Evaluation

## Literature calibration and ranking

I used recent primitive-level work only as calibration, not as proposal templates: Mamba/S6 introduced input-selective state-space dynamics and hardware-aware recurrent scans; Mamba-2/SSD sharpened the state-space/attention duality and reported faster core layers; xLSTM changed the LSTM memory/update algebra with exponential gates and scalar/matrix memories; KAN moved learnable nonlinearities onto edges; DeltaNet revived delta-rule linear attention with hardware-efficient parallelization. Sources: [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060), [xLSTM](https://arxiv.org/abs/2405.04517), [KAN](https://arxiv.org/abs/2404.19756), [DeltaNet](https://arxiv.org/abs/2406.06484).

Projection-style proposals below are scoped conservatively because differentiable optimization layers and differentiable sorting/ranking are established prior art. Sources: [OptNet](https://arxiv.org/abs/1703.00443), [Fast Differentiable Sorting and Ranking](https://proceedings.mlr.press/v119/blondel20a.html). Symmetry-related claims are also scoped against group-equivariant CNNs and 2024 canonicalization/frame-averaging work. Sources: [Group Equivariant CNNs](https://arxiv.org/abs/1602.07576), [A Canonicalization Perspective on Invariant and Equivariant Learning](https://arxiv.org/abs/2405.18378).

| Overall rank | Primitive | Plausibility of novelty | RTX 3070 demonstrability | Potential inference-speed advantage | Generalisation beyond chess |
|---:|---|---:|---:|---:|---:|
| 1 | Truncated Occupancy Polynomial Ledger | 2 | 1 | 1 | 2 |
| 2 | Grassmann Rook-Matching Pool | 1 | 4 | 3 | 3 |
| 3 | Matroid Sparsemax Pool | 3 | 3 | 2 | 1 |
| 4 | Chess-Group IrrepNorm | 4 | 2 | 4 | 4 |
| 5 | Poset ConeNorm | 5 | 2 | 3 | 2 |

Lower rank numbers are better. The list below is ordered by combined practicality for `chess-nn-playground`, not by pure mathematical novelty.

## Proposals

### primitive_polynomial_ledger

**Name:** Truncated Occupancy Polynomial Ledger

**One-line claim:** A fused elementary-symmetric polynomial operator that scores k-piece motifs without enumerating k-tuples.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times n\times d}\rightarrow\mathbb{R}^{B\times K\times d}\). Let \(z_i=\tanh(Wx_i+b)\in\mathbb{R}^{d}\). Initialize \(e_0=\mathbf{1}_d\) and \(e_{1:K}=0\). For \(i=1..n\), update descending:
\[
e_k\leftarrow e_k+z_i\odot e_{k-1},\quad k=K,\ldots,1.
\]
Return \(Y=[e_1,\ldots,e_K]\). The backward pass is well-defined by leave-one-out coefficients: \(\partial e_k/\partial z_i=e^{(-i)}_{k-1}\), where \(e^{(-i)}\) is the same polynomial ledger with item \(i\) removed.

**Why this does not decompose into existing PyTorch ops:**
This is not sum pooling, bilinear pooling, attention, or a polynomial activation. It is a triangular scan in the truncated polynomial ring \(\mathbb{R}[t]/t^{K+1}\), with a fused inverse/delete update and a leave-one-out adjoint. A Python loop of multiply/add nodes could emulate the value, but no existing `torch.nn` primitive exposes this state algebra, backward graph, or \(O(Kd)\) edit update.

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: sparse delta accumulators and delta pair selective bispectra. It is not a plain additive accumulator under bounded move deltas; its state is multiplicative elementary-symmetric structure \((e_0,\ldots,e_K)\), and deletion uses formal series inversion rather than subtracting a moved feature. It is not a pair/bispectrum primitive because it returns all degrees \(1..K\) and has no ray, legal-edge, Hessian, or semiring connectivity.

**Chess-specific motivation:**
Chess evaluation has compact multi-piece motifs: bishop-pair plus pawn color complex, rook+queen battery, knight+queen mating geometry, and material configurations where the third piece changes the sign of a pairwise feature. Small scouts often miss these interactions because enumerating tuple features is data-hungry. This primitive gives controlled k-piece interaction capacity while preserving NNUE-like bounded-change updates.

**Generalisation beyond chess:**
Useful for molecule fragments, recommender baskets, sparse event multisets, and scene-object sets where low-degree combinations matter more than token order.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BnKd)\) vs explicit k-tuple pooling \(O(Bn^K d)\) or dense attention \(O(Bn^2d)\)
- Backward: \(O(BnKd)\)
- Incremental update on a bounded-change input: \(O(Kd)\) per inserted/deleted piece feature using truncated inverse updates

**Scout-scale falsification test:**
Drop \(K=3,d=32\) after the i193 conv-only trunk's board-to-token pooling. Baseline: same parameter count with mean/sum pooling and a small MLP. Measure CRTK class-1 matched-recall near-puzzle FP rate, not just aggregate PR AUC. The primitive works if FP rate drops at least 5% at fixed recall with less than 15% latency hit; it fails if only easy-negative PR AUC improves.

**Failure mode catalogue:**
- Hidden rebrand objection: “This is just Deep Sets or a polynomial kernel.” Reject it if \(K=1\), if tuples are materialized, or if the delete/update law is not used.
- Numerical instability: degree-\(K\) products can explode or vanish; keep \(K\le3\), bound \(z_i\), and optionally normalize coefficients by \({n\choose k}\).
- Speed risk: a non-fused Python loop will be too slow; the actual primitive needs a custom CUDA/Triton scan.

**Status:** proposed

### primitive_grassmann_rook_pool

**Name:** Grassmann Rook-Matching Pool

**One-line claim:** An exterior-algebra pooling primitive that sums only row/column-disjoint edge interactions.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times r\times c\times d}\rightarrow\mathbb{R}^{B\times K\times d}\). For edge feature \(z_{ijh}\), define nilpotent basis \(g_{ij}=\epsilon_i\wedge\eta_j\), where \(\epsilon_i^2=\eta_j^2=0\). For each channel:
\[
P_h(t)=\prod_{i,j}\left(1+t\,z_{ijh}g_{ij}\right).
\]
Return \(Y_{kh}\), the summed grade-\(k\) coefficient. Nilpotency deletes every monomial that uses the same row or column twice. Gradients are matching-polynomial leave-one-out coefficients.

**Why this does not decompose into existing PyTorch ops:**
Bipartite attention normalizes edge weights independently, and Sinkhorn returns soft assignment matrices. This operator computes truncated matching-polynomial coefficients in a nilpotent exterior algebra. Row/column exclusivity is enforced algebraically inside the product, not by an attention mask, router, or iterative soft assignment.

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: legal-move graph accumulators and occlusion-semiring bilinear hyperedges. Although it consumes an edge tensor, it does not route messages along legal moves and does not update node states. It is also not ray/occlusion semiring work; the core object is a global disjoint-matching coefficient, not a ray-blocked reducer or legal-edge compiler.

**Chess-specific motivation:**
Tactics often depend on mutually exclusive resources: one defender cannot parry two threats, one attacker cannot be counted twice, and overloaded pieces decide many near-puzzle negatives. Standard pooling double-counts the same latent resource. This primitive scores compatible sets of relations instead of strong individual relations.

**Generalisation beyond chess:**
Reusable for bipartite resource allocation, multi-object tracking, scene matching, assignment-heavy routing, auction-style selection, and scheduling.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(Bd|E|S_K)\), where \(S_K=\sum_{q=0}^{K}{r\choose q}{c\choose q}\), vs explicit matching enumeration \(O(|E|^K)\)
- Backward: \(O(Bd|E|S_K)\)
- Incremental update on a bounded-change input: \(O(dS_K)\) per changed edge using \((1+zg)^{-1}=1-zg\)

**Scout-scale falsification test:**
Use i243's piece-token branch only as a harness. Feed the primitive a learned attacker-defender edge tensor; baseline is the same edge MLP plus sum-pool or sparsemax-pool. Test \(K=2\), cap \(r,c\le16\), and measure CRTK class-1 matched-recall FP rate. It works if class-1 FP drops at least 5% with less than 20% latency hit; it fails if exclusivity gives no class-1 gain.

**Failure mode catalogue:**
- Hidden rebrand objection: “This is just Sinkhorn or differentiable matching.” Reject it if the output is a doubly stochastic matrix rather than matching-polynomial coefficients.
- Numerical instability: high-degree coefficients can grow quickly; keep \(K\le3\), bound edge scores, or compute in a log-scaled coefficient basis.
- Speed risk: \(S_K\) grows fast; full \(K=4\) is probably engine-scale-only.

**Status:** proposed

### primitive_matroid_sparsemax

**Name:** Matroid Sparsemax Pool

**One-line claim:** A sparse pooling operator that selects a differentiable independent set under matroid constraints.

**Mathematical signature:**
Inputs \(s\in\mathbb{R}^{B\times n}\), \(V\in\mathbb{R}^{B\times n\times d}\), and a matroid independence polytope \(P(M)\subset[0,1]^n\):
\[
z^*(s)=\arg\max_{z\in P(M)} s^\top z-\frac{\tau}{2}\|z\|_2^2,\quad Y=V^\top z^*.
\]
For \(\tau>0\), the gradient is the active-face projection Jacobian. As \(\tau\rightarrow0\), it approaches max-weight matroid greedy selection.

**Why this does not decompose into existing PyTorch ops:**
Top-k and sparsemax constrain only cardinality. Matroid Sparsemax constrains the selected set by an exchange system, so the backward graph depends on independence constraints, not independent token scores. Generic differentiable optimization layers such as OptNet can express related constrained projections, so the novelty claim is scoped to a specialized neural primitive with matroid greedy/projection backward rather than a general QP layer.

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: Pareto Antichain Frontier and Regret Saddlepoint. This is not dominance-frontier extraction; it optimizes over an exchange system with augmentation/exchange axioms. It is also not minimax, witness/counterwitness, reply-capacity, or tail-copula logic; the returned object is a sparse convex combination over a matroid polytope.

**Chess-specific motivation:**
Overload, discovered defense, and “only one resource can be used” motifs are independence constraints. Near-puzzle false positives often come from counting the same piece as satisfying multiple latent tactical roles. A matroid pool makes that double-counting structurally impossible while keeping differentiable token selection.

**Generalisation beyond chess:**
Sensor selection, budgeted retrieval, sparse routing, scheduling, scene-object role assignment, and differentiable subset selection with independence constraints.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(n\log n+nT_{\text{oracle}})\) for hard or low-\(\tau\) greedy vs dense attention \(O(n^2d)\)
- Backward: \(O(nd)\) on the active face, plus oracle bookkeeping
- Incremental update on a bounded-change input: \(O(\log n+T_{\text{affected}})\) if scores are kept in an order-statistic structure

**Scout-scale falsification test:**
In i242's smallest token branch, replace top-k or attention pooling over candidate tactical tokens with Matroid Sparsemax using a partition or laminar matroid. Baseline: same branch with sparsemax/top-k. Measure CRTK class-1 matched-recall FP rate. It works if class-1 FP drops at least 5% with less than 15% latency hit; it fails if the selected set degenerates to ordinary top-k.

**Failure mode catalogue:**
- Hidden rebrand objection: “This is sparsemax with constraints.” Reject it if the matroid is only a uniform cardinality constraint.
- Numerical instability: active-set changes create gradient kinks; train with \(\tau>0\) and anneal only after the primitive proves useful.
- Speed risk: arbitrary matroid oracles are too slow; scout tests must start with partition or laminar matroids.

**Status:** proposed

### primitive_irrepnorm

**Name:** Chess-Group IrrepNorm

**One-line claim:** A LayerNorm analogue that normalizes irreducible symmetry subspaces of the chess group.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times C\times8\times8}\rightarrow\mathbb{R}^{B\times C\times8\times8}\). Let finite group \(G\) act by a signed permutation representation \(\rho(g)\) over spatial squares and feature channels. For real irrep \(\lambda\):
\[
P_\lambda=\frac{d_\lambda}{|G|}\sum_{g\in G}\chi_\lambda(g)\rho(g),\quad u_\lambda=P_\lambda X.
\]
Then
\[
Y=\sum_\lambda \gamma_\lambda\frac{u_\lambda-\mu_\lambda}{\sqrt{\|u_\lambda-\mu_\lambda\|^2/m_\lambda+\epsilon}}+\beta_\lambda.
\]
Gradients pass through fixed orthogonal projectors and variance terms.

**Why this does not decompose into existing PyTorch ops:**
LayerNorm normalizes coordinates; GroupNorm normalizes manually chosen channel groups. IrrepNorm normalizes character-projected isotypic components under a finite signed-permutation representation. Group-equivariant layers are established prior art, so the claim is not “new equivariance,” but a chess-specific reusable normalization primitive over irreducible components rather than convolutional weight tying or data augmentation.

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: CRELU/color-involution graph messages and piece-relabelling/involution gates. IrrepNorm sends no messages, constructs no adjacency, and performs no learned relabelling gate. It uses fixed representation projectors and per-irrep normalization; the computation graph is projection-normalization-reconstruction, not content-conditioned graph update or involution routing.

**Chess-specific motivation:**
Chess has useful symmetries beyond image-like dihedral transforms: color swap, file mirror, perspective flip, and piece-channel sign/permutation structure. Normalizing symmetric and antisymmetric components separately can reduce small-data variance without relying on transformer-scale data. The primitive also gives a clean falsification target for whether chess-group structure helps near-puzzle discrimination.

**Generalisation beyond chess:**
Any finite-symmetry domain: board games, symbolic grids, molecules with discrete automorphisms, multi-agent role swaps, and equivariant tabular systems.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(|G|BCHW)\) vs LayerNorm \(O(BCHW)\); cheaper than symmetry ensembling at \(O(|G|\cdot\text{model})\)
- Backward: \(O(|G|BCHW)\)
- Incremental update on a bounded-change input: \(O(|G|C)\) for one changed square if orbit sums and projected moments are cached

**Scout-scale falsification test:**
Replace one LayerNorm/BatchNorm site in i193 with IrrepNorm for \(G=\langle\)file mirror, color-swap plus 180° perspective flip\(\rangle\). Baseline: same trunk with LayerNorm and identical parameter count. Measure CRTK class-1 matched-recall near-puzzle FP rate and throughput. It works if class-1 FP drops at least 3% and throughput loss is below 8%; it fails if gains appear only when ordinary symmetry augmentation is added.

**Failure mode catalogue:**
- Hidden rebrand objection: “This is just group averaging.” Reject it if the output collapses all irreps into an invariant average rather than preserving each isotypic component.
- Numerical instability: small irrep subspaces can have low variance; merge one-dimensional unstable components or increase \(\epsilon\).
- Speed risk: naive tensor copies for every group element can dominate small models; implement projectors as precomputed signed gather/scatter tables.

**Status:** proposed

### primitive_poset_conenorm

**Name:** Poset ConeNorm

**One-line claim:** A normalization primitive that projects latent features onto a partial-order cone before affine scaling.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times n\times d}\rightarrow\mathbb{R}^{B\times n\times d}\). Given poset edges \(E\) and signs \(a_{uv}\in\{-1,+1\}\):
\[
Y^*=\arg\min_Y \frac12\|Y-X\|_F^2\quad\text{s.t.}\quad a_{uv}(Y_v-Y_u)\ge0,\ \forall(u,v)\in E.
\]
Return \(Z=\gamma\odot Y^*+\beta\). The backward pass is the isotonic active-block projection Jacobian.

**Why this does not decompose into existing PyTorch ops:**
LayerNorm rescales; it does not solve an inequality-constrained projection. Differentiable sorting/ranking already shows that projection-defined order operators can be primitive-level, so the novelty claim is scoped: ConeNorm is an arbitrary-poset normalization primitive, not a new theory of isotonic regression. It differs from total-order sorting because the constraint graph can be a chess-derived DAG, chain family, or laminar poset.

**Duplicate audit against existing primitive memory:**
Closest blocklisted families: Pareto Antichain Frontier and dynamic adjacency rank-order gates. ConeNorm does not extract a skyline/frontier and does not gate adjacency by rank. It returns the closest continuous point in an order cone with a KKT-defined gradient, so the core graph is projection onto inequalities, not graph routing or frontier selection.

**Chess-specific motivation:**
Some latent quantities should be monotone over known axes: pawn advancement pressure, king-distance danger, promotion-race urgency, material-phase resource count, or file/rank pressure accumulation. This is a small-data bias that prevents a scout model from fitting arbitrary nonmonotone noise where chess supplies a partial order. It should be tested specifically against near-puzzle false positives, where tactical exceptions punish naive monotonicity.

**Generalisation beyond chess:**
Risk scoring, calibrated ranking, survival models, monotone physics surrogates, tabular models with known order constraints, and scientific ML with conservation/order priors.

**Complexity (forward, backward, incremental-update):**
- Forward: chain/laminar posets \(O(Bdn)\); general DAG \(O(BdT_{\text{iso}}(n,|E|))\) vs generic QP \(O(n^3)\)
- Backward: \(O(Bdn)\) after active blocks are known for chain/laminar cases
- Incremental update on a bounded-change input: \(O(\log n)\) for chains, or affected-block recomputation for DAGs

**Scout-scale falsification test:**
Replace one normalization site in i193 with ConeNorm over rank/file chains on intermediate feature maps. Baseline: same model with LayerNorm. Measure CRTK class-1 matched-recall near-puzzle FP rate and latency. It works if class-1 FP drops at least 3% with no more than 5% latency hit; it fails if monotonicity hurts tactical exceptions or only improves calibration.

**Failure mode catalogue:**
- Hidden rebrand objection: “This is just isotonic regression.” Correct at the optimization-theory level; reject it if it cannot justify the `torch.nn`-style normalization interface and custom backward.
- Numerical instability: active blocks can collapse too aggressively; use residual mixing \(X+\alpha(Y^*-X)\) during the first scout test.
- Speed risk: arbitrary DAG projection is not scout-safe; start with chains or laminar posets only.

**Status:** proposed

## What I cut

- **Legal-move sparse attention:** duplicate of legal-move graph routers, sparse legal graph transitions, and content-conditioned sparse attention.
- **Ray SSM / ray-blocker scan:** duplicate of ray-scan, ray-occlusion dispatch, and ray-parallel SSM families.
- **Differentiable exchange minmax solver:** too close to Regret Saddlepoint and witness/counterwitness quantifier primitives.
- **Sinkhorn attacker-defender assignment:** mostly an OT/OptNet-style layer, not a new primitive-level operator.
- **Learned group canonicalizer:** substantial overlap with 2024 canonicalization/frame-averaging work; the remaining chess version looked like preprocessing or architecture glue.
- **Cubical persistence threat pooling:** plausible for king-safety fields, but differentiable topology layers such as [PersLay](https://arxiv.org/abs/1904.09378) and [PLLay](https://arxiv.org/abs/2002.02778) already cover the primitive class.
- **Boolean Fourier/Hadamard interaction layer:** too close to pair-resonance Hessian and bispectrum-style interaction families.
- **Incremental Bloom/count-sketch memory:** duplicate of sparse delta accumulators unless the sketch algebra itself becomes the core contribution.
- **Tropical shortest-path/race transform:** mostly softmin graph closure and too close to SLG diffusion/path-style primitives.
- **Promotion-fanout nilpotent algebra:** rejected because promotion-fanout counterfactual tensor operators are already blocklisted.
