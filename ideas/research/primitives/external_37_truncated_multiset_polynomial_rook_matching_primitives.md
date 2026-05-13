# New Neural-Network Primitives for Chess Evaluation

Prepared for `chess-nn-playground`.

The numbered proposal order is the overall rank. I used recent primitive-level literature as calibration: Mamba/selective SSMs introduced input-conditioned state-space recurrence ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)); Mamba-2/SSD reframed SSMs and attention via structured semiseparable matrices and reported 2–8× faster core layers ([Dao & Gu, 2024](https://arxiv.org/abs/2405.21060)); DeltaNet changed linear-attention memory updates and received a 2024 parallelization treatment ([Yang et al., 2024](https://openreview.net/forum?id=y8Rm4VNRPH)); xLSTM changed recurrent memory/gating ([Beck et al., 2024](https://arxiv.org/abs/2405.04517)); KAN replaced scalar weights with learnable edge functions ([Liu et al., 2024](https://arxiv.org/abs/2404.19756)); and Titans elevated test-time memory into a neural module ([Behrouz et al., 2025](https://arxiv.org/abs/2501.00663)).

## Ranking summary

| overall rank | primitive | novelty plausibility | RTX 3070 demonstrability | inference-speed advantage | generalises beyond chess |
|---:|---|---:|---:|---:|---:|
| 1 | `truncated_multiset_polynomial_pool` | 2 | 1 | 1 | 2 |
| 2 | `entropic_rook_matching_contraction` | 1 | 3 | 4 | 1 |
| 3 | `laplacian_forest_connectivity` | 3 | 2 | 3 | 3 |
| 4 | `weighted_hodge_flow_split` | 4 | 4 | 2 | 4 |
| 5 | `chess_group_irrep_norm` | 5 | 5 | 5 | 5 |

1. ### primitive_truncated_multiset_polynomial_pool

**Name:** Truncated Multiset Polynomial Pool

**One-line claim:** Pools unordered piece sets into exact low-order interaction coefficients with bounded-change updates.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times n\times d}\times\{0,1\}^{B\times n}\rightarrow\mathbb{R}^{B\times K\times d}\). Let \(u_{bic}=\phi(x_{bi})_c\). For each batch item \(b\) and channel \(c\),
\[
P_{bc}(z)=\prod_{i=1}^{n}(1+m_{bi}u_{bic}z),\qquad
 y_{bkc}=[z^k]P_{bc}(z),\quad k=1,\dots,K.
\]
Equivalently, initialise \(e_0=\mathbf{1}\), \(e_k=0\), then for every active token update \(e_k\leftarrow e_k+e_{k-1}\odot u_i\) for \(k=K,\dots,1\). Gradients are the reverse-mode derivative of this coefficient recurrence.

**Why this does not decompose into existing PyTorch ops:**
This is not DeepSets sum pooling: DeepSets-style invariant pooling collapses the set through \(\sum_i\phi(x_i)\), while this returns elementary-symmetric interaction coefficients over all \(k\)-subsets ([Zaheer et al., 2017](https://arxiv.org/abs/1703.06114)). A naive PyTorch graph would enumerate \(\binom{n}{k}\) products or unroll a dynamic program. The primitive claim is the fused truncated-generating-function scan with a custom reverse pass and cached polynomial-division update for token deletion.

**Duplicate audit against existing primitive memory:**
Closest blocklisted family 1: sparse delta accumulators. This has an incremental update, but the accumulator algebra is a truncated polynomial ring, not additive feature accumulation; deletion is polynomial division, not subtractive scatter. Closest family 2: Signed Piece-Existence Hessian / pair-resonance Hessian. This is not a finite-difference derivative over piece existence and is not limited to pair interactions; \(K=3,4\) produces true symmetric higher-order set coefficients.

**Chess-specific motivation:**
Chess positions are sparse multisets of pieces, and many hard negatives differ by one supporting piece, defender, or pawn. The operator gives a cheap interaction spectrum over material and local piece-token features without attention. It should be especially relevant for near-puzzle false positives where one missing defender invalidates an apparent tactic.

**Generalisation beyond chess:**
Useful for unordered sparse-event sets: molecules, object sets, recommender baskets, entity-state tables, and multimodal scene tokens.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BnKd)\) vs DeepSets \(O(Bnd)\) but first-order only; vs attention \(O(Bn^2d)\)
- Backward: \(O(BnKd)\)
- Incremental update on a bounded-change input: \(O(qKd)\) for \(q\) added/removed/changed tokens using cached coefficients

**Scout-scale falsification test:**
Drop the primitive into the i193 conv-only parent as a replacement for the existing global sum/mean side branch over piece tokens. Baseline: equal-parameter DeepSets sum pool over the same tokens. Measure matched-recall near-puzzle false-positive rate on CRTK class 1. “Works” means at least 10% relative FP reduction at the same recall with less than 5% slower inference. “Fails” means only aggregate PR AUC improves or latency rises enough to erase the gain.

**Failure mode catalogue:**
- Hidden rebrand risk: with \(K=1\), it degenerates to sum pooling; the experiment must use \(K\ge2\).
- Numerical risk: products can underflow or explode; use RMS-normalised coefficients or log-scaled variants.
- Speed risk: high \(K\) becomes pointless; cap at \(K=3\) or \(4\).

**Status:** proposed

2. ### primitive_entropic_rook_matching_contraction

**Name:** Entropic Rook-Matching Contraction

**One-line claim:** Replaces independent edge scoring with differentiable log-partition over one-to-one resource matchings.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times p\times q}\rightarrow\mathbb{R}^{B\times p\times q}\). For score matrix \(S_b\), define \(\mathcal{M}_{\le K}\) as all bipartite matchings of size at most \(K\):
\[
Z_b=\sum_{M\in\mathcal{M}_{\le K}}\exp\left(\tau^{-1}\sum_{i,j}M_{ij}S_{bij}\right),\qquad
P_{bij}=\frac{\partial}{\partial S_{bij}}\left(\tau\log Z_b\right).
\]
Output \(P\) is the exact edge-marginal tensor. Gradients are second-order matching cumulants.

**Why this does not decompose into existing PyTorch ops:**
This is not softmax attention: attention normalises each query independently, while this normalises over mutually exclusive global matchings, so incompatible edges receive coupled negative gradients. It is also not Sinkhorn: Sinkhorn gives an entropic transport relaxation, while this returns exact low-order matching marginals over \(\mathcal{M}_{\le K}\). Related differentiable optimisation layers such as OptNet show the value of constrained operators, but this is a discrete rook-polynomial contraction, not a QP layer ([Amos & Kolter, 2017](https://arxiv.org/abs/1703.00443)).

**Duplicate audit against existing primitive memory:**
Closest family 1: legal-move graph accumulators. This does not pass messages over legal edges; it consumes a dense or sparse score matrix and produces global matching marginals. Closest family 2: delta pair selective bispectra / bilinear hyperedges. The output edge probability depends on all other incompatible edges through \(Z\), not on independent bilinear pair scores or semiring exchange.

**Chess-specific motivation:**
Captures one-to-one competition between attackers, defenders, targets, and escape resources. A near-puzzle false positive often happens because two attacking ideas require the same piece, or one defender covers two apparent threats but only one can be realised. This primitive makes resource exclusivity native.

**Generalisation beyond chess:**
Object tracking, assignment, resource allocation, keypoint matching, sparse retrieval, and scene-graph grounding.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BpKq^K)\) for fixed small \(K\) via rook-polynomial DP vs attention \(O(Bpqd)\)
- Backward: same order using stored inside/outside DP tables
- Incremental update on a bounded-change input: \(O(Kq^K)\) for one changed row with cached partial polynomials; otherwise not \(O(1)\)

**Scout-scale falsification test:**
Drop it into the i242 piece-interaction harness as a replacement for the smallest cross-piece attention or bilinear block, with \(p,q\le16\), \(K=2\) first. Baseline: equal-parameter bilinear attention over the same piece tokens. Metric: matched-recall near-puzzle FP rate plus wall-clock evals/sec. “Works” means FP falls by at least 10% with no more than 20% speed loss. “Fails” if it merely improves easy-negative PR AUC or is slower than the attention block.

**Failure mode catalogue:**
- Hidden rebrand risk: if implemented as Sinkhorn or row-softmax, it no longer has exact matching marginals.
- Numerical risk: small \(\tau\) makes \(Z\) sharply peaked; use log-space DP.
- Speed risk: \(K=3\) may be too slow on RTX 3070; start with \(K=2\).

**Status:** proposed

3. ### primitive_laplacian_forest_connectivity

**Name:** Laplacian Forest Connectivity Transform

**One-line claim:** Turns edge logits into differentiable all-pairs soft connectivity through a random-forest Laplacian resolvent.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times m}\times\mathbb{R}^{B\times n\times d}\rightarrow\mathbb{R}^{B\times n\times d}\). Let fixed incidence matrix \(D\in\{-1,0,1\}^{m\times n}\), edge weights \(w_b=\mathrm{softplus}(a_b)\), and
\[
L_b=D^\top\mathrm{diag}(w_b)D,
\qquad
K_b=(L_b+qI)^{-1},
\qquad
Y_b=qK_bX_b.
\]
Backward uses \(dK=-K(dL)K\). The \(qK\) resolvent is the primitive’s connectivity kernel; matrix-forest theory gives a rooted-forest interpretation to inverses of shifted Laplacians ([Chebotarev & Shamis, 2006](https://arxiv.org/abs/math/0602575)).

**Why this does not decompose into existing PyTorch ops:**
This is not GCN diffusion or message passing; one edge-weight change globally changes \(K\) through a Laplacian resolvent. It is not merely `torch.linalg.solve`, because the primitive includes constrained Laplacian assembly, symmetric positive backward, forest-connectivity semantics, and Woodbury bounded-update caching as one operator. Matrix-tree methods have long used Laplacian determinants/inverses for structured marginals, but PyTorch has no forest-connectivity neural primitive ([Koo et al., 2007](https://aclanthology.org/D07-1015/)).

**Duplicate audit against existing primitive memory:**
Closest family 1: SLG diffusion / legal-state diffusion. This is not a legal-state transition or iterative diffusion over a move graph; it is an exact resolvent over a chosen structural graph such as pawn adjacency or king-shelter adjacency. Closest family 2: sparse legal graph transitions. No legal-move connectivity decides message flow; the graph can be static board adjacency, pawn-chain adjacency, or learned structural edges.

**Chess-specific motivation:**
Pawn chains, pawn islands, king shelters, and blockades are connectivity phenomena. A conv stack can learn local adjacency, but global “is this shelter connected?” or “is this passer supported?” is closer to a resolvent question. This should target hard negatives where local tactical cues look good but the structure is disconnected.

**Generalisation beyond chess:**
Graph semi-supervision, image segmentation, road networks, biological networks, and scene connectivity.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B(n^3+n^2d))\) via Cholesky vs \(T\)-step message passing \(O(BTmd)\); for \(n=64\), the cubic term is small
- Backward: \(O(B(n^3+n^2d))\)
- Incremental update on a bounded-change input: \(O(rn^2+n^2d)\) for rank-\(r\) edge changes via Sherman–Morrison/Woodbury

**Scout-scale falsification test:**
Use it in i193 as a replacement for a small pawn-structure conv branch: input edge logits from existing square features over fixed pawn/king-neighbour edges, output transformed square features. Baseline: equal-parameter two-layer 3×3 conv branch. Metric: CRTK class-1 matched-recall FP rate and evals/sec. “Works” means FP drops at least 8% with less than 10% latency overhead. “Fails” if only calibration or easy negatives improve.

**Failure mode catalogue:**
- Hidden rebrand risk: a reviewer may call it graph diffusion; the distinction must be exact resolvent plus forest-style edge gradients, not unrolled local propagation.
- Numerical risk: \(q\) too small makes \(L+qI\) ill-conditioned.
- Speed risk: per-position factorisation is wasteful unless \(n=64\) and the kernel is fused.

**Status:** proposed

4. ### primitive_weighted_hodge_flow_split

**Name:** Weighted Hodge Flow Split

**One-line claim:** Decomposes directed board-edge signals into source, cycle, and harmonic components by metric-dependent projection.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times m\times d}\times\mathbb{R}_{+}^{B\times m}\times\mathbb{R}_{+}^{B\times r}\rightarrow\mathbb{R}^{B\times m\times 3d}\). For a fixed square complex with node-edge incidence \(B_1\in\mathbb{R}^{n\times m}\) and edge-face incidence \(B_2\in\mathbb{R}^{m\times r}\), input edge flow \(U_b\), edge metric \(W_b=\mathrm{diag}(\rho_b)\), face metric \(F_b=\mathrm{diag}(\sigma_b)\):
\[
G_b=B_1^\top(B_1W_bB_1^\top+\epsilon I)^{-1}B_1W_bU_b,
\]
\[
C_b=B_2(B_2^\top W_bB_2+\epsilon I)^{-1}B_2^\top W_bU_b,
\qquad
H_b=U_b-G_b-C_b.
\]
Output is \([G_b,C_b,H_b]\). Gradients use differentiable linear solves.

**Why this does not decompose into existing PyTorch ops:**
This is not `Conv2d`, not attention, and not graph message passing. It is a metric-dependent orthogonal projection onto exact, coexact, and harmonic flow subspaces. Hodge-Laplacian learning is adjacent prior work, so the honest novelty claim is “underexplored primitive for chess,” not “new mathematics” ([Roddenberry & Segarra, 2019](https://arxiv.org/abs/1912.02354); [Papamarkou et al., 2024](https://arxiv.org/html/2402.08871v2)).

**Duplicate audit against existing primitive memory:**
Closest family 1: attack-ray sparse attention / ray-parallel SSMs. This uses fixed topological boundary operators, not rays, blockers, or legal move edges. Closest family 2: SLG diffusion. HodgeSplit is a one-shot orthogonal decomposition; it does not simulate diffusion or transition dynamics over a legal graph.

**Chess-specific motivation:**
Threat pressure is naturally a directed flow: sources, sinks, cycles, and trapped loops matter. Direct attacks on a king look like gradient flow; repeated manoeuvre pressure and blockades look more cyclic. Separating these components may reduce near-puzzle false positives caused by visually strong but topologically trapped pressure.

**Generalisation beyond chess:**
Traffic flow, fluid grids, electrical networks, mesh learning, social circulation, and any edge-flow dataset.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B(m^3+m^2d))\) if metric-dependent; \(O(Bm^2d)\) with precomputed projections vs conv \(O(Bnk^2d)\)
- Backward: same order as forward solves
- Incremental update on a bounded-change input: \(O(md)\) if metrics fixed; \(O(m^2d)\) if metrics change

**Scout-scale falsification test:**
In i193, replace one late 3×3 conv block with a square-edge flow projection: convert square features to oriented edge flows with a fixed difference stencil, apply HodgeSplit, project back. Baseline: same-parameter 3×3 conv. Metric: matched-recall near-puzzle FP rate and evals/sec. “Works” means at least 8% FP reduction with no more than 15% speed loss. “Fails” if it acts like a fixed linear layer and brings no hard-negative gain.

**Failure mode catalogue:**
- Hidden rebrand risk: with fixed metrics, it may collapse to a fixed linear projection; require input-dependent positive metrics.
- Numerical risk: pseudo-inverse instability; use \(\epsilon\)-regularised solves.
- Speed risk: metric-dependent solves are slower than conv unless fused and low-dimensional.

**Status:** proposed

5. ### primitive_chess_group_irrep_norm

**Name:** Chess-Group Irrep Normalization

**One-line claim:** Normalizes features by exact finite-group representation blocks instead of raw channels or batches.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times |\Omega|\times d}\rightarrow\mathbb{R}^{B\times |\Omega|\times d}\). Let finite group \(G\) act on feature indices \(\Omega\) by permutation or signed-linear maps \(T_g\). For each irreducible character \(\chi_\lambda\),
\[
P_\lambda=\frac{d_\lambda}{|G|}\sum_{g\in G}\chi_\lambda(g^{-1})T_g,
\]
\[
Y=\sum_\lambda \gamma_\lambda
\frac{P_\lambda X}{\sqrt{\frac{1}{B\,\mathrm{rank}(P_\lambda)d}\|P_\lambda X\|_F^2+\epsilon}}
+\beta_{\mathrm{triv}}.
\]
Only the trivial irrep gets a bias. Gradients are projected back into the same isotypic subspaces.

**Why this does not decompose into existing PyTorch ops:**
This is not LayerNorm: the normalization domains are group-theoretic subspaces, not feature dimensions. It is also not group convolution; group-equivariant CNNs use equivariant kernels, whereas this is a representation-spectrum normalization/projection primitive ([Cohen & Welling, 2016](https://arxiv.org/abs/1602.07576)). The novelty claim is specifically the irrep-normalization operator for chess-like finite action groups, not equivariance in general.

**Duplicate audit against existing primitive memory:**
Closest family 1: CRELU/color-involution graph messages. This passes no graph messages and applies no nonlinear color gate; it projects features using group-algebra idempotents. Closest family 2: piece-relabelling/involution gates. This has no learned relabelling decision. The group action is fixed, and gradients are constrained by exact irrep blocks.

**Chess-specific motivation:**
Chess has rule-preserving board symmetries and a color-swap involution; most networks only use augmentation or channel tying. This primitive makes symmetry energy and antisymmetry energy explicit. It can separate “same position from the other side” value symmetry from tactical asymmetries caused by side-to-move.

**Generalisation beyond chess:**
Other board games, molecules with finite symmetry groups, multi-agent role swaps, robotic symmetry groups, and finite-state physics systems.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|G||\Omega|d)\) vs LayerNorm \(O(B|\Omega|d)\)
- Backward: \(O(B|G||\Omega|d)\)
- Incremental update on a bounded-change input: \(O(|G|\Delta d)\) with cached orbit/projection sums

**Scout-scale falsification test:**
Replace LayerNorm/BatchNorm in the i193 evaluation head with IrrepNorm under the rule-preserving subgroup: left-right mirror, color-swap with board reversal, and side-to-move sign handling. Baseline: same head with LayerNorm. Metric: class-1 matched-recall FP rate plus symmetry-consistency error under transformed boards. “Works” means lower FP and lower symmetry inconsistency without slower inference beyond 10%. “Fails” if it only improves transformed-board consistency.

**Failure mode catalogue:**
- Hidden rebrand risk: if \(G\) is just one involution, reviewers may call it tied LayerNorm; use the full validated finite group.
- Numerical risk: small irrep blocks can over-normalize rare channels.
- Speed risk: \(|G|\) permutations per eval may be too costly unless projections are cached/fused.

**Status:** proposed

## What I cut

1. **Legal-move selective SSM:** rejected as a near-duplicate of legal-move kinematic state-space routers and sparse legal graph transitions.
2. **Ray-occlusion min-plus closure:** rejected because it collapses into ray semiring exchange / ray-blocked reducers.
3. **Counterfactual piece-drop Jacobian:** rejected as a renamed signed piece-existence Hessian / cross-derivative operator.
4. **Promotion-funnel tensor product:** rejected as too close to promotion-fanout counterfactual tensors.
5. **Complex Zobrist binding accumulator:** rejected because the core is complex-amplitude interference plus sparse delta accumulation.
6. **Cardinality Sinkhorn piece projector:** rejected as entropic optimal transport/Sinkhorn with chess counts; too close to existing differentiable assignment layers and more like an encoding constraint.
7. **Top-k threat order-statistic pool:** rejected because PyTorch already exposes sort/top-k-style computation and the idea is mostly a pooling tweak.
8. **Legal graph union-find connectivity:** rejected because making connectivity depend on legal moves pushes it back into blocklisted legal graph accumulators.
9. **Möbius subset transform over occupancies:** rejected as a higher-order finite-difference/counterfactual operator, too close to Hessian-over-piece-existence work.
10. **Learned mixture of Hodge + conv + attention:** rejected as an architecture composition, not a primitive.
