# New Neural Primitive Proposals for Chess Evaluation

Literature calibration: recent primitive-level work sets the novelty bar high. Mamba introduced input-selective state-space recurrence with linear sequence scaling ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)); Mamba-2/SSD reframed attention and SSMs through structured semiseparable matrices and reports a faster core layer ([Dao & Gu, 2024](https://arxiv.org/abs/2405.21060)); TTT layers update a test-time hidden-state model during inference ([Sun et al., 2024](https://arxiv.org/abs/2407.04620)); xLSTM changes recurrent memory with exponential gates and matrix memory ([Beck et al., 2024](https://arxiv.org/abs/2405.04517)); and KAN replaces scalar weights with learnable edge splines ([Liu et al., 2024](https://arxiv.org/abs/2404.19756)). The proposals below aim for that operator-level bar, while staying testable under the project’s RTX 3070 scout constraints.

Rank order is the proposal order. Scores use 5 = strongest.

| rank | primitive | novelty | RTX-3070 demonstrability | inference-speed advantage | generalisation |
|---:|---|---:|---:|---:|---:|
| 1 | `primitive_active_esp` | 4 | 5 | 5 | 4 |
| 2 | `primitive_conflict_matching_poly` | 5 | 4 | 3 | 5 |
| 3 | `primitive_occupancy_eikonal` | 3 | 5 | 4 | 5 |
| 4 | `primitive_clifford_accumulator` | 4 | 4 | 4 | 4 |
| 5 | `primitive_stabilizer_orbitnorm` | 3 | 4 | 4 | 4 |

1. ### primitive_active_esp

**Name:** Active-Set Elementary-Symmetric Interaction

**One-line claim:** Computes exact unordered k-way sparse-piece interactions with a polynomial coefficient operator and O(Kr) bounded-move update.

**Mathematical signature:**

\(f:\mathbb{R}^{B\times n\times d}\times[0,1]^{B\times n}\rightarrow\mathbb{R}^{B\times K\times r}\). Let \(z_i=\tanh(x_iP)\in\mathbb{R}^r\). Define an elementwise generating polynomial

\[
E(t)=\prod_{i=1}^{n}(1+m_i z_i t),\qquad
E_k=[t^k]E(t)=\sum_{|S|=k}\prod_{i\in S}m_i z_i.
\]

Output \(Y=\mathrm{concat}(E_1,\ldots,E_K)W\). Gradient:

\[
\partial E_k/\partial z_i=m_i[t^{k-1}]\prod_{j\ne i}(1+m_jz_jt).
\]

**Why this does not decompose into existing PyTorch ops:**

This is not `sum`, `pool`, attention, or a factorized MLP; it is coefficient extraction over a product polynomial. A naive PyTorch emulation needs an unrolled \(nK\) dynamic program or explicit subset enumeration, while the primitive would expose a fused forward/VJP and insertion/deletion algebra. Deep Sets characterises invariant sum-structured networks, but not coefficient-extracting elementary-symmetric operators ([Zaheer et al., 2017](https://arxiv.org/abs/1703.06114)).

**Duplicate audit against existing primitive memory:**

Closest blocklisted family 1: Signed Piece-Existence Hessian / pair-resonance Hessian. This is not a second derivative over piece-existence bits; \(E_k\) is a forward elementary-symmetric coefficient for arbitrary \(k\), with gradients from quotient polynomials. Closest family 2: sparse delta accumulators / delta-event routers. It supports incremental updates, but the algebra is multiplicative polynomial insertion/deletion, not additive latent accumulation under a move delta.

**Chess-specific motivation:**

Tactical motifs are often unordered sparse interactions: defender + pinned piece + mating square, or sacrifice + overloaded defender + king shelter. Attention must learn these k-way contacts from data; this gives exact low-degree set interactions at scout scale. A legal move changes only a few active factors, directly chasing the HalfKA update property.

**Generalisation beyond chess:**

Useful for recommender systems, sparse-event fraud detection, molecule/set modelling, and any domain needing low-degree unordered interactions.

**Complexity (forward, backward, incremental-update):**

- Forward: \(O(BnKr)\) vs explicit k-way enumeration \(O(Bn^Kr)\).
- Backward: \(O(BnKr)\) with saved prefix/suffix coefficients.
- Incremental update on a bounded-change input: \(O(BcKr)\), where \(c\le 4\) changed active factors.

**Scout-scale falsification test:**

Drop \(K=3,r=16\) after the first square-feature block of i193 conv-only; concatenate \(E_1,E_2,E_3\) into the value head. Baseline: same parameter count via MLP pooling. Measure matched-recall CRTK class-1 near-puzzle false-positive rate and per-position inference time. Works if FP rate falls by at least 5% relative with no more than 10% latency hit; fails if only aggregate PR AUC improves.

**Failure mode catalogue:**

- Hidden rebrand risk: reviewer says this is factorization-machine polynomial pooling; the defense requires the active-set fused VJP and formal inverse update.
- Numerical risk: products can explode or vanish; constrain \(z_i\) with `tanh` and clip coefficient norms.
- Speed risk: \(K>3\) is probably too slow and overfits 173k positions.

**Status:** proposed

2. ### primitive_conflict_matching_poly

**Name:** Conflict-Constrained Matching Polynomial Pool

**One-line claim:** Sums all mutually compatible tactical contacts as truncated graph-polynomial coefficients, not as message passing or one best matching.

**Mathematical signature:**

\(f:\mathbb{R}^{B\times m\times r}\times[0,1]^{B\times m}\times\{0,1\}^{B\times m\times q}\rightarrow\mathbb{R}^{B\times K\times r}\). Candidate contact \(e\) has feature \(u_e\), gate \(w_e\), and resource-incidence vector \(H_e\). A valid matching \(M\) satisfies \(\sum_{e\in M}H_{e,a}\le 1\) for every resource \(a\). Define

\[
C_k=\sum_{M: |M|=k,\ M\text{ valid}}\left(\prod_{e\in M}w_e\right)\left(\bigodot_{e\in M}u_e\right),\quad C_0=\mathbf{1}.
\]

Output \(Y=\mathrm{concat}(C_1,\ldots,C_K)W\). Matching-polynomial coefficients count matchings by size in graph theory ([Spielman notes, 2018](https://www.cs.yale.edu/homes/spielman/561/lect26-18.pdf)).

**Why this does not decompose into existing PyTorch ops:**

`scatter`, sparse attention, and graph convolution aggregate edges independently; they do not enforce “no two selected contacts share a resource” while summing all valid subsets. Sinkhorn/Hungarian-style layers return one soft assignment, while this primitive returns polynomial coefficients over all compatible matchings. A PyTorch emulation requires explicit combinatorial enumeration or a custom hard-core partition-function kernel.

**Duplicate audit against existing primitive memory:**

Closest blocklisted family 1: legal-move graph accumulators / sparse legal graph transitions. This is not routing values along legal edges; it computes coefficients of a constrained edge-subset polynomial. Closest family 2: factor-graph/tensor-product legal-state primitives. This does not compute legality or legal-state consistency; the constraint resources are tactical contacts, and the output is degree-indexed matching coefficients.

**Chess-specific motivation:**

A false tactic often double-counts one defender or assumes the same piece can satisfy two independent tactical contacts. This operator encodes resource exclusivity directly: one piece, square, tempo, or defender cannot be spent twice. That is exactly the kind of structure near-puzzle hard negatives exploit.

**Generalisation beyond chess:**

Applies to scene-graph relation selection, multi-object tracking, molecule matching, resource allocation, and bipartite recommender constraints.

**Complexity (forward, backward, incremental-update):**

- Forward: truncated exact \(O(Brm^K)\) worst case, practical only for \(K\le3\); closest one-matching solver \(O(n^3)\) returns a different object.
- Backward: same order; gradients are sums over valid matchings containing each edge.
- Incremental update on a bounded-change input: \(O(Br\Delta m\,m^{K-1})\), less with cached bitset compatibility tables.

**Scout-scale falsification test:**

Use only board-derived contact candidates, no Stockfish scores, PVs, node counts, or verification metadata. Insert \(K=2\) or \(3\) before the i193 value head. Baseline: same-size edge-MLP plus sum pooling. Works if matched-recall CRTK class-1 FP rate drops at least 5% and inference remains below 1.25× baseline; fails if gains vanish when easy negatives are removed.

**Failure mode catalogue:**

- Hidden rebrand risk: if implemented as ordinary maximum matching or Sinkhorn, it is not this primitive.
- Numerical risk: products of many \(w_e\) underflow; keep \(K\le3\) and use log-gates if needed.
- Speed risk: general high-degree matching polynomials are combinatorial; do not scale \(K\) casually.

**Status:** proposed

3. ### primitive_occupancy_eikonal

**Name:** Differentiable Occupancy Eikonal Transform

**One-line claim:** Computes learned arrival-time fields through blockers via a min-plus fixed point instead of attention over squares.

**Mathematical signature:**

\(f:\mathbb{R}_{+}^{B\times |E|\times q}\times\mathbb{R}^{B\times |V|\times q}\rightarrow\mathbb{R}^{B\times |V|\times q}\). For fixed graph \(G=(V,E)\), edge costs \(c_{uv}^{(h)}\), seed costs \(s_v^{(h)}\), and temperature \(\tau\):

\[
T_v^{(h)}=\operatorname{softmin}_{\tau}\left(s_v^{(h)},\{T_u^{(h)}+c_{uv}^{(h)}:(u,v)\in E\}\right),
\]

where \(\operatorname{softmin}_{\tau}(a)=-\tau\log\sum_i e^{-a_i/\tau}\). Solve by soft Bellman-Ford or fast-marching relaxation; gradients use implicit fixed-point differentiation or saved relaxations.

**Why this does not decompose into existing PyTorch ops:**

This is a min-plus transitive-closure operator, not convolution, pooling, or masked attention. Its computation graph is a shortest-path fixed point whose active predecessor set changes with costs. Prior work on differentiating through Dijkstra and other combinatorial solvers exists ([Vlastelica et al., 2020](https://openreview.net/forum?id=BkevoJSYPB)), so the claim is not “first differentiable shortest path”; the claim is a reusable bounded-update eikonal primitive for local dynamic boards.

**Duplicate audit against existing primitive memory:**

Closest blocklisted family 1: legal-move graph transitions / move-graph routers. This primitive does not route latent messages along legal moves; it solves an arrival-time equation on a cost field. Closest family 2: ray-scan / ray-parallel SSM / obstacle-pooling emitters. This is not a line or ray recurrence; it is global geodesic propagation over arbitrary local neighborhoods.

**Chess-specific motivation:**

King danger is often “how fast can force arrive,” not simply “how many attacks exist.” Blockers, pawn breaks, knight hops, and escape squares are naturally cost fields. This gives the network a cheap tactical-distance bias without a transformer.

**Generalisation beyond chess:**

Path planning, robotics, traffic prediction, medical/image segmentation, and any learned-cost grid or graph problem.

**Complexity (forward, backward, incremental-update):**

- Forward: exact \(O(Bq(|E|\log |V|))\) or soft \(O(BqR|E|)\); closest attention over squares is \(O(Bq|V|^2)\).
- Backward: same order with saved predecessors or relaxation states.
- Incremental update on a bounded-change input: dynamic affected-region update \(O(q(\Delta E\log |V|+A))\), where \(A\) is affected vertices.

**Scout-scale falsification test:**

Add four 8×8 arrival maps after i193’s first conv block: own/opponent arrival to king-zone and queen-zone seeds, with costs predicted from latent features. Baseline: four extra learned conv channels. Works if near-puzzle FP rate at matched recall drops at least 5% with no more than 20% latency hit; fails if it improves only easy negatives.

**Failure mode catalogue:**

- Hidden rebrand risk: if it becomes “soft attention with distance bias,” reject it.
- Numerical risk: softmin temperature can blur tactics; hard min can create brittle gradients.
- Speed risk: iterative relaxation can dominate runtime unless \(|V|=64\), \(q\) is small, and iterations are capped.

**Status:** proposed

4. ### primitive_clifford_accumulator

**Name:** Active Clifford Product Accumulator

**One-line claim:** Multiplies active piece events in a real graded algebra, preserving signs, orientations, and noncommuting interactions.

**Mathematical signature:**

\(f:\mathbb{R}^{B\times n\times d}\times[0,1]^{B\times n}\rightarrow\mathbb{R}^{B\times G}\), where \(G=2^{p+q}\). Map \(a_i=Ax_i\in Cl(p,q)\). With geometric-product tensor \(T_{\alpha\beta}^{\gamma}\),

\[
(P\star Q)_\gamma=\sum_{\alpha,\beta}P_\alpha Q_\beta T_{\alpha\beta}^{\gamma}.
\]

Define

\[
P=\prod_{i=1}^{n}(1+m_i a_i),\qquad
Y=[\langle P\rangle_0,\langle P\rangle_1,\ldots,\langle P\rangle_L]W.
\]

Backward uses prefix/suffix Clifford products; optional grade involution enforces color-swap sign rules.

**Why this does not decompose into existing PyTorch ops:**

A dense `einsum` can emulate one product, but no existing PyTorch primitive has a fixed Clifford multiplication table, grade projections, involutions, and prefix/suffix VJP as one algebraic operator. It is not complex-valued interference: complex multiplication is commutative and two-dimensional, whereas Clifford products are real, graded, and generally noncommutative. Clifford-equivariant networks show that geometric products can define expressive equivariant layers; this proposal isolates the active-set accumulator as the primitive ([Ruhe et al., 2023](https://openreview.net/forum?id=n84bzMrGUD&noteId=sQG6abJbs8)).

**Duplicate audit against existing primitive memory:**

Closest blocklisted family 1: Complex-Amplitude Interference operators. This is not amplitude/phase interference; it uses multivector grades, noncommutative products, and grade-selective outputs. Closest family 2: CRELU/color-involution graph messages and piece-relabelling gates. No graph messages or learned relabelling gates are used; color and piece symmetries are represented as algebra involutions.

**Chess-specific motivation:**

Pins, skewers, forks, and discovered attacks are oriented relations with sign flips under color swap. A real graded algebra can represent scalar material, vector direction, bivector interaction, and pseudoscalar “side” in one product. It may capture tactical composition with fewer examples than attention.

**Generalisation beyond chess:**

Rigid-body physics, robotics, molecular geometry, signed scene relations, and event streams with orientation or parity.

**Complexity (forward, backward, incremental-update):**

- Forward: \(O(BnG^2)\), or \(O(BnGs)\) with sparse multiplication table; closest MLP mixer \(O(Bnd^2)\).
- Backward: \(O(BnG^2)\) with prefix/suffix products.
- Incremental update on a bounded-change input: \(O(BcG^2\log n)\) with a product tree, \(O(BcG^2)\) if cached local replacement is enough.

**Scout-scale falsification test:**

Use active non-empty square tokens from i243 or i193 latent square features; set \(p+q=4\), so \(G=16\). Replace one equal-parameter dense token mixer with the accumulator. Works if class-1 matched-recall FP rate improves and latency is not worse than the prior attention ablation. Fails if gains disappear under color-swap consistency checks.

**Failure mode catalogue:**

- Hidden rebrand risk: a reviewer may call it a bilinear layer; the defense requires grade projections, involutions, and product-tree VJP.
- Numerical risk: repeated products can explode; constrain \(\|a_i\|\) or use Cayley-style normalization.
- Speed risk: \(G=32\) or \(64\) is likely too slow on RTX 3070.

**Status:** proposed

5. ### primitive_stabilizer_orbitnorm

**Name:** Stabilizer-Weighted OrbitNorm

**One-line claim:** Normalises features over the input’s approximate symmetry stabilizer, not over batch, channel, or fixed token axes.

**Mathematical signature:**

\(f:\mathbb{R}^{B\times n\times d}\rightarrow\mathbb{R}^{B\times n\times d}\). Let finite group \(G\) act by \(\rho_g\) on token-feature tensors. Define stabilizer weights

\[
\alpha_g(X)=\frac{\exp(-\|PX-\rho_gPX\|_F^2/\tau)}{\sum_{h\in G}\exp(-\|PX-\rho_hPX\|_F^2/\tau)}.
\]

Orbit statistics:

\[
\mu_i=\sum_{g\in G}\alpha_g(\rho_gX)_i,
\qquad
\sigma_i^2=\sum_{g\in G}\alpha_g\| (\rho_gX)_i-\mu_i\|^2.
\]

Output:

\[
Y_i=\gamma\odot\frac{X_i-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}+\beta.
\]

**Why this does not decompose into existing PyTorch ops:**

BatchNorm, LayerNorm, and GroupNorm normalise over fixed axes. OrbitNorm normalises over a finite group orbit whose weights are determined by the input’s approximate stabilizer; the normalization axis is therefore content-dependent but symmetry-constrained. Group-equivariant CNNs use fixed group convolutions to exploit symmetries, while this is a normalization primitive over group actions ([Cohen & Welling, 2016](https://arxiv.org/abs/1602.07576)).

**Duplicate audit against existing primitive memory:**

Closest blocklisted family 1: CRELU/color-involution graph messages. OrbitNorm sends no messages and performs no graph update; it computes group-orbit statistics. Closest family 2: piece-relabelling/involution gates. Those gate features under specific relabellings; this estimates the stabilizer distribution \(\alpha_g\) and normalises against its orbit mean/variance.

**Chess-specific motivation:**

Chess has dihedral board symmetries, color-swap involution, side-to-move sign flips, and partial piece-type symmetries. Many positions have approximate stabilizers: mirrored pawn shells, symmetric king safety, or color-swapped tactical motifs. A small-data scout should not relearn these from examples.

**Generalisation beyond chess:**

Molecular graphs, crystals, board games, cellular automata, and any finite-symmetry structured input.

**Complexity (forward, backward, incremental-update):**

- Forward: \(O(B|G|nd)\) vs LayerNorm \(O(Bnd)\); for chess keep \(|G|\le16\).
- Backward: \(O(B|G|nd)\).
- Incremental update on a bounded-change input: update stabilizer norms in \(O(B|G|cd)\); refreshing all normalised outputs is \(O(B|G|nd)\), or local-orbit only if cached.

**Scout-scale falsification test:**

Replace the first normalization in i193 or the i242 lightweight ablation with OrbitNorm over \(D_4\times C_2\) color swap, with no architecture change otherwise. Baseline: same model with LayerNorm/BatchNorm. Works if class-1 matched-recall FP rate improves at least 3% relative and color-swap consistency error drops; fails if only calibration improves.

**Failure mode catalogue:**

- Hidden rebrand risk: if \(\alpha_g\) is fixed uniform, this collapses to group augmentation plus normalization.
- Numerical risk: near-zero orbit variance can over-amplify rare asymmetric features; use \(\epsilon\) and variance clipping.
- Speed risk: large piece-relabelling groups are too expensive; restrict to tiny, testable subgroups.

**Status:** proposed

## What I cut

- **Legal-move attention variants:** rejected because they reduce to masked/sparse attention or legal-edge routing, both blocklisted.
- **Ray-occlusion distance scans:** rejected because ray scans, ray SSMs, and occlusion semiring reducers are already in the inventory.
- **Soft minimax / differentiable negamax backup:** rejected because it overlaps with regret saddlepoint and witness-counterwitness primitives and drifts into search rather than evaluation.
- **Persistent-homology king-shelter layer:** rejected because differentiable/topological layers already exist, including persistence-based layers such as PLLay ([Kim et al., 2020](https://papers.nips.cc/paper_files/paper/2020/file/b803a9254688e259cde2ec0361c8abe4-Paper.pdf)); this is underexplored for chess but not novel enough here.
- **Entropic exchange matching / Sinkhorn SEE:** rejected because differentiable optimization layers are already a known family, including OptNet-style optimization layers ([Amos & Kolter, 2017](https://arxiv.org/abs/1703.00443)); it also too easily becomes a legal-attack graph solver.
- **Weighted model-counting / BDD legality layer:** rejected as too close to blocklisted factor-graph/tensor-product legal-state primitives.
- **Tropical ray convolution:** rejected because semiring ray exchange is explicitly blocklisted.
- **Counterfactual one-piece removal influence:** rejected because it is too close to piece-existence Hessian and counterfactual tensor families.
- **KAN-style spline-on-piece-edge operator:** rejected because KAN already made learnable edge functions a primitive-level idea; chess would only provide a domain-specific placement.
- **TTT-style fast tactical memory:** rejected because it is a training/inference adaptation mechanism and close to TTT layers rather than a chess-specific new primitive.
