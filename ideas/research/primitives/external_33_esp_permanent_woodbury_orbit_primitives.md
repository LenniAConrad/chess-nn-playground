# New Neural-Network Primitives for Chess Evaluation

This document proposes five primitive-level operators for the `chess-nn-playground` research project. The proposals are intentionally not architectures, input encodings, losses, curricula, or hyperparameter variations. They are ranked by a composite judgment over: novelty plausibility, feasibility on a single RTX 3070, inference-speed upside, and generalisation beyond chess.

Recent primitive-level calibration references include Mamba/S6 state-space layers, Mamba-2, xLSTM, Differential Transformer, Kolmogorov-Arnold Networks, Titans memory modules, and FlashAttention. These are useful as examples of work where the computation graph, state update, normalization, routing, or IO/backward behavior changes at the primitive level rather than merely composing existing blocks. See the references section for links.

Composite rank axes `[novelty, RTX3070 demonstrability, inference-speed advantage, generalisation]`, where 1 is strongest:

- `primitive_esp_set`: `[2,1,2,1]`
- `primitive_permanent_roles`: `[1,3,4,2]`
- `primitive_woodbury_resolver`: `[3,2,3,3]`
- `primitive_orbit_canonicalizer`: `[4,4,1,4]`
- `primitive_component_pool`: `[5,5,5,5]`

1. ### primitive_esp_set

**Name:** Elementary-Symmetric Event Transform

**One-line claim:** Converts active sparse tokens into exact bounded-order set-polynomial features with O(Kd) bounded-change updates.

**Mathematical signature:**

`f: R^{B×N×d} × {0,1}^{B×N} → R^{B×K×c}`.

Let `a_{b,i}=m_{b,i}(x_{b,i}U) ∈ R^c`. Define channelwise coefficients of

`P_b(z)=∏_{i=1}^N (1+z a_{b,i})`.

Use the recurrence:

`E_{b,0}=1`

For `i=1..N` and `k=K..1`:

`E_{b,k} ← E_{b,k}+a_{b,i}⊙E_{b,k-1}`.

Output `E_{b,1:K}`. Gradients are ordinary polynomial gradients through the coefficient recurrence.

**Why this does not decompose into existing PyTorch ops:**

The closest existing pattern is DeepSets-style sum/mean pooling, whose invariant form is sum-decomposed; this operator returns exact elementary-symmetric coefficients, not a sum statistic. DeepSets theory already uses elementary symmetric polynomials as a mathematical reference object, but not as an incremental neural primitive with exposed coefficient state. A reference implementation can use loops and tensor ops, but no `torch.nn` primitive exposes “coefficient-of-product” semantics, coefficient-state backward, or O(Kd) downdates.

**Duplicate audit against existing primitive memory:**

Closest blocklisted family 1: sparse delta accumulators. This is not an additive accumulator; its algebra is the truncated polynomial ring `R[z]/z^{K+1}`, and removal uses polynomial division `Q_k=E_k-aQ_{k-1}`.

Closest blocklisted family 2: signed piece-existence Hessian / pair-resonance operators. This is not pairwise second-derivative logic; it simultaneously represents all subset orders `1..K` with multiplicative gradients.

**Chess-specific motivation:**

Chess tactics are often conjunctions: two attackers, one defender plus king exposure, pawn chain plus outpost. The operator gives order-insensitive high-order conjunction features without enumerating legal moves or ray edges. It is especially suited to near-puzzle false positives where one missing conjunct changes the answer.

**Generalisation beyond chess:**

Useful for sparse-event sequences, molecule atom sets, object sets, and any domain needing bounded-order set interactions with incremental updates.

**Complexity (forward, backward, incremental-update):**

- Forward: `O(BNKc)` vs pair or tuple enumeration `O(BN^Kc)`
- Backward: `O(BNKc)`
- Incremental update on a bounded-change input: `O(Kc)` per added or removed token via coefficient multiplication/division

**Scout-scale falsification test:**

Drop one ESET block into the i193 conv-only parent just before the value head, using existing square-stem features and the piece-occupancy mask; set `K=3,c=64`. Baseline: same head with global sum/mean pooling and equal parameter count. Measure CRTK class-1 matched-recall near-puzzle FP rate. The primitive works if FP falls by at least 5% with <10% latency overhead. It fails if aggregate PR improves but matched-recall near-puzzle FP does not.

**Failure mode catalogue:**

- Hidden-rebrand risk: a reviewer could argue this is “just DeepSets with a fancier pooling.” The distinction depends on exact coefficient recurrence and downdate state.
- Numerical risk: coefficients can explode with large `K`; use log scaling, bounded activations, or degree-wise normalization.
- Speed risk: if `K>4`, the primitive may become slower than the tactical signal is worth.

**Status:** proposed

2. ### primitive_permanent_roles

**Name:** Exchangeable Permanent Role Assignment

**One-line claim:** Assigns indistinguishable tokens to learned roles using exact permanent marginals, not arbitrary ordering or Sinkhorn relaxation.

**Mathematical signature:**

`f: R^{B×m×d} × R^{r×d} → R^{B×r×d_v}` for one exchangeable class, such as white pawns.

Scores:

`A_{b,i,j}=exp(φ(x_{b,i})ᵀr_j/τ)`.

Partition function over injective assignments:

`Z_b=Σ_{π∈Inj(m,r)} ∏_{i=1}^m A_{b,i,π(i)}`.

Marginal:

`P_{b,i,j}=A_{b,i,j} ∂log Z_b/∂A_{b,i,j}`.

Output:

`y_{b,j}=Σ_i P_{b,i,j} Vx_{b,i}`.

Gradients flow through the dynamic program for `Z`.

**Why this does not decompose into existing PyTorch ops:**

Attention normalizes rows independently; Sinkhorn approximates doubly stochastic transport iteratively. This operator computes exact one-to-one Gibbs assignment marginals through a matrix permanent, whose definition is a sum over permutation products with all-positive signs. PyTorch has no permanent/marginal primitive, and the backward graph is a combinatorial partition-function derivative, not softmax.

**Duplicate audit against existing primitive memory:**

Closest blocklisted family 1: piece-relabelling/involution gates. This does not relabel piece channels or choose a gate; it marginalizes over all bijections among identical pieces.

Closest blocklisted family 2: move-graph routers and legal-move accumulators. No legal edges, attack rays, or message passing appear. The exclusivity constraint is assignment-theoretic: one physical piece cannot fill two latent roles.

**Chess-specific motivation:**

Identical pieces are exchangeable, but roles are not: a pawn may be passer, shield, blocker, or target. Near-puzzle false positives often arise when evidence for two tactical roles is present but carried by the same piece; exact assignment prevents double-counting.

**Generalisation beyond chess:**

Applicable to molecules with identical atoms, multi-object tracking, scene graphs with exchangeable objects, and set-to-slot models.

**Complexity (forward, backward, incremental-update):**

- Forward: `O(B r 2^r d_v)` via subset DP, vs attention `O(Bmrd_v)` but without exclusivity
- Backward: `O(B r 2^r d_v)`
- Incremental update on a bounded-change input: `O(r2^r)` for the affected piece class; in chess, `r≤8` for a single piece type

**Scout-scale falsification test:**

Use existing piece tokens from i243 or i193’s square stem. Replace same-type DeepSets pooling with EPRA for pawns, knights, bishops, and rooks only; keep parameter count matched. Baseline: typewise sum pooling. Measure CRTK class-1 matched-recall near-puzzle FP and per-position eval time. The primitive works if FP falls by at least 3% with <15% latency overhead. It fails if it only improves easy negatives or materially slows inference.

**Failure mode catalogue:**

- Hidden-rebrand risk: a reviewer could say this is Sinkhorn assignment. It must use exact permanent marginals, not iterative row/column normalization.
- Numerical risk: log-domain DP is mandatory; raw `exp` permanents will underflow or overflow.
- Speed risk: pawns with `r=8` are feasible, but using this over all 64 squares would be unjustifiably slow.

**Status:** proposed

3. ### primitive_woodbury_resolver

**Name:** Woodbury Set Resolver

**One-line claim:** Resolves queries against an active set through an inverse covariance memory with rank-one move updates.

**Mathematical signature:**

`f: R^{B×N×d} × {0,1}^{B×N} × R^{B×Q×r} → R^{B×Q×d_v}`.

Let:

`U_i=φ(x_i) ∈ R^r`

`V_i=ψ(x_i) ∈ R^{d_v}`

`A_b=λI_r+Σ_i m_{b,i} U_{b,i}U_{b,i}ᵀ`

`S_b=Σ_i m_{b,i} U_{b,i}V_{b,i}ᵀ`.

Output:

`Y_b=Q_b A_b^{-1} S_b`.

Gradients use matrix-solve differentials.

**Why this does not decompose into existing PyTorch ops:**

The closest PyTorch operation is `torch.linalg.solve`, but that recomputes a dense solve and exposes no event-update state. The primitive’s signature is “active set → inverse-covariance resolver,” with Sherman-Morrison/Woodbury rank-one updates. It is also not attention: there is no softmax, no key-token normalization, and gradients pass through an SPD inverse.

**Duplicate audit against existing primitive memory:**

Closest blocklisted family 1: sparse delta accumulators. Although it caches state, the cached object is an inverse SPD matrix plus cross-covariance; bounded updates are non-additive rank-one inverse updates.

Closest blocklisted family 2: bilinear ray-blocked / delta-pair bispectrum operators. This uses global covariance geometry, not ray segments, hyperedges, or pair message products.

**Chess-specific motivation:**

Many false positives are caused by correlated evidence: several features all describe the same attacker, shield, or material fact. The inverse covariance term can downweight collinear active-set evidence and make query-square or query-piece features compete globally without legal-move attention.

**Generalisation beyond chess:**

Useful for online memory, recommender contexts, dynamic point sets, kernel ridge-style retrieval, and sparse-event systems.

**Complexity (forward, backward, incremental-update):**

- Forward: `O(B(Nr²+r³+Qrd_v))` vs attention `O(BQNd_v)`
- Backward: `O(B(Nr²+r³+Qrd_v))`
- Incremental update on a bounded-change input: `O(r²+rd_v)` per added or removed token, using rank-one inverse update

**Scout-scale falsification test:**

Insert WSR as the only global mixer before the value head in i193, with `r=24,d_v=64,Q=1+64` pooled/query tokens. Baseline: same parameter count with linear attention or global MLP pooling. Measure CRTK class-1 matched-recall near-puzzle FP and nodes/sec proxy. The primitive works if FP improves by at least 3%, or if it gives equal FP with at least 1.25× faster inference than the mixer baseline.

**Failure mode catalogue:**

- Hidden-rebrand risk: a reviewer could call this kernel attention or ridge regression. The primitive claim rests on cached inverse-covariance state and event-update backward.
- Numerical risk: ill-conditioned `A` can destabilize gradients; require `λ≥1e-3`, Cholesky solve, and spectral clipping.
- Speed risk: for `r>64`, the `r³` solve can erase the speed advantage.

**Status:** proposed

4. ### primitive_orbit_canonicalizer

**Name:** Straight-Through Orbit Canonicalizer

**One-line claim:** Chooses one canonical finite-group representative inside the graph, replacing symmetry ensembles with one routed branch.

**Mathematical signature:**

`f: R^{B×N×d} × {0,1}^{B×N×p} × G → R^{B×N×d}`.

For finite group actions `T_g` on board indices/channels, compute:

`g_b^*=argmin_{g∈G} Hash(T_g M_b)`

using deterministic lexicographic or Zobrist hash.

Forward:

`Y_b=T_{g_b^*}X_b`.

Backward:

`∂L/∂X_b=T_{g_b^*}^{-1}(∂L/∂Y_b)`.

No gradient is defined through the discrete hash choice.

**Why this does not decompose into existing PyTorch ops:**

Group-equivariant CNNs average or share weights over fixed group actions; this primitive performs input-dependent conditional canonical routing. G-CNNs exploit symmetries with fixed G-convolutions and weight sharing, but they do not select a single canonical representative by a discrete board-dependent argmin. It is closer to a non-learned MoE gate over group actions, but the gate is exact symbolic canonicalization.

**Duplicate audit against existing primitive memory:**

Closest blocklisted family 1: CRELU/color-involution messages and color-involution adjacency updates. This is not a color gate or message update; it routes the whole latent tensor through one group representative.

Closest blocklisted family 2: piece-relabelling/involution gates. It does not learn relabel scores; the branch is determined by exact orbit canonicality.

**Chess-specific motivation:**

Chess has label-preserving symmetries: board flips under perspective conventions, color-swap with score sign, side-to-move normalization, and piece-channel permutations. Canonicalization can reduce sample complexity without evaluating all group copies, which matters on 173k-position scout runs.

**Generalisation beyond chess:**

Useful for finite-symmetry domains: molecules, board games, symbolic grids, CAD layouts, and object scenes with known discrete transformations.

**Complexity (forward, backward, incremental-update):**

- Forward: `O(B|G|Np + BNd)` vs group ensemble/G-conv `O(B|G|Nd)`
- Backward: `O(BNd)`
- Incremental update on a bounded-change input: `O(|G|)` with cached per-transform rolling hashes

**Scout-scale falsification test:**

Place the primitive immediately after the first square-feature projection in i193; apply the inverse transform only to any squarewise auxiliary output if needed. Baseline: no canonicalizer. Secondary baseline: 8-way symmetry ensemble at inference only. The primitive works if it matches ensemble FP within 1% while being at least 4× faster, or beats no-canonicalizer FP by at least 3%.

**Failure mode catalogue:**

- Hidden-rebrand risk: a reviewer could say this is input preprocessing, not a primitive. It must be usable at arbitrary latent layers, not only raw planes.
- Numerical/discrete risk: hash ties near symmetric boards can create unstable branch choices; tie-breaking must be deterministic.
- Correctness risk: channel-action bugs under color swap can silently leak label sign errors.

**Status:** proposed

5. ### primitive_component_pool

**Name:** Masked Transitive-Component Pool

**One-line claim:** Pools features over exact connected components of a dynamic mask, not fixed windows or finite-depth message passing.

**Mathematical signature:**

`f: R^{B×N×d} × {0,1}^{B×N} × {0,1}^{N×N} → R^{B×N×d}`.

Given fixed adjacency `A₀`, define `i ~_b j` iff `m_{b,i}=m_{b,j}=1` and there is an `A₀` path through active vertices. Let `C_b(i)` be the component of `i`.

For active `i`:

`y_{b,i}=|C_b(i)|^{-1/2} Σ_{j∈C_b(i)} W x_{b,j}`.

For inactive `i`:

`y_{b,i}=0`.

Gradients distribute through the exact component sums.

**Why this does not decompose into existing PyTorch ops:**

MaxPool and AvgPool use fixed local windows; graph pooling methods such as DiffPool learn soft clusters, not exact transitive closure over a changing binary mask. Computing components is an algorithmic primitive: dynamic connectivity maintains connected components as edges change, classically via union-find or fully dynamic structures. A finite stack of message-passing layers only approximates this unless its depth reaches graph diameter.

**Duplicate audit against existing primitive memory:**

Closest blocklisted family 1: legal-move graph accumulators. This uses a fixed adjacency such as grid-neighbor or pawn-neighbor adjacency, not legal moves, attack edges, or rays.

Closest blocklisted family 2: sparse delta accumulators. Its state is an equivalence partition with merge/split operations; the gradient is componentwise pooling, not additive latent maintenance.

**Chess-specific motivation:**

Pawn islands, connected pawn chains, king-shelter blobs, open corridors, and blocked regions are transitive structures. A 3×3 convolution sees local contact; this primitive sees the whole connected structure in one operator.

**Generalisation beyond chess:**

Useful for segmentation masks, occupancy grids, dynamic scene components, robotics maps, and graph-structured sparse binary fields.

**Complexity (forward, backward, incremental-update):**

- Forward: `O(B(N+E)d)` vs `L`-layer local message passing `O(BLEd)`
- Backward: `O(B(N+E)d)`
- Incremental update on a bounded-change input: additions `O(α(N)d)` with union-find; deletions `O(log²N d)` with fully dynamic connectivity, or `O((N+E)d)` recompute on 8×8 chess boards

**Scout-scale falsification test:**

Add component pooling over own pawns, enemy pawns, all pieces, and empty squares using existing 8×8 square features in i193. Baseline: two extra 3×3 conv layers with matched parameters. Measure CRTK class-1 matched-recall near-puzzle FP and latency. The primitive works if FP drops by at least 2% with no latency regression, or if latency improves at equal FP.

**Failure mode catalogue:**

- Hidden-rebrand risk: a reviewer could call this graph pooling. It must compute exact connected components, not learned soft assignment or one-hop aggregation.
- Algorithmic risk: deletions are harder than insertions; naive recomputation may dominate on larger boards.
- Chess-value risk: it may mostly learn pawn-structure heuristics and fail the near-puzzle tactical discriminator.

**Status:** proposed

## What I cut

1. **Piece-conditioned legal-move attention** — rejected as masked/sparse attention over a content-dependent legal graph, directly covered by the legal-move graph and sparse attention blocklist.

2. **Ray finite-state automaton scan** — rejected because blocker-reset scans, ray SSMs, directional scans, and obstacle-pooling emitters are already in the project memory.

3. **Poisson-binomial defender-count layer** — rejected as a special case of the Elementary-Symmetric Event Transform; not distinct enough.

4. **Boolean Fourier / Möbius piece-existence transform** — rejected as too close to piece-existence Hessian and counterfactual piece-existence families.

5. **Differentiable minimax / soft Bellman reply operator** — rejected as overlapping reply-channel, regret-saddlepoint, quantifier, and legal-response primitives.

6. **DPP determinant subset-volume pooling** — rejected because it is another high-order set polynomial; ESET is simpler, cheaper, and more incrementally useful.

7. **Learned distance-transform / Voronoi influence primitive** — cut despite plausibility because differentiable distance-transform layers already exist in vision, and chess versions risk becoming renamed attack-distance/ray machinery.

8. **Constraint-projection legality layer** — rejected because differentiable optimization layers such as OptNet already establish the primitive family, and a chess version would mostly encode constraints rather than invent a new operator.

9. **Content-dependent sparse tactical router** — rejected because the core computation is routing/message passing on precomputed or derived edges, which falls under the legal-move graph router and sparse legal transition blocklist.

10. **Color-swap equivariant residual gate** — rejected as a near-duplicate of color-involution gates and CRELU/color-involution message operators.

## References

- Gu, A. and Dao, T. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.” 2023. https://arxiv.org/abs/2312.00752
- Dao, T. and Gu, A. “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.” 2024. https://arxiv.org/abs/2405.21060
- Beck, M. et al. “xLSTM: Extended Long Short-Term Memory.” 2024. https://arxiv.org/abs/2405.04517
- Ye, T. et al. “Differential Transformer.” 2024. https://arxiv.org/abs/2410.05258
- Liu, Z. et al. “KAN: Kolmogorov-Arnold Networks.” 2024. https://arxiv.org/abs/2404.19756
- Behrouz, A. et al. “Titans: Learning to Memorize at Test Time.” 2024. https://arxiv.org/abs/2501.00663
- Dao, T. et al. “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.” 2022. https://arxiv.org/abs/2205.14135
- Zaheer, M. et al. “Deep Sets.” 2017. https://arxiv.org/abs/1703.06114
- Cohen, T. and Welling, M. “Group Equivariant Convolutional Networks.” 2016. https://arxiv.org/abs/1602.07576
- Ying, R. et al. “Hierarchical Graph Representation Learning with Differentiable Pooling.” 2018. https://cs.stanford.edu/people/jure/pubs/diffpool-neurips18.pdf
- Amos, B. and Kolter, J. Z. “OptNet: Differentiable Optimization as a Layer in Neural Networks.” 2017. https://proceedings.mlr.press/v70/amos17a.html
- Woodbury matrix identity overview. https://en.wikipedia.org/wiki/Woodbury_matrix_identity
- Matrix permanent overview. https://en.wikipedia.org/wiki/Permanent_(mathematics)
- Dynamic connectivity overview. https://en.wikipedia.org/wiki/Dynamic_connectivity
