# Five Candidate Neural-Network Primitives for Chess Evaluation

**TL;DR**
- Two of the five proposals (`DeltaCReLU` and `InvolutionReynoldsAffine`) are the strongest bets because they have a clean non-decomposability argument (saturation-aware incremental computation; structured-weight Reynolds projection that is not a composition of linear + Reynolds-postprocess at graph level), provable O(|Δ|) incremental-update, and a 1-GPU falsifiable test.
- Two more (`InputGraphMessagePass`, `DiffUnionPool`) clear the non-decomposable bar through input-defined connectivity / merge-order gradients, but should be flagged as "underexplored primitive for chess" (precedent exists in dynamic-GNN and differentiable-clustering literature).
- The fifth (`PairRayContraction`) is presented as the lowest-confidence proposal — it has the strongest "is this just bilinear+gate?" devil's-advocate exposure and is included to give the project a structural-bias primitive to compare against, with the honest caveat that it may collapse on careful audit.

---

## Key Findings (literature survey, calibration, methodology)

I surveyed: KAN / Kolmogorov–Arnold layers (Liu et al. 2024, arXiv:2404.19756); Monarch matrices and Monarch Mixer (Dao et al. 2022; NeurIPS 2023); Hyena (Poli et al. 2023, ICML); xLSTM (Beck et al. 2024) and xLSTM-7B (2025); Test-Time Training (TTT) layers as architectural primitives (Sun et al. 2020; multiple 2024–2025 follow-ups); Stockfish NNUE / HalfKAv2_hm and the SFNNv11 accumulator (official-stockfish repo, Chessprogramming wiki); group-equivariant CNNs (Cohen & Welling 2016) and recent "Any-Subgroup Equivariant Networks via Symmetry Breaking" (2026 preprint); dynamic GNNs (Zheng et al. arXiv:2404.18211; Feng et al. arXiv:2405.00476); SparseProp event-based RSNN training (Engelken 2023, arXiv:2312.17216); Multiset Transformer / multiset-equivariance (Zhang et al. 2021/2024); hypernetwork review (Chauhan et al. 2024, Springer); NeoNeXt patch-wise matmul primitive (arXiv:2403.11251).

**Calibration takeaway**: The bar for "novel primitive" (GELU, LayerNorm, Mamba selective scan, RWKV receptance) is not just math; it is (i) a named operator (ii) with a custom kernel motif that does not factor into existing torch.nn modules at graph level (iii) reusable as `torch.nn.<NewOp>`. Many candidates that look novel (e.g., "color-swap equivariant linear", "legal-move masked attention") decompose at graph level and are therefore disqualified or demoted.

**Operating constraints** (from project context): single RTX 3070 (8 GiB), 173k positions × 12 epochs, single seed; matched-recall near-puzzle FP rate on CRTK class 1 as the discriminator; incremental update is the structural property to chase.

---

## Details — Five Proposals

### 1. `primitive_delta_crelu`

**Name:** DeltaCReLU — Saturation-Aware Δ-Accumulator with Differentiable Incremental Update

**One-line claim:** A stateful affine+clipped-ReLU operator whose forward and backward cost depend on the size of the input edit, not on the input dimension.

**Mathematical signature:**
- State: `h ∈ ℝ^d` (post-activation), `p ∈ ℝ^d` (pre-activation), `s ∈ {−1,0,+1}^d` (saturation regime: below 0, in-range, above clip max).
- Parameters: embedding table `E ∈ ℝ^{N×d}`, bias `b ∈ ℝ^d`, clip `c > 0`.
- Input: a delta event tape `Δ = [(i_k, σ_k)]_{k=1..m}` with `σ_k ∈ {+1,−1}` (feature insertions/removals).
- Forward: `p ← p + Σ_k σ_k · E[i_k]`; `h = clip(p, 0, c)`; `s` updated channel-wise.
- Backward: `∂L/∂E[i_k] = σ_k · (∂L/∂h ⊙ 1[s ∈ {0}])` masked by the in-range channels at *that* event's epoch. Cross-event credit assignment is recorded through `s_t` per epoch in a compressed tape.

**Why this does not decompose into existing PyTorch ops:**
The closest existing op is `F.embedding_bag` followed by `F.hardtanh`. That composition recomputes the full pre-activation each call; here, the operator's defining contract is that the *post*-activation can be advanced by O(|Δ|+|channels-that-cross-a-saturation-boundary|) work with a correct gradient that integrates the per-channel piecewise-linear regime over the entire Δ-tape. A pure-PyTorch composition does not have access to the saturation-regime tape and cannot avoid full recomputation while keeping a differentiable graph; the custom backward over `(p, s)` history is the non-trivial piece. Stockfish NNUE does the forward part in C++ but provides no autograd path — this primitive turns that engine optimization into a `torch.nn.<NewOp>`.

**Chess-specific motivation:**
HalfKA / HalfKAv2_hm feature deltas between plies are typically 2–4 sparse changes per move (Chessprogramming wiki / Stockfish NNUE docs). The current state-of-the-art (SFNNv11, Sept 2025) explicitly removes feature classes to save "approximately 0.8 threat feature updates per incremental accumulator update" — confirming that update-set size is the operative cost metric. A differentiable version unblocks end-to-end training that *rewards* delta-sparsity rather than treating it as a deploy-time hack.

**Generalisation beyond chess:**
Sparse-event sequences (event cameras, log-stream classification, order-book ticks), any domain where features are inserted/removed by small bounded edits and downstream activations are clip-bounded.

**Complexity (forward, backward, incremental-update):**
- Forward: O(|Δ|·d) vs `F.embedding + F.hardtanh` baseline O((|active features|)·d). Wins when |Δ| ≪ |active|.
- Backward: O(|Δ|·d + |crossing channels|).
- Incremental update on bounded-change input: O(|Δ|·d), *the defining property*.

**Scout-scale falsification test:**
Drop into the i243 HalfKA dual-stream as a replacement for the standard `Linear → ClippedReLU` of the feature transformer. Baseline: same architecture with `nn.EmbeddingBag + nn.Hardtanh`. Train 173k positions × 12 epochs, single RTX 3070. Metric: matched-recall (set recall@0.95 of conv-only parent i193) near-puzzle FP rate on CRTK class 1; secondary metric: wall-clock ms/position at batch=1 with simulated move-by-move evaluation (chain of 30 plies). Works iff: (a) FP rate within 1σ of baseline AND (b) ≥1.5× faster wall-clock on incremental-eval mode. Fails iff incremental-mode wall-clock not faster than recompute-mode, which would prove the saturation-tape bookkeeping eats the savings.

**Failure mode catalogue:**
- Hidden rebrand: a reviewer can claim "this is just `EmbeddingBag` + `Hardtanh` with a manual cache." The non-trivial defense is the joint gradient over the saturation tape — without it the cached version is incorrect when any channel crosses a saturation boundary between deltas.
- Numerically unstable: drifting `p` over hundreds of incremental updates can accumulate FP error; needs periodic full-refresh (Stockfish already does this and it is a hyperparameter, not a primitive flaw).
- Too slow even if it works: backward through a long Δ-tape can blow memory; mitigation is checkpointed refresh every K plies.

**Status:** proposed (novel primitive).

---

### 2. `primitive_involution_reynolds_affine`

**Name:** InvolutionReynoldsAffine — Affine Layer Equivariant under Z₂ ⋉ S_k (color involution × piece-type relabelling)

**One-line claim:** An affine layer whose weight tensor is intrinsically parameterized on the Reynolds-fixed subspace of a non-abelian finite group action, not via post-hoc symmetrization.

**Mathematical signature:**
- Let `G = Z_2 ⋉ S_6` act on the board × piece-type axis by (color-swap involution ι, piece-type permutation π). `ι` flips ranks AND swaps piece colors AND negates the evaluation sign.
- Free parameters live in `ℝ^{|orbits(G)|}`. The full weight `W ∈ ℝ^{m×n}` is materialized as `W = Σ_o θ_o · B_o` where `{B_o}` is a precomputed orthonormal basis of the G-equivariant subspace under the *semi-direct* product action.
- Forward: `y = W x + b`, with `b` similarly constrained.
- Backward: gradients automatically live in the equivariant subspace because parameters are the orbit coefficients.

**Why this does not decompose into existing PyTorch ops:**
A naive composition would be: store `W̃` freely, then compute `W = (1/|G|)·Σ_g g·W̃·g^{-1}` per forward pass — but that costs |G|·m·n per forward (|G| can exceed 1440 with the semi-direct product) and is provably suboptimal in parameter count. The primitive's contract is to parameterize directly on the fixed subspace with a structured basis `{B_o}` so forward cost matches an ordinary Linear and parameter count drops to the orbit dimension. The graph difference is: standard PyTorch has no op that takes orbit-coefficient inputs and emits a structured-equivariant matrix-vector product without materializing the |G|-fold sum.

**Chess-specific motivation:**
Chess has more than dihedral symmetry: also the color-swap involution and piece-type relabelling under restricted subsets (e.g., minor pieces). G-CNN literature (Cohen & Welling 2016) covers translations + reflections + rotations; the semi-direct product of an involution group with a permutation subgroup of S_6 is not standard. The user's project context explicitly flags this under-exploitation.

**Generalisation beyond chess:**
Any domain with a non-abelian finite symmetry group acting on a labelled feature axis: molecular property prediction (atom type permutation × chirality involution), particle physics (charge conjugation × particle-type permutation), card games (suit permutation × red/black involution).

**Complexity (forward, backward, incremental-update):**
- Forward: O(m·n) — identical to `nn.Linear`. Parameter count: |orbits| ≪ m·n.
- Backward: O(m·n) with O(|orbits|) parameter gradient.
- Incremental update: not applicable (no temporal state).

**Scout-scale falsification test:**
Drop into the i242 chess-decomposed attention path as a replacement for the value/output projections. Baseline: same architecture with standard `nn.Linear`. Train 173k × 12, single seed. Primary metric: matched-recall near-puzzle FP rate on CRTK class 1. Works iff: parameter count drops ≥3× AND FP rate is no worse. Fails iff: subgroup is too restrictive (validation loss strictly worse), which would prove the prior is mis-specified for chess.

**Failure mode catalogue:**
- Hidden rebrand: "this is just weight tying." Defense: the orbit structure is not a single tied pattern but a basis of dimension |orbits|, which is provably the unique equivariant subspace under the non-abelian semi-direct product — weight tying is a special case for abelian groups.
- Numerically unstable: basis `B_o` may be ill-conditioned for large |G|; mitigation is orthonormalize at construction time.
- Too slow even if it works: orbit-basis computation is one-time at init, but if mistakenly recomputed per forward, costs explode.

**Status:** proposed (underexplored primitive for chess; the math is standard in equivariant-ML literature but no `torch.nn.GroupEquivariantLinear` ships, and the specific group Z₂ ⋉ S_k has not been used as a chess primitive — escnn focuses on continuous groups and dihedral discrete groups).

---

### 3. `primitive_input_graph_message_pass`

**Name:** InputGraphMessagePass — Message Passing over Per-Sample Content-Defined Adjacency

**One-line claim:** A message-passing op whose adjacency tensor is computed from the input on the fly and through which gradients flow back to whatever produced the adjacency.

**Mathematical signature:**
- Input: node features `X ∈ ℝ^{N×d}`, edge logits `e_{ij}(X) ∈ ℝ` produced by a small differentiable scorer; for chess, masked to legal-move pairs.
- Adjacency: `A_{ij} = top-k(softmax_j(e_{ij}))_{ij}` with straight-through estimator (STE) for the top-k indicator.
- Forward: `y_i = Σ_{j: A_{ij}>0} A_{ij} · φ(x_i, x_j)`.
- Backward: gradient flows into both `φ` parameters AND into the edge-logit scorer through the STE.

**Why this does not decompose into existing PyTorch ops:**
The closest comparator is masked attention: `softmax(QK^T + M)V`. There, M is an *input-independent* mask (or a function of position, not content). Here the *support* of A is content-dependent and the backward differentiates through the top-k operator — `torch.topk` returns indices with no backward through index selection. The primitive's required behavior is a graph-level scatter where the scatter pattern is itself a differentiable function of the input, which no torch.nn module provides.

**Chess-specific motivation:**
The sparse legal-move graph changes per position; a fixed mask cannot capture it. A move from f3 only "exists" if no piece blocks the diagonal, which is a content-dependent connectivity, not a positional mask.

**Generalisation beyond chess:**
Dynamic-graph learning broadly (Zheng et al. 2024 survey on dynamic GNNs), scene-graph generation, biological pathway inference where edges are inferred from node states.

**Complexity (forward, backward, incremental-update):**
- Forward: O(N · k · d) where k is top-k degree, vs full attention O(N²·d). Wins for k ≪ N.
- Backward: O(N · k · d) plus O(N²) for the edge scorer (this is the bottleneck; can be downsampled).
- Incremental update: O(Δ-affected edges · k · d) when only a few squares change — chess-favorable.

**Scout-scale falsification test:**
Drop in as a replacement for one self-attention block in i242. Baseline: same block as masked-attention with legality mask. Metric: matched-recall near-puzzle FP rate on CRTK class 1. Works iff: matches baseline AND ≥1.3× faster forward at inference (k=8, N=64). Fails iff: STE through top-k destabilizes training (validation loss diverges within 4 epochs).

**Failure mode catalogue:**
- Hidden rebrand: "this is sparse attention with a learned mask." Defense: standard sparse/longformer attention uses a *fixed* sparsity pattern; here the support is per-input and differentiable, and the operator is intended as a single named torch.nn module rather than a recipe.
- Numerically unstable: STE bias can accumulate; mitigation via Gumbel-top-k relaxation with annealed temperature.
- Too slow even if it works: per-sample adjacency precludes batched GEMM speedups on small N; on N=64 (chess) this is fine, on N=10⁴ it dies.

**Status:** proposed (underexplored primitive for chess; dynamic-GNN literature has many specific instances but no canonical torch.nn op).

---

### 4. `primitive_diff_union_pool`

**Name:** DiffUnionPool — Differentiable Union-Find Pooling with Merge-Order Gradients

**One-line claim:** A pooling operator that performs soft Kruskal-style cluster merges over a per-input weighted edge list and back-propagates through merge order.

**Mathematical signature:**
- Input: node features `X ∈ ℝ^{N×d}`, edge weights `w_e ∈ ℝ^{|E|}` (for chess: edges = pairs of squares connected by attack or defense relations, weights from a learned scorer).
- Process: sort edges by w; iteratively merge endpoints if not already in same component (path compression, union by rank); record merge tree T.
- Output: per-cluster pooled features `Y ∈ ℝ^{K×d}` where K is the number of components at a learned threshold τ; cluster membership is a soft assignment from a sigmoid of the merging-edge weights.
- Backward: gradients propagate (i) through the soft cluster-assignment sigmoid into `w_e`, and (ii) through the pool aggregation into `X`, with custom backward over the merge tree.

**Why this does not decompose into existing PyTorch ops:**
The closest comparator is `torch_scatter.scatter_mean` with a precomputed `cluster_id`. That requires cluster IDs to be input — here they are *computed* by a union-find subroutine inside the op and the merge-order dependency means a small change in `w_e` flips the cluster topology in a piecewise-constant way, which `scatter_mean` cannot differentiate. Differentiable-clustering work (e.g., deep K-means, soft DBSCAN) is closest, but no torch.nn op implements union-find with α(N) complexity and a merge-tree gradient.

**Chess-specific motivation:**
Chess positions have natural connected components: pawn chains, attack/defense webs, king-safety clusters. A primitive that surfaces them as a differentiable pooling operation matches a class of human-expert features that current convolutional and attention nets must rediscover from scratch.

**Generalisation beyond chess:**
Scene-graph partitioning, biological network module discovery, agglomerative clustering as a layer in any tabular/graph model.

**Complexity (forward, backward, incremental-update):**
- Forward: O(|E| · α(N) · d) where α is the inverse Ackermann — practically linear.
- Backward: O(|E| · log N · d) due to merge-tree traversal.
- Incremental update on bounded-change input: O((edges affected by Δ) · α(N)) — partially favorable but the merge tree can need a localized rebuild.

**Scout-scale falsification test:**
Append as a global-pooling stage of i193 (conv-only baseline) in place of `nn.AdaptiveAvgPool2d`. Edge scorer = learned MLP over piece-type pairs. Metric: matched-recall near-puzzle FP rate on CRTK class 1. Works iff: outperforms global average pool by ≥10% relative FP-rate reduction AND interpretable cluster heatmaps emerge. Fails iff: cluster assignments collapse to all-one or singletons within 6 epochs.

**Failure mode catalogue:**
- Hidden rebrand: "this is hierarchical agglomerative clustering with soft assignments." Defense: the precise combo of α(N) forward, merge-tree backward, and the merge-order STE has no off-the-shelf PyTorch equivalent.
- Numerically unstable: discrete merge ordering creates flat-then-discontinuous gradient regimes; mitigation via Sinkhorn or Gumbel-sort relaxation on the edge ranking.
- Too slow even if it works: custom CUDA kernel for union-find with path compression is needed to hit α(N) — pure-Python forward would dominate runtime.

**Status:** proposed (underexplored primitive for chess; differentiable clustering and soft union-find appear in graph-pooling literature but not as a canonical torch.nn primitive).

---

### 5. `primitive_pair_ray_contraction`

**Name:** PairRayContraction — Geometry-Indexed 3-Tensor Contraction with Content-Gated Support

**One-line claim:** A pairwise feature operator whose 3-way contraction support is determined by the sliding-piece ray geometry between square pairs and gated by intervening-piece occupancy.

**Mathematical signature:**
- Input: square features `X ∈ ℝ^{64×d}`, occupancy gate `g ∈ {0,1}^{64}` (or soft).
- For each ordered pair (i, j), define `R(i,j) ⊂ {0,…,63}` as the set of squares on the chess ray from i to j (empty if i,j are not on a common rank/file/diagonal), and let `m(i,j,X,g) = ∏_{k ∈ R(i,j)} (1 − g_k)` (soft "is the ray clear").
- Output: `Y[i,j,:] = m(i,j) · Σ_{k ∈ R(i,j)} T_k · (x_i ⊗ x_j)` where `T_k ∈ ℝ^{d×d×d}` is a learned tensor per intervening square (or, more parsimoniously, a single T modulated by k's positional embedding).
- Backward: gradients flow through `m` into `g`, through `T_k` parameters, and through `x_i, x_j`.

**Why this does not decompose into existing PyTorch ops:**
The combination of (a) a contraction support set R(i,j) that is a fixed *but non-trivially structured* function of (i,j), (b) a multiplicative gate over that support that is content-dependent, and (c) the explicit (i,j,k) 3-way contraction is not equivalent to bilinear (`nn.Bilinear` only handles `x_i ⊗ x_j` with no third-axis structured support) and not equivalent to attention (no softmax, no Q/K). A naive PyTorch implementation requires explicit index-gather per (i,j), which is precisely the missing primitive.

**Chess-specific motivation:**
Sliding-piece tactics (skewers, pins, batteries, x-rays) are defined by exactly this ray geometry with blocker gating. A standard 8×8 conv has the wrong receptive-field shape; standard attention is content-blind to the ray structure. This primitive injects the inductive bias.

**Generalisation beyond chess:**
This one is chess-leaning. Closest analogs: line-of-sight reasoning in 3D scene graphs, raycasting features in occupancy-grid networks. Mark as **chess-leaning** with weak transfer.

**Complexity (forward, backward, incremental-update):**
- Forward: O(64² · |R̄| · d²) where |R̄| ≤ 7 is the max ray length, vs full bilinear O(64² · d²). Higher constant factor.
- Backward: same order.
- Incremental update: O(|Δ-pairs| · |R̄| · d²); when a piece moves, only the pairs sharing a ray with the moved square need recomputation — chess-favorable.

**Scout-scale falsification test:**
Add as a single block atop i193 conv features. Baseline: same depth with an extra `nn.Bilinear` block of matched parameter count. Metric: matched-recall near-puzzle FP rate on CRTK class 1, with stratified analysis on positions containing pins/skewers. Works iff: pin/skewer-stratified FP rate improves ≥15% relative AND aggregate FP rate not worse. Fails iff: aggregate FP rate worse — would prove the inductive bias is too narrow.

**Failure mode catalogue:**
- Hidden rebrand: "this is bilinear with a precomputed ray mask × occupancy gate." This is the strongest objection. The defense relies on (a) the explicit 3rd-axis indexed contraction over `T_k` (not just a 2-way bilinear with a scalar mask) and (b) the soft occupancy gate being differentiable through R(i,j)'s support. If T_k is collapsed to a single tensor T modulated only positionally, this proposal **does** collapse to "bilinear with structured mask" — drop it.
- Numerically unstable: multiplicative chain `∏(1−g_k)` can vanish on long rays; mitigation via log-domain accumulation.
- Too slow even if it works: explicit per-(i,j) gather kills throughput; needs a custom CUDA kernel or it is dead on arrival.

**Status:** proposed (lowest confidence — flagged as borderline-decomposable; include only if devil's-advocate audit on `T_k` indexing survives).

---

## Recommendations (staged, decision-ready)

**Stage 1 (immediate, 1–2 weekends on the 3070):** Implement and falsify `DeltaCReLU` (proposal 1) first. It has the cleanest mapping to the project's empirical priors (HalfKA accumulator update is exactly its motivation), the test harness is i243, and a single ablation against `EmbeddingBag + Hardtanh` cleanly decides novelty-vs-rebrand. **Threshold to continue:** ≥1.5× wall-clock speedup at batch=1 in incremental-eval mode and FP rate within 1σ of baseline.

**Stage 2 (if Stage 1 succeeds):** Implement `InvolutionReynoldsAffine` (proposal 2). It is parameter-only (no temporal state), drops into i242 with one line, and its falsification is unambiguous: parameter count must drop ≥3× without hurting matched-recall FP rate. **Threshold to continue:** parameter savings realized and FP rate not strictly worse.

**Stage 3 (one of the harder two):** `InputGraphMessagePass` (proposal 3) if the project is moving toward attention-based architectures (it is the natural successor to i242), `DiffUnionPool` (proposal 4) if moving toward interpretability and pawn-structure features. Both require custom backward kernels; budget ≥1 week of engineering.

**Stage 4 (only if motivated):** Audit `PairRayContraction` (proposal 5) by writing the explicit `T_k` indexed contraction and checking whether the gradient graph genuinely differs from `nn.Bilinear` + mask. If not, drop it and do not publish.

**Thresholds that change the plan:**
- If `DeltaCReLU` does NOT speed up inference, abandon the incremental-update angle entirely — the gradient bookkeeping costs more than it saves and the user's "HalfKA O(1) is the property to chase" thesis was probably defeated by autograd overhead at scout scale.
- If `InvolutionReynoldsAffine` HURTS validation loss, the chess semi-direct group is too restrictive a prior at 173k positions; relax to color-only involution before discarding.
- If neither survives, the structural bet for the project should shift from primitives to data scale (the i242 lesson) rather than to new primitives.

---

## What I Cut (self-audit)

- **"Color-Swap Equivariant Linear" as a standalone primitive** — cut: provably decomposable into `Linear` + Reynolds postprocess at graph level. Subsumed into `InvolutionReynoldsAffine` only because the *parameterization* on the orbit basis is the non-decomposable piece, not the color-swap itself.

- **"Legal-Move Masked Attention"** — cut: this is just `softmax(QK^T + M)V` with a structured M. Decomposable. Explicitly flagged as an anti-example by the user's spec.

- **"Spline-Edge KAN Layer for chess"** — cut: KAN (Liu et al. 2024, arXiv:2404.19756) decomposes into per-edge spline evaluation + sum, both supported by `torch.nn.Functional` and B-spline ops. It is a parameterization, not a new operator at graph level; KAN-GNNs (Li et al., Nature Mach. Intell. 2025) inherit the same property.

- **"Selective-Scan SSM applied to MCTS-PV sequences"** — cut twice: (a) Mamba is already a published primitive (Gu & Dao 2023) so it is not new; (b) PVs are verification metadata which the project rules explicitly forbid as input features.

- **"Hypernetwork as a chess primitive that generates the evaluator's weights from the king square"** — cut: hypernetworks (Chauhan et al. 2024 review, Springer) are a known primitive class; a king-conditioned hypernet is an instantiation, not a new operator. Additionally, HalfKA *already* king-conditions the feature transformer indexing — so this would not be net-new at graph level for chess.

---

## Caveats

- **Novelty bar honesty.** Three of the five proposals are flagged as "underexplored primitive for chess" rather than "new primitive" because precedent exists in adjacent literature (dynamic GNNs, equivariant ML, differentiable clustering). Per the user's spec this is the correct calibration; the project should not market them as net-new operators without further literature audit.
- **Predicted-not-measured.** Every speedup and FP-rate figure cited above is a target threshold for the falsification test, not a measured result. No empirical claim is made until the 3070 run is executed.
- **Subagent budget not used.** The system prompt described a `run_blocking_subagent` and an `enrich_draft` tool; neither was present in the runtime tool palette, so the integration/enrichment passes specified in the research process could not be executed. I disclose this rather than fabricate enrichment.
- **Search budget exhausted at 12 queries** of the planned 20; literature coverage on (a) "differentiable union-find as a layer" and (b) "input-defined adjacency MP as a torch.nn primitive" is the thinnest. If either of those turns out to have a direct 2024–2026 NeurIPS/ICLR paper proposing them as a named primitive, demote proposals 3 and 4 from "underexplored" to "existing primitive applied to chess" and the project should cite that work.
- **Devil's-advocate residual risk.** Proposal 5 (`PairRayContraction`) has the highest probability of collapsing to "bilinear + structured mask" on careful audit. It is included for completeness but should be the first to be dropped if the implementation does not preserve a true 3-way indexed contraction over `T_k`.
- **Reproducibility.** All proposals respect the single-RTX-3070, 173k-position, single-seed, 12-epoch constraint, and none requires Stockfish scores, PVs, node counts, or verification metadata in the compute graph.