# Five Candidate Neural-Network Primitives for Chess Evaluation

**TL;DR**
- The single most promising direction is a *primitive that generalises HalfKA's O(|Δ|) accumulator update into a typed, differentiable, stateful operator* (`primitive_sda` below); it is the only proposal that directly attacks the 2×-wall-clock → ~30 Elo lever and has a credible scout-scale falsification test.
- Of the five proposals, two (`primitive_sda`, `primitive_csg`) are claimed as genuinely structurally new operators; the other three (`primitive_cgec`, `primitive_rstm`, `primitive_sessm`) are best honestly framed as **underexplored primitives** with close 2023–2025 literature precedent (G-CNNs / Cohen-Welling; Bai et al. DEQ + reversible nets; Gu & Dao Mamba/S6) that have never been instantiated as the specific chess-shaped operator described.
- All five must be validated against the matched-recall near-puzzle FP rate on CRTK class 1 — not aggregate PR AUC — at the 173k-position / 12-epoch / single-RTX-3070 scout scale, with the explicit prediction that genuine architectural lift will appear in hard-negative discrimination rather than in easy-negative throughput.

---

## Key Findings

The chess-evaluation primitive landscape decomposes into two non-overlapping deficiencies in `torch.nn`:

1. **No primitive captures "forward cost depends on input *change* rather than input *size*."** HalfKA's accumulator is hand-coded in C++ SIMD inside Stockfish; PyTorch's `nn.EmbeddingBag` recomputes the bag every forward. No `torch.nn` operator carries an explicit *prior-state* tensor that is updated by a sparse delta with a well-defined gradient threaded through a stateful session. This is the single largest inference-speed gap and the most defensible novelty axis.

2. **No primitive carries a content-determined sparse connectivity pattern as its compute graph.** Masked attention always pays O(n²) FLOPs even when the mask is 99% zero; Routing Transformer materialises clusters but still pays O(n^1.5·d); GATs assume a *fixed* graph. The chess legal-move graph is content-determined, sparse (~30 edges/board on ~64 squares), and changes every position — the "right" primitive is a gather/scatter whose support is a hard, input-conditional set, with gradients flowing only along the selected edges.

Secondary opportunities (chess-group equivariance, reversible-tree state for MCTS, event-driven SSMs) are real but should be marked as underexplored rather than genuinely new operators, because each has a close 2016–2024 precedent that a reviewer will surface.

The Stockfish NNUE/HalfKA reference confirms that ~0.07% input density and ~30 active features per position is the actual operating regime (cf. Stockfish NNUE architecture documentation, ~40,960–60,000 features, ~30 active, SFNNv4–v13). Mamba/S6 (Gu & Dao 2023) provides the closest "input-dependent dynamics" analogue but processes all L tokens; Routing Transformer (Roy et al. 2020) is the closest content-routed analogue but still pays O(n^1.5·d) and uses online k-means rather than a discrete legal-move predicate; G-CNNs (Cohen & Welling 2016) cover D₄ but not the colour-swap × piece-type involution.

---

## Details

### primitive_sda

**Name:** Sparse-Delta Accumulator (Differentiable Stateful HalfKA Generalisation)

**One-line claim:** A persistent, reversible state vector that is updated by a typed (add-index, remove-index) sparse delta in O(|Δ|·d) per call, with gradients defined over the delta-stream.

**Mathematical signature:**
f : (h_{t-1} ∈ ℝ^d, I⁺_t ∈ ℕ^{k⁺}, I⁻_t ∈ ℕ^{k⁻}, W ∈ ℝ^{V×d}) → h_t ∈ ℝ^d
Forward: h_t = h_{t-1} + Σ_{i∈I⁺_t} W[i] − Σ_{j∈I⁻_t} W[j], with an attached *session tape* τ_t = (τ_{t-1}, I⁺_t, I⁻_t).
Backward: ∂L/∂W[v] = Σ_{t: v∈I⁺_t} (∂L/∂h_t) − Σ_{t: v∈I⁻_t} (∂L/∂h_t), recovered by replaying τ — never materialising the full t × d activation buffer.

**Why this does not decompose into existing PyTorch ops:** `nn.EmbeddingBag` is stateless: every call rebuilds the bag from scratch in O(n·d). `nn.Linear` on a one-hot vector is O(V·d). Neither carries the *prior state* h_{t-1} as a typed primitive input, nor exposes a backward that threads through a session tape. The closest analogue, RNN cells, perform a full d×d matmul per step — O(d²), not O(|Δ|·d). The primitive's compute graph has no edge from t-1 to t through a weight matrix — only an additive splice — which is what makes the gradient sparse and locally recoverable.

**Chess-specific motivation:** HalfKA changes ≤4 input features per move (one piece moves; one capture; one castle = at most two pieces; promotion adds/removes ≤2). The current Stockfish implementation hand-rolls this in C++; no PyTorch operator captures it. Exposing it as a primitive lets researchers compose it with downstream non-linear heads while preserving the make/unmake speed property.

**Generalisation beyond chess:** Online recommender state (user-impression deltas), event-stream sensor fusion, incremental scene-graph features in robotics, KV-cache delta updates in autoregressive decoding.

**Complexity (forward, backward, incremental-update):**
- Forward: O((k⁺+k⁻)·d) vs `nn.EmbeddingBag` O(n·d) and `nn.Linear` O(V·d)
- Backward: O((k⁺+k⁻)·d) per step, amortised over session
- Incremental update on a bounded-change input: O(|Δ|·d) — this is the defining property

**Scout-scale falsification test:** Replace the first layer of the existing 234-architecture scout's best NNUE-style baseline with `SparseDeltaAccumulator(V=40960, d=256)`. Train on 173k positions × 12 epochs, single seed, single RTX 3070. Metric: (a) wall-clock NPS in a 100-position eval-only benchmark vs. the current `nn.EmbeddingBag` baseline (must show ≥1.5× speedup with identical FP32 outputs), and (b) **matched-recall near-puzzle FP rate on CRTK class 1** within ±1 absolute point of the baseline (must not regress). "Works" = both pass; "fails" = either regression in FP rate or <1.2× speedup.

**Failure mode catalogue:**
- Hidden rebrand of `nn.EmbeddingBag`: refutable because EmbeddingBag has no state input and recomputes O(n·d).
- Numerical drift across a long session (h_t = h_0 + Σ deltas in FP16 will accumulate error); mitigated by periodic FP32 refresh, but reviewer will demand a bound.
- Backward through long sessions can blow up activation memory if tape is naive; needs reversible-tape implementation to be honest.

**Status:** proposed

---

### primitive_csg

**Name:** Content-Sparse Gather (Hard Input-Conditional Edge Aggregation)

**One-line claim:** A gather/aggregate operator whose edge set is a hard, discrete function of the input — compute graph carries only |E(x)| operations, not n².

**Mathematical signature:**
f : (X ∈ ℝ^{B×n×d}, π : ℝ^{B×n×d} → 𝒫([n]×[n]), W ∈ ℝ^{d×d}) → Y ∈ ℝ^{B×n×d}
E_b = π(X_b) ⊆ [n]×[n], with |E_b| ≤ K(x); Y_b[i] = Σ_{(i,j)∈E_b} W X_b[j].
Gradient through X via standard scatter-add on the realised edges; gradient through π via a straight-through estimator on a learned scoring function s_θ(X)_{ij} that selects edges (top-K with hard threshold).

**Why this does not decompose into existing PyTorch ops:** Masked attention (`F.scaled_dot_product_attention` with attn_mask) still allocates an n×n score tensor and pays O(n²d) FLOPs regardless of mask density. Routing Transformer (Roy et al. 2020, *TACL*) reduces to O(n^1.5·d) but routes via online k-means over content, not via a hard external predicate. PyG's message passing supports dynamic graphs but only as Python-level iteration, not as a fused primitive with a content-conditional edge selector and straight-through gradient. The compute graph here literally has |E| edges, not n²; that is the structural distinction.

**Chess-specific motivation:** The legal-move graph on 64 squares has ~30–40 edges and changes every position. A primitive whose FLOPs scale with legal-move count, not square count, would let strong-move reasoning concentrate compute exactly where chess concentrates information (i.e., the captures and tactical motifs).

**Generalisation beyond chess:** Dynamic scene graphs in video understanding, particle interactions in molecular dynamics (cutoff radius), sparse-event social-graph reasoning, code-AST edge prediction.

**Complexity (forward, backward, incremental-update):**
- Forward: O(|E(x)|·d) vs masked-attention O(n²·d), Routing Transformer O(n^1.5·d)
- Backward: O(|E(x)|·d) plus a small O(n²) score-tensor pass *only if* π is learned end-to-end; if π is supplied externally (legal-move bitboard), backward is exactly O(|E|·d)
- Incremental update: O(|ΔE|·d) when the edge set itself only changes slightly between consecutive ply — applicable in MCTS branch propagation

**Scout-scale falsification test:** Drop CSG into the i242 decomposed-attention follow-up replacing one self-attention layer. Use the externally supplied legal-move bitboard as π (so the test isolates the gather, not the selector). Train at 173k × 12 epochs. Metric: **matched-recall near-puzzle FP rate on CRTK class 1** must lift by ≥0.5 absolute point over the attention-only i242 ablation, while wall-clock per-position eval must not exceed 1.2× the attention baseline. Aggregate PR AUC is not allowed to count.

**Failure mode catalogue:**
- Reviewer will say "this is just sparse-matrix attention with a precomputed mask"; rebuttal must show *fused* O(|E|·d) kernel, not a masked O(n²) kernel that throws away results.
- Straight-through estimator for π may bias gradients; mitigation is to first test with an externally supplied π.
- Sparse-CUDA kernels at |E|≈30 may be slower than dense O(n²) on small n=64 due to launch overhead; a small-n cutoff may be required, which would defeat the chess use case if poorly tuned.

**Status:** proposed

---

### primitive_cgec

**Name:** Chess-Group Equivariant Contraction (Colour-Coupled Permutation-Equivariant Linear)

**One-line claim:** A linear operator parameter-tied to be equivariant under the product group G = D₄ × Z₂^{colour-flip-with-board-flip} × S_p (piece-type permutations preserving chess legality classes).

**Mathematical signature:**
f : ℝ^{B×64×C} → ℝ^{B×64×C'}, with weight tensor W ∈ ℝ^{C×C'×64×64} constrained to the G-invariant subspace W = Π_G W, implemented by Schur-decomposing C and C' into irreps of G. Equivariance: f(ρ_in(g)·X) = ρ_out(g)·f(X) for all g ∈ G.

**Why this does not decompose into existing PyTorch ops:** `nn.Conv2d` is translation-equivariant; G-CNNs (Cohen & Welling 2016) cover the D₄ piece of G but not the colour-flip-coupled-to-board-flip involution that also permutes feature channels (white-king channel ↔ black-king channel). No PyTorch operator carries that coupled spatial-and-channel involution as a parameter-tying constraint. The closest, `e2cnn`/`escnn`, supports finite groups acting on spatial dims but not the simultaneous coupled feature-channel action required by chess colour symmetry.

**Chess-specific motivation:** Standard data augmentation by colour-swap gives the network the symmetry as a regulariser, but does not *constrain* the parameter count. CGEC bakes the symmetry into the operator, cutting parameter count by ~|G|× and reducing the data-hunger flagged by the i242 result that attention underperformed conv at 173k positions. This is a direct attack on the small-scale falsification regime.

**Generalisation beyond chess:** Other turn-based zero-sum games with colour involutions (Go, Othello, Shogi), bipartite molecular graphs with charge symmetry, two-team sports analytics.

**Complexity (forward, backward, incremental-update):**
- Forward: same FLOPs as `nn.Linear`/`nn.Conv2d` at inference (G-tying is a parameter constraint, not extra ops); ~|G|× fewer *parameters* (G has order 16 = 8 × 2 for D₄ × colour, before piece-permutation)
- Backward: same as Conv2d plus a cheap O(|G|·C·C') projection of weight grads onto the invariant subspace
- Incremental update: not applicable

**Scout-scale falsification test:** Replace the conv stem of i193 with a CGEC stem of equal parameter budget (so CGEC has more *effective* capacity since fewer parameters are wasted on duplicating colour-flipped features). Train at 173k × 12 epochs. Metric: matched-recall near-puzzle FP rate on CRTK class 1 must lift by ≥0.5 absolute point AND test-set loss curve must be visibly below the i193 baseline by epoch 6 (data efficiency claim). If only aggregate PR AUC moves but CRTK class 1 does not, the primitive does *not* count as working.

**Failure mode catalogue:**
- This is honestly an **underexplored primitive** — G-CNN math is 2016 and `escnn` exists; the novelty is only the specific chess group, particularly the colour-channel-coupled involution. Reviewer will rightly demand we frame it as "G-CNN with G=chess group" rather than a new primitive class.
- The piece-type permutation subgroup S_p does *not* actually preserve chess semantics (knight ≠ bishop), so the S_p factor must be restricted to trivial — leaving only D₄ × Z₂, which may be too small a group to give a meaningful parameter saving.
- Numerical: weight projection onto invariant subspace each step can drift if implemented as a soft penalty; must use hard reparameterisation.

**Status:** proposed (underexplored)

---

### primitive_rstm

**Name:** Reversible Tree-State Monad (Make/Unmake as a Typed Primitive)

**One-line claim:** A primitive exposing a pair (push : (h, a) → h', pop : (h', a) → h) of exact bijective functions, enabling O(1) MCTS branch backtracking on a learned state.

**Mathematical signature:**
f, f⁻¹ : ℝ^d × 𝒜 → ℝ^d, with f⁻¹(f(h, a), a) = h *exactly* in floating point (not only as a gradient identity). Implemented as a stack of additive coupling layers (RealNVP-style) parameterised on a; the primitive bundles the pair (f, f⁻¹) plus a typed "session stack" of applied actions so a search tree can be traversed without recomputing h from the root.

**Why this does not decompose into existing PyTorch ops:** Reversible networks (RevNet, i-RevNet, MEMnet) exist as *training-memory* tricks: forward and backward both walk the same linear chain. None expose `(forward, inverse, action)` as a single typed primitive supporting *non-linear* tree traversal where pop is called on demand from any node. The compute graph of RTSO has the shape of a search tree, not a chain — which `nn.Module` cannot natively represent without per-call Python-level inversion.

**Chess-specific motivation:** MCTS expands branches, then backtracks. If the value-network's input features are derived from a learned state h, then make/unmake on h must be exact, or the search tree's h diverges from the root. NNUE solves this for the accumulator by linearity (`primitive_sda` above); RTSO extends the property to *non-linear* learned states.

**Generalisation beyond chess:** Tree search in code synthesis, theorem proving, planning in robotics with discrete action sets, game-tree search in any deterministic environment.

**Complexity (forward, backward, incremental-update):**
- Forward (push): O(d) per move, same as a coupling-layer flow
- Inverse (pop): O(d) per move, *exact*
- Backward: O(d · depth) using the inverse to reconstruct intermediate activations on demand — O(1) memory in d
- Incremental update on bounded-change input: O(d) per move — this is the defining property in a tree, not a chain

**Scout-scale falsification test:** Build a 4-coupling-layer RTSO over d=128 conditioned on action ∈ {from, to, piece}. Train it to predict Stockfish eval at depth-2 leaves of a small MCTS-style rollout (held within 173k positions; rollouts of depth ≤4). Metric: leaf-eval MSE relative to a non-reversible baseline of the same depth/width; **and** wall-clock per leaf in a 1000-leaf benchmark (must show ≥1.3× speedup vs. recomputing-from-root baseline). Failure: no speedup or MSE regresses by >5%.

**Failure mode catalogue:**
- **Underexplored primitive**: reversible nets are 2017–2019; the novelty is only the tree-state interface, which a reviewer may call a wrapper, not a primitive.
- Coupling layers are weaker function approximators than free MLPs; primitive may underfit chess evaluation.
- Numerical: composing many coupling layers in FP16 may break exact invertibility; needs FP32 state with FP16 weights, or periodic recomputation.

**Status:** proposed (underexplored)

---

### primitive_sessm

**Name:** Sparse-Event State-Space Operator (Event-Driven S6 with Closed-Form Jump-Ahead)

**One-line claim:** A selective state-space recurrence whose state update is triggered only by sparse *events*; empty steps advance the state by a closed-form A^Δt jump in O(log Δt) time.

**Mathematical signature:**
f : (h_{t-1} ∈ ℝ^N, e_t ∈ ℝ^d ∪ {∅}, Δt ∈ ℕ) → h_t ∈ ℝ^N
If e_t = ∅: h_t = Ā^{Δt} h_{t-1} (matrix power via repeated squaring on a structured Ā — e.g., HiPPO or diagonal plus low-rank)
If e_t ≠ ∅: h_t = Ā^{Δt} h_{t-1} + B(e_t) · e_t, with B input-dependent as in S6.
Output y_t = C(h_t) emitted only on event steps.

**Why this does not decompose into existing PyTorch ops:** Mamba/S6 (Gu & Dao 2023) computes a parallel scan over *every* token in a length-L sequence — its complexity is O(L). SE-SSM treats Δt as a runtime input and exploits Ā's structure (diagonal in Mamba-2) to perform Ā^{Δt} in O(log Δt) per skip, so a sequence with M events over length L costs O(M log(L/M)·N) rather than O(L·N). No PyTorch primitive carries (state, event, delay) as a unified signature; the existing `mamba_ssm` kernel always iterates one token per step.

**Chess-specific motivation:** Sparse-move sequences (opening lines, endgame studies) are event streams over a much larger latent "ply clock." Capturing the latent dynamics of who-moves-when as a continuous-time SSM, with event-only updates, matches chess's structure: positions are interesting at branch points, not between them. Also fits MCTS visit-count streams.

**Generalisation beyond chess:** Asynchronous event-camera vision, irregular-time-series finance/healthcare, neuromorphic spike processing, sparse-edit document streams.

**Complexity (forward, backward, incremental-update):**
- Forward: O(M log(L/M) · N) vs Mamba O(L · N) and self-attention O(L²·d)
- Backward: O(M log(L/M) · N) via reverse parallel scan
- Incremental update: O(log Δt · N) per new event — directly applicable to MCTS sequential rollouts

**Scout-scale falsification test:** Use the move-history feature of i243 (≤40 ply) as the event stream feeding an SE-SSM (N=64, d=128) head; freeze HalfKA accumulator; replace the existing positional-history layer with SE-SSM. 173k × 12 epochs. Metric: matched-recall near-puzzle FP rate on CRTK class 1 vs. a fixed-time Mamba baseline of equal parameter count. Must lift ≥0.3 absolute points OR show ≥1.5× wall-clock speedup on a 1000-position move-history benchmark.

**Failure mode catalogue:**
- **Underexplored**: Mamba/S6's per-token Δt parameter already lets the model effectively "skip" tokens softly; reviewer will argue SE-SSM is just Mamba with an external Δt. Defence is the closed-form Ā^{Δt} algorithm and the API that doesn't process empty steps at all.
- Matrix power of Ā may be unstable if eigenvalues exit the unit circle; needs HiPPO-style or diagonal-stable parameterisation.
- For dense chess move-streams (one move per ply), event sparsity is low and the constant factor wins go away.

**Status:** proposed (underexplored)

---

## Recommendations

Stage the work in this order, with explicit kill criteria:

1. **First (week 1–2): build `primitive_sda` as a CUDA kernel.** This has the largest potential Elo payoff (the HalfKA generaliser), is the most defensibly novel of the five, and the scout-scale falsification (≥1.5× wall-clock, no CRTK class 1 regression) is mechanical. If it cannot beat 1.2× speedup vs. `nn.EmbeddingBag` for k⁺+k⁻ ≤ 4, kill it and reuse the kernel for inference of the existing HalfKA baseline.
2. **Second (week 3–4): build `primitive_csg` with externally supplied π.** Isolate the gather from the selector; this is the cleanest test of whether content-conditional sparse connectivity actually lifts CRTK class 1 hard-negative discrimination. Threshold: ≥0.5 absolute point lift in matched-recall near-puzzle FP rate at no worse than 1.2× wall-clock. Below 0.3 lift → kill.
3. **Third (week 5–6): build `primitive_cgec`.** Cheap to implement (parameter projection only), no kernel work. Decision rule: if CRTK class 1 lifts and loss curve is below i193 by epoch 6, keep; if only aggregate PR AUC moves, kill — the project rubric explicitly says easy-negative lifts are uninteresting.
4. **Hold `primitive_rstm` and `primitive_sessm` for a v2 pass** after MCTS integration of the engine is ready; until then there is no test harness for the tree-monad property or for the event-stream property that distinguishes them from existing reversible/SSM baselines. Don't burn the single seed on them yet.
5. **Benchmark threshold for "engine integration":** any primitive that survives steps 1–3 must additionally hold ≥1.3× SIMD-friendliness on the int8/int16 quantised forward path before it earns a Stockfish-fork prototype.

---

## Caveats

- Three of the five proposals (`primitive_cgec`, `primitive_rstm`, `primitive_sessm`) have close 2016–2024 literature precedents (Cohen-Welling G-CNNs; reversible networks of the i-RevNet/RevNet line; Gu & Dao's selective SSMs) and are honestly **underexplored primitives** for chess rather than wholly new operators. The report flags these as such; do not let later writeups soften that framing.
- `primitive_csg` overlaps with the 2020 Routing Transformer in the "content-routed sparse attention" sense; the structural distinction (hard predicate, O(|E|·d) compute graph, no n×n score tensor) must be defended kernel-level, not just architecturally.
- All five tests assume the existing 234-architecture scout's CRTK class 1 hard-negative FP rate is a reliable signal of strong-vs-weak architectures. If post-hoc analysis shows that scout was noisy in CRTK class 1 specifically, the falsification tests need higher-N replicates than the single-seed budget allows — which would force every primitive to be deferred until a larger validation regime is feasible.
- I was unable to call `run_blocking_subagent` or `enrich_draft` (not present in the available tool set for this session); I would normally have spent the subagent budget on resolving whether the *exact* "content-determined hard-sparse gather with fused kernel" already exists as a published 2024–2026 primitive — readers should check recent PyG / xFormers / FlashAttention releases for collisions before publishing `primitive_csg` as new.
- Stockfish/HalfKA citations in this report refer to publicly documented behaviour (SFNNv4–v13, Nasu 2018) rather than primary-source code reads; if a primary-source verification is required for the `primitive_sda` claim of O(|Δ|·d) being absent from `torch.nn`, that needs an independent PyTorch source-tree audit.

## What I Cut (and why)

- **"Hyperbolic king-distance embedding"** — this is an *encoding*, explicitly banned by the rules. Geometry is set by input mapping, not by a new operator.
- **"Tactical-motif contrastive head"** — this is a *training trick* (a new loss), explicitly banned.
- **"Krylov-projected self-attention" / "linearised softmax variant"** — fails the calibration bar: it decomposes into matmul + feature map + softmax, i.e. just an activation-function tweak on an existing primitive.
- **"Deep-Equilibrium chess head"** — DEQ (Bai et al. 2019) is a real primitive but the chess motivation is weak: positions don't have an obvious fixed-point semantics, and the iterative solver kills the per-MCTS-node inference speed that the project explicitly prizes. Wrong inductive bias for the chess speed regime.
- **"Hash-embedding piece-square table"** — Kang et al. 2021's DHE is a primitive, but applied to chess it reduces to a different sparse-feature lookup, i.e. an encoding-layer choice rather than a structurally new operator. Cut as a hidden encoding proposal.