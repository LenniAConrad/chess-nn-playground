# Five Candidate Neural-Network Primitives for Chess Position Evaluation

**TL;DR**
- The two most defensible novel primitives are **Δ-Pair Accumulator** (a HalfKA-style accumulator extended to *input-dependent* pairwise interactions, where Rendle's factorization-machine diagonal trick provably fails) and **Ray-Selective Scan** (a Mamba/S6 selective scan whose per-ray length and termination are determined by the input board, not by fixed positions).
- Three further candidates — **D4×Z2 Bispectrum Pool**, **Bitboard PopConv (STE)**, and **Threat-Diffusion Implicit Layer** — are flagged honestly as *underexplored primitives for chess* rather than genuine inventions: each maps onto a published 2016–2024 primitive (Sanborn G-triple-correlation; XNOR-Net binary conv; Bai-Kolter-Koltun DEQ) but has not been instantiated against the chess group structure, the bitboard data type, or the transitive-threat fixed point.
- Five candidates were rejected during self-audit, almost all because they collapsed into compositions of existing PyTorch ops once the computation graph was drawn out (FiLM, sparsemax, MoE-of-attention, etc.). The deliverable schema is filled exactly as requested below.

---

## Key Findings

1. **Only one structural property genuinely separates chess from generic sequence/grid problems at the primitive level: the input determines the connectivity.** Legal moves, attack/defense rays, and piece occupancy all change per board. Any primitive whose forward/backward graph depends on a fixed adjacency (Conv2d, vanilla Attention, standard Mamba scan) cannot exploit this. Primitives 1, 2, and 5 below all hinge on this property.
2. **The incremental-update (O(|Δ|)) property is structurally underrepresented in `torch.nn`.** Stockfish's HalfKAv2 accumulator (Nasu 2018; Sobczyk et al.) is the existence proof that this property buys ≈100 Elo and is the engine-speed lever. No PyTorch op currently exposes a `(state, Δ_add, Δ_remove) → state'` API as a fused primitive; `nn.EmbeddingBag` is the closest cousin but is recompute-only. This is the lowest-hanging primitive-shaped fruit.
3. **The chess group is richer than D4.** D4 (board rotations/reflections) × Z2 (color swap, with simultaneous value-sign flip) gives a 16-element group with a non-trivial sign representation. The Sanborn et al. G-Triple-Correlation primitive (NeurIPS 2023) is the right shape for this but has not, to my knowledge, been applied to chess.
4. **Most "chess-novel" ideas collapse on inspection.** Color-swap-equivariant linear, king-conditional FiLM, castling-aware attention masks, sparsemax-over-legal-moves — all decompose cleanly into existing ops and were cut.
5. **Two 2024–2025 papers are unavoidable prior art and must be cited honestly.** Native Sparse Attention (Yuan et al., arXiv 2502.11089, 2025) and Mixture of Sparse Attention (MoSA, arXiv 2505.00315, 2025) both implement content-dependent block sparsity. Any "legal-move sparse attention" primitive overlaps with these and was downgraded.

---

## Details

### 1. primitive_delta_pair_accumulator

**Name:** Δ-Pair Accumulator (DPA)

**One-line claim:** A sparse-feature accumulator that maintains both first-order and *input-dependent* pairwise interaction terms with O(|Δ|·k) incremental updates per move.

**Mathematical signature:**
Let S ⊂ {1,…,N} be the set of active sparse-feature indices (e.g., HalfKA piece-square features), and let E(S) ⊂ S×S be an input-dependent edge set (e.g., attacker→defender pairs, determined from the same sparse vector via a deterministic, fixed function of S only, never from Stockfish scores).
Maintain state vector
 A(S) = Σ_{i∈S} u_i + Σ_{(i,j)∈E(S)} W_{type(i),type(j),Δsq(i,j)} ∈ ℝ^k
The op exposes three entry points:
 forward_full(S) → A(S)
 delta(A, S, Δ_add, Δ_remove, E_add, E_remove) → A'(S')
 backward: dL/du_i, dL/dW receive gradients only through the indices touched in the forward (i.e., dL/dW_{abc} = Σ over the (i,j) pairs in E that hit (a,b,c)).

**Why this does not decompose into existing PyTorch ops:**
The closest existing op is `nn.EmbeddingBag` (sum-only, recompute-only) and the factorization-machine identity (Rendle 2010), which computes Σ_{i<j} ⟨v_i,v_j⟩x_i x_j in O(k·|S|) via ((Σ v_i x_i)² − Σ v_i² x_i²)/2. **Rendle's trick collapses because it sums over all pairs**; here the pair set E(S) is a strict, input-dependent subset of S×S, so the diagonal subtraction cannot recover it. A faithful implementation therefore requires explicit per-pair iteration, with a fused `(state, Δ) → state'` kernel that no PyTorch op currently exposes.

**Chess-specific motivation:**
Stockfish's HalfKAv2 accumulator updates first-order features in O(|Δ|·hidden) per move (chessprogramming.org/NNUE). DPA extends this to genuine pair interactions — attacker/defender, pinner/pinned, battery alignment — which classical eval encodes by hand. When a piece moves, the set of legal-attack pairs changes by O(1) in the quiet case and O(piece-fan-out) in tactical cases, well below |E(S)| ≤ ~250 for any chess position.

**Generalisation beyond chess:**
Any domain with a sparse binary feature vector and an input-dependent pairwise interaction that changes by O(1) per timestep: streaming recommender systems (user–item co-clicks with temporal masking), incremental molecular property prediction (bond formation/breaking), online graph anomaly detection (edge additions/deletions).

**Complexity (forward, backward, incremental-update):**
- Forward: O(|S| + |E(S)|)·k vs nn.EmbeddingBag O(|S|·k) [first-order only]; vs dense FM O(|S|²·k); vs Rendle FM O(|S|·k) but only for the full pair set
- Backward: O(|Δ_E|·k) for the touched pairs in a streaming setting; O((|S|+|E(S)|)·k) in batch training
- Incremental update on a bounded-change input: **O((|Δ_S| + |Δ_E|)·k)** — bounded by ~30·k in chess

**Scout-scale falsification test:**
Drop DPA in as a replacement for the HalfKA accumulator in the i243 dual-stream proposal (scout dataset, 173k positions × 12 epochs, single RTX 3070). Baseline: i243 with sum-only accumulator. Metric: matched-recall near-puzzle FP rate on CRTK class 1 at recall = baseline's recall at FPR 0.05. **Pass:** DPA reduces FPR by ≥10% relative at matched recall, *and* the fused delta-CUDA kernel produces inference wall-clock ≤1.25× the sum-only baseline at the same hidden width. **Fail:** either condition violated, or the pair contribution is dominated by the first-order term in a learned scale ablation (i.e., learned pair gain → 0).

**Failure mode catalogue:**
- **Hidden rebrand:** If E(S) collapses to a fixed-pattern relation (e.g., "same diagonal"), DPA degenerates into a structured-mask sparse bilinear layer expressible via two `EmbeddingBag` + Hadamard. Mitigation: assert E(S) is sample-varying in the test.
- **Numerical instability:** Pair terms can grow quadratically in piece count; without per-pair normalization the accumulator saturates int16 quantization (the very property NNUE needs for CPU inference). Mitigation: per-piece-type scale parameter, calibration ablation.
- **Too slow:** If |E(S)| approaches |S|² in middlegame positions with many sliding pieces, the speed advantage over a recomputed bilinear layer disappears. Mitigation: cap |E(S)| via attacker-only or attacker∪defender restriction; measure pair fan-out empirically before declaring victory.

**Status:** proposed

---

### 2. primitive_ray_selective_scan

**Name:** Ray-Selective Scan (RaySSM)

**One-line claim:** A Mamba/S6-style selective scan whose scan order is the eight compass rays of each square, terminated at the first occupied square — so the scan topology is determined by the input board.

**Mathematical signature:**
For each square s ∈ {0,…,63} and each of 8 ray directions d ∈ D:
 Let R(s,d,x) = (s₁, s₂, …, s_{ℓ(s,d,x)}) be the ordered sequence of squares along direction d starting from s, where ℓ(s,d,x) is the smallest index such that x at s_{ℓ} is occupied (and s_{ℓ} is included).
 Run the selective-scan recurrence (Gu & Dao 2023, S6):
  h_{t} = exp(Δ_t A) h_{t-1} + Δ_t B_t x_{s_t}
  y_{s,d} = C_{s_t} h_{ℓ(s,d,x)}
 where (Δ_t, B_t, C_t) are produced from x by linear projections at each scanned square.
 Output per square: y_s = Σ_d y_{s,d}, shape [B, 64, d_model].

**Why this does not decompose into existing PyTorch ops:**
Mamba's selective_scan_cuda (Gu & Dao 2023, arXiv 2312.00752) operates on a tensor of fixed shape [B, L, D] with a *fixed* scan order over the L axis. Vision-Mamba (Zhu et al. 2024) and LocalMamba (Huang et al. 2024) extend this to 2D by running 4 fixed scan directions. RaySSM's scan path **is itself input-dependent**: ℓ(s,d,x) is determined by piece occupancy, so the gradient through the scan must terminate at a position the input chose. PyTorch has no fused kernel that scans an indexed jagged tensor of input-determined length per row; the closest substitute (Python for-loop with dynamic slicing) loses both throughput and graph fusibility.

**Chess-specific motivation:**
Sliding pieces (bishop, rook, queen) interact precisely along rays terminated by the first occupant — this is the chess concept of an "X-ray" or "battery." Conv2d and standard attention treat the board as a fixed grid; neither natively expresses "feature flows along the bishop's diagonal up to the first piece in the way." RaySSM is the smallest primitive that does.

**Generalisation beyond chess:**
Any domain with input-dependent path traversal terminated by an event: optical raycasting (light bounces off the first surface hit), event-camera asynchronous scans, computational geometry (visibility polygons), shortest-light-path PDE solvers.

**Complexity (forward, backward, incremental-update):**
- Forward: O(64 · 8 · L̄ · d) where L̄ ≤ 7 is the average ray length, vs full Conv2d's O(64 · k² · d²) and full Attention's O(64² · d)
- Backward: same scaling; selective-scan recomputation trick (Gu & Dao 2023, App. D) keeps memory O(64·d) without storing per-step states
- Incremental update on a bounded-change input: O(rays affected · ray-length) ≈ O(8 · 7) per moved piece — **not strictly O(1) but O(constant in board size)**

**Scout-scale falsification test:**
Drop RaySSM in as the global-mixing block of the i242 chess-decomposed-attention follow-up, replacing the full-attention layer. Train 12 epochs on 173k positions. Baseline: i242 unchanged. Metric: near-puzzle FP rate at matched recall on CRTK class 1, plus wall-clock per forward pass. **Pass:** ≥5% relative FPR reduction *and* forward-pass wall-clock ≤ 0.8× the attention baseline. **Fail:** no FPR improvement, or wall-clock parity not achieved (the chess-decomposed attention is already small, so a slow ray kernel kills the proposal even at equal accuracy).

**Failure mode catalogue:**
- **Hidden rebrand:** If rays are not actually terminated by occupancy (e.g., always run to board edge), the primitive degenerates into 8 fixed-direction 1D Mambas, which is just Vision-Mamba's 4-way scan ×2 — published. Mitigation: explicit ablation with `terminate_at_occupant=False`.
- **Numerical instability:** Selective scans accumulate exp(Δ A) along the ray; for short rays (1–2 squares) Δ is poorly identified and Δ-init matters. Mitigation: follow Mamba's Δ initialization (uniform in [log 0.001, log 0.1]).
- **Too slow:** Triton/CUDA kernel for jagged ray-scan over 64 starting squares × 8 directions may not amortize over batch when L̄ is small (mid-/endgame). Mitigation: pad-and-mask to fixed L=7 in the kernel; benchmark before claiming engine-relevance.

**Status:** proposed

---

### 3. primitive_d4_z2_bispectrum

**Name:** D4×Z2 Bispectrum Pool (BispecPool)

**One-line claim:** A complete (information-preserving) invariant pool over the board's 16-element symmetry group, replacing max-pool's lossy invariance.

**Mathematical signature:**
Let G = D4 × Z2 (8 board symmetries × color swap with simultaneous value-sign flip). For an input signal f: G → ℝ^d, the G-bispectrum is
 β(f)(ρ_i, ρ_j) = f̂(ρ_i) ⊗ f̂(ρ_j) · (f̂(ρ_i ⊗ ρ_j))*
where f̂ is the G-Fourier transform over irreducible representations ρ_i (Sanborn et al., NeurIPS 2023, arXiv 2310.18564). For |G|=16, the Fourier basis is finite and explicit; β is computed as a fixed contraction. Output: a vector of dimension equal to the number of (ρ_i, ρ_j, ρ_k) triples in the Clebsch–Gordan decomposition.

**Why this does not decompose into existing PyTorch ops:**
`max_pool` and `mean_pool` over G-orbits produce invariants but are *lossy* (Sanborn et al. 2023 prove non-injectivity). The bispectrum is the unique lowest-order *complete* polynomial G-invariant. Its computation requires explicit Clebsch–Gordan coefficients for the chosen group; there is no PyTorch op that performs a group-Fourier transform plus CG contraction generically. (Compare: `torch.fft` covers cyclic groups only.)

**Chess-specific motivation:**
Position evaluation must be invariant to board flip + simultaneous color swap + value negation (a Z2 acting by sign). It must also be equivariant/invariant to D4 (corner-axis symmetries). Standard data augmentation samples 8/16 of these orbits per gradient step; a hard-coded complete invariant uses zero augmentation slots and removes a known source of label noise.

**Generalisation beyond chess:**
Any compact-group invariance task with a small finite group: molecular point-group invariants (C2v, C3v), crystallographic space-group classification, particle-physics discrete-symmetry classification (CPT-like Z2 products).

**Honest novelty:** The bispectrum-pool primitive is published (Sanborn, Shewmake, Olshausen, Hillar — NeurIPS 2023). This proposal is **"underexplored primitive for chess"**: chess-specific extension is the inclusion of the Z2-with-sign representation (color swap induces a sign flip on the evaluation output, which is *not* the trivial Z2 the original paper handles).

**Complexity (forward, backward, incremental-update):**
- Forward: O(|G|² · d) = O(256·d) per forward pass vs max-pool's O(|G|·d)
- Backward: O(|G|² · d)
- Incremental update on a bounded-change input: **not applicable** (pooling is global)

**Scout-scale falsification test:**
Drop BispecPool in as the final aggregation layer of the i193 conv-only baseline, with **no D4/color-swap data augmentation**. Compare to (i) i193 with augmentation but mean-pool, (ii) i193 without augmentation, mean-pool. Metric: near-puzzle FP rate at matched recall. **Pass:** BispecPool (no aug) ≥ baseline (with aug) at matched recall, and beats baseline (no aug) by ≥10% relative. **Fail:** does not match the augmented baseline.

**Failure mode catalogue:**
- **Hidden rebrand:** For abelian subgroups, bispectrum reduces to triple-products of Fourier coefficients — implementable as 3 FFTs + einsum. For the full D4×Z2 the non-abelian part is real; mitigation: ablate against the abelian-only version.
- **Numerical instability:** CG coefficients for D4 are sparse but introduce sign cancellations; fp16 will break invariance. Mitigation: fp32 in the pool, fp16 elsewhere.
- **Too slow:** O(|G|²·d) = 256·d FLOPs per board is negligible at the pool, but if applied per-square instead of globally, cost is 64·256·d. Mitigation: restrict to a single global pool.

**Status:** proposed

---

### 4. primitive_popconv

**Name:** Bitboard PopCount Convolution (PopConv)

**One-line claim:** A "convolution" whose multiply-accumulate is replaced by 64-bit `AND` + `popcount`, operating natively on bitboards rather than float planes.

**Mathematical signature:**
Input: a stack of P 64-bit bitboards X ∈ {0,1}^{P×64} (e.g., 12 piece-type bitboards plus derived attack bitboards). Learned kernels K ∈ {0,1}^{F×P×64}, parameterised by a real underlying tensor K̃ ∈ ℝ^{F×P×64} with K = (K̃ > 0).
Forward (per output filter f):
 y_f = Σ_{p=1..P} popcount(K_{f,p} AND X_p) − b_f
Backward (straight-through estimator, Bengio et al. 2013):
 dL/dK̃_{f,p,i} = dL/dy_f · X_{p,i}, clipped to |K̃| ≤ 1.

**Why this does not decompose into existing PyTorch ops:**
PyTorch has no integer-popcount op that is differentiable end-to-end, and no Conv2d-style layer over `torch.bool` tensors with a bit-AND inner product. XNOR-Net (Rastegari et al. 2016) is the closest cousin but uses *signed* binary (-1/+1) multiplication, which factors into Hadamard-then-sum and is implemented as float matmul with scale factors. PopConv uses unsigned AND-popcount, which is structurally a different graph (no sign cancellation, no scale factor, no float multiplication anywhere in the forward integer kernel).

**Chess-specific motivation:**
Chess engines store the board *natively* as bitboards. Stockfish's attack tables, magic bitboards, and king-safety patterns are all popcount-style operations. PopConv directly replicates classical eval terms like "popcount(king-zone AND opponent-attacks)" inside a learnable layer, on CPU, with one machine instruction per inner product (POPCNT, ~3-cycle latency).

**Generalisation beyond chess:**
Genomics (k-mer presence/absence with popcount AND fingerprints), cybersecurity (binary feature hashes for malware classification), database query estimation (Bloom-filter-like learnable predicates).

**Honest novelty:** Binary networks (Courbariaux et al. 2015, Rastegari et al. 2016) and bit-serial accelerators are extensively published; the STE is from Bengio et al. (2013, arXiv 1308.3432). This proposal is **"underexplored primitive for chess"** — the chess-specific contribution is that the input *is already* a stack of bitboards, so PopConv has no quantization loss on the input side; only the kernels are quantized.

**Complexity (forward, backward, incremental-update):**
- Forward: O(F·P) POPCNT instructions per board vs Conv2d's O(F·P·k²) FMAs — typically 30–60× fewer cycles
- Backward: O(F·P·64) STE updates per board
- Incremental update on a bounded-change input: **O(F·P)** per moved piece if a piece-wise bitboard delta is fed; the AND-popcount is monotonic in X, so contributions can be added/subtracted exactly

**Scout-scale falsification test:**
Replace the first convolution of the i193 baseline with PopConv on a 12-channel piece bitboard input. Train 12 epochs on 173k positions. Metric: (a) near-puzzle FP rate at matched recall, (b) CPU wall-clock per inference (single-threaded). **Pass:** matched-recall FPR within 5% relative of the float Conv2d baseline *and* CPU inference ≥3× faster. **Fail:** either FPR > 1.1× baseline or speedup < 2×.

**Failure mode catalogue:**
- **Hidden rebrand:** PopConv with kernels relaxed to floats during training is exactly XNOR-Net up to sign convention. Mitigation: insist on integer-only forward at inference; report dequantization error.
- **Numerical instability:** STE has no fundamental gradient signal for kernels far from the decision boundary; learning stalls on cold features. Mitigation: warm-start kernels from a thresholded pretrained float Conv2d.
- **Too slow:** GPU popcount is not as well-optimized as integer matmul; on the RTX 3070 the speedup is CPU-only. Mitigation: report training-time on GPU with float surrogate; inference benchmark on CPU only.

**Status:** proposed

---

### 5. primitive_threat_diffusion_deq

**Name:** Threat-Diffusion Implicit Layer (TDIL)

**One-line claim:** A deep-equilibrium layer whose fixed point is the stationary "threat-defense" distribution on the attack graph, capturing transitive tactical patterns (pins, X-rays, batteries) in a single primitive.

**Mathematical signature:**
Let A(x) ∈ ℝ^{64×64} be the (per-position) attack adjacency derived from x (entry (s,t) = learned scalar if piece at s attacks square t, else 0). Let h(x) ∈ ℝ^{64×d} be per-square input features.
 The primitive solves for z* ∈ ℝ^{64×d}:
  z* = σ(D^{-1} A(x) z* W + h(x))
 where D is the row-normaliser, W ∈ ℝ^{d×d} is learned, σ is a contractive nonlinearity (e.g., tanh).
 Forward: z* found via Anderson acceleration; iterate to ‖z_{k+1}−z_k‖ < ε.
 Backward (Bai, Kolter, Koltun 2019, arXiv 1909.01377):
  dL/dθ = (dL/dz*) (I − Jσ ∘ A(x))^{-1} (∂f/∂θ)
 computed via one linear solve, **not** by unrolling iterations.

**Why this does not decompose into existing PyTorch ops:**
A k-step explicit message-passing GNN computes an *approximation* with a fixed depth k; its computation graph has k stacked matmuls. The DEQ primitive instead computes the exact fixed point and differentiates through it via the implicit-function theorem, requiring no stored intermediate iterates and producing a *different* gradient (Bai et al. 2019, Thm. 1). `torch.autograd` has no native fixed-point op; the implicit backward must call a linear solver (GMRES/Anderson) inside `Function.backward`.

**Chess-specific motivation:**
Tactical patterns are transitively closed: a pin involves attacker → pinned → defended piece; an X-ray battery involves attacker → friendly piece → enemy piece behind. These are fixed-point properties of the attack graph, not k-hop properties for any fixed k. A DEQ layer computes them in one shot at a depth determined by the position itself.

**Generalisation beyond chess:**
Any task with transitive structural propagation: PageRank-style influence on dynamic graphs, fluid-pressure propagation in PDEs, equilibrium-finding in game-theoretic settings (Nash, correlated eq.), influence maximization in social networks.

**Honest novelty:** Deep Equilibrium Models are published (Bai, Kolter, Koltun, NeurIPS 2019; Bai et al. 2020 multi-scale DEQ). This proposal is **"underexplored primitive for chess"** — chess contribution is the use of a per-position attack adjacency A(x) and the claim that the fixed point natively expresses tactical motifs.

**Complexity (forward, backward, incremental-update):**
- Forward: O(K · 64² · d) where K is the iteration count to convergence (typically 5–20 for contractive σ), vs k-stacked GNN O(k · 64² · d) for fixed k
- Backward: O(K' · 64² · d) for one linear solve, K' typically smaller than K
- Incremental update on a bounded-change input: **not directly applicable** — but warm-starting z from the parent-node fixed point typically halves K in MCTS, which is a *de facto* engine-scale incremental property

**Scout-scale falsification test:**
Drop TDIL in as a single global layer in the i243 architecture, replacing one transformer block. Restrict to a tractable adjacency: per-square attacker count + binary attack indicator. Train 12 epochs on 173k positions. Metric: near-puzzle FP rate at matched recall on CRTK class 1, with a tactical-puzzle subset breakout. **Pass:** ≥15% relative FPR reduction on the tactical subset (where transitive threats matter) *and* convergence in ≤10 iterations at inference. **Fail:** no tactical-subset improvement, or K > 20 iterations needed (engine speed killer).

**Failure mode catalogue:**
- **Hidden rebrand:** If σ is linear, TDIL is a closed-form linear solve z* = (I − AW)^{-1} h, expressible as one PyTorch `linalg.solve`. Mitigation: enforce σ = tanh and report nonlinear fixed-point residuals.
- **Numerical instability:** Fixed-point existence requires the spectral radius of (Jσ ∘ A) < 1. Empirically DEQs need spectral regularisation (Bai et al. 2021). Mitigation: explicit Jacobian regularisation term in training.
- **Too slow:** A 10-iter inner loop at every MCTS node is fatal for engine deployment. Mitigation: present as an *evaluator-quality* primitive, not an engine-NNUE replacement; or amortize via warm-start across MCTS siblings.

**Status:** proposed

---

## Recommendations

**Stage 1 — implement now (≤4 GPU-hours total on the RTX 3070):**
Implement **Δ-Pair Accumulator** and **PopConv** first. Both have CPU-inference angles aligned with NNUE-style deployment, both have small reference implementations (DPA = scatter-pair kernel; PopConv = bit-tensor + STE), and both are the most likely to *not* be hidden rebrands. Drop them into i243 (DPA) and i193 (PopConv) respectively. The benchmark threshold that promotes either to "primitive works" is the matched-recall near-puzzle FP rate reduction in §1 and §4 above; the threshold that demotes either is wall-clock parity failure.

**Stage 2 — implement after Stage 1 lands a result:**
**Ray-Selective Scan**. This requires a Triton kernel for jagged scans over 64×8 ragged segments; the engineering cost is the highest of the five. Postpone until DPA/PopConv have either succeeded (validating the "input-dependent connectivity wins at scout scale" thesis) or failed (in which case RaySSM is unlikely to clear the bar either).

**Stage 3 — implement only if Stage 1 succeeds and bigger data is available:**
**Bispectrum Pool** and **Threat-Diffusion DEQ**. Both have failure modes that are most likely to bite at scout scale (Bispec: hard invariance may overfit at 173k; DEQ: spectral-radius training instability needs ≥1M positions to converge). Re-evaluate when LC0-scale data becomes accessible or when a larger compute budget than a single 3070 is available.

**Thresholds that change the staging:**
- If DPA loses on FPR but wins on wall-clock by ≥3×, retain DPA as an engine-only primitive (a 3× CPU inference speedup is worth ~50 Elo per the user's "2× wall-clock = 30 Elo" calibration).
- If PopConv loses by >10% FPR but the kernels are interpretable (visualizable as classical-eval-like masks), retain as an *interpretability* primitive only, not for performance.
- If RaySSM's fused kernel cannot beat 1.0× attention-baseline wall-clock, abandon — the scout-scale data is too small to justify the operator on accuracy alone.

---

## Caveats

1. **No primitive has been trained.** Every "Pass" criterion above is a prediction conditioned on prior literature, not a result. The user's request explicitly forbids inventing results, and this report obeys.
2. **The novelty bar is "underexplored for chess" for three of five proposals (Bispec, PopConv, DEQ).** Each cites the original primitive and identifies the chess-specific extension. Primitives 1 (DPA) and 2 (RaySSM) are the candidates with the most plausible claim to *genuine* novel computation graphs, but both have adjacent published work (factorization machines for DPA; Graph-Mamba variants for RaySSM) that a careful reviewer might cite as prior art.
3. **Web-search budget exhausted before completing planned follow-ups.** Specifically: I could not verify whether (a) any 2024–2026 paper instantiates an "input-dependent edge" version of factorization machines (would weaken DPA's novelty claim), (b) Graph-Mamba (Wang et al. 2024) or its successors have published with input-dependent scan termination (would weaken RaySSM's novelty), or (c) the Sanborn G-Triple-Correlation has been applied to non-trivial-Z2 actions on regression outputs. A reviewer should check these before publication.
4. **Subagent and enrich_draft tooling referenced in the instructions was not exposed in this environment**, so the deliverable was produced from the lead researcher's own searches alone. This means the "thinnest sourcing" gap — most likely the existence of a 2024–2025 paper that implements input-dependent sparse pairwise factorization machines — remains unverified.
5. **Reproducibility constraints from the prior scout are tight** (single RTX 3070, 8 GiB, single seed, 12 epochs, 173k positions). Each proposal's falsification test was sized to fit; none requires more than 2 GPU-hours per run, and DPA + PopConv together should fit in one afternoon.

---

## What I Cut

Five candidates were generated and rejected during self-audit. All were rejected for the same reason: when the computation graph was drawn out, the candidate collapsed into a composition of existing PyTorch operators.

1. **King-Conditional FiLM.** Proposal: modulate every layer's activations by an MLP of the king square. **Cut because:** this is exactly FiLM (Perez et al., AAAI 2018, arXiv 1709.07871) — `γ(k) ⊙ x + β(k)`, fully decomposable into a small MLP and `mul`/`add`. The chess-specific bit is just the choice of conditioning variable, which is an input-encoding decision, not a primitive.

2. **Sparsemax-over-Legal-Moves Policy Head.** Proposal: replace the policy-softmax with sparsemax restricted to legal moves. **Cut because:** sparsemax exists (Martins & Astudillo, ICML 2016) and restriction-to-legal is a mask; the composition `sparsemax(logits + legal_mask)` is two existing ops. This is an output-head trick, not a primitive.

3. **Mixture of Attention + Conv with Learned Gate.** Proposal: per-token soft mixture of self-attention output and depthwise conv output, gated by an MLP. **Cut because:** the anti-example list in the brief explicitly disqualifies "learned mixture of attention + conv." It is `gate · attn(x) + (1 − gate) · conv(x)`, fully decomposable. Demoted to architectural composition.

4. **Color-Swap-Equivariant Linear Layer.** Proposal: a `nn.Linear` whose weights are tied so that color-swapping the input swaps and sign-flips the output. **Cut because:** this is a standard G-equivariant linear layer for Z2 (Cohen & Welling 2016, ICML 2016, arXiv 1602.07576). Hard parameter tying is a one-line index trick on top of `nn.Linear`, not a new primitive — it's a constraint on weights. Subsumed by Bispectrum Pool's group story above, which retained the *complete-invariant* property as the distinguishing feature.

5. **Per-Square Mixture-of-Experts Routing by Piece Type.** Proposal: an MoE gate whose router is the one-hot piece type at each square. **Cut because:** this is a deterministic MoE with a hand-set router, which reduces to a `torch.embedding` lookup followed by per-expert linear layers — two existing PyTorch ops. The "MoE-gate" primitive (Jacobs et al. 1991, Shazeer et al. 2017) is the soft, learned-router version; deterministic routing by piece type does not extend that primitive, it specializes (degrades) it. Demoted to an input-encoding trick.
