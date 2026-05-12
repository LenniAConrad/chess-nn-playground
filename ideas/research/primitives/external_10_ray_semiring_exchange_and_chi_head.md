# Five Candidate Novel Neural-Network Primitives for Chess Evaluation

**TL;DR**
- Five primitives proposed below clear the calibration bar (Conv2d / Adam / LayerNorm / MoE-gate / Mamba-S6 / RWKV) on at least two of: distinct computation graph, distinct gradient flow, distinct complexity class, distinct connectivity pattern. The three with the cleanest novelty are **RSEA** (reversible sparse-event accumulator), **BTRS** (blocker-terminated ray scan), and the **χ-equivariant value head**. **P-Tens** and **DAJP** are flagged as *underexplored-for-chess* rather than fully novel.
- Ranking on the four criteria the brief specified: (a) plausibility-of-novelty: χ-head > RSEA > BTRS > DAJP > P-Tens; (b) demonstrability on a single RTX 3070 in <2 GPU-hours: P-Tens > χ-head > RSEA > DAJP > BTRS; (c) inference-speed advantage: RSEA ≫ BTRS > DAJP > χ-head ≈ P-Tens; (d) generalisation beyond chess: RSEA > BTRS > χ-head > P-Tens > DAJP.
- Devil's-advocate self-audit on the top two (RSEA, BTRS) is included; both survive but with explicit risks named.

---

## Key Findings

The 2024-2026 literature contains exactly one operator family that gets close to what NNUE's accumulator does *while being differentiable end-to-end through state changes*: the DeltaNet line (Yang, Wang, Zhang, Shen, Kim, NeurIPS 2024, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", arXiv:2406.06484), and Gated DeltaNet (Yang et al., ICLR 2025). These are sequential and have no rollback semantics. NNUE itself (Nasu 2018; chessprogramming.org/NNUE; official-stockfish/nnue-pytorch) is integer-quantised, inference-only, and not differentiable through make/unmake. No primitive currently combines the two — that is the cleanest gap.

Mamba/S6 (Gu & Dao, ICLR 2024) establishes input-conditioned parameters as a legitimate new primitive class — that gives a precedent for proposing content-dependent prefix-scans on 2D board topology.

The only published chess-group-equivariant primitive is FGNN (Carroll & Beel 2020, "Finite Group Equivariant Neural Networks for Games", arXiv:2009.05027), which handles color-swap by output reshuffling on the *policy* head. No work I could find treats the color involution as a non-trivial **1-dimensional character** ("sign-graded equivariance") on the *value* head, even though that is the mathematically correct structure for chess evaluation (eval(τx) = −eval(x)).

Partition-algebra-equivariant linear layers (Pearce-Crump, arXiv:2212.08648) give a principled S_n-equivariant operator basis but have never been applied to the piece-type axis in chess networks.

---

## Details — Five Proposals

### 1. primitive_rsea

**Name:** Reversible Sparse-Event Accumulator (RSEA)

**One-line claim:** A differentiable stateful primitive that supports sparse-event forward, exact O(|Δ|·d) make/unmake, and gradient flow through both apply and undo.

**Mathematical signature:**
State `h ∈ R^{B,d}`. Two operators sharing a learned weight `W ∈ R^{F,d}` and bias `b ∈ R^d`:
- `apply : (h, Δ⁺ ⊂ {1..F}, Δ⁻ ⊂ {1..F}) → h'` with `h' = h + Σ_{i∈Δ⁺} W[i] − Σ_{j∈Δ⁻} W[j]`
- `read : h → y = σ(h)` (nonlinear head, computed on demand, not on every apply)

Crucially the primitive registers `(Δ⁺, Δ⁻)` on a tape so that backward can attribute `∂L/∂W[i]` to *every* state at which feature `i` was active, even across millions of MCTS tree edges in the same forward graph.

**Why this does not decompose into existing PyTorch ops:** A `nn.EmbeddingBag` with sum-reduce gives the forward but not the *reversible*, state-carrying graph: PyTorch has no primitive that maintains a persistent tensor edited by add/sub deltas while keeping a per-edit gradient tape that supports rollback to any prior state and re-application along a different branch. DeltaNet (Yang et al. NeurIPS 2024) is sequential and offers no `unmake`; RevNet (Gomez 2017) is reversible for memory-saving but takes dense inputs, not sparse-event deltas; NNUE's accumulator is integer and not differentiable through pop.

**Chess-specific motivation:** The HalfKA accumulator is *the* reason NNUE is fast — typically a handful of feature flips per ply (chessprogramming.org/NNUE; official-stockfish/nnue-pytorch docs). RSEA generalises that property from inference-time-only to a fully differentiable, gradient-trained primitive, so that an engine doing MCTS with batched make/unmake can run gradient updates on the *same* state machine the engine uses at inference.

**Generalisation beyond chess:** Any domain with bounded-change inputs in a tree-rolled-out setting: SAT/SMT solvers with learned heuristics, theorem-prover state, online recommender systems with single-item-at-a-time user state edits.

**Complexity:**
- Forward (full): O(F·d) (one-time per root)
- Apply/Unmake on bounded delta of size k: **O(k·d)** vs O(F·d) for re-running `nn.Linear`
- Backward: O((Σ |Δ_t|) · d) total across a search tree of T edges

**Scout-scale falsification test:** Drop RSEA in as the feature-transformer layer of a 768-wide NNUE-style net on the user's 173k-position dataset. Baseline: same architecture without RSEA (i.e. standard `nn.Linear` recomputed each board). Both nets identical otherwise. Metric: matched-recall (R=0.5) near-puzzle FP rate on CRTK class 1. "Works" = RSEA matches baseline FP rate within 1 pp AND yields ≥3× wall-clock speedup at *inference* on a batched make/unmake harness. "Fails" = FP rate worse by >2 pp.

**Failure mode catalogue:**
- Hidden rebrand of `EmbeddingBag` + manual undo loop — true only if you ignore the gradient-tape design; the differentiability through *both* apply and unmake on a shared state tensor is what is new.
- Numerically unstable in fp16 because errors accumulate over a long branch (dogeystamp.com/chess6 documents this exact failure mode in floating-point NNUE). Mitigation: bf16 accumulator with periodic fp32 re-grounding.
- Too slow in practice if the tape overhead per edit exceeds the savings; would mean primitive is only useful at very small d.

**Status:** proposed

---

### 2. primitive_btrs

**Name:** Blocker-Terminated Ray Scan (BTRS)

**One-line claim:** A 2D prefix-scan primitive along 8 board directions whose *per-ray termination point* is determined by an input-dependent blocker score.

**Mathematical signature:**
Input `x ∈ R^{B,8,8,d}`, blocker logits `β ∈ R^{B,8,8}`. For each of 8 directions `θ ∈ {N,NE,E,SE,S,SW,W,NW}` and each origin square `s`, define `y_θ(s) = Σ_{t=1..L_θ(s)} α_t · φ(x[s + t·θ])`, where `L_θ(s)` is the smallest t such that the soft termination indicator `τ(β[s + t·θ])` ≥ 0.5 (differentiable via straight-through estimator), and `α_t = ∏_{u<t}(1 − τ(β[s+u·θ]))` is the continuous survival weight.

Output `y ∈ R^{B,8,8,8,d}` (8 directions stacked).

**Why this does not decompose into existing PyTorch ops:** Mamba/S6 (Gu & Dao 2023) introduced input-conditioned 1D recurrences; BTRS is structurally different in (a) 2D planar topology with 8 simultaneous independent direction scans sharing the same blocker field, and (b) *terminating* prefix-scan semantics rather than exponential decay — the survival weight `α_t` collapses to zero past a single sharp blocker rather than decaying smoothly. Masked attention cannot express this because the mask is *causal along each ray independently*, not a fixed bidirectional pattern.

**Chess-specific motivation:** Sliding pieces (bishops, rooks, queens) attack until they hit an occupied square — exactly the magic-bitboard ray-termination semantics (chessprogramming.org/Sliding_Piece_Attacks). Existing chess transformers (Lc0 transformer progress blog, Feb 2024; "Mastering Chess with a Transformer Model", arXiv:2409.12272) put significant work into encoding this geometry via position encoding; BTRS bakes it into the operator's connectivity.

**Generalisation beyond chess:** Any 2D grid task with directional propagation that should stop at content-defined boundaries — light/shadow rendering networks, room-segmentation in indoor scenes, neural ODE integration with content-defined termination.

**Complexity:**
- Forward: O(8·64·max_ray·d) = O(8·64·8·d) per board, ~32× a 3×3 conv at d=128; comparable to a single MHA layer
- Backward: same; straight-through estimator gives clean gradients
- Incremental update: not applicable (each board recomputed from scratch)

**Scout-scale falsification test:** Insert BTRS as a single replacement layer for the first attention block in the user's existing i242 transformer architecture, keeping all other parameters identical. Baseline: the i242 attention block at 173k positions × 12 epochs. Metric: matched-recall near-puzzle FP rate on CRTK class 1 AND param-matched FLOPs. "Works" = ≥1 pp FP-rate improvement on near-puzzles at equal FLOPs. "Fails" = no improvement OR FLOPs >1.5× baseline.

**Failure mode catalogue:**
- Hidden rebrand of Mamba's selective scan, just on 8 stacked sequences — strongest reviewer objection. Counter: the termination semantics (hard cut via STE rather than soft decay) and 2D direction-sharing of `β` make the compute graph distinct, but reviewer can fairly demand an ablation against a 1D-Mamba-per-direction baseline.
- STE on the termination indicator is biased; could destabilise training. Mitigation: anneal sharpness during training.
- 8× memory and FLOPs over a single conv; could be too slow at scout scale if applied at every layer rather than as a one-shot geometric injection.

**Status:** proposed

---

### 3. primitive_chi_head

**Name:** Sign-Graded χ-Equivariant Value Head

**One-line claim:** A bilinear readout primitive that satisfies `f(τx) = −f(x)` for the color-swap involution τ by construction, via Z₂-graded feature algebra.

**Mathematical signature:**
Internal feature space `h ∈ R^{2k}` is split into `h = (h^+, h^−)` with `h^± ∈ R^k`. Bilinear primitive `BG(h) = Σ_{ij} M^{++}_{ij} h^+_i h^+_j + Σ_{ij} M^{+−}_{ij} h^+_i h^−_j + Σ_{ij} M^{−−}_{ij} h^−_i h^−_j`. Z₂-graded constraint: even × even and odd × odd terms produce *even* output (set to zero in the value head); only even × odd cross terms survive, yielding an *odd* scalar output. The color-swap operator acts as `τ : (h^+, h^−) ↦ (h^+, −h^−)`, giving `f(τh) = −f(h)` by construction.

**Why this does not decompose into existing PyTorch ops:** Standard `nn.Linear` + `nn.Bilinear` followed by `tanh` does not enforce the χ-equivariance constraint — the gradient flows freely over an unconstrained weight matrix, so the network must *learn* the symmetry from data (and Carroll & Beel 2020 §2 show data-augmentation cannot recover this without infinite samples). χ-head's weights live on a constrained manifold (zero blocks in `M^{++}` and `M^{−−}`), and gradient flow respects the constraint exactly. FGNN (arXiv:2009.05027) handles color-swap on the *policy* head via output reshuffling, but not as a 1-dimensional character on the value head; there is no Z₂-graded bilinear primitive in `torch.nn`.

**Chess-specific motivation:** Chess evaluation is *exactly* sign-antisymmetric under color swap with board flip (chessprogramming.org/NNUE; Stockfish "transform" step adds a Tempo bonus on top of an inherently antisymmetric eval). The χ-head bakes this into the operator instead of training-time augmentation.

**Generalisation beyond chess:** Any zero-sum two-player game (shogi, Go), any physical-system output that flips sign under an exact involution (parity-violating amplitudes in physics), any preference-learning model where eval(A,B) = −eval(B,A) is required.

**Complexity:**
- Forward: O(k²) — same as `nn.Bilinear` of the same width
- Backward: same, on the constrained manifold (only `M^{+−}` parameters are free)
- Incremental update: not applicable (head only)

**Scout-scale falsification test:** Replace the user's i243 dual-stream output head with χ-head at the same parameter count (k chosen to match). Training-time color-swap augmentation **turned off** for both nets. Metric: (i) symmetry error `|f(x) + f(τx)|` averaged across val set — must be machine-zero for χ-head; (ii) matched-recall FP rate on CRTK class 1. "Works" = symmetry error <1e-5 AND FP rate ≤ baseline. "Fails" = FP rate >1 pp worse than baseline-with-augmentation.

**Failure mode catalogue:**
- Hidden rebrand of "split features and only use cross terms," which is two `nn.Linear` layers and an outer product — strongest objection. Counter: the *constraint* on the weight Hessian (forcing the diagonal blocks to zero) and the resulting different gradient subspace is what makes it a primitive, analogous to how LayerNorm differs from "subtract mean, divide std" because of the constrained gradient.
- Halving expressive capacity (only `k²` cross terms vs `4k²`) may underfit on positions where intrinsic asymmetry matters (e.g. en-passant rights, castling). Mitigation: keep one un-graded skip connection for state-flags.
- Too easy to demonstrate the *invariance* but hard to show it improves CRTK near-puzzles — reviewer will demand the second metric.

**Status:** proposed

---

### 4. primitive_ptens

**Name:** Piece-Type Partition-Algebra Equivariant Layer (P-Tens)

**One-line claim:** An S_T-equivariant linear primitive over the piece-type axis built from the partition-algebra basis, with capacity *constant* in T rather than O(T²).

**Mathematical signature:**
Input tensor `X ∈ R^{B,T,64,d_in}` (T = piece types, e.g. 6 or 12). Linear map `Y = P-Tens(X) ∈ R^{B,T,64,d_out}` parameterised as `Y[:,t,s,:] = A · X[:,t,s,:] + B · Σ_{t'≠t} X[:,t',s,:]`, with only `A, B ∈ R^{d_out × d_in}` learned. This is the order-2 partition-algebra basis (Pearce-Crump, arXiv:2212.08648, Schur-Weyl duality: dim of equivariant maps `M_T → M_T` is B(2) = 2).

**Why this does not decompose into existing PyTorch ops:** Yes you *can* write it as a structured-weight `nn.Linear`, but the structural constraint reduces parameter count from `T²·d_in·d_out` to `2·d_in·d_out` — and crucially the gradient is averaged over the S_T orbit, not free per (t,t'). `torch.nn` has no equivariant linear primitive that exposes the partition-algebra basis directly. Implementing it as plain `nn.Linear` would forfeit the gradient-averaging property and would need infinite-sample augmentation to recover.

**Chess-specific motivation:** The function "evaluate a position" should be exchangeable under arbitrary relabellings of piece types provided the relabelling commutes with the rules — and in many feature-extraction stages (early embedding of which-piece-on-which-square) the rules have not yet been applied, so full S_T symmetry holds locally. Piece-type relabelling structure was flagged in the brief as under-exploited.

**Generalisation beyond chess:** Multi-type particle systems (graph nets for chemistry where atom-type labelling is exchangeable), multi-class detection where class indices are arbitrary, set-of-types-of-sets problems.

**Complexity:**
- Forward: O(2·d_in·d_out·T·64·B) ≈ same as standard `nn.Linear` on flattened channels, with 1/T fewer params
- Backward: same; gradient is computed in the basis, not in the full T×T grid
- Incremental update: composes with RSEA — only changed piece-types update

**Scout-scale falsification test:** Replace the first embedding layer of the user's i243 architecture with P-Tens applied across the 6 (own-side) piece-type planes. Match parameter count of baseline. Metric: matched-recall FP rate on CRTK class 1 at 12 epochs × 173k positions. "Works" = matches baseline FP rate while using ≥2× fewer parameters in that layer. "Fails" = worse FP rate at equal params.

**Failure mode catalogue:**
- **Underexplored, not novel:** I am flagging this honestly — Pearce-Crump (2022) establishes the basis; applying it to chess is new, but the primitive's mathematical content is not. Status downgraded to "underexplored primitive for chess."
- Could be implemented as `Linear` + augmentation and recover similar performance at scout scale; the *parameter saving* (not the accuracy) is the only honest benefit at small data.
- Piece-type exchange does not commute with the chess rules (pawns are special), so the operator must be confined to upstream feature-mixing layers where the structural facts have not yet been injected — limits applicability.

**Status:** proposed

---

### 5. primitive_dajp

**Name:** Discrete-Action Jacobian Primitive (DAJP)

**One-line claim:** A primitive whose output, for input board `s`, is the vector `{f(s) − f(make(s, m)) : m ∈ legal(s)}` evaluated in one operator call with shared accumulator state.

**Mathematical signature:**
Operator `D : R^{B,F} × LegalMoveSet → R^{B,|legal(s)|}`. Internally maintains a single RSEA accumulator; for each legal move `m_i` applies the move-delta `Δ_i`, reads the head, undoes `Δ_i`, accumulates into output position `i`. Gradient flows through `f` for both the base position *and* every successor, with the legal-move iteration treated as a sum-reduction along a varying-length axis.

**Why this does not decompose into existing PyTorch ops:** You could write a Python `for m in legal_moves: y.append(f(s) − f(make(s, m)))` and `torch.stack`, but (a) it materialises |legal_moves| separate forward graphs rather than one, blowing up memory; (b) the gradient w.r.t. `W` requires *de-duplicated* attribution across the shared base state, which manual composition gets wrong (it double-counts). DAJP packages the legal-move iteration as a single op with a custom backward that does the correct credit assignment in one pass — analogous to how `nn.MultiheadAttention` is a primitive even though you *can* write QKV via three `nn.Linear`s.

**Chess-specific motivation:** Hard-negative discrimination on CRTK class 1 is precisely the case where the engine must distinguish a position from its one-ply successors. A primitive whose *output* is the local Jacobian of eval w.r.t. legal moves forces the network to learn a coherent local potential and gives a much sharper training signal than position-level MSE alone.

**Generalisation beyond chess:** Any RL/search setting with a small discrete legal-action set — Go, Atari with frame-skip, theorem proving with a finite tactic library, combinatorial optimisation.

**Complexity (with RSEA underneath):**
- Forward: O(|legal(s)| · k · d) where k = average move-delta size (≈4 for chess), vs O(|legal(s)| · F · d) for naïve full re-forward — roughly 100–500× speedup at typical F
- Backward: O(|legal(s)| · k · d) with single base-state attribution
- Incremental update: same primitive — that *is* its purpose

**Scout-scale falsification test:** Add DAJP as an auxiliary head producing per-legal-move eval-deltas on top of the user's i193 conv-only baseline. Train with MSE on Stockfish-labelled position eval *only* (not on PV data; per the brief, PVs are audit fields). Metric: matched-recall FP rate on CRTK class 1 — the discriminating metric. "Works" = ≥2 pp reduction in near-puzzle FP rate at matched recall. "Fails" = no improvement.

**Failure mode catalogue:**
- Hidden rebrand of "batched forward + subtract" — strongest objection. Counter: the primitive's contribution is the *shared-base credit assignment* in backward; without it, naïve composition double-counts. This is a real distinction in the gradient graph but reviewers will demand a head-to-head with naïve composition.
- Numerically the difference `f(s) − f(make(s,m))` is small (move evaluation deltas are typically tens of centipawns out of thousands), causing catastrophic cancellation in fp16.
- Only useful in tandem with RSEA; standalone, it is too slow because it requires |legal(s)| full forwards (~30× cost per position).

**Status:** proposed

---

## What I Cut (rejected candidates and why)

1. **"Magic-bitboard" sparse convolution with fixed per-piece-type connectivity tensors.** Cut because despite the chess-specific framing, the operator is structurally *just* `Conv` with a non-square fixed receptive field — i.e., a new mask shape rather than a new primitive (explicitly listed as an anti-example in the brief).

2. **Color-equivariant convolution via channel-doubling and weight tying.** Cut because, on inspection, this decomposes into two parallel `Conv2d`s with shared weights and a sum at the end — pure composition, which the brief explicitly forbids ("two streams that share weights").

3. **Eigenvalue-gap value head.** Cut because `torch.linalg.eigh` exists and is differentiable; the "gap" output is just an indexing + subtraction. Fully composable; no new computation graph beyond existing PyTorch eigh autograd rules.

4. **Persistent-homology pawn-structure layer.** The differentiable filtration of pawn-square sublevel sets would be genuinely novel as a primitive, but it cannot be demonstrated on a single RTX 3070 in <2 GPU-hours at 173k positions (current differentiable-PH libraries are CPU-bottlenecked). Violates the reproducibility constraint.

5. **Learned-mixture-of-conv-and-Mamba primitive.** Cut: explicitly listed as anti-example ("learned mixture of attention + conv (composition)"). Composition, not primitive.

---

## Recommendations

1. **Implement RSEA first (highest expected ROI).** Drop into your i243 dual-stream HalfKA harness as a literal differentiable wrapper around the existing accumulator code. Target: 3× wall-clock inference speedup at zero loss-rate cost on 173k positions × 12 epochs. Threshold to escalate to a full ablation suite: the implementation already passes a unit test where `read(apply(unmake(apply(h, Δ)), Δ)) == read(apply(h, Δ))` to fp32 precision *and* gradients pass `torch.autograd.gradcheck` on a 4-feature toy.

2. **Run χ-head as a small, fast, falsifiable side experiment in parallel** (<30 GPU-minutes). It either gives you a free symmetry guarantee at no quality cost — a clean win — or it materially hurts CRTK class-1 FP rate, in which case the data tells you the dual-stream design's asymmetry-handling is doing more work than you expected. Either outcome is publishable as a small note.

3. **Defer BTRS until after RSEA lands.** BTRS is the most likely to be subsumed by an existing operator (Mamba on 8 parallel sequences) and the most expensive to falsify cleanly. Run only if you have a clear 2 GPU-hour slot and a clean Mamba-1D-per-direction baseline ready.

4. **Treat P-Tens as a free architectural deparameterisation** in your *next* scout sweep, not as a hero result. Honest framing: "underexplored for chess," not "novel primitive."

5. **DAJP is contingent on RSEA.** Do not implement standalone — it is 30× slower than per-position eval without the shared-accumulator backbone.

**Stop-conditions / benchmark thresholds:** Drop any primitive that (i) fails `gradcheck` at fp32, (ii) does not improve OR match CRTK class-1 matched-recall FP rate within 1 pp of baseline at equal FLOPs, OR (iii) takes >2× the GPU-time budget you allocated. Promote any primitive that gives ≥2 pp FP-rate improvement on CRTK class 1 *or* ≥2× wall-clock speedup at matched accuracy — both of those clear the "30 Elo from 2× speedup" bar named in the brief.

---

## Caveats

- **The brief's required workflow could not be fully executed.** The `run_blocking_subagent` and `enrich_draft` tools returned "tool was not provided" in this environment; the web_search budget exhausted at 12 calls. The most-vulnerable gap I would have asked a subagent to verify is whether any 2024-2026 paper has independently introduced a "differentiable push/pop accumulator with sparse-event API" — if such a paper exists, RSEA must be re-classed from "novel primitive" to "underexplored primitive for chess."
- **Novelty claims are vulnerable on RSEA and BTRS in particular.** RSEA is closest to NNUE (not differentiable through pop) and DeltaNet (no pop semantics). BTRS is closest to multi-directional Mamba. I have flagged each risk in the relevant failure-mode bullet rather than hiding it.
- **No empirical numbers are quoted in this document.** The brief explicitly forbade inventing results; every "works/fails" criterion is stated as a falsification target, not a prediction.
- **CRTK class 1 (near-puzzle hard negatives) is the load-bearing metric across all five proposals.** If your CRTK class-1 labelling has any quality issues, all five falsification tests fail simultaneously and silently — worth a sanity-check audit before running the suite.
- **All five primitives are flagged as testable on the user's single-RTX-3070 / 173k-positions / 12-epochs harness in <2 GPU-hours**, with the explicit exception that **DAJP requires RSEA** and **BTRS** may exceed 2 hours if applied at every layer rather than as a single one-shot geometric injection layer.
