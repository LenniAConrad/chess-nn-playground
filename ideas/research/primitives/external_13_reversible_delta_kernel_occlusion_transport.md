# Deep Research Primitive Results: New Neural Primitives for Chess Evaluation

The five proposals below are ordered by overall rank: novelty plausibility, RTX 3070 demonstrability, inference-speed upside, and generalisation beyond chess.

### primitive_01_reversible_delta_kernel_memory

**Name:** Reversible Delta Kernel Memory

**One-line claim:** A linear-attention-like set memory with exact signed insert/delete updates for bounded-change inputs.

**Mathematical signature:**
Let dynamic set state be `(M,z)`, with `M ∈ R^{B×h×v}`, `z ∈ R^{B×h}`. Inputs are queries `Q ∈ R^{B×q×d}` and signed events `Δ={(s_l,u_l)}_{l=1..k}`, `s_l∈{-1,+1}`, `u_l∈R^d`. With learned maps `φ:R^d→R_+^h`, `ν:R^d→R^v`:
\[
M' = M + \sum_{l=1}^k s_l\,φ(u_l)ν(u_l)^T,\quad
z' = z + \sum_{l=1}^k s_l\,φ(u_l)
\]
\[
Y_{b,j}=\frac{φ(Q_{b,j})^TM'_b}{φ(Q_{b,j})^Tz'_b+\epsilon}
\]
`fθ: (Q, Δ, M, z) → (Y, M', z')`. Gradients are standard except where an implementation deliberately detaches old cached state during inference.

**Why this does not decompose into existing PyTorch ops:**
The closest PyTorch primitive is `MultiheadAttention`, which forms attention heads from `Q,K,V`, masks optional positions, and evaluates dense or masked attention maps; PyTorch sparse softmax normalises already-specified sparse entries but does not define reversible set-state updates. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html?utm_source=chatgpt.com)) This operator’s signature is state plus signed delta events, not `Q,K,V`. A naive emulation can be written with gathers and adds, but the primitive’s computation graph has no dependency on unchanged inactive items and admits row-sparse event gradients.

**Chess-specific motivation:**
Stockfish NNUE’s accumulator update subtracts removed feature columns and adds new feature columns instead of recomputing the first layer; the official NNUE docs show exactly this add/subtract update pattern. ([official-stockfish.github.io](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)) This primitive generalises that idea from unary HalfKA-style accumulation to kernel memory, so moved/captured pieces update interaction memory without rescanning all 64 squares.

**Generalisation beyond chess:**
Useful for dynamic scene graphs, recommender sessions, molecular simulations with atom insert/delete events, and sparse-event time series.

**Complexity (forward, backward, incremental-update):**
- Forward: build-from-scratch `O(B·(n+q)·h·v)` vs dense attention `O(B·q·n·d)`
- Backward: `O(B·(n+q)·h·v)` in training
- Incremental update on a bounded-change input: `O(B·k·h·v + B·q_changed·h·v)`

**Scout-scale falsification test:**
Drop into i243 as the interaction primitive over active piece tokens, replacing the proposed dual-stream attention interaction while keeping parameter count within ±10%. Baseline: i243 dual-stream and i193 conv-only. Metric: CRTK class-1 matched-recall near-puzzle false-positive rate plus batch inference latency. “Works” if FP rate drops by ≥5% versus i243 at ≤1.25× i193 latency; “fails” if no FP gain or latency exceeds 1.5× i193.

**Failure mode catalogue:**
- Strongest reviewer objection: “This is just linear attention with an accumulator.” The differentiator is exact signed deletion over dynamic sets, not causal sequence accumulation.
- Signed deletion can make `z'` small or negative unless `φ` and damping are constrained.
- On GPU batches with no temporal locality, the stateful advantage may disappear.

**Status:** proposed

### primitive_02_occlusion_scanned_move_transport

**Name:** Occlusion-Scanned Move Transport

**One-line claim:** A differentiable ray-scan transport operator whose connectivity is created by occupancy and first-blocker structure.

**Mathematical signature:**
Input board tokens `X∈R^{B×64×d}`, occupancy probabilities `o∈[0,1]^{B×64}`, piece labels or learned type embeddings `p`, directions `r∈R` with valid ray squares `π(s,r,l)`. For square `s`, direction `r`, distance `l`:
\[
v_{s,r,l}=g_\theta(p_s,p_{\pi(s,r,l)},r,l)\prod_{u=1}^{l-1}(1-o_{\pi(s,r,u)})
\]
\[
Y_{b,s}=\sum_{r∈R}\sum_{l:π(s,r,l)\ valid} v_{s,r,l}\,W_{r,l}X_{b,π(s,r,l)}
\]
`fθ: R^{B×64×d}×[0,1]^{B×64}×P^{B×64}→R^{B×64×d'}`. Gradients through `o` use the product rule; with hard occupancy, gradient to occupancy is N/A.

**Why this does not decompose into existing PyTorch ops:**
This is not masked attention: there is no `QK^T`, no softmax, and no externally supplied fixed mask. PyTorch sparse softmax treats unspecified entries as `−∞` and normalises specified entries; it does not compute content-dependent line-of-sight edges. ([docs.pytorch.org](https://docs.pytorch.org/docs/2.12/generated/torch.sparse.softmax.html)) A hand-written `cumprod + gather + matmul` version is an implementation sketch, but the primitive’s mathematical connectivity is a prefix-scan first-blocker graph.

**Chess-specific motivation:**
Sliding-piece legality is occlusion-structured: rooks, bishops, and queens see until a blocker, while knights and kings are local leapers. This primitive directly represents attack rays, pins, skewers, and discovered attacks without asking attention to learn board geometry from small data.

**Generalisation beyond chess:**
Line-of-sight robotics, grid-world planning, visibility in scene graphs, ray-based medical imaging, and sparse cellular simulations.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(B·64·|R|·L·d')`, with `L≤7`, vs attention `O(B·64²·d')`
- Backward: `O(B·64·|R|·L·d')`
- Incremental update on a bounded-change input: `O(B·c·|R|·L·d')` for `c` changed squares, assuming cached rays

**Scout-scale falsification test:**
Drop one instance into i193 by replacing one same-width `3×3` convolution with this primitive. Baseline: i193 conv-only at the same width. Metric: CRTK class-1 matched-recall near-puzzle FP rate and positions/sec. “Works” if FP rate improves by ≥5% while positions/sec stays within 30% of baseline; “fails” if it only improves easy negatives or is slower than 1.3× baseline.

**Failure mode catalogue:**
- Reviewer objection: “It is dynamic sparse attention with a handcrafted mask.” Counter: the operator is a differentiable visibility scan, not a score-mask-softmax-value pipeline.
- Products of many `(1-o)` terms can underflow or saturate; log-domain scan may be needed.
- If implemented naively with Python loops, it will be slower than conv despite better asymptotics.

**Status:** proposed

### primitive_03_incremental_pair_accumulator

**Name:** Incremental Pair Accumulator

**One-line claim:** A second-order set primitive that caches unordered pair interactions and updates only pairs touched by a move.

**Mathematical signature:**
For active items `S_t={u_i,r_i}_{i=1..n}`, item embedding `u_i∈R^d`, relation code `r_{ij}`, and symmetric learned pair map `Φθ(u_i,u_j,r_{ij})∈R^{d_o}`:
\[
P_t=\sum_{1≤i<j≤n} Φ_\theta(u_i,u_j,r_{ij})
\]
For signed item deltas `Δ^-` removed from `S_{t-1}` and `Δ^+` added to `S_t`:
\[
P_t=P_{t-1}
-\sum_{a∈Δ^-}\sum_{j∈S_{t-1}\setminus\{a\}}Φ_\theta(a,j,r_{aj})
+\sum_{a∈Δ^+}\sum_{j∈S_t\setminus\{a\}}Φ_\theta(a,j,r_{aj})
\]
`fθ:(S,Δ,P_{t-1})→P_t`, optionally returning nodewise `Y_i=Σ_{j≠i}Φθ(i,j,r_{ij})`.

**Why this does not decompose into existing PyTorch ops:**
The closest implementation is `einsum` or pairwise attention over an explicit `n×n` tensor. That recomputes all pairs and has a dense pair graph every forward. This primitive’s graph is an unordered-pair cache with exact remove/add semantics; unchanged pairs are absent from the incremental computation graph.

**Chess-specific motivation:**
Many hard positions are pair-structured: king-piece distance, pinned piece plus pinner, defender plus target, discovered attack pair, and overloaded defender pair. Unary NNUE accumulation is fast but weak on second-order interactions; this primitive keeps the NNUE update philosophy while adding explicit pair terms.

**Generalisation beyond chess:**
Dynamic molecular graphs, object-centric video, recommender co-occurrence state, and physics systems where pair potentials are updated after sparse events.

**Complexity (forward, backward, incremental-update):**
- Forward: scratch `O(n²·d_o)` vs attention `O(n²·d)` plus softmax; no score matrix required
- Backward: scratch `O(n²·d_o)`
- Incremental update on a bounded-change input: `O(|Δ|·n·d_o)`; in chess, `n≤32`, so effectively constant with respect to board squares

**Scout-scale falsification test:**
Drop into i242 as the replacement for one pairwise/attention interaction stage, using the same token embeddings and output width. Baseline: i242 full chess-decomposed attention and i193 conv-only. Metric: CRTK class-1 matched-recall FP rate and wall-clock evaluation speed. “Works” if it matches or beats i242 FP rate with ≥1.5× faster inference; “fails” if it is merely slower attention without FP improvement.

**Failure mode catalogue:**
- Reviewer objection: “This is just all-pairs MLP.” The proposed primitive is the incremental cached pair state, not the scratch pair function.
- Pair cache drift can occur if move/unmove bookkeeping is inconsistent.
- In non-chess domains with large `n`, `O(|Δ|n)` may still be too slow.

**Status:** proposed

### primitive_04_alternating_soft_exchange_scan

**Name:** Alternating Soft-Exchange Scan

**One-line claim:** A differentiable minimax-style scan over sparse attacker streams for exchange-value discrimination.

**Mathematical signature:**
For each target `t`, input an ordered sparse stream of candidate exchange gains `G∈R^{B×T×K}` and valid mask `m∈{0,1}^{B×T×K}`. The stream may be produced by any upstream edge extractor; this primitive only consumes the stream. Define:
\[
r_{K+1}=0,\quad r_k=m_k\left(G_k-\operatorname{softplus}_\tau(r_{k+1})\right)+(1-m_k)r_{k+1}
\]
\[
Y_t=[r_1,\max_k r_k,\operatorname{mean}_k m_k r_k]
\]
`fτ:R^{B×T×K}×{0,1}^{B×T×K}→R^{B×T×3}`. A stronger variant internally uses differentiable sparse top-k ordering; sparse differentiable top-k is an active primitive-level research line, not a stock PyTorch module. ([proceedings.mlr.press](https://proceedings.mlr.press/v202/sander23a.html))

**Why this does not decompose into existing PyTorch ops:**
This is neither pooling nor attention: the output is an alternating adversarial recurrence, where each step’s value is subtracted through a soft minimax continuation. A loop can emulate it, but the primitive is a scan with a custom backward over a sparse variable-length stream. PyTorch has no `nn` primitive for alternating soft minimax contraction.

**Chess-specific motivation:**
Near-puzzle false positives often look positionally plausible but fail because a capture sequence loses material or because a defender recaptures. This primitive targets static-exchange-like discrimination without feeding Stockfish scores, PVs, or node metadata into the graph.

**Generalisation beyond chess:**
Adversarial auctions, resource capture games, security patrolling, negotiation rollouts, and any alternating ownership process with sparse candidate actions.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(B·T·K)` after sparse stream construction vs attention-style `O(B·T·K²·d)`
- Backward: `O(B·T·K)`
- Incremental update on a bounded-change input: `O(K)` for affected targets, or `O(K log K)` if internal soft ordering is recomputed

**Scout-scale falsification test:**
Drop into i193 as a three-channel auxiliary primitive before the final classifier, using attack streams derived only from current board tokens. Baseline: same i193 with a parameter-matched MLP channel. Metric: CRTK class-1 matched-recall FP rate. “Works” if class-1 FP drops ≥7% without reducing aggregate PR AUC; “fails” if gains appear only on easy negatives.

**Failure mode catalogue:**
- Reviewer objection: “This is handcrafted static exchange evaluation.” Counter: gains and ordering can be learned; only the alternating contraction is fixed.
- Softplus temperature can make gradients vanish if too low or blur tactics if too high.
- Stream construction may dominate runtime unless fused with a sparse move-edge primitive.

**Status:** proposed

### primitive_05_signed_chess_orbit_norm

**Name:** Signed Chess Orbit Normalization

**One-line claim:** A normalization primitive over chess symmetry orbits, including color-swap antisymmetry.

**Mathematical signature:**
Let finite group `G` act on token indices by `π_g` and feature channels by signed/permutation representation `R_g`. Let `χ_g∈{−1,+1}` encode whether the output value should flip sign under that transform. For `X∈R^{B×n×d}`:
\[
\tilde X_g = χ_g\,R_g X_{\pi_g}
\]
\[
\mu_{b,i,c}=\frac1{|G|}\sum_{g∈G}\tilde X_{g,b,i,c},\quad
\sigma^2_{b,i,c}=\frac1{|G|}\sum_{g∈G}(\tilde X_{g,b,i,c}-\mu_{b,i,c})^2
\]
\[
Y_{b,i,c}=\gamma_c\frac{X_{b,i,c}-\mu_{b,i,c}}{\sqrt{\sigma^2_{b,i,c}+\epsilon}}+\beta_c
\]
with `γ,β` constrained to the same signed representation. `f_G:R^{B×n×d}→R^{B×n×d}`.

**Why this does not decompose into existing PyTorch ops:**
LayerNorm normalises over feature dimensions inside one sample; this normalises over transformed group orbits with signed channel actions. Group-equivariant convolution already exists and uses weight sharing over group actions, but this is a normalization primitive, not a convolution. Group-equivariant CNNs and EGNNs establish the broader equivariant-operator precedent; this proposal is narrower and should be claimed as underexplored for chess, not as the first equivariant primitive. ([proceedings.mlr.press](https://proceedings.mlr.press/v48/cohenc16.html))

**Chess-specific motivation:**
Chess has legal symmetries beyond plain board translation: file mirror, side-to-move handling, and color-swap with board reorientation. NNUE documentation explicitly discusses using mirrored positions and color reversal for the black perspective. ([official-stockfish.github.io](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)) This primitive forces normalization statistics to respect that structure.

**Generalisation beyond chess:**
Finite-symmetry board games, molecules with discrete automorphisms, robotics under mirror symmetries, and equivariant vision models with signed labels.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(B·|G|·n·d)` vs LayerNorm `O(B·n·d)`
- Backward: `O(B·|G|·n·d)`
- Incremental update on a bounded-change input: `O(B·|G|·d)` for affected orbit entries, if cached statistics are maintained

**Scout-scale falsification test:**
Replace LayerNorm/BatchNorm sites in i193 with this primitive; no architecture changes. Baseline: i193 with its original normalization. Metric: CRTK class-1 matched-recall FP rate plus positions/sec. “Works” if FP improves ≥3% with <10% speed loss; “fails” if only calibration improves or speed loss exceeds 20%.

**Failure mode catalogue:**
- Reviewer objection: “This is LayerNorm plus permutations.” The differentiator is signed group-orbit statistics and constrained affine parameters.
- Wrong chess group choice can bake in false equivalences.
- Orbit expansion may be too expensive unless transforms are fused.

**Status:** proposed

## Self-audit notes for the top two

**primitive_01_reversible_delta_kernel_memory:** The strongest hidden-rebrand proof attempt is: “linear attention already stores `Σφ(k)v^T`; adding signed removals is just subtraction.” I do not think that proves rebrand, because published linear-attention and delta-rule systems are sequence-memory operators, while this proposal’s primitive signature is dynamic set state plus signed insert/delete deltas. It overlaps with the research direction of Gated DeltaNet and Kimi Delta Attention, so the novelty claim should be “reversible dynamic-set kernel memory,” not “first kernel memory.” Gated DeltaNet explicitly combines gating with a delta update mechanism, and Kimi Delta Attention extends that family with finer-grained gating. ([huggingface.co](https://huggingface.co/papers/2412.06464))

**primitive_02_occlusion_scanned_move_transport:** The strongest hidden-rebrand proof attempt is: “compute legal-move edges, feed them as a sparse attention mask.” That fails because the proposed operator has no attention score matrix, no softmax normalisation, and no externally supplied mask; its edges and weights are the result of a differentiable prefix visibility scan. It is related in spirit to dynamic sparse attention work, but the primitive claim is the first-blocker scan transport, not generic dynamic sparsity. Recent content-based sparse attention work reinforces why this distinction matters: dynamic selection is an active primitive-level area rather than a mere mask shape. ([papers.cool](https://papers.cool/arxiv/2505.00315))

## What I cut during self-audit

- **Legal-move attention mask:** cut because it is still standard masked attention; PyTorch `MultiheadAttention` already supports masks, and sparse softmax already treats unspecified entries as excluded. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html?utm_source=chatgpt.com))
- **Raw chess group convolution:** cut because group convolution is already a known primitive from G-CNNs; the retained proposal is OrbitNorm, a normalization analogue rather than another group convolution. ([proceedings.mlr.press](https://proceedings.mlr.press/v48/cohenc16.html))
- **Piece-type MoE router:** cut because it is a straightforward mixture-of-experts gate with chess labels.
- **Color-swap data augmentation / canonicalization:** cut because it is an input encoding or training trick, not a primitive.
- **New activation for tactical sharpness:** cut because “GELU but chess-shaped” is only an activation tweak unless it changes signature, connectivity, or gradient flow.

## Ranking matrix

| Rank | Primitive | Novelty plausibility | RTX 3070 demonstrability | Inference-speed advantage | Generalisation beyond chess |
|---:|---|---|---|---|---|
| 1 | Reversible Delta Kernel Memory | High | Medium | Very high if temporal locality is used | High |
| 2 | Occlusion-Scanned Move Transport | High | High | High on 64-square boards | Medium-high |
| 3 | Incremental Pair Accumulator | Medium-high | High | High for chess-sized active sets | High |
| 4 | Alternating Soft-Exchange Scan | Medium-high | Medium | Medium | Medium |
| 5 | Signed Chess Orbit Normalization | Medium | Very high | Low-medium | High |

## Bibliography

- Gu and Dao, **“Mamba: Linear-Time Sequence Modeling with Selective State Spaces”**. Used as a calibration point for input-conditioned recurrent/SSM primitives and hardware-aware linear-time sequence modeling. ([huggingface.co](https://huggingface.co/papers/2312.00752))
- Dao and Gu, **“Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality”**. Used as calibration for Mamba-2/SSD and the importance of faster primitive kernels. ([huggingface.co](https://huggingface.co/papers/2405.21060))
- Beck et al., **“xLSTM: Extended Long Short-Term Memory”**. Used as a recent example of primitive-level recurrence/memory modifications: exponential gating, scalar memory, and matrix memory. ([apointa.github.io](https://apointa.github.io/publication/2024-xlstm.html))
- Yang, Kautz, and Hatamizadeh, **“Gated Delta Networks: Improving Mamba2 with Delta Rule”**. Used as the closest prior line for delta-rule memory updates. ([huggingface.co](https://huggingface.co/papers/2412.06464))
- Kimi Team, **“Kimi Linear: An Expressive, Efficient Attention Architecture”**. Used as 2025 evidence that gated delta/linear-attention primitives remain active and speed-critical. ([huggingface.co](https://huggingface.co/papers/2510.26692))
- Behrouz, Zhong, and Mirrokni, **“Titans: Learning to Memorize at Test Time”**. Used as recent context for neural long-term memory primitives with fast training/inference claims. ([research.google](https://research.google/pubs/titans-learning-to-memorize-at-test-time/))
- Sander et al., **“Fast, Differentiable and Sparse Top-k: a Convex Analysis Perspective”**. Used for the differentiable sparse ordering/top-k reference in the exchange-scan proposal. ([proceedings.mlr.press](https://proceedings.mlr.press/v202/sander23a.html))
- Cohen and Welling, **“Group Equivariant Convolutional Networks”**. Used to separate true new group-orbit normalization from already-known group convolution. ([proceedings.mlr.press](https://proceedings.mlr.press/v48/cohenc16.html))
- Satorras, Hoogeboom, and Welling, **“E(n) Equivariant Graph Neural Networks”**. Used as broader equivariant-graph precedent and generalisation context. ([proceedings.mlr.press](https://proceedings.mlr.press/v139/satorras21a.html))
- PyTorch documentation, **`torch.nn.MultiheadAttention`** and **`torch.sparse.softmax`**. Used to audit whether proposals collapse into standard attention masks or sparse softmax. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html?utm_source=chatgpt.com))
- Stockfish NNUE documentation, **“NNUE”**. Used for accumulator update mechanics, perspective handling, and HalfKP/HalfKA-style update motivation. ([official-stockfish.github.io](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html))
