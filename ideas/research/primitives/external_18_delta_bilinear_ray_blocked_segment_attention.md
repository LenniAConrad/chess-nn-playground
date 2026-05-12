# Invent New Neural Primitives for Chess Evaluation

This document follows the supplied primitive-only scope: propose operators, not architectures, encodings, or training tricks.

## Literature calibration

The recent primitive-level bar is high. Mamba introduced input-conditioned selective state-space dynamics, explicitly allowing the model to propagate or forget information depending on the current token, with linear sequence scaling and fast inference claims relative to Transformers. ([openreview.net](https://openreview.net/forum?id=AL1fq05o7H)) Mamba-2 / SSD then connected SSMs and attention through structured semiseparable matrices and reported a refined selective-SSM core that is 2–8× faster than Mamba’s earlier core. ([arxiv.org](https://arxiv.org/abs/2405.21060)) xLSTM is another genuine primitive-level example: exponential gates plus scalar/matrix memory structures, not merely a bigger LSTM stack. ([huggingface.co](https://huggingface.co/papers/2405.04517)) Gated DeltaNet combines adaptive memory erasure with a delta-rule update for targeted memory modification, and was accepted as ICLR 2025 camera-ready. ([arxiv.org](https://arxiv.org/abs/2412.06464))

Other 2024–2026 calibration points: KAN replaces fixed node activations and linear weights with learnable spline-like edge functions, which is a real operator-level change even though it is not directly ideal for chess speed. ([huggingface.co](https://huggingface.co/papers/2404.19756)) TTT layers and Titans push test-time memory as a primitive: TTT makes the hidden state itself a learned model updated during inference, while Titans proposes a neural long-term memory with fast parallel training and fast inference. ([arxiv.org](https://arxiv.org/abs/2407.04620)) Conditional computation remains relevant because it dynamically activates or deactivates parts of the computation graph based on input. ([journals.sagepub.com](https://journals.sagepub.com/doi/10.3233/IA-240035)) Dynamic GNN surveys also confirm that changing graph topology is a live research direction, but most work treats the graph as an input object rather than compiling domain rules into the operator itself. ([link.springer.com](https://link.springer.com/article/10.1007/s11704-024-3853-2)) Equivariant neural networks are mature around symmetry groups, but chess-specific finite-group operators remain underexplored. ([link.springer.com](https://link.springer.com/article/10.1007/s10462-023-10502-7))

## Ranking summary

Scores are 1–5, where 5 is best. “Novelty” means plausibility of being a primitive rather than a layer; “3070 demo” means likelihood of falsifying in under two GPU-hours; “speed” means plausible inference advantage; “generalisation” means usefulness outside chess.

| Rank | Primitive | Novelty | 3070 demo | Speed | Generalisation | Main risk |
|---:|---|---:|---:|---:|---:|---|
| 1 | `primitive_delta_bilinear_accumulator` | 4 | 5 | 5 | 4 | Could be dismissed as cached factorization machine |
| 2 | `primitive_ray_blocked_scan` | 4 | 5 | 4 | 4 | Could be dismissed as multiple masked selective scans |
| 3 | `primitive_legal_segment_attention` | 4 | 4 | 4 | 5 | Could collapse into graph attention with a generated mask |
| 4 | `primitive_exchange_bellman_reducer` | 5 | 3 | 3 | 3 | May be too chess-specific and numerically sharp |
| 5 | `primitive_orbit_canonicalizer` | 3 | 4 | 3 | 5 | Could be dismissed as canonical input preprocessing |

## Self-audit of top two

**Top 1 devil’s advocate: `primitive_delta_bilinear_accumulator`.** Static second-order sparse interactions are not new: factorization machines already model pairwise interactions and reduce naive $O(kn^2)$ computation to $O(kn)$. ([gabormelli.com](https://www.gabormelli.com/RKB/2010_FactorizationMachines)) Therefore the novelty claim must not be “pairwise sparse interaction.” The only defensible primitive claim is the event-sourced, stateful, custom-autograd update whose forward and backward cost depend on changed active features, not active feature count. If implemented as a normal `EmbeddingBag` plus recomputed FM term, reject it.

**Top 2 devil’s advocate: `primitive_ray_blocked_scan`.** A reviewer can flatten each ray into a sequence and call it “eight Mamba-like scans with masks.” That objection is strong because Mamba already made input-conditioned recurrence a recognized primitive. ([openreview.net](https://openreview.net/forum?id=AL1fq05o7H)) The proposal survives only if the primitive exposes segment boundaries, occlusion resets, and bounded-change ray invalidation as first-class semantics. If every forward scans all rays with a precomputed mask, it is not new enough.

## 1. Proposal

### primitive_delta_bilinear_accumulator

**Name:** Event-Delta Bilinear Accumulator

**One-line claim:** Maintains first- and second-order sparse-set features with forward cost proportional to changed events.

**Mathematical signature:**
$f:(A_{t-1},B_{t-1},Q_{t-1},\Delta_t;U,V)\rightarrow (Y_t,A_t,B_t,Q_t)$, with $U,V\in\mathbb{R}^{M\times d}$, $A,B,Q,Y\in\mathbb{R}^{B\times d}$, and $\Delta_b=\{(i,\sigma_i):i\in[1,M],\sigma_i\in\{-1,+1\}\}$.  
For active set $S_t$:
$A_t=\sum_{i\in S_t}U_i$, $B_t=\sum_{i\in S_t}V_i$,  
$Q_t=\sum_{i<j,\ i,j\in S_t}(U_i\odot V_j+U_j\odot V_i)$,  
$Y_t=\phi(W[A_t;B_t;Q_t]+c)$.  
The primitive computes this exactly by event updates. Add $i$: $Q\leftarrow Q+U_i\odot B+V_i\odot A$, then update $A,B$. Remove $i$: $Q\leftarrow Q-U_i\odot(B-V_i)-V_i\odot(A-U_i)$, then update $A,B$.

**Why this does not decompose into existing PyTorch ops:**
A static call can be emulated by `EmbeddingBag` plus factorization-machine algebra, so the static formula alone is not novel. The primitive claim is the stateful event-log autograd object: it receives a previous accumulator and a bounded change list, updates only touched rows, and produces sparse parameter gradients tied to the event path. `torch.nn.Linear`, `EmbeddingBag`, and ordinary sparse tensors do not expose cross-forward persistent delta semantics.

**Chess-specific motivation:**
NNUE’s power is not just linear accumulation; it is bounded-change accumulation. Chess moves alter a few piece-square facts, but tactics often depend on pairwise relations: king–attacker, pinned piece–slider, bishop pair, rook–open-file, defender–target. This primitive preserves HalfKA-like update cost while adding a controlled second-order term.

**Generalisation beyond chess:**
Useful for recommender sessions, sparse event streams, fraud graphs, and any model over a slowly changing active set.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(|\Delta|d)$ incremental vs `EmbeddingBag` $O(|S|d)$ and dense pairwise $O(|S|^2d)$
- Backward: $O(|\Delta|d)$ along an event path; $O(|S|d)$ if trained as independent static boards
- Incremental update on a bounded-change input: $O(d)$ per added or removed feature

**Scout-scale falsification test:**
Use the i243 HalfKA+dual-stream harness only as a test bed. Replace the HalfKA accumulator with this primitive at the same hidden width. Baseline: original HalfKA accumulator. Train one seed on the 173k-position scout set for the existing 12-epoch budget or a 50k/4-epoch smoke test if time-constrained. Metric: matched-recall CRTK class-1 near-puzzle false-positive rate plus engine-style nodes/sec on legal-move trajectories. Works if near-puzzle FP rate drops by at least 5% at matched recall and cached inference is not more than 20% slower than HalfKA. Fails if improvement appears only in aggregate PR AUC or if static recomputation is required.

**Failure mode catalogue:**
- Hidden rebrand objection: “This is just a factorization machine plus cache.” Correct unless event-sourced autograd is implemented as the primitive.
- Numerical issue: $Q_t$ can drift after long update chains; require periodic exact refresh and fp32 accumulators.
- Speed issue: cache invalidation across batched transpositions may erase the $O(|\Delta|d)$ advantage.

**Status:** proposed

## 2. Proposal

### primitive_ray_blocked_scan

**Name:** Ray-Blocked Selective Scan

**One-line claim:** Performs selective recurrence along visibility rays, with blockers creating input-dependent scan segments.

**Mathematical signature:**
$f:\mathbb{R}^{B\times H\times W\times d}\times[0,1]^{B\times H\times W}\rightarrow\mathbb{R}^{B\times H\times W\times d}$.  
For directions $\delta\in D$ and square $p$, let $p-\delta$ be the previous square on that ray. With opacity $o_p$ and diagonal gates $\alpha_\delta(x_p),\beta_\delta(x_p)\in[0,1]^d$:
$h_{p,\delta}=(1-o_p)\alpha_\delta(x_p)\odot h_{p-\delta,\delta}+\beta_\delta(x_p)\odot W_\delta x_p$, with $h_{\mathrm{offboard},\delta}=0$.  
$y_p=W_0x_p+\sum_{\delta\in D}C_\delta h_{p-\delta,\delta}$.

**Why this does not decompose into existing PyTorch ops:**
It is not `Conv2d`: the receptive field is not a fixed kernel but a content-blocked line segment. It is not ordinary Mamba/selective scan because the recurrence is over multiple 2D ragged ray segments whose reset boundaries are induced by board occupancy. A PyTorch implementation can emulate it with loops, masks, and scans, but the primitive’s graph exposes occlusion resets and changed-ray invalidation directly.

**Chess-specific motivation:**
Sliding attacks, pins, skewers, discovered attacks, and rook/bishop/queen mobility are all ray phenomena. A one-square move changes only a small number of ranks, files, and diagonals, exactly matching the bounded-change requirement. This is likely more scout-scale-friendly than global attention because it hard-codes the right sparsity without using engine labels.

**Generalisation beyond chess:**
Visibility in robotics, grid-world planning, LiDAR occupancy maps, differentiable rendering approximations, and line-of-sight scene graphs.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(BHW|D|d)$ vs dense attention $O(B(HW)^2d)$ and large-kernel conv $O(BHWk^2d^2)$
- Backward: $O(BHW|D|d)$
- Incremental update on a bounded-change input: $O(|R_\Delta|Ld)$, where $R_\Delta$ is the set of rays crossing changed cells; on 8×8 chess, this is constant-bounded

**Scout-scale falsification test:**
Use i193 conv-only as harness. Replace one middle spatial mixing convolution with this primitive at the same channel count. Baseline: same i193 checkpoint/config with the original convolution. Metric: matched-recall CRTK class-1 near-puzzle FP rate and per-position latency. Works if FP rate improves by at least 5% and latency is less than 1.25× baseline. Fails if it only improves easy negatives or needs more than scout-scale data.

**Failure mode catalogue:**
- Hidden rebrand objection: “This is eight masked Mamba scans.” It fails novelty if implemented as full-board masked scans without segment/incremental semantics.
- Numerical issue: repeated multiplicative gates can vanish; use log-domain or bounded residual carry for long rays.
- Speed issue: Python loops over rays will lose; it needs a fused CUDA/Triton segmented-scan kernel.

**Status:** proposed

## 3. Proposal

### primitive_legal_segment_attention

**Name:** Compiled Legal-Edge Segment Attention

**One-line claim:** Builds legal-move edges inside the operator and performs ragged segment attention only on those edges.

**Mathematical signature:**
$f:\mathbb{R}^{B\times64\times d}\times\{0,\dots,12\}^{B\times64}\times\{\pm1\}^B\rightarrow\mathbb{R}^{B\times64\times d}$.  
For board $b$, the primitive internally constructs $E_b=\mathrm{LegalEdges}(P_b,s_b)$, where each edge $e=(u,v,t)$ has source square $u$, destination square $v$, and move type $t$.  
$z_e=((W_Q^t x_v)^\top(W_K^t x_u))/\sqrt{d}+r_t$.  
$\alpha_e=\exp(z_e)/\sum_{e':\mathrm{dst}(e')=v}\exp(z_{e'})$.  
$y_v=W_0x_v+\sum_{e:\mathrm{dst}(e)=v}\alpha_eW_V^t x_u$.

**Why this does not decompose into existing PyTorch ops:**
If $E_b$ is supplied as an external mask, this is merely graph attention or masked attention. The primitive claim is that legal-edge construction, ragged edge packing, segment softmax, and scatter-reduce are one autograd operator with discrete topology generated from the current board. `nn.MultiheadAttention` assumes a tensor mask already exists; it does not compile a variable legal graph from input state.

**Chess-specific motivation:**
The sparse legal-move graph is the most chess-specific connectivity prior. Near-puzzle false positives often come from positions that look tactically similar but differ by one legal move, pinned attacker, or blocked escape square. Attention over all 64² square pairs wastes data and compute; legal-edge attention asks only rule-reachable questions.

**Generalisation beyond chess:**
Dynamic graphs whose edges are generated by rules: traffic right-of-way graphs, molecular reaction candidates, program-analysis graphs, and game-state evaluators.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(E d)$ with $E$ legal or pseudo-legal edges vs dense attention $O(64^2d)$
- Backward: $O(E d)$ plus rule-edge bookkeeping
- Incremental update on a bounded-change input: $O(\Delta E\,d)$ if legal-edge deltas are cached; otherwise $O(E d)$

**Scout-scale falsification test:**
Use i242’s attention-like harness only for evaluation, not as a new architecture. Replace one standard square-attention call with this primitive; keep width, head count, and downstream layers fixed. Baseline: identical harness with dense or masked attention. Metric: matched-recall near-puzzle FP rate, not aggregate PR AUC. Works if class-1 FP rate drops by at least 5% and latency is below dense attention. Fails if edge construction dominates runtime or if gains vanish when easy negatives are removed.

**Failure mode catalogue:**
- Hidden rebrand objection: “This is masked attention.” True if a dense 64×64 mask is materialized outside the op.
- Numerical issue: very low-degree segments can produce brittle attention entropy; include degree-aware temperature.
- Speed issue: legal generation on CPU will erase all gains; bitboard edge generation must be fused or prepacked on GPU.

**Status:** proposed

## 4. Proposal

### primitive_exchange_bellman_reducer

**Name:** Alternating Exchange Bellman Reducer

**One-line claim:** Computes differentiable attack-defense backups over an internally generated capture graph.

**Mathematical signature:**
$f:\mathbb{R}^{B\times64\times d}\times\{0,\dots,12\}^{B\times64}\rightarrow\mathbb{R}^{B\times64\times d'}$.  
For each board, construct capture/defense edges $C=\{e=(u\rightarrow v,c,t)\}$ internally. Let $r_e=w_t^\top[x_u;x_v;x_u\odot x_v]$. For side $c$ and depth $\ell$:
$z_{v,c}^{0}=0$,  
$z_{v,c}^{\ell+1}=\tau\log\sum_{e=(u\rightarrow v,c,t)\in C}\exp((r_e-z_{u,1-c}^{\ell})/\tau)$.  
Output $y_v=R[x_v;z_{v,\mathrm{white}}^K;z_{v,\mathrm{black}}^K]$. Gradients flow through $r_e$, $R$, and all soft Bellman backups; graph topology is discrete.

**Why this does not decompose into existing PyTorch ops:**
This is not GNN message passing with sum, mean, or attention aggregation. It is an alternating max-plus / soft-Bellman reduction over a rule-generated capture graph, with sign reversal between players. It can be simulated with scatter operations and loops, but PyTorch has no primitive whose backward graph is a differentiable adversarial dynamic program over ragged edges.

**Chess-specific motivation:**
Static exchange evaluation is one of the few hand-engineered chess concepts that directly targets tactical false positives. Near-puzzle negatives often differ by whether a sacrifice actually wins material after recaptures. This primitive gives the network a differentiable exchange operator without feeding it Stockfish scores, PVs, or verification metadata.

**Generalisation beyond chess:**
Adversarial resource graphs, cybersecurity attack-defense chains, contested auctions, pursuit-evasion games, and multi-agent planning on sparse interaction graphs.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(K|C|d)$ vs dense attention $O(64^2d)$; closest GAT is $O(K|C|d)$ but with a different semiring
- Backward: $O(K|C|d)$
- Incremental update on a bounded-change input: $O(K\Delta C\,d)$ with cached capture-edge deltas; otherwise $O(K|C|d)$

**Scout-scale falsification test:**
Use i193 as the test harness. Replace one late spatial mixing block with this primitive projected back to the same channel count. Baseline: the original block and, separately, a GAT-style capture-edge message-passing control. Metric: matched-recall CRTK class-1 near-puzzle FP rate. Works if it beats both baselines on class-1 FP rate while keeping latency under 1.5× i193. Fails if it only improves positions with obvious material imbalance.

**Failure mode catalogue:**
- Hidden rebrand objection: “This is just a GNN.” The counterclaim survives only if the alternating Bellman semiring is implemented as the primitive, not as attention.
- Numerical issue: low $\tau$ causes hard-max saturation; high $\tau$ blurs exchange order.
- Speed issue: depth $K>4$ may be too slow for engine inference.

**Status:** proposed

## 5. Proposal

### primitive_orbit_canonicalizer

**Name:** Straight-Through Chess-Orbit Canonicalizer

**One-line claim:** Selects a symmetry-canonical feature frame by hard group routing with a custom equivariant backward pass.

**Mathematical signature:**
$f:\mathbb{R}^{B\times64\times d}\rightarrow\mathbb{R}^{B\times64\times d}$.  
Let finite group $G$ act on square indices and signed feature channels. For each $g\in G$, compute additive score $s_g(X)=a^\top\sum_i\psi((T_gX)_i)$.  
$g^\*=\arg\max_{g\in G}(s_g+\epsilon_g)$ with deterministic tie-break $\epsilon_g$.  
Forward: $Y=T_{g^\*}^{-1}X$.  
Backward: straight-through $\partial Y/\partial X=T_{g^\*}^{-1}$, and optional score-gradient surrogate $\partial g^\*/\partial s\approx\partial\mathrm{softmax}(s/\tau)/\partial s$.

**Why this does not decompose into existing PyTorch ops:**
It is not data augmentation and not group convolution. It is conditional computation over a finite symmetry group: one group action is selected and applied to the entire tensor, with custom straight-through gradient and stabilizer-aware tie-breaking. `max`, `gather`, and `permute` can emulate parts, but not the intended group-routing autograd primitive as a reusable operator.

**Chess-specific motivation:**
Chess has board symmetries plus color-swap / side-to-move involutions. Existing small-data scouts waste capacity learning equivalent orientations. This primitive canonicalizes hidden features inside the model, avoiding a pure input-encoding proposal while still exploiting the chess group.

**Generalisation beyond chess:**
Molecular conformer canonicalization, crystal symmetries, board games, robotic scene frames, and any finite-group equivariant model that wants hard canonical routing rather than averaging.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(|G|64d)$ for scoring plus $O(64d)$ for selected transform vs group convolution $O(|G|64d^2)$
- Backward: $O(64d)$ for straight-through path, plus $O(|G|64d)$ if training the scoring surrogate
- Incremental update on a bounded-change input: $O(|G|\Delta d)$ if scores are additive and cached; otherwise $O(|G|64d)$

**Scout-scale falsification test:**
Apply after the first hidden feature map in i193, not as raw input preprocessing. Baseline: same model with normal D4/color augmentation disabled and enabled as two controls. Metric: matched-recall near-puzzle FP rate plus calibration drift between original and transformed boards. Works if it reduces transform inconsistency and class-1 FP rate without hurting latency by more than 20%. Fails if it merely duplicates augmentation benefits.

**Failure mode catalogue:**
- Hidden rebrand objection: “This is input canonicalization.” It fails if used only before the network; it must operate on hidden tensors.
- Numerical issue: near-tie boards create discontinuous routing; deterministic stabilizer averaging is needed.
- Speed issue: scoring all group elements may dominate small conv models unless cached additively.

**Status:** proposed

## What I cut

1. **External-mask legal attention.** Rejected because it is exactly the anti-example: `(QK^T + mask).softmax()V`. Only the compiled legal-edge version survives by making edge construction internal and ragged.

2. **KAN-style chess edge splines.** KAN is already a 2024 primitive family with learnable edge functions, so a chess KAN layer would be “underexplored for chess,” not a new primitive. ([huggingface.co](https://huggingface.co/papers/2404.19756))

3. **Differential-attention-for-tactics.** Differential Transformer subtracts two softmax attention maps to reduce irrelevant context, but a chess version would still be an attention variant unless the topology or gradient semantics changed. ([huggingface.co](https://huggingface.co/papers/2410.05258))

4. **Two-stream HalfKA plus conv/attention hybrids.** These are architecture compositions. They may be useful, but they violate the primitive-only requirement.

5. **Piece-type relabel equivariant linear layer.** Too close to standard group convolution / equivariant linear maps. The chess group is interesting, but a plain orbit-tied linear map is not novel enough given the existing equivariant-network literature. ([link.springer.com](https://link.springer.com/article/10.1007/s10462-023-10502-7))

## Bibliography

- Gu and Dao, **“Mamba: Linear-Time Sequence Modeling with Selective State Spaces,”** ICLR 2024 / OpenReview. ([openreview.net](https://openreview.net/forum?id=AL1fq05o7H))
- Dao and Gu, **“Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality,”** ICML 2024. ([arxiv.org](https://arxiv.org/abs/2405.21060))
- Beck et al., **“xLSTM: Extended Long Short-Term Memory,”** 2024. ([huggingface.co](https://huggingface.co/papers/2405.04517))
- Yang, Kautz, and Hatamizadeh, **“Gated Delta Networks: Improving Mamba2 with Delta Rule,”** ICLR 2025 camera-ready. ([arxiv.org](https://arxiv.org/abs/2412.06464))
- Liu et al., **“KAN: Kolmogorov-Arnold Networks,”** 2024. ([huggingface.co](https://huggingface.co/papers/2404.19756))
- Sun et al., **“Learning to (Learn at Test Time): RNNs with Expressive Hidden States,”** 2024 / revised 2025. ([arxiv.org](https://arxiv.org/abs/2407.04620))
- Behrouz, Zhong, and Mirrokni, **“Titans: Learning to Memorize at Test Time,”** Google Research, 2025. ([research.google](https://research.google/pubs/titans-learning-to-memorize-at-test-time/))
- Scardapane et al., **“Conditional computation in neural networks: Principles and research trends,”** 2024. ([journals.sagepub.com](https://journals.sagepub.com/doi/10.3233/IA-240035))
- Zheng, Yi, and Wei, **“A survey of dynamic graph neural networks,”** Frontiers of Computer Science, 2025. ([link.springer.com](https://link.springer.com/article/10.1007/s11704-024-3853-2))
- Gerken et al., **“Geometric deep learning and equivariant neural networks,”** Artificial Intelligence Review, 2023. ([link.springer.com](https://link.springer.com/article/10.1007/s10462-023-10502-7))
- Rendle, **“Factorization Machines,”** ICDM 2010. ([gabormelli.com](https://www.gabormelli.com/RKB/2010_FactorizationMachines))
