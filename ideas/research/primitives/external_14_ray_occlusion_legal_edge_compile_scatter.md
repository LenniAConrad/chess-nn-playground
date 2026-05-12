# Invent New Neural Primitives for Chess Evaluation

This document proposes neural-network primitives for `chess-nn-playground` under the constraint that the object of invention is the operator itself, not an architecture, input encoding, loss, scheduler, or training trick.

Notation: \(B\)=batch size, \(n\)=board tokens/squares, \(s=\sqrt n\), \(d\)=channel width, \(R\)=ray directions, \(e\)=compiled sparse edges, \(k\)=bounded edit count after a move, \(M\)=candidate legal moves, \(G_\chi\)=finite chess-rule symmetry group.

## Literature calibration used

Recent primitive-level work raises the novelty bar mostly through new state-update laws, connectivity patterns, and hardware-aware scan/attention kernels rather than through new block diagrams. Relevant calibration points include [Mamba / selective SSM](https://arxiv.org/abs/2312.00752), [Mamba-2 / structured state-space duality](https://arxiv.org/abs/2405.21060), [xLSTM](https://arxiv.org/abs/2405.04517), [Griffin / gated linear recurrence](https://arxiv.org/abs/2402.19427), [DeltaNet parallelisation](https://arxiv.org/abs/2406.06484), [Gated DeltaNet](https://arxiv.org/abs/2412.06464), [Log-Linear Attention](https://arxiv.org/abs/2506.04761), [DeltaProduct](https://arxiv.org/abs/2502.10297), and [Kimi Linear / Kimi Delta Attention](https://arxiv.org/abs/2510.26692). For overlap checks, the most important older references are [Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html), [Graph Attention Networks](https://arxiv.org/abs/1710.10903), [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212), and the [official Stockfish NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html).

## Numbered proposals

### primitive_ray_occlusion_scan

**Name:** Occlusion-Gated Ray Scan

**One-line claim:** A differentiable line-of-sight scan where blockers modulate what information can pass along each ray.

**Mathematical signature:**
\[
f: \mathbb{R}^{B\times n\times d_v}\times[0,1]^{B\times n\times R}\rightarrow\mathbb{R}^{B\times n\times R\times d_o}.
\]
For square \(i\), ray \(r\), ray cells \(\pi_r(i,t)\), and \(L_{i,r}\le s\):
\[
y_{b,i,r}=\sum_{t=1}^{L_{i,r}}
\left(\prod_{u=1}^{t-1}(1-g_{b,\pi_r(i,u),r})\right)
V_{b,\pi_r(i,t)}W_{r,t}.
\]
The gradient is defined through \(V\), \(W\), and blocker gates \(g\); the geometric ray index \(\pi\) is discrete and fixed.

**Why this does not decompose into existing PyTorch ops:**
The closest existing primitive is `Conv2d`, but convolution has fixed local kernels and cannot express multiplicative “stop when blocked” visibility. Masked attention can emulate ray reachability only by materializing an \(n\times n\) mask and using softmax, while this operator is a directional prefix-product scan with a different sparse adjoint. It is closer in spirit to scan-style sequence primitives such as Mamba, but Mamba’s recurrence is along a sequence with input-conditioned SSM parameters, not along board rays with occlusion products.

**Chess-specific motivation:**
Sliding-piece tactics are ray geometry: pins, skewers, discovered attacks, batteries, and x-rays all depend on blockers between two squares. Standard \(3\times3\) convs need depth to discover this; attention sees it but pays \(O(n^2)\) and was data-hungry in the prior scout. This primitive directly gives bishops, rooks, queens, and king lines the right inductive bias.

**Generalisation beyond chess:**
Useful for visibility in gridworlds, robotics, occupancy maps, radiology line integrals, and scene graphs where occluders determine long-range interaction.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BRnsd)\) vs full attention \(O(Bn^2d)\); on \(8\times8\), \(s\le8\) is constant.
- Backward: \(O(BRnsd)\), with prefix/suffix products for blocker gradients.
- Incremental update on a bounded-change input: \(O(Rsd)\); constant on a fixed chessboard.

**Scout-scale falsification test:**
Drop one `Conv2d(3x3)` block in i193 with `RayScan + 1x1 projection`, matching parameter count. Baseline: original i193 conv-only. Train on the existing 173k positions × 12 epochs. Primary metric: matched-recall CRTK class-1 near-puzzle false-positive rate. Works if FP rate drops by at least 5% relative at matched recall and inference slowdown is <1.25×. Fails if gains appear only in aggregate PR AUC or nodes/sec drops >1.5×.

**Failure mode catalogue:**
- Hidden rebrand risk: “This is just `cumprod + gather + matmul`.” Response: that emulation is possible but not a PyTorch primitive; the proposed op’s sparse ray topology and custom adjoint are the primitive.
- Blocker gates near 0 or 1 can cause vanishing gradients along long rays.
- On non-chess grids with large \(s\), \(O(ns)\) may lose to sparse attention unless rays are short or cached.

**Status:** proposed

### primitive_legal_edge_compile_scatter

**Name:** Content-Compiled Legal Edge Scatter

**One-line claim:** A sparse operator that builds legal-move edges inside the primitive, then scatters typed messages over them.

**Mathematical signature:**
\[
f:\{0,\ldots,C\}^{B\times n}\times\mathbb{R}^{B\times n\times d}\rightarrow\mathbb{R}^{B\times n\times d_o}.
\]
For board symbols \(P_b\), compile typed edges:
\[
E_b=\mathcal{C}(P_b)=\{(i,j,\tau)\},
\]
where \(\mathcal{C}\) is a deterministic legal/attack-edge compiler. Then:
\[
y_{b,j}=\sum_{(i,j,\tau)\in E_b}
\sigma(a_\tau^\top[x_{b,i},x_{b,j}])\; W_\tau x_{b,i}.
\]
Gradients flow through \(x,a,W\), not through discrete edge existence.

**Why this does not decompose into existing PyTorch ops:**
If \(E_b\) is precomputed outside the graph, this degenerates into a Graph Attention Network or MPNN-style message pass; those are known operators over supplied graphs. The proposed primitive’s signature includes the discrete state \(P\) and compiles content-dependent sparse connectivity inside the op, avoiding an \(n\times n\) mask and giving a different forward/backward graph. It is not standard masked attention, because the mask is neither fixed nor an input tensor.

**Chess-specific motivation:**
Chess connectivity changes every position: legal moves, attacks, pins, and discovered lines depend on piece identity, blockers, side to move, and king safety. This primitive attacks the specific failure mode where generic attention wastes sample budget relearning legal connectivity from \(10^5\)-scale data. It is especially aimed at CRTK near-puzzle false positives, where plausible-looking but illegal or tactically pinned relations matter.

**Generalisation beyond chess:**
Applies to dynamic graphs where edges are compiled from state: program control-flow graphs, simulators with contact constraints, routing networks, molecular reaction graphs, and event-driven scene graphs.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B(e d+d\cdot \text{compile}))\) vs masked attention \(O(Bn^2d)\).
- Backward: \(O(Bed)\), sparse over compiled edges.
- Incremental update on a bounded-change input: \(O(\Delta e\,d)\); for chess, \(\Delta e=O(s)\) rays plus moved-piece edges, constant on \(8\times8\).

**Scout-scale falsification test:**
Drop one instance into i193 after the first spatial block; baseline against i193 and a GAT-style layer using externally precomputed pseudo-legal edges. Same 173k × 12-epoch scout. Works if it beats both baselines on matched-recall CRTK class-1 FP rate and keeps batch inference within 1.3× i193. Fails if only the external-edge GAT improves, proving the compiler is unnecessary.

**Failure mode catalogue:**
- Hidden rebrand risk: if edge compilation is moved outside the op, this is just GAT/MPNN.
- Legal-edge compilation may be branchy and GPU-unfriendly unless implemented with table-driven bitboards.
- Gradients do not flow through edge existence, so near-threshold structural mistakes cannot be learned away.

**Status:** proposed

### primitive_delta_apply_linear

**Name:** Delta-Apply Linear

**One-line claim:** A linear layer whose input is a bounded edit script against a cached feature accumulator.

**Mathematical signature:**
\[
f:\mathbb{R}^{B\times d_a}\times \mathbb{Z}^{B\times M\times k}\times\{-1,+1\}^{B\times M\times k}\rightarrow\mathbb{R}^{B\times M\times d_a}.
\]
With cached accumulator \(A_b\), feature table \(T\in\mathbb{R}^{F\times d_a}\), edit feature ids \(q_{b,m,\ell}\), and signs \(s_{b,m,\ell}\):
\[
Y_{b,m}=A_b+\sum_{\ell=1}^{k}s_{b,m,\ell}T_{q_{b,m,\ell}}.
\]
Optional fused projection:
\[
Z_{b,m}=Y_{b,m}P
\]
without materializing \(Y\).

**Why this does not decompose into existing PyTorch ops:**
The closest emulation is `EmbeddingBag + add + Linear`, but that loses the primitive property: the state is a persistent accumulator updated by an edit script, and the backward pass writes only touched table rows. Stockfish NNUE’s speed comes from sparse inputs, small changes between evaluations, and incremental accumulator updates; PyTorch has sparse embeddings, but not a first-class “bounded edit against cached dense state” layer. The computation graph is edit-size-driven rather than input-size-driven.

**Chess-specific motivation:**
A chess move changes a bounded number of piece-square features. HalfKA/NNUE exploits this for \(O(1)\) update; this primitive generalises that into a reusable differentiable layer. It also allows all legal successor accumulators to be evaluated cheaply, which is useful for hard-negative discrimination without feeding Stockfish scores, PVs, or node counts into the model.

**Generalisation beyond chess:**
Useful for sparse-event sequences, recommender updates, dynamic knowledge graphs, simulator state deltas, and any model whose state changes by a small edit script.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BMkd_a)\) vs dense recomputation \(O(BMFd_a)\); fused projection can avoid \(B\times M\times d_a\) materialization.
- Backward: \(O(BMkd_a)\), sparse over touched table rows.
- Incremental update on a bounded-change input: \(O(kd_a)\) for the chosen successor; \(O(\Delta Mkd_a)\) if the legal successor set changes.

**Scout-scale falsification test:**
Drop into the i243 HalfKA-style branch as the feature-transformer primitive; baseline against ordinary HalfKA accumulator recomputation and `EmbeddingBag + add`. Measure matched-recall CRTK class-1 FP rate plus candidate-evaluation throughput. Works if quality is non-worse and candidate throughput improves by ≥1.5×. Fails if it is only an implementation speed trick with no graph-level difference versus `EmbeddingBag`.

**Failure mode catalogue:**
- Strongest objection: “This is a stateful `EmbeddingBag`.” That is the main novelty risk.
- Sparse-gradient accumulation can become nondeterministic on GPU unless row updates are sorted or segmented.
- If downstream layers dominate runtime, \(O(1)\) accumulator update will not matter.

**Status:** proposed

### primitive_rule_automaton_scan

**Name:** Rule-Automaton Selective Scan

**One-line claim:** A selective scan whose transition regime is chosen by a finite symbolic automaton.

**Mathematical signature:**
\[
f:\{0,\ldots,C\}^{B\times L}\times\mathbb{R}^{B\times L\times d}\rightarrow\mathbb{R}^{B\times L\times h}.
\]
For symbolic input \(p_t\), continuous input \(x_t\), automaton state \(q_t\in Q\):
\[
q_t=\delta(q_{t-1},p_t),
\]
\[
h_t=A_{q_t,p_t}(x_t)h_{t-1}+B_{q_t,p_t}(x_t),
\qquad
y_t=C_{q_t,p_t}h_t.
\]
The automaton transition \(\delta\) is discrete; gradients flow through \(A,B,C,x\).

**Why this does not decompose into existing PyTorch ops:**
Mamba and Mamba-2 make SSM parameters input-conditioned and exploit scan-style efficiency, while Gated DeltaNet combines gating with delta-rule memory updates. This primitive adds a finite symbolic control state that changes the recurrence regime by rule, not merely by continuous gates. Existing `RNN`, `GRU`, or SSM modules do not expose a deterministic automaton-controlled transition table with sparse symbolic resets.

**Chess-specific motivation:**
Pins and x-rays can be recognized by scanning a line with finite states: empty, own king seen, first blocker seen, enemy slider seen, etc. The primitive lets a small network represent “king-blocker-attacker” line structure without learning it from scratch. It should be much less data-hungry than attention at scout scale.

**Generalisation beyond chess:**
Useful for regex-like sequence modeling, genomics motifs, packet inspection, program analysis, and symbolic-event streams where finite-state structure controls continuous updates.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BLh^2)\) or \(O(BLh)\) for diagonal/low-rank \(A\), vs attention \(O(BL^2h)\).
- Backward: \(O(BLh^2)\), with stored automaton states.
- Incremental update on a bounded-change input: \(O(\Delta Lh^2)\); for chess lines, \(\Delta L=O(s)\), constant on \(8\times8\).

**Scout-scale falsification test:**
Drop into i193 as four rank/file scans and four diagonal scans before the final head, replacing one parameter-matched conv block. Baselines: i193 and a small Mamba-style scan of the same width. Works if it lowers matched-recall CRTK class-1 FP rate and latency stays within 1.3× i193. Fails if Mamba-style continuous scan matches it.

**Failure mode catalogue:**
- Hidden rebrand risk: if \(Q\) has one state, it collapses to a standard selective scan.
- Bad automaton design can hard-code the wrong chess semantics and suppress learning.
- Branching over \(Q\) can be slow unless compiled as table lookup over small integer states.

**Status:** proposed

### primitive_chess_orbit_linear

**Name:** Chess-Orbit Linear

**One-line claim:** A finite-group linear map tying weights by legal-rule symmetry orbits across squares, colors, and piece roles.

**Mathematical signature:**
\[
f:\mathbb{R}^{B\times n\times c\times d_i}\rightarrow\mathbb{R}^{B\times n\times c\times d_o}.
\]
Let \(u=(\text{square},\text{color},\text{piece-role})\), and let \(G_\chi\) act on \(u\). Kernel parameters are indexed by pair-orbits:
\[
K_{u,v}=K_{\mathcal{O}(u,v)},\qquad
\mathcal{O}(u,v)=\{(gu,gv):g\in G_\chi\}.
\]
Then:
\[
y_{b,u}=\sum_v K_{\mathcal{O}(u,v)}x_{b,v}.
\]
Equivariance condition:
\[
f(\rho(g)x)=\rho(g)f(x),\quad g\in G_\chi.
\]

**Why this does not decompose into existing PyTorch ops:**
This is not a standard `Linear`, because parameter identity is determined by group orbits over square, color, and role indices. It overlaps with group-equivariant convolution, which is established prior work, so the honest novelty claim is not “new equivariance,” but “underexplored finite chess-rule orbit primitive.” Group-equivariant CNNs showed how symmetry-aware weight sharing can reduce sample complexity, but those layers target translation/reflection/rotation groups rather than chess-rule automorphisms over typed pieces.

**Chess-specific motivation:**
Chess has color-swap and board symmetries, but also typed piece roles with rule-dependent relations. Scout-scale data is small, so exact weight tying may matter more than expressivity. This primitive tries to prevent the network from separately relearning symmetric tactical relations.

**Generalisation beyond chess:**
Reusable for finite-rule domains: board games, card games, typed program graphs, molecules with atom-type symmetries, and any finite relational system with known automorphisms.

**Complexity (forward, backward, incremental-update):**
- Forward: dense \(O(Bn^2cd_id_o)\), or sparse/orbit-pruned \(O(Bo d_id_o)\), where \(o\) is active orbit-pair count; closest `Linear` is \(O(Bn^2cd_id_o)\) untied.
- Backward: same asymptotic, with gradient accumulation over tied orbit parameters.
- Incremental update on a bounded-change input: \(O(\Delta o\,d_id_o)\) if cached; otherwise not applicable.

**Scout-scale falsification test:**
Replace the final square/piece relational linear head in i193 with `ChessOrbitLinear`, matching or reducing parameter count. Baselines: untied linear head and ordinary D4-tied head. Works if matched-recall CRTK class-1 FP rate improves or stays equal with materially fewer parameters. Fails if D4-only tying matches it.

**Failure mode catalogue:**
- Hidden rebrand risk: reviewer may call it group convolution with a chess-specific group.
- Over-tying can erase real asymmetries, especially pawn direction and side-to-move effects.
- Dense orbit contraction may be too slow unless orbit indices are precomputed and sparsified.

**Status:** proposed

## Ranking matrix

| Rank | Primitive | Plausibility of novelty | RTX 3070 demonstrability | Inference-speed advantage | Generalisation beyond chess | Verdict |
|---:|---|---|---|---|---|---|
| 1 | `primitive_ray_occlusion_scan` | High | High | Medium-high | High | Best scout-scale bet. |
| 2 | `primitive_legal_edge_compile_scatter` | High but fragile | Medium-high | High if compiled well | Medium-high | Most chess-structural. |
| 3 | `primitive_delta_apply_linear` | Medium | High | Very high | High | Best speed primitive, biggest rebrand risk. |
| 4 | `primitive_rule_automaton_scan` | Medium-high | Medium | High | High | Good if ray tactics dominate CRTK failures. |
| 5 | `primitive_chess_orbit_linear` | Medium-low | High | Medium | High | Honest “underexplored for chess,” not fully new equivariance. |

## Top-2 self-audit

### Self-audit: `primitive_ray_occlusion_scan`

Devil’s advocate: this can be written as `gather → cumprod → matmul → sum`, so maybe it is just a composition of PyTorch tensor ops. That objection is real for a prototype implementation. The reason I keep it is that the primitive signature is a ray-topology semiring scan with a custom sparse adjoint and bounded-change update; no existing `torch.nn` operator has visibility-prefix products as its connectivity rule. It is also not just Mamba/SSM, because the recurrence topology is geometric and multi-ray rather than a single sequence recurrence with learned continuous transition parameters.

### Self-audit: `primitive_legal_edge_compile_scatter`

Devil’s advocate: if the legal edge list is precomputed, this becomes graph attention/message passing, which already exists conceptually. That would fail the novelty bar. I keep it only under the stricter definition that edge compilation is inside the primitive’s compute graph signature: \(P\mapsto E(P)\mapsto\text{sparse scatter}\), with no materialized attention mask and no externally supplied `edge_index`. If reviewers reject discrete edge compilation as part of a neural primitive, this proposal should be dropped before implementation.

## What I cut

- **Legal-move masked attention:** rejected because it is exactly masked attention over a different mask shape.
- **King-conditioned convolution:** rejected because it is a composition of existing convs plus conditioning, not a primitive.
- **Piece-type MoE gate:** rejected because MoE routing is an existing primitive family; piece type is just a routing key.
- **Polynomial tactical activation:** rejected because it is an activation tweak, not a new operator.
- **Stockfish-guided verification primitive:** rejected because Stockfish scores, PVs, node counts, and verification metadata are labels/audit fields, not valid compute-graph inputs.

## Bibliography

- Albert Gu and Tri Dao, **“Mamba: Linear-Time Sequence Modeling with Selective State Spaces.”** arXiv:2312.00752. <https://arxiv.org/abs/2312.00752>
- Tri Dao and Albert Gu, **“Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.”** arXiv:2405.21060. <https://arxiv.org/abs/2405.21060>
- Maximilian Beck et al., **“xLSTM: Extended Long Short-Term Memory.”** arXiv:2405.04517. <https://arxiv.org/abs/2405.04517>
- Soham De et al., **“Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.”** arXiv:2402.19427. <https://arxiv.org/abs/2402.19427>
- Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim, **“Parallelizing Linear Transformers with the Delta Rule over Sequence Length.”** arXiv:2406.06484. <https://arxiv.org/abs/2406.06484>
- Songlin Yang, Jan Kautz, and Ali Hatamizadeh, **“Gated Delta Networks: Improving Mamba2 with Delta Rule.”** arXiv:2412.06464. <https://arxiv.org/abs/2412.06464>
- Han Guo, Songlin Yang, Tarushii Goel, Eric P. Xing, Tri Dao, and Yoon Kim, **“Log-Linear Attention.”** arXiv:2506.04761. <https://arxiv.org/abs/2506.04761>
- Julien N. Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, and Riccardo Grazzi, **“DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products.”** arXiv:2502.10297. <https://arxiv.org/abs/2502.10297>
- Kimi Team, **“Kimi Linear: An Expressive, Efficient Attention Architecture.”** arXiv:2510.26692. <https://arxiv.org/abs/2510.26692>
- Taco Cohen and Max Welling, **“Group Equivariant Convolutional Networks.”** ICML 2016 / PMLR. <https://proceedings.mlr.press/v48/cohenc16.html>
- Petar Veličković et al., **“Graph Attention Networks.”** arXiv:1710.10903. <https://arxiv.org/abs/1710.10903>
- Justin Gilmer et al., **“Neural Message Passing for Quantum Chemistry.”** arXiv:1704.01212. <https://arxiv.org/abs/1704.01212>
- Official Stockfish NNUE PyTorch Wiki, **“NNUE.”** <https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html>
