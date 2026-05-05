# Codex Handoff Packet: Tactical Transport Imbalance Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0512_tuesday_local_transport_imbalance.md`
- Generated at: 2026-04-21 05:12 America/Los_Angeles
- Weekday: Tuesday
- Timezone: local / America/Los_Angeles
- Idea slug: `transport_imbalance`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Tactical Transport Imbalance Network
- One-sentence thesis: A chess puzzle-like position should often exhibit an asymmetric low-cost, low-entropy transport alignment from the side-to-move's active material toward the opponent king and high-value targets, and this global geometric imbalance can be learned without engine scores, move trees, attack-sheaves, or one-ply move-delta pooling.
- Idea fingerprint: `current-board piece-square measures + side-to-move canonicalization + differentiable entropic optimal transport between own force mass and opponent target mass + reverse-direction imbalance summaries + binary puzzle-likeness target + no engine metadata, no legal move tree, no attack/sheaf/Hodge operator, no one-ply move-delta bag`.
- Closest baseline or common method it resembles: A small CNN with an added differentiable Sinkhorn/optimal-transport feature layer; conceptually closest to differentiable optimization layers and OT pooling, not to LC0, ResNet scaling, square Transformers, tactical sheaves, or one-ply move landscapes.
- Why this is not a common CNN/ResNet/Transformer variant: The central nonlocal computation is an explicit variational matching problem over two learned probability measures on the 64 squares; replacing it with another convolutional block, residual block, square self-attention layer, or wider stem removes the defining operator.
- Current-data minimal experiment: Train `transport_imbalance_net` on `simple_18` using the existing `crtk_sample_3class` train/val/test Parquet splits, coarse binary labels with fine-label diagnostics, 3 epochs, balanced class weighting, and the shared trainer/reporting pipeline.
- Smallest central falsification ablation: Keep the CNN, mass heads, parameter count, Sinkhorn iterations, and marginal distributions unchanged, but replace the chess-geometric cost bank with a fixed square-permuted cost bank that preserves the multiset of pairwise costs while destroying rank/file/diagonal/knight/forward semantics; if this matches the main model, the transport geometry is not doing useful work.
- Expected information gain if it fails: Failure would show that board-wide geometric mass alignment is either already captured by the existing CNN baselines, dominated by material/source shortcuts, or irrelevant relative to exact legal move consequences; the next cycle should then avoid OT-over-piece-measures as a central mechanism.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The task is chess puzzle-likeness classification from a single board position.

Binary target:

- output `0`: non-puzzle.
- output `1`: puzzle-like.

Fine labels used for supervision/diagnostics:

- fine label `0`: known non-puzzle, mapped to binary `0`.
- fine label `1`: verified near-puzzle, mapped to binary `1`.
- fine label `2`: verified puzzle, mapped to binary `1`.

Required input/output contract:

- Model type: PyTorch `torch.nn.Module`.
- Input tensor: `(batch, C, 8, 8)`.
- Output tensor: logits `(batch, 2)`.
- Shared trainer, reports, confusion matrices, predictions, and leaderboards must keep working.
- Main diagnostic matrix remains rectangular: true fine label `0/1/2 -> predicted binary output 0/1`.

Current benchmark split:

- `data/splits/crtk_sample_3class/split_train.parquet`
- `data/splits/crtk_sample_3class/split_val.parquet`
- `data/splits/crtk_sample_3class/split_test.parquet`

Available encodings:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant.
- `lc0_static_112`.
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists.

Leakage checklist:

- Allowed: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board.
- This idea intentionally avoids pseudo-legal attack/defense incidence, x-ray graphs, legal move generation, and one-ply move-delta sets as central operators, because those families are already heavily covered in imported packets.
- Leakage-prone unless explicitly justified, label-independent, engine-free, and ablated: full legal-move generation, legal move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences.
- Never use as neural-network inputs: engine evaluation, Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, unresolved-candidate status, or dataset provenance.
- Fine labels may be used only for supervised targets and reports, not as input features.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry may use only registered current-board piece channels and side-to-move semantics. History channels, if present, may be consumed only by learned neural adapters and must not be parsed into deterministic transport geometry unless the channel semantics are explicitly registered and tested. Unknown channel semantics must fail closed.

Boundary between safe rule-derived features and leakage:

- Safe: a fixed 64-square coordinate system, side-to-move perspective canonicalization, piece occupancy masks, deterministic king-ring masks from current-board king squares, and precomputed square-pair coordinate costs such as normalized Manhattan distance, Chebyshev distance, same-rank/file indicators, diagonal indicators, knight-graph distance, color parity, and forward-rank displacement.
- Not used here: search depth, engine evaluation, move legality outcomes, check/mate detection, or tactical verification artifacts.
- The transport layer does not ask whether a move is legal or good; it only asks whether learned source mass and target mass can be globally matched through a chess-shaped coordinate cost.

## 4. Research Map

| Source | What is borrowed | What is not copied |
|---|---|---|
| Marco Cuturi, “Sinkhorn Distances: Lightspeed Computation of Optimal Transport,” NeurIPS 2013. URL: https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport | Entropic regularization and Sinkhorn matrix scaling as a fast differentiable approximation to optimal transport between histograms. | No image-retrieval task, no MNIST setup, no claim that Sinkhorn alone solves chess tactics. |
| Gabriel Peyré and Marco Cuturi, “Computational Optimal Transport,” Foundations and Trends in Machine Learning / arXiv 2018-2019. URL: https://arxiv.org/abs/1803.00567 | The formulation of OT as a global matching cost between probability measures and the use of numerical OT summaries as machine-learning features. | No generic OT benchmark, no Wasserstein generative model, no high-dimensional continuous OT solver. |
| Brandon Amos and J. Zico Kolter, “OptNet: Differentiable Optimization as a Layer in Neural Networks,” ICML 2017. URL: https://proceedings.mlr.press/v70/amos17a.html | The design principle that a constrained optimization problem can be placed inside a neural forward pass and trained end-to-end. | No quadratic program, no Sudoku task, no exact implicit-differentiation requirement; the proposed layer uses unrolled Sinkhorn iterations. |
| David Silver et al., “Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm,” arXiv 2017 / Science 2018. URL: https://arxiv.org/abs/1712.01815 | A reminder that chess neural nets can exploit board-plane encodings and side-to-move perspective without hand-coded engine evaluations. | No reinforcement learning, no MCTS, no policy/value target, no LC0 architecture copy, no search-derived supervision. |
| Imported `chess-nn-playground` research packets on tactical sheaves and one-ply counterfactual move-delta landscapes | Negative guidance: avoid attack/defense incidence sheaves, Hodge/Laplacian/tension/curvature operators, and one-ply move-delta set pooling. | None of their central operators are reused. This proposal uses OT between piece-square probability measures instead of graph/sheaf/move-set computations. |

All citations above are stable research anchors verified by public paper pages. No unverified bibliographic details are needed for implementation.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on `simple_18` | Existing `src/chess_nn_playground/models/cnn.py` simple CNN | Too ordinary and already implemented; it does not introduce a falsifiable nonlocal chess-specific operator. |
| Residual CNN variants | Existing `src/chess_nn_playground/models/residual_cnn.py` | Scaling residual depth/width is a baseline-family extension, not a new research mechanism. |
| LC0-style CNN/residual CNN on `lc0_bt4_112` | Existing LC0 BT4-style CNN and residual variants | Too close to the current suite and would mainly test encoding capacity rather than a new hypothesis about puzzle-likeness. |
| Ordinary ViT over 64 square tokens | Common square-token Transformer | Self-attention is a generic global mixer and would be hard to distinguish from “use a vanilla Transformer,” which is explicitly disallowed. |
| Plain GNN on squares | Generic graph neural network over board adjacency or piece attacks | Without a new operator it is just message passing on a chess graph, and attack/defense graph variants are already saturated by imported sheaf packets. |
| Hyperparameter tuning | Any existing CNN/ResNet config | Tuning epochs, width, dropout, optimizer, or learning rate is useful engineering but not a research idea. |
| Ensembling existing models | Leaderboard ensemble | Ensembling hides which inductive bias works, increases compute, and is explicitly disallowed as the core idea. |
| Static attack-defense sheaf/Hodge/Laplacian/tension/curvature model | Imported tactical sheaf/Hodge packets | Already researched; changing edge labels, pooling, or terminology would be a duplicate. |
| One-ply move-delta DeepSets/attention/spectrum/free-energy model | Imported counterfactual move-delta packets | Already researched; this proposal must not enumerate pseudo-legal one-ply move deltas as the central set. |
| Legal move-count or checkmate-oracle features | Rule-engine feature model | Leakage-prone and would confound puzzle-likeness with explicit move-tree or terminal-state computation. |
| Material-count logistic baseline as main model | Nuisance/material baseline | Useful as an ablation, but too weak and too shortcut-prone to be the central architecture. |
| Ordinal-only `0 < 1 < 2` calibration head | Standard ordinal classification | Potentially useful later, but by itself it does not add a board-structure mechanism and may overfit label ambiguity rather than explain it. |

## 6. Mathematical Thesis

Input space definition:

Let `E` be an encoding family. A board tensor is

\[
X \in \mathcal X_E \subseteq \mathbb R^{C_E \times 8 \times 8}.
\]

A deterministic, fail-closed adapter

\[
A_E : \mathcal X_E \to (P, r)
\]

extracts current-board piece occupancy and metadata when channel semantics are known. Here

\[
P \in \{0,1\}^{2 \times 6 \times 64}
\]

stores color-by-piece-type occupancy on the 64 squares, and `r` contains side-to-move and optional castling/en-passant metadata already present in the encoding. For the first experiment, `E = simple_18`. For LC0 encodings, `A_E` must raise a clear error unless current-board piece channels and side-to-move channels are registered.

Target definition:

Let fine label

\[
L \in \{0,1,2\}
\]

where `0` is known non-puzzle, `1` is verified near-puzzle, and `2` is verified puzzle. The binary target is

\[
Y = \mathbf 1\{L \in \{1,2\}\}.
\]

Data distribution assumptions:

There is an unknown training distribution \(D_{\text{train}}\) over \((X,L,Y)\) and an unknown benchmark distribution \(D_{\text{test}}\) induced by the provided splits. The dataset may contain nuisance correlations between \(Y\) and material, phase, side-to-move, opening/endgame composition, or source construction. The model may use board geometry, but it must not use source labels, engine verification signals, or provenance metadata.

Allowed symmetry or equivariance assumptions:

Chess is not invariant under arbitrary rotations/reflections because pawns, castling, and side-to-move break most board symmetries. The only symmetry assumption used here is a side-to-move perspective canonicalization: if black is to move, swap colors and flip ranks so the player to move is represented as `own` and their pawns advance in the canonical forward direction. This is a rule automorphism for piece geometry when castling/en-passant metadata is transformed consistently. The OT branch uses this canonical piece view; the learned CNN adapter may still consume the original encoding.

Core hypothesis:

After conditioning on material and phase, many puzzle-like positions have an asymmetric concentration of tactical potential: own active pieces can be matched to opponent king-zone/high-value targets through low chess-geometric transport cost with a sharper, lower-entropy plan than the reverse direction. Non-puzzles with the same material often lack this directional concentration. Therefore, a differentiable transport imbalance feature should improve class `1` and class `2` detection at a fixed fine-label-`0` false-positive rate.

Formal object introduced by the idea:

Let \(S = \{1,\ldots,64\}\) be the square set. For each transport head \(h \in \{1,\ldots,H\}\), the model constructs positive probability measures

\[
\mu_h^{+}, \nu_h^{-}, \mu_h^{-}, \nu_h^{+} \in \Delta^{63}
\]

where \(\mu_h^{+}\) is own source/force mass, \(\nu_h^{-}\) is opponent target mass, \(\mu_h^{-}\) is opponent source/force mass, and \(\nu_h^{+}\) is own target mass. These are learned from current-board piece occupancy and square embeddings, with small positive floor mass for numerical stability.

A head-specific cost matrix is

\[
C_h(s,t)=\operatorname{softplus}\left(\beta_h + \sum_{m=1}^{M} \alpha_{hm} B_m(s,t)\right) + c_{\min},
\]

where \(B_m\) are fixed square-pair geometry bases such as normalized Manhattan distance, Chebyshev distance, same-rank/file indicator cost, diagonal indicator cost, knight-graph distance, color-parity mismatch, queen-line compatibility, and forward-rank displacement in the canonical side-to-move frame. These bases are coordinate geometry, not current-board attack/defense incidence.

The entropic transport value is

\[
W_{\varepsilon,h}(\mu,\nu)
= \min_{\pi \in \Pi(\mu,\nu)}
\left\langle C_h, \pi \right\rangle
+ \varepsilon \sum_{s,t}\pi_{st}(\log \pi_{st} - 1),
\]

where

\[
\Pi(\mu,\nu)=\left\{\pi \in \mathbb R_+^{64\times64}:\sum_t \pi_{st}=\mu_s,\;\sum_s \pi_{st}=\nu_t\right\}.
\]

The transport imbalance summary for head \(h\) is a vector

\[
\Delta_h = \psi(\pi_h^{+\to -}, W_{\varepsilon,h}(\mu_h^{+},\nu_h^{-}))
- \psi(\pi_h^{-\to +}, W_{\varepsilon,h}(\mu_h^{-},\nu_h^{+})),
\]

where \(\psi\) includes cost, plan entropy, maximum transport entry, squared plan mass \(\|\pi\|_2^2\), and source-target distance moments. The final classifier receives pooled CNN features plus \(\Delta_1,\ldots,\Delta_H\).

Proposition:

For strictly positive marginals and \(\varepsilon > 0\), each head's entropic OT problem has a unique optimizer \(\pi_h^\star\). The value and the plan are differentiable almost everywhere with respect to the learned mass logits and cost parameters. Further, for any square permutation \(g\) that preserves the chosen canonical chess geometry and transforms the cost by \(C_h^g(gs,gt)=C_h(s,t)\),

\[
W_{\varepsilon,h}(g_\#\mu,g_\#\nu; C_h^g)=W_{\varepsilon,h}(\mu,\nu; C_h).
\]

Proof sketch or derivation:

The feasible set \(\Pi(\mu,\nu)\) is compact and convex for positive probability marginals. The entropic term is strictly convex on the positive transport polytope, so the minimizer is unique. Sinkhorn scaling computes the optimizer in the form

\[
\pi^\star = \operatorname{diag}(u) K \operatorname{diag}(v),
\quad K_{st}=\exp(-C_{st}/\varepsilon),
\]

with positive scaling vectors \(u,v\). Unrolled Sinkhorn iterations are differentiable compositions of multiplication, division, exponentiation, and normalization when a positive floor is used. For the permutation statement, push any feasible plan `pi` forward by \((g,g)\); row and column marginals become \(g_\#\mu\) and \(g_\#\nu\), and the transport objective is unchanged because the cost is transformed compatibly. Taking the minimum in both directions gives equality.

Optimization objective:

The main training objective is

\[
\min_\theta \; \mathbb E_{(X,Y)\sim D_{\text{train}}}
\left[\operatorname{CE}_w(f_\theta(X),Y)\right]
+ \lambda_{\text{div}} R_{\text{head-diversity}}(\theta),
\]

where \(\operatorname{CE}_w\) is balanced cross-entropy and the optional diversity penalty discourages all transport heads from learning identical cost weights. The central model remains valid with \(\lambda_{\text{div}}=0\).

What is actually proven:

The OT layer is a well-defined differentiable global matching operator under positive marginals, and its value is invariant/equivariant under the explicitly allowed side-to-move canonical square transformation when the cost bank is transformed consistently.

What remains only hypothesized:

It is not proven that puzzle-likeness causally depends on transport imbalance. It is only hypothesized that the benchmark labels contain enough positions where tactical salience is correlated with directional low-cost, low-entropy piece-target alignment beyond material counts and local CNN features.

Counterexamples where the idea should fail:

- Quiet endgame studies, zugzwang, opposition, fortress, or tempo puzzles whose solution is not visible as force-to-target mass alignment.
- Puzzles where exact legality, check evasion, or a long forcing line matters more than current-board geometry.
- Positions with spectacular attacking alignment that are not puzzles because the tactic fails tactically or strategically.
- Material-imbalance artifacts where a nuisance-only model is already optimal.
- Underpromotion, stalemate, or repetition motifs whose signal is in move consequences rather than current-board transport geometry.

## 7. Architecture Specification

Module names:

- `EncodingAdapter`
- `SideToMoveCanonicalizer`
- `LocalBoardEncoder`
- `TransportMassHead`
- `ChessCostBank`
- `SinkhornTransportBlock`
- `TransportFeaturePool`
- `TransportImbalanceNet`

First experiment encoding:

- Use `simple_18` first because its 12 current-board piece planes and side-to-move channel are explicitly available.
- Add LC0 support only after Codex registers exact current-board piece-plane indices for `lc0_static_112` and `lc0_bt4_112`.
- Unknown channel semantics must raise a clear error rather than silently extracting the wrong planes.

Forward-pass steps and shapes:

1. Input:
   - `x`: `(B, C, 8, 8)`.
   - Main config: `C=18` for `simple_18`.

2. `EncodingAdapter(x, encoding)`:
   - For `simple_18`, extract 12 piece planes and side-to-move from the known project encoding contract.
   - Output piece tensor `P_raw`: `(B, 2, 6, 8, 8)`.
   - Output metadata `r`: side-to-move `(B,)` plus castling/en-passant metadata if needed by learned adapters.
   - Fail closed if a piece-channel map is missing.

3. `SideToMoveCanonicalizer(P_raw, r)`:
   - If white to move: own=white, opponent=black, no rank flip.
   - If black to move: own=black, opponent=white, flip ranks so own pawns advance in the canonical positive direction.
   - Output `P`: `(B, 12, 8, 8)` ordered as own six piece planes followed by opponent six piece planes.
   - Output flattened canonical occupancy `P64`: `(B, 12, 64)`.

4. `LocalBoardEncoder(x)`:
   - `Conv2d(C, 64, kernel_size=3, padding=1)` -> `(B,64,8,8)`.
   - 3 small residual blocks at width 64 -> `(B,64,8,8)`.
   - `Conv2d(64, 128, kernel_size=3, padding=1)` -> `(B,128,8,8)`.
   - global average pool -> `z_cnn`: `(B,128)`.
   - This is intentionally small; the research mechanism is the transport branch, not CNN scaling.

5. `TransportMassHead(P64)`:
   - Number of heads: `H=8` by default.
   - Learn source piece-type embeddings for own/opponent force mass: `(H,6)`.
   - Learn target piece-type embeddings for king/queen/rook/minor/pawn value demand: `(H,6)`.
   - Add square positional bias: `(H,64)`.
   - Construct king-ring demand by fixed 3x3 dilation around the current-board opponent king in canonical coordinates; this is a deterministic current-board mask, not a check oracle.
   - Produce four positive raw mass arrays:
     - `own_source_raw`: `(B,H,64)`
     - `opp_target_raw`: `(B,H,64)`
     - `opp_source_raw`: `(B,H,64)`
     - `own_target_raw`: `(B,H,64)`
   - Normalize each with `softmax(log(raw + mass_floor))` to probability marginals:
     - `mu_own`, `nu_opp`, `mu_opp`, `nu_own`: each `(B,H,64)`.

6. `ChessCostBank`:
   - Precompute fixed basis tensor `B_cost`: `(M,64,64)` with `M=8` geometry bases.
   - Learn head weights `alpha`: `(H,M)` and bias `beta`: `(H,)`.
   - Output cost matrices `C_head`: `(H,64,64)` via positive softplus transform.
   - No occupancy-dependent attack edges, no legal moves, and no x-ray incidence are computed.

7. `SinkhornTransportBlock`:
   - Inputs: `mu`: `(B,H,64)`, `nu`: `(B,H,64)`, `C_head`: `(H,64,64)`.
   - Use entropic regularization `epsilon=0.07` by default and `sinkhorn_iters=8`.
   - Compute forward plan `pi_fwd`: `(B,H,64,64)` for own-source to opponent-target.
   - Compute reverse plan `pi_rev`: `(B,H,64,64)` for opponent-source to own-target.
   - Numerical plan: use log-domain Sinkhorn if underflow appears; otherwise standard scaling with clamped kernel is acceptable for the first pass.

8. `TransportFeaturePool`:
   - Per head and direction compute:
     - expected cost `sum(pi*C)`
     - plan entropy `-sum(pi*log(pi))`
     - max entry `max(pi)`
     - squared concentration `sum(pi^2)`
     - mean source rank/file and target rank/file under the plan, or equivalent first moments
   - Combine as forward, reverse, and forward-minus-reverse summaries.
   - Output `z_ot`: `(B, H*10)` with default `(B,80)`.

9. `TransportImbalanceNet` classifier:
   - Concatenate `[z_cnn, z_ot]`: `(B,208)`.
   - MLP: `Linear(208,128)`, GELU, dropout `0.10`, `Linear(128,2)`.
   - Return logits `(B,2)`.

Parameter-count estimate:

- `simple_18` local stem and residual blocks: about 310k-360k parameters depending on exact block normalization.
- Transport mass embeddings, cost weights, and positional biases: under 10k parameters.
- Final MLP: about 27k parameters.
- Expected total: about 350k-450k parameters for `simple_18`; about 410k-520k if the first convolution is widened to accept 112 LC0 channels.

FLOP and complexity estimate:

- CNN branch: `O(B * 8 * 8 * width^2 * blocks)`; small relative to ordinary residual baselines.
- OT branch: `O(B * H * I * 64^2)` for Sinkhorn iterations, where `H=8` and `I=8` by default.
- For `B=512`, `H=8`, `I=8`, the two-direction transport layer performs roughly `2 * 512 * 8 * 8 * 4096`, about 268 million simple multiply/divide operations before pooling.

Memory and chunking plan:

- The square-pair plan has size `(B,H,64,64)` floats per direction.
- Memory per direction is approximately `4 * B * H * 4096` bytes in fp32.
- At `B=512`, `H=8`, one plan is about 64 MiB; forward+reverse plans are about 128 MiB before autograd overhead.
- If memory is high, chunk over heads or batch: compute `H_chunk=2` heads at a time, pool transport features immediately, and discard the plan before the next chunk.
- Mixed precision should remain off in the first deterministic benchmark because Sinkhorn can be numerically fragile.

Required config fields:

- `model.name: transport_imbalance_net`
- `model.input_channels`
- `model.num_classes: 2`
- `model.encoding: simple_18`
- `model.transport_heads: 8`
- `model.sinkhorn_iters: 8`
- `model.sinkhorn_epsilon: 0.07`
- `model.mass_floor: 1.0e-4`
- `model.cost_basis: chess_geometry_v1`
- `model.canonicalize_side_to_move: true`
- `model.transport_feature_dim_per_head: 10`
- `model.dropout: 0.10`
- `model.fail_closed_unknown_channels: true`

Encoding-adapter assumptions:

- `simple_18`: deterministic extraction is allowed using the existing 12 piece planes and side-to-move plane. Castling and en-passant planes may be passed to the learned CNN branch but are not central to the OT mass construction.
- `lc0_static_112`: deterministic OT extraction is allowed only from registered current-board piece planes. Non-current or unknown channels may enter only through the learned CNN adapter.
- `lc0_bt4_112`: deterministic OT extraction is allowed only from the current-board slice. Zero-filled or future history planes must not be interpreted as move history for rule-derived geometry. History planes may be consumed by learned convolutions but not by deterministic transport mass generation unless semantics are registered.
- All adapters must fail closed when channel semantics are unknown.

Pseudocode, not final implementation:

```text
forward(x):
    P_raw, meta = adapter.extract_current_board(x)
    P = canonicalizer.to_side_to_move_frame(P_raw, meta.side_to_move)

    z_cnn = local_encoder(x)                  # (B, 128)

    mu_own, nu_opp, mu_opp, nu_own = mass_head(P)   # four (B,H,64)
    C = cost_bank()                                  # (H,64,64)

    pi_fwd = sinkhorn(mu_own, nu_opp, C)       # (B,H,64,64)
    pi_rev = sinkhorn(mu_opp, nu_own, C)       # (B,H,64,64)

    z_ot = pool_transport(pi_fwd, pi_rev, C)   # (B, H*10)
    logits = classifier(concat(z_cnn, z_ot))   # (B, 2)
    return logits
```

## 8. Loss, Training, And Regularization

Primary loss:

- Balanced cross-entropy over the binary target `Y = 0` for fine label `0`, `Y = 1` for fine labels `1` and `2`.

Optional auxiliary loss:

- `head_diversity_loss`: small pairwise cosine penalty on normalized cost-weight vectors `alpha_h` so transport heads do not collapse to identical geometry. Default `lambda_head_diversity = 1.0e-3`.
- This auxiliary loss is optional. The central claim must still be tested with and without it if the first run is inconclusive.

Class weighting:

- Use the same balanced class weighting mechanism as existing benchmark configs.
- Do not rebalance fine labels separately unless the shared trainer already supports it; the main target is coarse binary.

Batch size expectations:

- Start with `batch_size: 512` on the sample split.
- If GPU memory is exceeded, first enable head chunking in the Sinkhorn block before reducing batch size.
- Keep `mixed_precision: false` initially.

Learning-rate and optimizer defaults:

- Optimizer: AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Epochs: `3` for the minimal current-data experiment.
- Early stopping patience: `2`.

Regularizers:

- Dropout `0.10` before final classifier.
- Mass floor `1.0e-4` for all OT marginals.
- Cost floor `1.0e-3` after softplus.
- Optional head diversity penalty as above.
- No label smoothing in the first benchmark, so probability calibration can be compared directly to baselines.

Determinism requirements:

- Use `seed: 42`.
- Set PyTorch deterministic flags according to existing project practice.
- Precompute cost bases deterministically and store their construction test expectations.
- Random-cost ablations must use a recorded seed and save the square permutation.

What must stay unchanged for fair comparison:

- Same train/val/test split paths.
- Same coarse-binary mapping.
- Same metric/reporting scripts.
- Same maximum epochs and early stopping policy as the matched baseline run.
- Same encoding for the main comparison, preferably `simple_18` first.
- No extra data, no full 45M-row Parquet training, and no streaming assumption.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `random_cost_bank_preserve_multiset` | Replace each chess-geometric cost matrix by a fixed square-permuted matrix preserving the multiset of pairwise costs but destroying rank/file/diagonal/knight/forward semantics; keep masses and Sinkhorn unchanged. | The geometry of the transport cost, not just extra parameters or global pooling, carries the signal. | If this matches the main model, abandon the central transport-geometry claim. |
| `uniform_cost_sinkhorn` | Set all pairwise costs equal so the optimal plan reduces to product-like coupling determined only by marginals. | Nontrivial square-pair costs matter beyond supply/demand histograms. | If performance remains, the model is likely using mass marginals/material shortcuts. |
| `marginal_count_only` | Replace OT features with material counts, phase, own/opponent value totals, king-square coordinates, and side-to-move metadata of matched dimension. | Transport over positions adds information beyond obvious board-summary nuisances. | If equal or better, do not scale OT; the gain is nuisance prediction. |
| `demand_square_shuffle_same_piece_counts` | Within each batch, shuffle opponent target square locations among examples with the same coarse material bucket while preserving piece counts and target mass totals. | Correct spatial alignment of sources and targets matters. | If unchanged, OT is not using semantic source-target geometry. |
| `source_square_shuffle_same_piece_counts` | Shuffle own source square locations under the same material-bucket constraint while preserving moving-side material and target mass. | Own active piece placement matters independently of target placement. | If unchanged, the model likely uses opponent king/material shortcuts only. |
| `no_reverse_direction` | Use only own-source to opponent-target transport features and remove reverse-direction imbalance. | Directional asymmetry is important for side-to-move puzzle-likeness. | If unchanged, reverse imbalance may be unnecessary; simplify the model. |
| `transport_only_no_cnn` | Remove the CNN branch and classify only from OT summaries. | OT features alone capture meaningful nonlocal signal. | If very weak, OT is only complementary; if strong, inspect for material shortcuts. |
| `cnn_only_matched_params` | Remove the OT branch and add a parameter-matched MLP or small convolutional block. | The explicit transport operator helps more than equal parameter count. | If equal or better, the OT layer is unnecessary complexity. |
| `drop_line_geometry_bases` | Remove rank/file/diagonal/queen-line cost bases, keeping Manhattan/Chebyshev/knight/parity/forward bases. | Sliding-piece alignment contributes to puzzle-likeness. | If unchanged, line geometry is not the useful part. |
| `drop_king_ring_demand` | Remove fixed opponent king-ring demand and use occupied opponent pieces only. | King-zone pressure is a useful target distribution. | If unchanged, high-value target placement may dominate king geometry. |
| `cost_head_diversity_off` | Set `lambda_head_diversity=0`. | Head diversity regularization is not responsible for any gain. | If main gains vanish only with diversity off, treat the regularizer as part of the mechanism and report it transparently. |

The smallest ablation that can falsify the central mathematical claim is `random_cost_bank_preserve_multiset`. It preserves candidate count, plan tensor shape, mass heads, material, side-to-move, source-square marginal construction, target histograms, and parameter count while destroying the proposed chess-geometric semantics.

This model does not generate move sets. Nevertheless, the ablations above include count-only and nuisance-preserving controls that preserve obvious shortcuts such as material, side-to-move, source/target mass totals, piece identity marginals, and target-value histograms while destroying the proposed square-pair transport semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN with the same split and training budget.
- Existing `simple_18` residual CNN with the same split and training budget.
- A parameter-matched CNN-only ablation created from this architecture.
- If LC0 adapters are later registered, compare against existing `lc0_static_112` and `lc0_bt4_112` CNN/residual CNN variants, but do not make the first experiment depend on LC0.

Metrics to inspect:

- Test accuracy.
- Balanced accuracy.
- Macro-F1.
- Binary AUROC and AUPRC if already reported by the project.
- Calibration: Brier score and ECE if available without changing the shared benchmark contract.
- Required rectangular diagnostic matrix: true fine label `0/1/2 -> predicted binary output 0/1`.

Required diagnostics for main model and every central ablation:

- Fine-label `0` false-positive rate.
- Fine-label `1` recall.
- Fine-label `2` recall.
- Fine-label `1` precision among examples predicted `1`, if computable.
- Fine-label `1` recall at a threshold chosen to match the residual CNN's fine-label-`0` false-positive rate.
- Score histograms or reliability summaries split by fine label `0`, `1`, and `2`.

Near-puzzle diagnostic:

- Primary near-puzzle diagnostic: class `1` recall at matched fine-label-`0` false-positive rate.
- Secondary diagnostic: separation between fine-label `1` and fine-label `0` predicted probabilities, measured by AUROC restricted to labels `{0,1}`.

Required artifacts:

- Trained model checkpoint.
- `metrics.json` or equivalent shared report.
- Main confusion matrix and `3x2` fine-label diagnostic matrix.
- Prediction Parquet/CSV with example IDs, fine label, binary label, predicted probability, and predicted class if existing reports support this.
- Ablation reports for `random_cost_bank_preserve_multiset`, `uniform_cost_sinkhorn`, `marginal_count_only`, and `cnn_only_matched_params` at minimum.
- Saved config files for each run.

Success threshold:

- Main model improves test macro-F1 or balanced accuracy over the best matched `simple_18` CNN/residual baseline by at least `+1.0` percentage point, or improves fine-label `1` recall at matched fine-label-`0` FPR by at least `+2.0` percentage points without reducing fine-label `2` recall by more than `1.0` percentage point.
- At least one central semantics-destroying ablation, preferably `random_cost_bank_preserve_multiset`, loses at least half of the main model's gain over the matched CNN-only baseline.

Failure threshold:

- Main model is within noise of the best matched baseline on macro-F1, balanced accuracy, and near-puzzle recall, and `random_cost_bank_preserve_multiset` performs the same or better.
- `marginal_count_only` matches the main model, indicating material/phase shortcuts explain the effect.
- Training is numerically unstable even with log-domain Sinkhorn and head chunking.

What result would make me abandon the idea:

- If `random_cost_bank_preserve_multiset`, `uniform_cost_sinkhorn`, and `marginal_count_only` all match the main model while the main model does not clearly beat the CNN-only baseline, stop exploring OT-over-piece-square-measures for this task.

What result would justify scaling:

- A consistent improvement on validation and test, concentrated in fine-label `1` and/or label `2` recall at matched label-`0` FPR, with clear degradation under randomized-cost or square-shuffle ablations.
- If achieved, scale only after adding LC0 current-board adapters and checking that the transport effect survives an encoding-family comparison.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_transport_imbalance/idea.yaml` | Create | Machine-readable summary of the Tactical Transport Imbalance Network, status, paths, and central falsification ablation. |
| `ideas/20260421_transport_imbalance/math_thesis.md` | Create | Copy Section 6 with equations, proof sketch, hypotheses, and counterexamples. |
| `ideas/20260421_transport_imbalance/architecture.md` | Create | Copy Section 7 with module specs, shapes, parameter estimates, memory estimates, and pseudocode. |
| `ideas/20260421_transport_imbalance/implementation_notes.md` | Create | Adapter fail-closed behavior, cost-bank construction, Sinkhorn numerical stability, deterministic tests, and no-leakage notes. |
| `ideas/20260421_transport_imbalance/trainer_notes.md` | Create | Loss, training defaults, fair-comparison requirements, deterministic settings, and reporting notes. |
| `ideas/20260421_transport_imbalance/ablations.md` | Create | Copy Section 9 and add run-name conventions for each ablation. |
| `ideas/20260421_transport_imbalance/train.py` | Create | Thin wrapper or documented command entrypoint that calls the existing shared trainer with `configs/transport_imbalance_simple18.yaml`; do not fork training logic unless necessary. |
| `ideas/20260421_transport_imbalance/config.yaml` | Create | Local copy of the config block from Section 12 for this idea. |
| `ideas/20260421_transport_imbalance/report_template.md` | Create | Required metrics, fine-label `3x2` diagnostic matrix, ablation comparison table, and final go/no-go conclusion fields. |
| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported research memory after implementation; add anti-duplicate guidance for OT-over-piece-square transport if it fails; preserve all leakage and label rules. |
| `src/chess_nn_playground/models/transport_imbalance_net.py` | Create | Implement `EncodingAdapter`, `SideToMoveCanonicalizer`, `ChessCostBank`, `SinkhornTransportBlock`, `TransportFeaturePool`, and `TransportImbalanceNet` as PyTorch modules. |
| `src/chess_nn_playground/models/registry.py` | Update | Register builder function `build_transport_imbalance_net` under model name `transport_imbalance_net`. |
| `configs/transport_imbalance_simple18.yaml` | Create | Main benchmark config using `simple_18`, balanced class weighting, 3 epochs, deterministic seed 42, and model defaults. |
| `configs/transport_imbalance_simple18_random_cost.yaml` | Create | Central randomized-cost ablation config with fixed recorded permutation seed. |
| `configs/transport_imbalance_simple18_uniform_cost.yaml` | Create | Uniform-cost Sinkhorn ablation config. |
| `configs/transport_imbalance_simple18_marginal_only.yaml` | Create | Marginal/material-count-only ablation config. |
| `configs/transport_imbalance_simple18_cnn_only.yaml` | Create | Parameter-matched CNN-only ablation config. |
| `tests/test_transport_cost_bank.py` | Create | Verify cost-bank shapes, determinism, nonnegativity, square-pair basis sanity, and random-cost multiset preservation. |
| `tests/test_transport_sinkhorn.py` | Create | Verify positive marginals, approximate row/column sums, finite gradients, and deterministic output on small tensors. |
| `tests/test_transport_adapter.py` | Create | Verify `simple_18` extraction using known synthetic positions and fail-closed behavior for unknown LC0 mappings. |
| `tests/test_transport_model_forward.py` | Create | Verify `(B,C,8,8) -> (B,2)` logits and compatibility with CPU forward pass. |

For `ideas/chatgpt_pro_deep_math_research_prompt.md`, Codex must update the prompt after consuming this output. The update should preserve hard constraints while adding reusable lessons from implementation and benchmarking, especially whether OT-over-piece-square mass is now an already-researched family.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0512_tuesday_local_transport_imbalance.md
  generated_at: 2026-04-21 05:12 America/Los_Angeles
  weekday: Tuesday
  timezone: local
  idea_slug: transport_imbalance
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_transport_imbalance
  name: Tactical Transport Imbalance Network
  slug: transport_imbalance
  status: draft
  created_at: 2026-04-21 05:12 America/Los_Angeles
  author: ChatGPT Pro
  short_thesis: Learn side-to-move directional entropic optimal transport between own force mass and opponent target mass as a global puzzle-likeness feature.
  novelty_claim: Uses differentiable OT over current-board piece-square measures rather than CNN scaling, LC0 copying, attack/sheaf/Hodge incidence, or one-ply move-delta pooling.
  expected_advantage: Better near-puzzle and true-puzzle recall at matched non-puzzle false-positive rate when tactical geometry is globally aligned but not captured by local convolutions.
  central_falsification_ablation: random_cost_bank_preserve_multiset
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112 and lc0_bt4_112 only with registered current-board channel maps and fail-closed adapters
  output_heads: binary_logits
  compute_notes: Sinkhorn cost O(batch * heads * iterations * 64^2); chunk heads if memory exceeds budget.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/transport_imbalance_simple18.yaml
  model_path: src/chess_nn_playground/models/transport_imbalance_net.py
  latest_result_path: null
  notes: Do not use engine scores, legal move trees, source labels, verification metadata, or one-ply move-delta sets; required diagnostic is fine-label 0/1/2 to predicted binary 0/1.
```

```yaml
config_yaml:
  run:
    name: transport_imbalance_simple18
    output_dir: results
  seed: 42
  deterministic: true
  mode: coarse_binary
  device: nvidia
  data:
    train_path: data/splits/crtk_sample_3class/split_train.parquet
    val_path: data/splits/crtk_sample_3class/split_val.parquet
    test_path: data/splits/crtk_sample_3class/split_test.parquet
    encoding: simple_18
    cache_features: false
  model:
    name: transport_imbalance_net
    input_channels: 18
    num_classes: 2
  training:
    epochs: 3
    batch_size: 512
    num_workers: 0
    learning_rate: 0.001
    weight_decay: 0.0001
    class_weighting: balanced
    early_stopping_patience: 2
    mixed_precision: false
```

```yaml
model_spec:
  model_name: transport_imbalance_net
  file_path: src/chess_nn_playground/models/transport_imbalance_net.py
  builder_function: build_transport_imbalance_net
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - EncodingAdapter
    - SideToMoveCanonicalizer
    - LocalBoardEncoder
    - TransportMassHead
    - ChessCostBank
    - SinkhornTransportBlock
    - TransportFeaturePool
    - TransportImbalanceNet
  required_config_fields:
    - model.input_channels
    - model.num_classes
    - model.transport_heads
    - model.sinkhorn_iters
    - model.sinkhorn_epsilon
    - model.mass_floor
    - model.cost_basis
    - model.canonicalize_side_to_move
    - model.fail_closed_unknown_channels
  expected_parameter_count: 350k-450k for simple_18
  expected_memory_notes: Plan tensor memory is about 4 * batch * heads * 4096 bytes per direction in fp32; chunk over heads if needed.
```

```yaml
research_continuity:
  idea_fingerprint: current-board piece-square probability measures + side-to-move canonicalization + entropic optimal transport + forward/reverse imbalance pooling + binary puzzle-likeness target
  already_researched_family_overlap: Avoids imported tactical sheaf/Hodge/attack-defense incidence families and avoids one-ply move-delta set, spectrum, entropy, or landscape pooling.
  closest_duplicate_risk: Could be mistaken for generic global pooling or material-count shortcut if ablations are weak; randomized-cost and marginal-only ablations are mandatory.
  do_not_repeat_if_this_fails:
    - Entropic OT between own piece-square mass and opponent target mass with Sinkhorn pooling
    - Forward-minus-reverse transport imbalance over 64-square cost matrices
    - Chess-geometric cost-bank variants that only add more distance bases or more transport heads
    - King-ring/high-value target demand mass with no legal move consequences
  suggested_next_search_directions:
    - Label-safe selective prediction for near-puzzle ambiguity
    - Causal invariance across encoding family and material phase without OT transport
    - Ordinal/calibration models that expose fine-label 1 uncertainty without fabricating labels
    - Information bottlenecks that explicitly suppress material/source artifacts while preserving board geometry
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Tactical Transport Imbalance Network` to the imported research memory after implementation. | Prevents the next research pass from proposing the same Sinkhorn/OT-over-piece-square mechanism as fresh. | `Imported Research Memory` |
| If the central randomized-cost ablation fails, add an anti-duplicate rule: do not propose entropic OT/Sinkhorn between own/opponent piece-square mass and king/value target mass unless the operator changes beyond cost bases, head count, or pooling statistics. | Avoids superficial variants such as more cost heads, different distance bases, or renamed “Wasserstein tactics.” | `Research Continuity` and anti-duplicate paragraph |
| Add a reusable adapter requirement: any deterministic geometry module for LC0 encodings must identify current-board channels and fail closed on history/unknown channels. | This closes a likely implementation leakage/error path for any future geometry-based idea. | `Project Context You Must Respect` or `Non-Negotiable Constraints` |
| Add a reporting requirement for semantics-destroying randomized ablations whenever a structured differentiable operator is proposed. | Forces future ideas to distinguish meaningful structure from extra parameters and nuisance shortcuts. | `Required Markdown File Content`, especially ablation requirements |
| If OT succeeds, add a prompt note that future ideas should compare against `transport_imbalance_net` as a structured-global baseline. | A successful OT layer becomes a meaningful baseline, not just an isolated experiment. | `Current baselines already exist` |
| Preserve the strict ban on engine scores, PVs, node counts, verification metadata, source labels, proposed labels, and fabricated near-puzzle labels. | The idea does not require weakening any leakage or label rules. | No weakening; only reaffirm in `Non-Negotiable Constraints` |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0512_tuesday_local_transport_imbalance.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes, `simple_18` on the existing `crtk_sample_3class` train/val/test split
- Falsification criterion is concrete: yes, randomized cost-bank preserving cost multiset and nuisance-preserving ablations
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
