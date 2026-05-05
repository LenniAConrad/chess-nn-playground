# Codex Handoff Packet: One-Ply Counterfactual Move Landscape Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0429_tuesday_local_move_landscape.md`
- Generated at: 2026-04-21 04:29:41 America/Los_Angeles
- Weekday: Tuesday
- Timezone: local (America/Los_Angeles)
- Idea slug: `move_landscape`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: One-Ply Counterfactual Move Landscape Network, abbreviated `CML-Net`
- One-sentence thesis: A chess position is puzzle-like when the deterministic, engine-free one-ply pseudo-legal consequences from the current board form a sharply anisotropic latent landscape: a few rule-allowed deltas look structurally exceptional while most alternatives look ordinary.
- Idea fingerprint: `current board tensor -> deterministic side-to-move pseudo-legal move-delta multiset -> shared move-delta encoder -> entropic free-energy / attention landscape pooling -> binary puzzle-like logits; no engine scores, no PVs, no source labels, no move tree`.
- Why this is not a common CNN/ResNet/Transformer variant: the core operator is not another spatial feature extractor over the 8x8 board; it constructs an unordered set of explicit one-ply counterfactual board deltas and classifies the geometry of that set with permutation-invariant landscape functionals.
- Current-data minimal experiment: train `CML-Net` on `simple_18` using the existing `crtk_sample_3class` train/val/test parquet split, 3 epochs, balanced binary cross-entropy or cross-entropy, and compare against the existing simple CNN and residual CNN under the same trainer and reporting stack.
- Smallest central falsification ablation: keep the same original board, side to move, valid-mask shape, moving-piece identities, source squares, and per-position move count, but randomly permute or reassign destination squares within each position and piece type before encoding the delta; if performance barely changes, the model is exploiting move-count/material artifacts rather than the semantics of rule-derived consequences.
- Expected information gain if it fails: a clean failure would say that one-ply rule consequences do not add useful signal over static occupancy for this benchmark, or that the useful signal is dominated by count/material/source artifacts rather than counterfactual move geometry.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification from a single current board position. The model receives a tensor shaped `(batch, C, 8, 8)` and returns logits shaped `(batch, 2)`. The binary output convention is:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The default binary target is `0` for fine label `0` and `1` for fine labels `1` and `2`. Reports must still include the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Current supported encodings are:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant
- `lc0_static_112`
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists

The benchmark split to use is:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The full parquet dataset has roughly 45M rows, but this idea must not point the current trainer directly at the full file until streaming support exists.

Leakage checklist:

- Allowed neural inputs: board occupancy, deterministic board coordinates, side to move, castling/en-passant planes already present in the encoding, and rule-derived pseudo-legal move geometry from the current board only.
- Forbidden neural inputs: Stockfish scores, engine evaluations, principal variations, mate scores, node counts, tablebase outcomes, puzzle verification metadata, source labels, proposed labels, dataset provenance, and any unresolved candidate-pool status.
- Label safety: do not fabricate class `1` or class `2`; unresolved candidates remain unresolved and must not be treated as verified near-puzzles or verified puzzles.
- Search safety: do not generate a move tree, do not evaluate candidates with an engine, and do not use forced-line, checkmate, or stalemate oracles.

Boundary between safe rule-derived features and leakage-prone features:

- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack or move geometry derived only from the current board are allowed.
- Full legal-move generation, legal move counts, checkmate/stalemate oracles, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated. This packet avoids full legal filtering in the main model and uses pseudo-legal moves only.
- Engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance must never be neural-network inputs.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry may only use known current-board channels. History channels may be consumed by a learned root adapter, but not by the pseudo-legal move enumerator. If the current-board channel map is not known exactly, the adapter must fail closed instead of guessing.

## 4. Research Map

| Source | URL | What is borrowed | What is not copied |
|---|---|---|---|
| Zaheer et al., “Deep Sets” | https://arxiv.org/abs/1703.06114 | The theorem-level idea that permutation-invariant functions over unordered sets can be represented by shared per-item maps plus symmetric pooling. Here, the unordered set is the set of pseudo-legal move deltas. | No point-cloud task, no generic set benchmark, and no claim that simple sum pooling alone is sufficient for chess. |
| Lee et al., “Set Transformer” | https://arxiv.org/abs/1810.00825 | The design principle that set elements may be pooled by learned attention while preserving permutation invariance. | Do not implement a vanilla Transformer over 64 board squares; do not make self-attention over moves the central mechanism in the first experiment. |
| Ilse, Tomczak, and Welling, “Attention-based Deep Multiple Instance Learning” | https://arxiv.org/abs/1802.04712 | The bag-level classification pattern: a label is assigned to a bag of instances, and attention can expose which instances drive the bag prediction. Here, the bag is the move-delta multiset and there are no instance labels. | No histopathology assumptions, no fabricated move labels, and no claim that attention weights are ground-truth explanations. |
| Wachter, Mittelstadt, and Russell, “Counterfactual Explanations without Opening the Black Box” | https://arxiv.org/abs/1711.00399 | The broad counterfactual idea that useful information can live in “what would change if an action were taken?” rather than only in the original static state. | No GDPR/legal framework, no recourse objective, and no post-hoc explanation system. The counterfactuals here are rule-generated chess deltas used inside the classifier. |
| Veličković and Blundell, “Neural Algorithmic Reasoning” | https://arxiv.org/abs/2105.02761 | The inductive-bias idea that a network can be aligned with a discrete algorithmic object. Here, the discrete object is pseudo-legal one-ply move generation, not engine search. | No value iteration, no planning rollout, no learned chess engine, and no claim of algorithmic correctness beyond deterministic enumeration. |
| python-chess core documentation | https://python-chess.readthedocs.io/en/latest/core.html | The operational distinction that pseudo-legal moves may leave or put the king in check but otherwise follow chess movement rules. This supports the no-check-oracle boundary. | Do not require python-chess as a runtime dependency if the repository prefers its own deterministic tensor/bitboard helper; do not use legal move generation in the main model. |
| Chessprogramming Wiki, “Pseudo-Legal Move” and “Move Generation” | https://www.chessprogramming.org/Pseudo-Legal_Move and https://www.chessprogramming.org/Move_Generation | The standard chess-programming distinction between pseudo-legal and legal generation, especially that legal filtering requires extra king-safety logic. | Do not import engine heuristics, move ordering, killer moves, alpha-beta search, or evaluation terms. |
| FIDE Laws of Chess 2023 | https://rcc.fide.com/wp-content/uploads/2022/11/Laws_of_Chess-2023.pdf | The basic piece movement rules are the normative source for deterministic move-delta construction. | Do not use game-result adjudication, competition rules, or draw/mate/stalemate adjudication as model inputs. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Make the existing simple CNN wider or deeper | `src/chess_nn_playground/models/cnn.py` | This is ordinary capacity scaling and does not test a new inductive bias about puzzle structure. |
| Add more residual blocks to the residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | This is already covered by the residual baseline family and would mainly test depth/width tuning. |
| Train another LC0-style CNN/residual CNN | Existing LC0 BT4-style CNN/residual variants | This is too close to the current LC0-style baseline suite and would not use unavailable history in a principled new way. |
| Vanilla ViT over 64 squares | Common square-token Transformer | This is explicitly disallowed and would be a generic architecture swap rather than a chess-specific operator. |
| Plain GNN on squares with adjacency/attack edges | Common graph neural network over board squares | This is too ordinary and too close to static attack/defense graph approaches already researched. |
| Another attack-defense sheaf, Hodge, Laplacian, curvature, or tension model | Imported sheaf/Hodge research packets | The imported research memory already covers static tactical incidence/sheaf families; adding edge labels or renaming the energy would be a duplicate. |
| Hyperparameter tuning of optimizer, batch size, learning rate, or weight decay | All existing trained baselines | Useful for engineering but not a research idea and not falsifiable as a mechanism. |
| Ensembling multiple CNNs or encodings | Any collection of existing baselines | Ensembling may improve leaderboard metrics but gives little mechanistic information and is explicitly not the core idea requested. |
| Supervise on Stockfish scores, PV move, node count, mate score, or tablebase outcome | Engine-assisted chess models | These features are forbidden leakage for this classification task. |
| Use full legal-move tree, minimax, proof search, or mate-distance labels | Engine/search-surrogate baselines | This would be leakage-prone and would collapse puzzle classification into hidden engine analysis. |
| Treat unresolved candidates as near-puzzles | Dataset-label shortcut | This fabricates class `1`/`2` labels and violates the data contract. |
| Material-only phase classifier with handcrafted piece values | Static material heuristic | It would likely learn source/material artifacts and not the tactical-likeness mechanism this task is meant to test. |

## 6. Mathematical Thesis

### Input space definition

Let `C` be the number of channels in the selected encoding. The neural input is

\[
x \in \mathcal{X}_C \subset \mathbb{R}^{C \times 8 \times 8}.
\]

For `simple_18`, assume a deterministic adapter

\[
A_{18}: \mathcal{X}_{18} \to \mathcal{B}
\]

that extracts a board state

\[
b=(O, s, c, e),
\]

where `O` is 12-plane piece occupancy, `s` is side to move, `c` is castling-right information, and `e` is en-passant target information. For LC0-style encodings, the adapter must only expose deterministic geometry when the current-board channel map is explicitly known.

### Label/target definition

Let the fine label be

\[
\ell \in \{0,1,2\}.
\]

The binary target is

\[
y = \mathbf{1}\{\ell \in \{1,2\}\}.
\]

The model estimates

\[
f_\theta(x) \in \mathbb{R}^2,
\]

and is trained with cross-entropy against `y`. Fine labels are only used for diagnostics and the rectangular `3x2` confusion matrix.

### Data distribution assumptions

Assume the benchmark split samples from an empirical distribution

\[
\widehat{D}\;\text{over}\;(x,\ell).
\]

The central modeling assumption is not that puzzles are always tactically forced in one ply. The weaker assumption is:

\[
P(y=1 \mid x)\;\text{contains signal in the distribution of current-board, one-ply, rule-only consequences beyond static occupancy alone.}
\]

This assumption can be false. The ablations below are designed to detect that.

### Allowed symmetry or equivariance assumptions

Chess is not fully invariant under board rotations or reflections because pawn direction, castling, en-passant, and side to move matter. This model should not impose full dihedral board symmetry.

The one symmetry imposed by the core operator is permutation invariance over the arbitrary enumeration order of pseudo-legal moves:

\[
\{m_1,\dots,m_n\} = \{m_{\pi(1)},\dots,m_{\pi(n)}\}
\]

for every permutation `π`. This is safe because move-list order has no chess meaning.

Optional future extensions may canonicalize from the side-to-move perspective, but the first implementation should not silently mirror or rotate boards unless every affected channel, including castling and en-passant, is transformed correctly and ablated.

### Core hypothesis

Let

\[
\widetilde{M}(b)
\]

be the deterministic pseudo-legal side-to-move move set from board `b`, generated without king-safety filtering, checkmate detection, stalemate detection, engine evaluation, or search. Every move

\[
m \in \widetilde{M}(b)
\]

induces a sparse board transformation

\[
T_m b
\]

and a signed move-delta object

\[
\Delta_m(b)=T_m b-b.
\]

The hypothesis is that puzzle-like positions are often characterized by a learned latent energy landscape

\[
e_\theta(b,m)=q_\theta\big(r_\theta(b),\phi_\theta(b,\Delta_m(b),m)\big)
\]

whose mass is more concentrated or more anisotropic than in non-puzzle positions. In plain language: the model should learn whether one or a few rule-derived consequences look unusually structurally important relative to the other current-board alternatives.

### Formal object introduced

Define the Counterfactual Move Landscape operator

\[
\mathcal{L}_\theta(b)
=\left(
 r_\theta(b),
 \operatorname{Mean}_{m \in \widetilde{M}(b)} z_m,
 \operatorname{Var}_{m \in \widetilde{M}(b)} z_m,
 \sum_{m \in \widetilde{M}(b)} p_m z_m,
 G_\tau(e),
 H(p)
\right),
\]

where

\[
z_m=\phi_\theta(b,\Delta_m(b),m),
\]

\[
p_m=\frac{\exp(e_\theta(b,m)/\tau)}{\sum_{m' \in \widetilde{M}(b)}\exp(e_\theta(b,m')/\tau)},
\]

\[
G_\tau(e)=\tau\log\left(\frac{1}{|\widetilde{M}(b)|}\sum_{m \in \widetilde{M}(b)}\exp(e_\theta(b,m)/\tau)\right)-\frac{1}{|\widetilde{M}(b)|}\sum_{m \in \widetilde{M}(b)}e_\theta(b,m),
\]

and

\[
H(p)=-\frac{1}{\log |\widetilde{M}(b)|}\sum_{m \in \widetilde{M}(b)}p_m\log(p_m+\epsilon).
\]

The classifier is

\[
f_\theta(x)=h_\theta\big(\mathcal{L}_\theta(A(x))\big).
\]

### Variational principle

The entropic free-energy term satisfies

\[
\tau\log\left(\frac{1}{n}\sum_{i=1}^n \exp(e_i/\tau)\right)
=\max_{p\in\Delta_n}\left[\sum_{i=1}^n p_i e_i + \tau H(p)\right]-\tau\log n.
\]

The optimizer is

\[
p_i \propto \exp(e_i/\tau).
\]

Therefore, the learned attention distribution is the solution of a smooth maximum-over-counterfactuals problem. The model can interpolate between mean-like behavior at high temperature and max-like behavior at low temperature, without selecting a hard move or using an engine.

### Proposition

For any fixed board `b`, `CML-Net` is invariant to the enumeration order of `\widetilde{M}(b)`. More generally, if a target function of a board can be written as a continuous permutation-invariant function of the multiset

\[
\{\Delta_m(b),m : m\in \widetilde{M}(b)\}
\]

together with a root board embedding, then a sufficiently wide shared move encoder plus symmetric landscape pooling can approximate that function on the finite benchmark domain.

### Proof sketch or derivation

Each move embedding `z_m` is computed by the same function `φθ`, independent of the move-list index. Mean, variance, masked softmax-weighted sum, entropy, and log-sum-exp are symmetric functions of the indexed collection. Applying a permutation to the move-list order permutes the intermediate `z_m` and `e_m` values but leaves every pooled statistic unchanged. This proves enumeration-order invariance.

The approximation statement follows the Deep Sets representation pattern: continuous invariant functions over sets can be represented or approximated by shared per-element maps followed by a symmetric aggregation and an output map, under the usual compactness and capacity assumptions. The current setting is easier in one respect because the board domain is finite after discretized encodings and a fixed maximum move slot count, but harder in practice because finite neural capacity and optimization may still fail.

The free-energy formula is the standard entropy-regularized maximum identity. Differentiating the right-hand side under the simplex constraint yields the softmax distribution.

### What is actually proven

- The core move-set pooling is invariant to arbitrary pseudo-legal move enumeration order.
- The free-energy pooling has a precise entropy-regularized maximum interpretation.
- The model can represent functions that depend on pooled statistics of one-ply pseudo-legal move deltas.

### What remains only hypothesized

- That verified near-puzzles and verified puzzles are measurably more anisotropic in this learned one-ply landscape than known non-puzzles.
- That this signal improves class `1` near-puzzle diagnostics, not only class `2` puzzle detection.
- That pseudo-legal moves are the right leakage-safe level of rule consequence; legal filtering might help but is deliberately not used in the main experiment.
- That the move-delta bottleneck suppresses material/source artifacts better than a plain CNN.

### Counterexamples where the idea should fail

- Quiet strategic positions whose puzzle-likeness depends on multi-ply zugzwang, fortress, or long maneuvering rather than immediate one-ply consequences.
- Puzzles where the critical feature is specifically checkmate or stalemate legality, which the main model does not compute.
- Non-puzzle positions with many captures or promotions that create a sharp-looking but tactically meaningless one-ply landscape.
- Positions whose labels are dominated by dataset provenance or material distribution rather than board-tactical structure.
- Positions where the correct move is quiet and its one-ply board delta is not locally distinctive until after an opponent reply.

## 7. Architecture Specification

### Module names

Implement the main model in:

```text
src/chess_nn_playground/models/move_landscape_net.py
```

Recommended classes/functions:

- `MoveLandscapeNet(torch.nn.Module)`
- `Simple18BoardAdapter`
- `LC0CurrentBoardAdapter`
- `PseudoLegalDeltaEnumerator`
- `MoveRecordEncoder`
- `LandscapeSetPool`
- `build_move_landscape_net(config)`

Registry model name:

```text
move_landscape_net
```

### Forward-pass steps

Input:

```text
x: float tensor [B, C, 8, 8]
```

Step 1: adapter validation and board extraction.

- `Simple18BoardAdapter` maps `x` into:
  - piece planes: `[B, 12, 8, 8]`
  - side-to-move scalar/plane: `[B, 1]` or `[B, 1, 8, 8]`
  - castling/en-passant auxiliary planes as provided
- If encoding is `lc0_static_112` or `lc0_bt4_112`, deterministic geometry extraction is allowed only if the current-board piece-plane and side-to-move channel map is explicitly configured and tested. Otherwise raise a clear error.

Step 2: root board encoding.

- Apply a small learned stem to the full input tensor:
  - `Conv2d(C, root_channels=48, kernel_size=3, padding=1)`
  - normalization, activation
  - two light 3x3 convolution blocks at width 48
- Output root grid:

```text
F: [B, 48, 8, 8]
```

- Global mean and max pooling over squares, then projection:

```text
r: [B, 96]
```

Step 3: deterministic pseudo-legal move-delta enumeration.

The enumerator uses only current-board occupancy, side to move, and already-present castling/en-passant planes. It must not use check, mate, stalemate, legal filtering, engine scoring, PVs, or labels.

For the side to move, generate pseudo-legal records for:

- pawns: one-step forward if empty, two-step from initial rank if clear, diagonal captures, en-passant capture if the en-passant plane indicates a target, and promotions to `q/r/b/n` when reaching the promotion rank
- knights: normal knight jumps to empty or opponent-occupied squares
- bishops, rooks, queens: ray moves stopped by first occupied square; include capture if first occupied square has opponent piece
- king: one-square moves to empty or opponent-occupied squares, without checking attacked squares
- castling: optional, default disabled in the first experiment unless implemented as a purely geometric candidate using castling-right and empty-path information only; never test whether the king is in check or passes through check

Return padded tensors with `max_moves=256`:

```text
from_sq:      [B, K] int64, 0..63
to_sq:        [B, K] int64, 0..63
piece_id:     [B, K] int64, 0..11
capture_id:   [B, K] int64, -1 or 0..11
promo_id:     [B, K] int64, -1 or q/r/b/n code
special_id:   [B, K] int64, normal/en-passant/castle/promotion
valid_mask:   [B, K] bool
```

Use deterministic ordering, for example `(piece_id, from_sq, move_kind, to_sq, promo_id)`, only so padding is reproducible. The model must be invariant to this order; the ordering is not a feature.

If more than `K=256` pseudo-legal moves are encountered, the first implementation should raise a clear exception during tests. If real data hits this exception, Codex may increase `max_moves` to 320 and document the observed maximum. Do not silently drop moves in the main benchmark.

Step 4: gather local board features and encode move records.

Gather from root grid `F`:

```text
F_from: [B, K, 48]
F_to:   [B, K, 48]
F_diff: [B, K, 48] = F_to - F_from
```

For sliding moves, optionally add a deterministic path summary:

```text
F_path_mean: [B, K, 48]
```

Set `F_path_mean=0` for non-sliding moves or if path summary is not implemented in the minimal version.

Add learned embeddings:

```text
piece_emb:   [B, K, 16]
capture_emb: [B, K, 16]
promo_emb:   [B, K, 8]
special_emb: [B, K, 8]
rel_emb:     [B, K, 16]  # relative displacement bucket, side-to-move aware
```

Concatenate these into `u_m` with approximate width 208-256, then use a shared MLP:

```text
z: [B, K, move_dim=96]
```

Invalid padded moves must be masked and should not contribute to pooling.

Step 5: compute move energies.

Broadcast root embedding:

```text
r_expand: [B, K, 96]
```

Compute:

```text
e: [B, K] = EnergyMLP(concat(z, r_expand))
```

Set invalid entries to a large negative value before softmax. Do not feed raw move count as a default scalar in the main model.

Step 6: landscape pooling.

With temperature `tau=0.5` by default:

```text
p = masked_softmax(e / tau, valid_mask)      # [B, K]
h_attn = sum_m p_m z_m                       # [B, 96]
h_mean = masked_mean(z, valid_mask)          # [B, 96]
h_var = masked_mean((z - h_mean)^2, mask)    # [B, 96]
```

Compute scalar landscape diagnostics:

```text
energy_mean:       [B, 1]
energy_max:        [B, 1]
energy_lse_gap:    [B, 1]
energy_top2_gap:   [B, 1]
entropy_norm:      [B, 1]
```

The main model should not include `num_valid_moves` as a classifier input. A count-only ablation may expose it deliberately.

Step 7: classifier head.

Concatenate:

```text
h = concat(r, h_mean, h_var, h_attn, scalar_features)
# shape about [B, 96*4 + 5] = [B, 389]
```

Classifier:

```text
Linear(389, 128) -> activation -> dropout(0.1) -> Linear(128, 2)
```

Return:

```text
logits: [B, 2]
```

compatible with the shared trainer.

### Parameter-count estimate

For `simple_18` with `root_channels=48` and `move_dim=96`, expected trainable parameters are roughly 250k-400k depending on normalization and path-summary choices. For `lc0_*` encodings, the first convolution increases the count by about `(112-18)*48*3*3 ≈ 40.6k`, so the model should remain well below one million parameters.

### FLOP or complexity estimate

Let `B` be batch size, `K` the padded move count, `d=96` the move dimension, and `R=48` the root grid width.

- Root CNN complexity: `O(B * 64 * C * R * 3 * 3 + B * 64 * R^2 * 3 * 3)`.
- Move gather complexity: `O(B*K*R)`.
- Move MLP and energy complexity: approximately `O(B*K*d^2)`.
- Landscape pooling complexity: `O(B*K*d)`.

There is no `K^2` move self-attention in the first implementation. This is intentional: the research claim is about counterfactual deltas and entropic landscape pooling, not a generic Set Transformer.

### Required config fields

Recommended config fields:

```yaml
model:
  name: move_landscape_net
  input_channels: 18
  num_classes: 2
  encoding: simple_18
  root_channels: 48
  root_embedding_dim: 96
  move_dim: 96
  max_moves: 256
  landscape_temperature: 0.5
  use_count_scalar: false
  include_path_summary: false
  include_castling_candidates: false
  adapter_strict: true
  lc0_geometry_channel_map: null
```

### Encoding support

First experiment: use `simple_18` only. This is the safest choice because the channel semantics are explicit enough to derive pseudo-legal deltas.

`simple_18` assumptions:

- The 12 piece planes are known and ordered by the repository's existing `simple_18` convention.
- Side-to-move, castling, and en-passant planes are interpreted exactly as in the current exporter.
- The adapter should include tests that reconstruct a few simple FEN-like positions from tensors and verify pseudo-legal moves.

`lc0_static_112` assumptions:

- The learned root CNN may consume all 112 channels.
- The deterministic move enumerator may only consume current-board piece and side-to-move channels if an explicit mapping is available.
- If that mapping is absent or ambiguous, raise an error and instruct the user to run `simple_18`.

`lc0_bt4_112` assumptions:

- The learned root CNN may consume all 112 channels, including zero-filled history channels.
- The deterministic move enumerator must not infer history semantics.
- Use current-board channels only for rule geometry, and fail closed if unknown.

### Pseudocode

```text
forward(x):
    board = adapter.extract_current_board_or_fail(x)
    F = root_stem(x)                               # [B, 48, 8, 8]
    r = root_pool_project(F)                       # [B, 96]

    moves = pseudo_legal_enumerator(board)         # padded records, mask [B, K]
    F_from = gather_square(F, moves.from_sq)       # [B, K, 48]
    F_to = gather_square(F, moves.to_sq)           # [B, K, 48]
    F_diff = F_to - F_from                         # [B, K, 48]

    record_emb = embed(piece, capture, promo, special, rel_disp)
    u = concat(F_from, F_to, F_diff, record_emb)
    z = move_record_mlp(u)                         # [B, K, 96]

    e = energy_mlp(concat(z, broadcast(r)))        # [B, K]
    pooled = landscape_pool(z, e, moves.valid_mask)

    h = concat(r, pooled.mean, pooled.var, pooled.attn, pooled.scalars)
    return classifier(h)                           # [B, 2]
```

This is pseudocode only. Codex should implement repository-style PyTorch modules and tests rather than copy this text as final code.

## 8. Loss, Training, And Regularization

Primary loss:

- Use `torch.nn.CrossEntropyLoss` on binary logits and binary labels.
- Use the existing benchmark's balanced class weighting if available. If not, compute class weights from the training split only.

Optional auxiliary loss:

- `subset_consistency_loss`, default off for the first fair run.
- If enabled, form two deterministic random subsets of valid moves during training, run landscape pooling on each subset, and penalize `KL(logits_subset_a || logits_subset_b)` or squared logit difference with weight `0.02`.
- This is meant to prevent brittle dependence on a single enumerated slot. It must not become the central result unless separately ablated.

Class weighting:

- `class_weighting: balanced`.
- Do not weight fine labels `1` and `2` separately in the main binary loss unless the shared trainer already supports that baseline fairly.

Batch size expectations:

- Start with batch size `512`, matching the provided config skeleton.
- If memory is high because of `K=256`, reduce to `256` and keep all baselines rerun or compared under documented constraints.

Optimizer and learning rate defaults:

- Optimizer: `AdamW`.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Epochs: `3` for the minimal experiment.
- Early stopping patience: `2`.
- Mixed precision: false for first reproducibility run.

Regularizers:

- Dropout `0.1` in the classifier head.
- Weight decay as above.
- No entropy regularizer in the first run. Entropy is a diagnostic and classifier feature, not a target.
- Optional move-record dropout may be added only as an ablation or later robustness run.

Determinism requirements:

- Use seed `42`.
- Deterministic pseudo-legal enumeration order.
- Deterministic padding.
- If any random subset auxiliary loss is enabled, seed it from the global trainer seed and epoch/batch index.
- The move shuffling ablation must be reproducible and logged.

What must stay unchanged for fair comparison:

- Same train/val/test files.
- Same binary target definition.
- Same reporting code, confusion matrices, predictions, and leaderboards.
- Same input encoding for the main comparison; compare the first `CML-Net` run against `simple_18` baselines before attempting LC0-style encodings.
- Same epoch budget unless Codex explicitly reports both budget-matched and best-effort runs.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Degree-preserving destination shuffle, central falsification | Preserve board, valid mask, moving piece, source square, and per-position/piece-type move count, but randomize destination squares among compatible records before gathering `F_to` and delta metadata | The actual semantics of rule-derived destinations and deltas matter | If this matches the main model, the central counterfactual-move claim fails; the model is likely using counts, source-square distribution, or root CNN features. |
| Zero-delta count-preserving ablation | Keep pseudo-legal move records and mask but set `F_to`, `F_diff`, capture/promo/special embeddings, and relative displacement embeddings to zero | Move consequences, not just move availability, drive the gain | If performance is unchanged, consequence encoding is not being used. |
| Count-only/mask-only ablation | Expose only the number of valid pseudo-legal records or mask-derived scalar summaries plus root CNN | Tests whether pseudo-legal mobility count is the hidden signal | If count-only is close to the main model, the method is too artifact-prone and should not be scaled. |
| Root-only ablation | Disable the move branch and use the root CNN head with a parameter count close to `CML-Net` | Tests whether gains come from the board stem only | If root-only matches main, the move landscape is unnecessary. |
| Move-only ablation | Disable root pooled embedding in the classifier except for local gathered features already used by moves | Tests whether the move landscape can classify without global static context | If move-only collapses, root context is required; if it matches main, the model may be a move-list classifier. |
| Mean-pool only | Remove softmax attention, free-energy gap, entropy, and top-2 gap; use masked mean/variance only | Tests the anisotropic-landscape thesis | If mean-pool matches main, sharpness/free-energy is not the important mechanism. |
| High-temperature landscape | Set `tau` very high, making attention nearly uniform | Tests whether extreme candidate concentration matters | If high-temperature equals main, hard-ish consequence salience is not useful. |
| Random move set from matched material bucket | Replace each position's move records with records from another position matched by side-to-move and coarse material count, while keeping the root board | Tests whether contextual compatibility between board and its own deltas matters | If this matches main, the move branch is learning generic material/mobility priors, not board-specific consequences. |
| Captures-only move set | Keep only pseudo-legal captures and promotions, pad otherwise | Tests whether the signal is merely capture/tactical material volatility | If captures-only matches main, quiet move consequences add little. |
| Quiet-only move set | Remove captures, promotions, and en-passant; keep quiet pseudo-legal moves | Tests whether non-capturing tactical structure contributes | If quiet-only performs well, the model may capture positional motifs beyond obvious captures. |

Every central ablation must produce the same report artifacts as the main model, including the `3x2` fine-label diagnostic matrix.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN under `src/chess_nn_playground/models/cnn.py`.
- Existing `simple_18` residual CNN under `src/chess_nn_playground/models/residual_cnn.py`.
- Existing small/medium/deep variants if already in the leaderboard.
- Existing LC0 BT4-style CNN/residual CNN variants only after the `simple_18` experiment is complete and reported.

Metrics to inspect:

- Binary ROC-AUC.
- Binary PR-AUC.
- Accuracy.
- Balanced accuracy.
- F1 for puzzle-like class.
- Precision and recall for binary output `1`.
- Calibration summary if the existing reports include it.
- Required fine-label `0/1/2 -> predicted 0/1` confusion matrix.

Near-puzzle diagnostic:

- Use validation set to choose a threshold matching a fixed fine-label-`0` false-positive rate, preferably the false-positive rate of the strongest `simple_18` baseline at its default threshold.
- On test, report class `1` recall and precision at that matched fine-label-`0` false-positive rate.
- Also report class `2` recall at the same threshold, because an improvement only on class `2` may mean the model is finding obvious tactical puzzles but not near-puzzles.

Required artifacts:

- Model config used for the main run.
- Checkpoint or final state path if the trainer normally saves it.
- Predictions parquet/csv including example id, fine label, binary target, predicted probability for output `1`, thresholded prediction, and split.
- Standard binary confusion matrix.
- Required rectangular `3x2` fine-label diagnostic matrix.
- The same artifacts for the central destination-shuffle ablation, zero-delta ablation, and root-only ablation.
- Report containing runtime, parameter count, maximum observed pseudo-legal move count, and whether any overflow occurred.

Success threshold:

- On `simple_18`, improve test ROC-AUC by at least `+0.01` absolute over the strongest existing `simple_18` CNN/residual CNN baseline under the same data split, or improve class `1` recall by at least `+0.03` absolute at matched fine-label-`0` false-positive rate without reducing class `2` recall by more than `0.02` absolute.
- The central destination-shuffle ablation should lose at least half of the main model's improvement over the root-only or best baseline comparator, or lose at least `0.005` ROC-AUC absolute. Otherwise the structural claim is weak.

Failure threshold:

- Main model is within `±0.003` ROC-AUC of the best `simple_18` baseline and does not improve the matched-FPR class `1` diagnostic.
- Destination-shuffle, zero-delta, or count-only ablations are statistically indistinguishable from the main model across at least three seeds or bootstrap confidence intervals.
- The model improves only by increasing fine-label-`0` false positives without improving class `1` or class `2` recall at matched FPR.

What result would make me abandon the idea:

- Across three seeds, the destination-shuffle ablation and count-only ablation match the main model within noise while the main model fails to improve class `1` recall at matched fine-label-`0` FPR. That would strongly suggest pseudo-legal move-delta semantics are not the useful signal here.

What result would justify scaling:

- The main model beats the best `simple_18` baseline on ROC-AUC or class `1` matched-FPR recall, and the central destination-shuffle ablation clearly loses performance.
- The `3x2` matrix shows improvement on fine label `1` without merely overpredicting output `1` for fine label `0`.
- No pseudo-legal overflow occurs at `K=256`, and runtime is acceptable enough to rerun at more epochs or try LC0 root-channel adapters.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0429_move_landscape/idea.yaml` | Create | Machine-readable metadata for `CML-Net`, including idea fingerprint, novelty claim, central ablation, config path, and model path. |
| `ideas/20260421_0429_move_landscape/math_thesis.md` | Create | Copy and refine Section 6, including input/target definitions, landscape operator, variational principle, proven claims, hypotheses, and counterexamples. |
| `ideas/20260421_0429_move_landscape/architecture.md` | Create | Codex-facing design notes from Section 7 with tensor shapes, module boundaries, adapter behavior, and parameter/FLOP estimates. |
| `ideas/20260421_0429_move_landscape/implementation_notes.md` | Create | Detailed notes for deterministic pseudo-legal enumeration, fail-closed channel adapters, overflow handling, and no-leakage constraints. |
| `ideas/20260421_0429_move_landscape/trainer_notes.md` | Create | Loss, optimizer, class weighting, determinism, fair-comparison settings, and artifact expectations. |
| `ideas/20260421_0429_move_landscape/ablations.md` | Create | Full ablation table and exact central falsification instructions, including destination shuffling and count-only controls. |
| `ideas/20260421_0429_move_landscape/train.py` | Create | Lightweight entrypoint that calls the shared trainer with `configs/move_landscape_simple18.yaml`; do not duplicate trainer logic. |
| `ideas/20260421_0429_move_landscape/config.yaml` | Create | Idea-local copy of the benchmark config for `simple_18`, model name `move_landscape_net`, 3 epochs, batch size 512, balanced class weighting. |
| `ideas/20260421_0429_move_landscape/report_template.md` | Create | Template requiring baseline comparison, metrics, `3x2` fine-label matrix, near-puzzle matched-FPR diagnostic, ablation results, and final decision. |
| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Update | Preserve hard constraints and add this packet to imported research memory after implementation; include anti-duplicate rules for one-ply pseudo-legal counterfactual move-landscape models if this fails. |
| `src/chess_nn_playground/models/move_landscape_net.py` | Create | PyTorch implementation of `MoveLandscapeNet`, adapters, pseudo-legal delta enumerator, move record encoder, landscape pooling, and builder function. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `move_landscape_net` and builder function without breaking existing model names. |
| `configs/move_landscape_simple18.yaml` | Create | Main benchmark config using current split paths, `simple_18`, `input_channels: 18`, `num_classes: 2`, and model-specific fields. |
| `configs/move_landscape_simple18_dest_shuffle.yaml` | Create | Central destination-shuffle ablation config if the trainer supports model ablation flags; otherwise document the command-line override. |
| `configs/move_landscape_simple18_zero_delta.yaml` | Create | Zero-delta count-preserving ablation config. |
| `configs/move_landscape_simple18_root_only.yaml` | Create | Root-only ablation config with comparable parameter budget if feasible. |
| `tests/test_move_landscape_net.py` | Create | Shape tests: input `[2, 18, 8, 8]` returns logits `[2, 2]`; deterministic behavior under seed; invalid LC0 map fails closed. |
| `tests/test_pseudo_legal_delta_enumerator.py` | Create | Focused pseudo-legal tests for pawn moves, knight moves, sliders, captures, promotions, en-passant plane handling, padding/mask shape, and no legal-check filtering. |
| `tests/test_move_landscape_ablations.py` | Create | Verify destination shuffle preserves valid count/source/piece fields, zero-delta removes destination consequence fields, and count-only exposes no board-delta semantics. |

For `ideas/chatgpt_pro_deep_math_research_prompt.md`, Codex should update the prompt only after consuming and implementing or rejecting this output. The update should preserve leakage rules, label rules, falsification requirements, and anti-duplicate requirements. It should add the result of this pass to research memory so the next ChatGPT Pro cycle does not repeat a one-ply pseudo-legal move-delta set model with entropic/attention landscape pooling under a new name.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0429_tuesday_local_move_landscape.md
  generated_at: 2026-04-21T04:29:41-07:00
  weekday: Tuesday
  timezone: local
  idea_slug: move_landscape
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0429_move_landscape
  name: One-Ply Counterfactual Move Landscape Network
  slug: move_landscape
  status: draft
  created_at: 2026-04-21T04:29:41-07:00
  author: ChatGPT Pro
  short_thesis: Puzzle-like positions may be detected by the anisotropy of engine-free one-ply pseudo-legal move-delta consequences rather than static occupancy alone.
  novelty_claim: Constructs a deterministic current-board pseudo-legal counterfactual move-delta multiset and classifies entropic landscape shape; not a CNN scaling, square Transformer, LC0 copy, static attack-defense graph, or sheaf/Hodge variant.
  expected_advantage: Better class 1 near-puzzle recall at matched fine-label-0 false-positive rate if tactical ambiguity is visible in one-ply consequence landscapes.
  central_falsification_ablation: Degree-preserving destination shuffle preserving board, move count, moving piece, and source square while destroying true destination/delta semantics.
  target_task: coarse_binary
  input_representation: simple_18 first; optional lc0_static_112/lc0_bt4_112 only with explicit fail-closed current-board channel maps
  output_heads: binary_logits
  compute_notes: About 250k-400k parameters for simple_18; O(B*K*d^2) move MLP with K=256 and d=96; no K^2 move attention.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/move_landscape_simple18.yaml
  model_path: src/chess_nn_playground/models/move_landscape_net.py
  latest_result_path: null
  notes: Main model uses pseudo-legal moves only, no full legal filtering, no engine features, no move tree, no source/proposed labels.
```

```yaml
config_yaml:
  run:
    name: move_landscape_simple18
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
    name: move_landscape_net
    input_channels: 18
    num_classes: 2
    encoding: simple_18
    root_channels: 48
    root_embedding_dim: 96
    move_dim: 96
    max_moves: 256
    landscape_temperature: 0.5
    use_count_scalar: false
    include_path_summary: false
    include_castling_candidates: false
    adapter_strict: true
    lc0_geometry_channel_map: null
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
  model_name: move_landscape_net
  file_path: src/chess_nn_playground/models/move_landscape_net.py
  builder_function: build_move_landscape_net
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18BoardAdapter
    - LC0CurrentBoardAdapter
    - PseudoLegalDeltaEnumerator
    - MoveRecordEncoder
    - LandscapeSetPool
    - MoveLandscapeNet
  required_config_fields:
    - input_channels
    - num_classes
    - encoding
    - root_channels
    - root_embedding_dim
    - move_dim
    - max_moves
    - landscape_temperature
    - use_count_scalar
    - adapter_strict
  expected_parameter_count: 250k-400k for simple_18 depending on normalization and optional path summary
  expected_memory_notes: Main extra activation is [batch, max_moves, move_dim]; with batch 512, max_moves 256, move_dim 96 this is about 50 MB per float32 tensor before overhead. Reduce batch size to 256 if needed and document.
```

```yaml
research_continuity:
  idea_fingerprint: current-board pseudo-legal side-to-move move-delta multiset + shared move record encoder + entropic free-energy/attention landscape pooling + binary puzzle-likeness target + no engine metadata
  already_researched_family_overlap: Avoids imported static attack-defense sheaf/Hodge/Laplacian/curvature/tension family; closest overlap is rule-derived chess geometry, but the falsifiable operator is one-ply counterfactual board deltas rather than static incidence energy.
  closest_duplicate_risk: Could be mistaken for attention-based multiple-instance learning or a legal-move GNN; distinguish by pseudo-legal current-board deltas, entropy/free-energy landscape diagnostics, and degree-preserving destination-shuffle falsification.
  do_not_repeat_if_this_fails:
    - One-ply pseudo-legal move-delta bag with DeepSets or attention pooling
    - Entropic free-energy or top-gap pooling over current-board move consequences
    - Destination-shuffle/count-only variants of the same move-landscape mechanism
    - Simple replacements of pseudo-legal moves with legal moves unless leakage is separately justified and ablated
  suggested_next_search_directions:
    - Causal invariance across source/material/phase environments with adversarial bottlenecks
    - Label-safe uncertainty or ordinal heads separating non-puzzle, near-puzzle, and puzzle diagnostics without fabricating labels
    - Optimal transport over material-change distributions that does not reuse attack-defense incidence or sheaf maps
    - Information bottlenecks that suppress material/source artifacts while preserving side-to-move tactical signal
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `One-Ply Counterfactual Move Landscape Network` to the imported research memory after implementation, with its fingerprint and outcome. | Prevents the next cycle from renaming the same pseudo-legal move-delta bag/free-energy pooling idea. | `Imported Research Memory` |
| Add an anti-duplicate rule: do not propose one-ply pseudo-legal move-delta multiset models with attention, DeepSets, MIL, log-sum-exp, entropy, or free-energy pooling if this packet fails central ablations. | The mechanism has many easy aliases; this makes the duplicate boundary clear. | `Research Continuity` or `What Counts As Creative Enough` |
| Preserve the distinction between pseudo-legal current-board deltas and full legal-move/search consequences; require any future legal-filtered idea to justify legality as rule-only, label-independent, engine-free, and ablated. | This is the most likely leakage boundary to blur in future prompts. | `Problem Restatement And Data Contract` |
| Require every structured-operator proposal to include a semantics-destroying randomized ablation that preserves obvious nuisance statistics such as count, degree, material, and side to move. | Makes falsification stronger than comparing against a weak no-structure baseline. | `Ablation Plan` requirements |
| Add a reusable near-puzzle diagnostic requirement: class `1` recall or precision at matched fine-label-`0` false-positive rate. | Prevents future ideas from looking good only by catching easy class `2` puzzles or overpredicting positives. | `Benchmark And Falsification Criteria` |
| Require fail-closed encoding adapters whenever a deterministic rule-derived feature depends on channel semantics. | Avoids silent misuse of LC0 history/current-board planes. | `Required Markdown File Content`, Section 7 guidance |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0429_tuesday_local_move_landscape.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes, `simple_18` on the current `crtk_sample_3class` split
- Falsification criterion is concrete: yes, degree-preserving destination shuffle plus zero-delta/count-only controls
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes; this is not another static attack-defense graph, sheaf, Hodge/Laplacian, curvature, or tension-energy variant
