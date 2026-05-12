# Codex Handoff Packet: Rule-Only Counterfactual Move-Delta Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0436_tuesday_los_angeles_move_delta_bottleneck.md`
- Generated at: 2026-04-21 04:36:33 America/Los_Angeles
- Weekday: Tuesday
- Timezone: America/Los_Angeles, filename token `los_angeles`
- Idea slug: `move_delta_bottleneck`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Rule-Only Counterfactual Move-Delta Bottleneck, abbreviated `CDBN`.
- One-sentence thesis: A position is puzzle-like when the side-to-move's rule-only one-ply counterfactual move deltas form a sparse, anisotropic consequence distribution, where a few interventions look structurally different from the ordinary legal-looking background even without engine scores or search.
- Idea fingerprint: `current-board simple_18 occupancy + optional side-to-move canonicalization + pseudo-legal one-ply board-delta multiset + permutation-invariant sparse move bottleneck + binary puzzle-likeness target + no engine/legal-search metadata`.
- Why this is not a common CNN/ResNet/Transformer variant: The network does not only convolve over the current board and does not attend over 64 static squares; it constructs a deterministic, label-independent set of counterfactual move interventions from the current board, encodes each intervention as a sparse board delta, and classifies from the geometry of that move-delta distribution.
- Current-data minimal experiment: Train `counterfactual_delta_bottleneck` on `data/splits/crtk_sample_3class/split_train.parquet`, validate on `split_val.parquet`, test on `split_test.parquet`, using `simple_18`, `mode: coarse_binary`, and the existing reporting stack with the required `3x2` fine-label diagnostic.
- Smallest central falsification ablation: Replace each position's pseudo-legal move-delta set by a degree-preserving, moving-piece/captured-piece-histogram-preserving set sampled from other positions in the same batch, while leaving the static board encoder and classifier unchanged.
- Expected information gain if it fails: Failure would show that one-ply rule-only counterfactual consequence geometry adds little beyond static occupancy and mobility artifacts for this dataset; next cycles should avoid move-delta pooling and look instead at label-safe uncertainty, causal environment invariance, or non-move-based compression.

Closest baseline or common method it resembles: an attention-based multiple-instance model over a bag of rule-generated move instances, with a small CNN context encoder. The novel part for this project is the deterministic chess intervention operator and the falsification against degree-preserving semantic randomization.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification from a single board position. The model returns:

- output `0`: non-puzzle;
- output `1`: puzzle-like.

The available fine labels are:

- fine label `0`: known non-puzzle;
- fine label `1`: verified near-puzzle;
- fine label `2`: verified puzzle.

The default training target is coarse binary, with `y = 0` for fine label `0` and `y = 1` for fine labels `1` and `2`. The benchmark must still report the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

The model must be a PyTorch `torch.nn.Module` accepting an input tensor of shape:

```text
(batch, C, 8, 8)
```

and returning logits of shape:

```text
(batch, 2)
```

The current benchmark split is fixed:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The full Parquet dataset has roughly 45M rows. This idea must not point the current trainer directly at the full file until streaming support exists.

Allowed encodings currently known to the project:

- `simple_18`: 12 piece planes plus side-to-move, castling, and en-passant state;
- `lc0_static_112`;
- `lc0_bt4_112`, where unavailable history planes are zero-filled until exporter support exists.

First experiment recommendation: use `simple_18` only. This idea depends on exact channel semantics for deterministic move-delta generation, and `simple_18` is the only encoding described tightly enough in the prompt. LC0 adapters may be added only if Codex can verify and hard-code the current-board channel map from existing exporter code.

Leakage checklist:

- No Stockfish score, win/draw/loss estimate, PV, mate score, node count, search depth, verification metadata, source label, proposed label, dataset provenance, puzzle ID, or generator metadata may be a neural-network input.
- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack or movement geometry derived only from the current board are allowed.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated.
- This proposal uses pseudo-legal one-ply movement deltas only. It must not filter moves by king safety, must not detect checkmate or stalemate, must not score positions, and must not look at opponent replies.
- Pseudo-legal move cardinality is not concatenated as a feature. Because the valid-move mask can still leak mobility information indirectly, Codex must run the count-only and degree-preserving shuffle ablations.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry may use only verified current-board piece/state channels. History channels may be consumed only by learned neural adapters, never by the rule-derived move generator. If current-board semantics are unknown, the adapter must fail closed.

Boundary between safe rule-derived features and leakage:

- Safe: source square, destination square, moving piece type, captured piece type visible on the current board, promotion type implied by current-board pawn movement, en-passant square if already present, deterministic path occupancy for sliders, and a sparse representation of `T_m x - x` for a pseudo-legal one-ply move `m`.
- Unsafe by default: true legal filtering by self-check, check flags, mate/stalemate flags, number of legal moves, engine-evaluated child positions, and any line beyond one side-to-move intervention.
- Allowed only as separately reported ablations: a full legal generator implemented by rules alone, if Codex clearly marks it as non-mainline and reports whether it changes metrics. The main model should not need it.

## 4. Research Map

| Source | URL | Borrowed | Not copied |
|---|---|---|---|
| Zaheer et al., “Deep Sets,” NeurIPS 2017 | https://arxiv.org/abs/1703.06114 and https://papers.nips.cc/paper/6931-deep-sets | The theorem that permutation-invariant set functions can be represented by shared element maps followed by pooled aggregation and a final map. | No generic set benchmark, no point-cloud architecture, and no claim that Deep Sets alone is novel here. |
| Lee et al., “Set Transformer,” ICML 2019 | https://arxiv.org/abs/1810.00825 and https://proceedings.mlr.press/v97/lee19d.html | The idea that set-structured inputs can use attention while preserving permutation invariance. | No full Set Transformer stack and no vanilla transformer over the 64 board squares. |
| Ilse, Tomczak, and Welling, “Attention-based Deep Multiple Instance Learning,” ICML 2018 | https://arxiv.org/abs/1802.04712 and https://proceedings.mlr.press/v80/ilse18a.html | The bag-label framing: a board has one label, while generated move deltas are unlabeled instances whose contributions may be sparse. | No medical-image MIL pipeline and no assumption that attention weights are faithful explanations. |
| Martins and Astudillo, “From Softmax to Sparsemax,” ICML 2016 | https://arxiv.org/abs/1602.02068 and https://proceedings.mlr.press/v48/martins16.html | Sparse attention over candidate move deltas via projection onto the probability simplex. | No sparsemax loss requirement and no multi-label task copy. |
| Peters, Niculae, and Martins, “Sparse Sequence-to-Sequence Models,” ACL 2019 | https://arxiv.org/abs/1905.05702 and https://aclanthology.org/P19-1146/ | Optional `entmax15` as a smoother sparse-attention replacement if sparsemax is unstable. | No sequence-to-sequence decoder, beam search, or text-generation mechanism. |
| Alemi et al., “Deep Variational Information Bottleneck,” ICLR 2017 | https://arxiv.org/abs/1612.00410 and https://openreview.net/forum?id=HyxQzBceg | The general regularization intuition that narrow stochastic representations can suppress nuisance information. | The first implementation should use a deterministic low-dimensional bottleneck; VIB is optional and must not be the central claim. |
| Arjovsky et al., “Invariant Risk Minimization,” 2019 | https://arxiv.org/abs/1907.02893 | Motivation for separating stable causal structure from source/material artifacts. | No IRM objective is proposed for the first experiment because reliable environment labels are not specified in the prompt. |
| Chessprogramming Wiki, “Move Generation” | https://www.chessprogramming.org/Move_Generation | Terminology distinction between pseudo-legal moves and fully legal moves, especially the fact that pseudo-legal generation does not check whether the king is left in check. | No engine search, alpha-beta, quiescence search, transposition table, or evaluation heuristic. |

All citations above were checked at packet generation time. This packet borrows mathematical and architectural motifs, not model weights, datasets, engine evaluations, or chess-engine labels.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN over `simple_18` | `src/chess_nn_playground/models/cnn.py` | Already present and too likely to learn static material or source artifacts without testing counterfactual structure. |
| Residual CNN over `simple_18` | `src/chess_nn_playground/models/residual_cnn.py` | Already present; adding residual depth is ordinary capacity scaling, not a new inductive bias. |
| LC0-style CNN or residual CNN over `lc0_static_112` / `lc0_bt4_112` | Existing LC0 BT4-style CNN/residual variants | Already covered by the baseline suite and too close to “copy LC0 but smaller.” |
| Ordinary ViT over 64 squares | Common square-token transformer | Rejected because it is a vanilla transformer over board squares and has no chess-specific falsifiable operator. |
| Plain GNN on squares or pieces | Generic graph neural network over 64 nodes or piece nodes | Too ordinary unless the edge semantics are new; static attack/defense graph variants are already covered by imported sheaf packets. |
| Static attack-defense sheaf / Hodge / Laplacian / curvature model | Recent research packets in `ideas/all_ideas/research/packets/classic/` | Explicitly disallowed as a repeated family unless the operator changes substantially; this packet avoids that family. |
| Hyperparameter tuning of the residual baseline | Existing trainer/config variants | Tuning learning rate, width, depth, or optimizer settings is not a research idea and gives weak scientific information. |
| Ensemble of CNN, residual CNN, and LC0-style models | Any leaderboard ensemble | Ensembling can improve leaderboard metrics but obscures whether a new mechanism works. |
| “Use more data” or train directly on the 45M-row Parquet file | Current non-streaming trainer | Not valid until streaming support exists, and data scale alone is not a model hypothesis. |
| Legal-move-count or mobility feature classifier | Hand-crafted mobility baseline | Too leakage-prone and likely spurious; it is included only as a diagnostic ablation, not as the main model. |
| Engine-free full legal move tree to depth two or three | Search surrogate | Too close to forced-line search and likely to leak puzzle verification structure; the main proposal stops at one pseudo-legal intervention. |
| Standard supervised contrastive pretraining on fine labels | Generic representation learning | It risks overusing fine-label boundaries and does not address the current binary benchmark with a distinct chess operator. |

## 6. Mathematical Thesis

### Input space definition

Let `B` be the set of board states representable by the current encoding. For the first experiment, an input is

```text
x ∈ X = {0,1}^{12×8×8} × S
```

where the first factor is current-board piece occupancy and `S` contains side-to-move, castling, and en-passant state available in `simple_18`. The raw tensor passed to the PyTorch module is in `R^{18×8×8}`.

Let `c(x)` denote an optional canonicalization that flips ranks and swaps colors when black is to move, so that the moving side is represented consistently. This is not a full chess-board symmetry assumption; it is a color-perspective normalization. It must correctly transform castling and en-passant state or be disabled.

### Label/target definition

The fine label is

```text
t ∈ {0,1,2}
```

with meanings `0 = known non-puzzle`, `1 = verified near-puzzle`, `2 = verified puzzle`. The training label is

```text
y = 1[t ∈ {1,2}] ∈ {0,1}.
```

The model estimates `P(y=1 | x)` and returns two logits.

### Data distribution assumptions

The training, validation, and test splits are samples from a project-specific distribution `P_split(x,t)`. The distribution may contain nuisance correlations from material balance, phase, opening/endgame mix, and source construction. The thesis does not assume that labels are generated only by tactics; it assumes only that a useful part of the signal is expressed by the geometry of one-ply rule-only interventions.

### Allowed symmetry or equivariance assumptions

Chess is not invariant under arbitrary rotations or reflections because pawn direction, castling semantics, en-passant, and side-to-move matter. The only optional symmetry used here is color-perspective normalization:

```text
black-to-move position -> flip ranks, swap colors, transform state, make moving side canonical
```

This is valid only as a rule-preserving relabeling of the same chess situation. No `D4` equivariance, no 90-degree rotations, and no file reflection are assumed by default.

The move-delta aggregator must be permutation-invariant to the arbitrary ordering of generated pseudo-legal moves:

```text
F({δ_m : m ∈ M(x)}) = F({δ_{π(m)} : m ∈ M(x)})
```

for any permutation `π` of the generated move list.

### Core hypothesis

Define `M(x)` as the set of pseudo-legal side-to-move interventions generated from the current board without king-safety filtering and without terminal-state tests. For each `m ∈ M(x)`, let `T_m x` be the board after applying the intervention by chess movement rules, and let

```text
δ_m(x) = T_m x - x
```

be the sparse board delta. A puzzle-like board often has a small subset of interventions whose deltas create a sharply different local consequence pattern from the background of ordinary interventions. Non-puzzles are hypothesized to have a less anisotropic or less semantically organized move-delta distribution after controlling for material and mobility.

This is a hypothesis about the current dataset, not a theorem about chess.

### Formal object introduced by the idea

The central object is the empirical counterfactual move-delta measure:

```text
μ_x = (1 / |M(x)|) Σ_{m∈M(x)} δ_{r_θ(x,m)}
```

where `δ_z` is a Dirac measure at vector `z`, and `r_θ(x,m)` is a learned embedding of the sparse intervention `δ_m(x)` in the context of the parent board.

A learned score function

```text
s_θ(x,m) = a_θ(z_θ(x), r_θ(x,m))
```

produces masked sparse weights

```text
α_θ(x) = sparsemax({s_θ(x,m) / τ : m ∈ M(x)}).
```

The move-cone bottleneck is

```text
b_θ(x) = Σ_{m∈M(x)} α_θ(x,m) r_θ(x,m).
```

The classifier receives

```text
h_θ(x) = [z_θ(x), b_θ(x), mean_{m∈M(x)} r_θ(x,m), second_moment_{m∈M(x)} r_θ(x,m), κ_θ(x)]
```

where `z_θ(x)` is a parent-board context vector and

```text
κ_θ(x) = max_m s_θ(x,m) - log((1/|M(x)|) Σ_m exp(s_θ(x,m)))
```

is a normalized anisotropy statistic. `κ_θ` is optional in the first implementation; if included, it must use `logmeanexp`, not `logsumexp`, to reduce direct dependence on move count.

### Variational principle / objective

The primary training objective is empirical risk minimization:

```text
min_θ E_{(x,y)∼P_train} CE(y, f_θ(x)) + λ_sparse R_sparse(α_θ(x)) + λ_wd ||θ||_2^2.
```

The first implementation should set `λ_sparse = 0` or a tiny value and rely on sparsemax/entmax for sparse selection. A later optional version may add a deterministic information bottleneck penalty by reducing the dimensionality of `b_θ` or a variational bottleneck penalty

```text
β KL(q_θ(u | x) || N(0,I))
```

but VIB is not required for the central test.

### Proposition

For any fixed maximum pseudo-legal move count `M_max` and compact move-embedding domain `K ⊂ R^d`, every continuous permutation-invariant function

```text
F: ⋃_{n≤M_max} K^n / S_n -> R
```

can be uniformly approximated by a function of the form

```text
ρ( (1/n) Σ_{i=1}^n φ(v_i) )
```

for suitable continuous maps `φ` and `ρ`, up to the usual Deep Sets assumptions on the domain. Moreover, sparse simplex attention can approximate an existence-like or extreme-element statistic

```text
max_i q(v_i)
```

as the attention temperature decreases and the score network `q` becomes expressive.

### Proof sketch or derivation

The first claim is the Deep Sets representation theorem applied to the finite set of move-delta embeddings. The use of a mean rather than a sum keeps the representation less directly tied to move count, though count can still leak through padding/masking and must be ablated. The second claim follows from the fact that a simplex-weighted sum with weights concentrated on an argmax element returns the embedding of an extreme-scoring element. Sparsemax is the Euclidean projection of scores onto the probability simplex, so for separated scores it assigns zero mass to lower-scoring candidates and positive mass only to a finite active set.

Thus the model class can represent functions such as:

```text
there exists a move delta in a learned tactical cone, and most other move deltas are outside it
```

while remaining invariant to generated move ordering.

### What is actually proven

- The aggregation part can be made permutation-invariant over the generated move-delta set.
- A Deep Sets-style or sparse-attention set module is expressive enough to approximate continuous invariant functions of finite move-delta multisets under bounded-cardinality and compactness assumptions.
- The rule generator can be deterministic and label-independent if it uses only current-board channels.

### What remains only hypothesized

- Puzzle-likeness in the current CRTK sample is strongly correlated with sparse anisotropy of one-ply pseudo-legal move deltas.
- The move-delta bottleneck suppresses source/material artifacts better than a static CNN.
- Near-puzzles, fine label `1`, will show intermediate behavior that improves class `1` recall or precision at matched fine-label-`0` false-positive rate.

### Counterexamples where the idea should fail

- A puzzle whose key move is a quiet waiting move with no distinctive one-ply board delta until several replies later.
- Zugzwang or fortress-like endgames where puzzle-likeness depends on full legal move constraints and opposition, not a sparse side-to-move intervention cone.
- Positions where the label is mostly a dataset artifact, such as material imbalance, source identity, or opening/endgame distribution.
- Positions where pseudo-legal moves leaving the king in check dominate the generated set and obscure true legal tactical structure.
- Problems whose puzzle-likeness depends on checkmate/stalemate detection; this model intentionally refuses terminal oracles.
- Positions with castling tactics if the first implementation omits pseudo-castling deltas.

## 7. Architecture Specification

### Module names

Implement the model in:

```text
src/chess_nn_playground/models/counterfactual_delta_bottleneck.py
```

Recommended classes/functions:

- `CounterfactualDeltaBottleneckNet`
- `EncodingAdapterBase`
- `Simple18BoardAdapter`
- `Lc0CurrentBoardAdapter`
- `SideToMoveCanonicalizer`
- `PseudoLegalMoveDeltaGenerator`
- `BoardContextEncoder`
- `MoveDeltaTupleEncoder`
- `MaskedSparsemax` or `MaskedEntmax15`
- `MoveConeBottleneck`
- `CounterfactualDeltaClassifierHead`
- builder function: `build_counterfactual_delta_bottleneck(config)`

### Forward-pass steps and shapes

Assume first experiment uses `simple_18`, `C = 18`, `B = batch`, hidden board channels `H = 64`, board vector dimension `D = 128`, move embedding dimension `R = 128`, and padded maximum generated moves `M = 256`.

1. Input:

```text
x: [B, C, 8, 8]
```

2. Encoding adapter:

```text
board: [B, 12, 8, 8]
state: dict with side_to_move, castling, en_passant
learned_input: [B, C, 8, 8]
```

For `simple_18`, the adapter uses known current-board piece planes and state planes. For LC0 encodings, the adapter must fail closed unless an explicit channel map is supplied.

3. Optional side-to-move canonicalization:

```text
board_canon: [B, 12, 8, 8]
learned_input_canon: [B, C, 8, 8]
state_canon: transformed state
```

Disable this if castling/en-passant transformation is not implemented correctly.

4. Parent board context encoder:

```text
F, z = BoardContextEncoder(learned_input_canon)
F: [B, H, 8, 8]
z: [B, D]
```

`BoardContextEncoder` may be a small residual CNN. It is a context provider, not the research idea. Keep it comparable to existing small baselines.

5. Pseudo-legal move-delta generation from current-board occupancy only:

```text
move_from: [B, M] int64 in 0..63
move_to: [B, M] int64 in 0..63
moving_piece: [B, M] int64 in 0..11
captured_piece: [B, M] int64 in 0..12, where 12 means none
promotion_piece: [B, M] int64, none/knight/bishop/rook/queen
move_flags: [B, M, Fm]
path_mask: optional [B, M, 64]
valid_mask: [B, M] bool
```

Generation rules:

- Include side-to-move piece moves only.
- Pawns: one-step push if empty, two-step from start rank if path empty, diagonal captures if occupied by opponent, en-passant capture if the en-passant plane/state is known, promotions with four promotion piece types.
- Knights/kings: normal pseudo-legal destinations on board, excluding own occupied destinations.
- Bishops/rooks/queens: ray moves until blocked; include first opponent-occupied square as a capture; stop after any occupied square.
- Castling: default `include_pseudo_castling: false` for the first experiment to avoid attacked-square logic. If enabled later, use only rights plus empty path, never attacked-square checks.
- Do not test whether the own king is left in check.
- Do not generate check, mate, stalemate, or engine-evaluation flags.
- Sort moves deterministically by `(from, to, promotion_piece, flag_code)` before padding/truncation.

6. Gather local context:

```text
src_ctx = gather(F, move_from): [B, M, H]
dst_ctx = gather(F, move_to): [B, M, H]
path_ctx = masked mean of F over path_mask: [B, M, H]  # zeros for non-sliders or if disabled
```

7. Embedding lookup and sparse delta representation:

```text
piece_emb: [B, M, Ep]
captured_emb: [B, M, Ep]
promo_emb: [B, M, Ep]
from_square_emb: [B, M, Es]
to_square_emb: [B, M, Es]
flag_emb_or_flags: [B, M, Ef]
```

The sparse board delta `δ_m = T_m x - x` is represented by the tuple `(from, to, moving_piece, captured_piece, promotion_piece, flags)` plus parent context at source, destination, and path. Codex does not need to materialize `[B,M,12,8,8]` dense child boards.

8. Move-delta tuple encoder:

```text
r = MoveDeltaTupleEncoder(concat(src_ctx, dst_ctx, path_ctx, embeddings, flags, z broadcast))
r: [B, M, R]
```

Invalid padded moves must be zeroed after encoding.

9. Sparse bottleneck:

```text
scores = score_mlp(concat(r, z broadcast)): [B, M]
alpha = masked_sparsemax(scores / temperature, valid_mask): [B, M]
b_sparse = sum_m alpha_m r_m: [B, R]
b_mean = masked_mean_m r_m: [B, R]
b_second = masked_mean_m r_m^2: [B, R]
kappa = max(scores) - logmeanexp(scores over valid moves): [B, 1]
```

Do not concatenate raw valid-move count. If `kappa` proves too count-correlated, remove it in an ablation.

10. Classifier head:

```text
h = concat(z, b_sparse, b_mean, b_second, kappa): [B, D + 3R + 1]
logits = MLP(h): [B, 2]
```

The returned tensor must be exactly `[B, num_classes]` and compatible with the shared trainer.

### Parameter-count estimate

For `simple_18`, `H=64`, `D=128`, `R=128`, `Ep=16`, `Es=8`, and a 2-block context CNN:

- `BoardContextEncoder`: about 160k to 260k parameters depending on residual-block count.
- Square and piece embeddings: under 5k parameters.
- `MoveDeltaTupleEncoder`: about 80k to 140k parameters.
- Sparse score MLP and classifier head: about 60k to 120k parameters.
- Total expected range: roughly 320k to 550k parameters.

Keep the first implementation below 750k parameters so improvements cannot be dismissed as ordinary capacity scaling.

### FLOP and complexity estimate

Let `M` be the padded move cap, usually `256`, and `L ≤ 7` the maximum ray length.

- Move generation: `O(B * M * L)` rule operations, mostly integer/boolean tensor work.
- Context CNN: `O(B * H^2 * 8 * 8 * residual_blocks)`.
- Move encoding: `O(B * M * R * W)` where `W` is the move-MLP width.
- Bottleneck and pooling: `O(B * M * R)`.

The dominant memory term is `O(B * M * R)`. With `B=512`, `M=256`, `R=128`, the move embedding tensor is about 67 MB in fp32 before intermediate MLP activations. If this is too large, use internal chunking over moves while preserving exact outputs, or lower `R` to `96`. Do not change the benchmark split or labels to solve memory issues.

### Required config fields

Add these model config fields outside the minimal machine-readable block if the repo config schema allows them:

```yaml
model:
  name: counterfactual_delta_bottleneck
  input_channels: 18
  num_classes: 2
  encoding: simple_18
  hidden_channels: 64
  board_dim: 128
  move_dim: 128
  max_moves: 256
  sparse_attention: sparsemax
  attention_temperature: 1.0
  canonicalize_side_to_move: true
  include_pseudo_castling: false
  include_en_passant: true
  use_kappa: true
  fail_closed_unknown_channels: true
```

### Encoding-adapter assumptions

`simple_18`:

- Must expose 12 current-board piece planes and state planes for side-to-move, castling, and en-passant.
- Codex must verify the exact piece-plane order in the existing exporter. If the order differs from assumed white-piece-then-black-piece convention, update the adapter and tests.
- This is the only encoding recommended for the first benchmark.

`lc0_static_112`:

- Deterministic geometry is allowed only if a verified channel map identifies the current-board piece planes and side-to-move/state channels.
- Learned context may consume all 112 channels, but move generation must use only verified current-board channels.
- If current-board channels cannot be verified, raise a clear exception before training.

`lc0_bt4_112`:

- Same fail-closed rule as `lc0_static_112`.
- History channels may be seen by the learned context encoder, but unavailable zero-filled history must not be interpreted by the move generator.
- The first experiment should not rely on BT4 history because exporter support is incomplete.

### Pseudocode, not final implementation

```text
forward(x):
    board, state, learned = adapter(x)
    if canonicalize:
        board, state, learned = canonicalizer(board, state, learned)

    F, z = context_encoder(learned)

    moves = pseudo_legal_generator(board, state, max_moves=M)
    # moves contains from, to, piece, capture, promo, flags, path_mask, valid_mask

    src = gather_square(F, moves.from)
    dst = gather_square(F, moves.to)
    path = path_pool(F, moves.path_mask)

    e = concat(src, dst, path,
               piece_embedding(moves.piece),
               capture_embedding(moves.capture),
               promo_embedding(moves.promo),
               square_embedding(moves.from),
               square_embedding(moves.to),
               moves.flags,
               broadcast(z))

    r = move_encoder(e)
    r = zero_invalid(r, moves.valid_mask)

    scores = score_mlp(concat(r, broadcast(z)))
    alpha = masked_sparsemax(scores / temperature, moves.valid_mask)

    b_sparse = weighted_sum(alpha, r)
    b_mean = masked_mean(r, moves.valid_mask)
    b_second = masked_mean(r * r, moves.valid_mask)
    kappa = masked_max(scores) - masked_logmeanexp(scores)

    h = concat(z, b_sparse, b_mean, b_second, kappa)
    return classifier(h)
```

## 8. Loss, Training, And Regularization

- Primary loss: weighted cross-entropy over the coarse binary target.
- Optional auxiliary loss: none by default. A tiny sparse-attention entropy penalty may be added only after the no-auxiliary baseline is recorded:

```text
L = CE_weighted + λ_entropy * mean(H(alpha) / log(|M(x)|))
```

with `λ_entropy ≤ 1e-4`. This penalty should be minimized only if sparsemax/entmax alone is too diffuse. It must not be needed for the central claim.

- Class weighting: use the existing balanced class weighting already supported by the shared trainer.
- Batch size expectations: start with `512` to match the required config and baselines. If memory fails, use internal move-chunking or gradient accumulation before reducing batch size. Any reduction must be reported.
- Optimizer defaults: `AdamW`, learning rate `0.001`, weight decay `0.0001`.
- Epochs: use the existing short benchmark value of `3` for the first pass, with early stopping patience `2`.
- Mixed precision: `false` for the first deterministic comparison unless the existing baseline configs use it.
- Regularizers: weight decay, dropout `0.05` to `0.10` in the classifier head, and optional piece-plane dropout only as a later ablation. Do not add strong data augmentation before the central ablation is run.
- Determinism requirements: fixed seed `42`; deterministic move ordering; deterministic padding/truncation; no random move sampling in the main model; deterministic sparsemax/entmax implementation; `torch.use_deterministic_algorithms(True)` if compatible with the repo.
- What must stay unchanged for a fair comparison: split files, target construction, `mode: coarse_binary`, batch reporting, confusion matrices, fine-label `3x2` diagnostics, prediction artifact format, baseline training epoch budget, class weighting policy, and no use of the full 45M-row file.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Degree-preserving cross-position move-delta shuffle | Replaces each board's move-delta instances with instances from other boards while preserving valid count and moving/captured/promotion-piece histograms as closely as possible | The board-specific semantics of `T_m x - x` matter beyond mobility and piece mix | If performance stays within 0.5 percentage points or improves, the central counterfactual operator is not carrying useful semantic signal. |
| Static-only context encoder | Bypasses `PseudoLegalMoveDeltaGenerator` and classifies from `z` only with a parameter-matched head | Move-delta distribution adds information beyond a small CNN context encoder | If static-only matches the main model, the idea reduces to an ordinary CNN. |
| Count-only / mobility probe | Feeds only pseudo-legal move count and coarse move-type counts to a tiny classifier | The model is not just exploiting mobility artifacts | If count-only approaches the main model, abandon this mechanism or add stronger count-invariance constraints. |
| Mean-only set pooling | Replaces sparse bottleneck `b_sparse` with `masked_mean(r)` only | Sparse anisotropy, not just average move statistics, is predictive | If mean-only matches the main model, the sparse “few forcing interventions” thesis is unsupported. |
| Uniform valid-mask random templates | Keeps valid move count but assigns random from/to squares and random piece labels consistent with occupancy only at the source | Chess movement semantics matter, not just having a bag of tokens | If random templates match the main model, the move generator is unnecessary. |
| No parent local context | Encodes move tuple `(from,to,piece,capture,promo,flags)` without `src_ctx`, `dst_ctx`, or `path_ctx` | Counterfactual deltas need board context, not only move notation | If this matches, the model may be using piece/mobility priors rather than position-specific consequence structure. |
| Remove `kappa` anisotropy scalar | Classifier receives `z`, `b_sparse`, `b_mean`, and `b_second` but not `kappa` | Explicit score anisotropy is useful and not merely count-correlated | If removing `kappa` improves, keep it out; sparse pooling may already encode the needed signal. |
| Captures/promotions-only generator | Generates only pseudo-legal captures, en-passant captures, and promotions | Puzzle-likeness may be dominated by tactical forcing captures | If this matches full moves, quiet-move deltas are not adding value; next cycles should specialize or drop quiet moves. |
| Disable side-to-move canonicalization | Uses raw board orientation and side-to-move planes | Canonicalization improves sample efficiency without invalid symmetry assumptions | If disabling helps, the canonicalizer likely mishandles state or discards useful orientation information. |
| Move-cap sensitivity | Runs `max_moves` at 64, 128, and 256 with deterministic sorted truncation | The operator is robust to padding/truncation choices | If results swing strongly, generator ordering or cap is a confound that must be fixed before scaling. |
| Optional full-legal-filter ablation, non-mainline | Filters pseudo-legal moves by self-check using a rule-only legal generator but no terminal/search oracle | Whether pseudo-legal illegal moves are harming signal | If legal filtering greatly helps, future work may justify rule-only legal moves, but this must remain separate from engine/search leakage. |

The smallest central falsification is the first row: degree-preserving cross-position move-delta shuffle. It destroys the semantics of the structured operator while preserving the most obvious nuisance variables.

## 10. Benchmark And Falsification Criteria

### Baselines to compare against

Compare against the best already available single-model baselines trained under the same split and epoch budget:

- `simple_18` simple CNN, small/medium if available;
- `simple_18` residual CNN, small/medium/deep if available;
- LC0-style baselines only as secondary context, because the first CDBN run uses `simple_18`.

Do not compare the main `simple_18` CDBN only to weak baselines. Use the strongest existing `simple_18` single model as the primary baseline.

### Metrics to inspect

- Test accuracy.
- Positive-class precision, recall, and F1.
- AUROC and AUPRC if the reporting stack already supports them.
- Calibration or Brier score if already available.
- Required `3x2` matrix: true fine label `0/1/2` by predicted binary output `0/1`.
- Class `1` near-puzzle recall at a matched fine-label-`0` false-positive rate.
- Class `1` precision among predicted positives at the same threshold used for the main binary report.

### Near-puzzle diagnostic

Codex should compute a threshold on validation predictions such that fine-label-`0` false-positive rate matches the primary baseline's fine-label-`0` false-positive rate. At that matched false-positive rate, report:

```text
class_1_recall_matched_fpr
class_2_recall_matched_fpr
positive_precision_matched_fpr
```

This is important because near-puzzles are where a puzzle-likeness model should show graded tactical sensitivity without fabricating new labels.

### Required artifacts

For the main model and every central ablation, save:

- config YAML actually used;
- model summary and parameter count;
- training log;
- validation metrics by epoch;
- test metrics;
- binary confusion matrix;
- fine-label `3x2` diagnostic matrix;
- predictions Parquet/CSV compatible with existing report tooling;
- ablation comparison table;
- failure notes, including whether move-count probes explain the result.

### Success threshold

Treat the idea as successful enough to keep if all are true:

1. Main CDBN improves positive-class F1 by at least 2.0 absolute percentage points over the strongest same-encoding `simple_18` single-model baseline, or improves class `1` recall at matched fine-label-`0` FPR by at least 5.0 absolute percentage points without reducing class `2` recall by more than 2.0 points.
2. The degree-preserving move-delta shuffle loses at least half of the main model's gain over the static-only baseline, or loses at least 1.5 absolute F1 points if the gain is small.
3. Count-only/mobility probe remains far below the main model and cannot explain most of the improvement.
4. The required `3x2` matrix shows improved handling of fine label `1`, not only over-predicting all positives.

### Failure threshold

Treat the idea as failed if any are true:

- Main CDBN is within ±0.5 F1 points of the static-only parameter-matched context encoder.
- Degree-preserving shuffled deltas match or beat the main model.
- Count-only/mobility probe matches within 1.0 F1 point of the main model.
- Fine-label `0` false positives rise substantially while class `1` recall does not improve at matched FPR.
- Training is unstable or OOM without an obvious implementation bug, even after move-chunking or reducing `move_dim` to `96`.

### What result would make me abandon the idea

Abandon this family if the semantic shuffle, random-template, and count-only ablations together show that the model's apparent gain comes from mobility, piece-type histograms, or static CNN capacity rather than board-specific one-ply deltas.

### What result would justify scaling

Scale only if CDBN clears the success threshold and the central shuffle ablation drops meaningfully. Next scaling steps should be modest: better vectorized generator, `move_dim=160`, more epochs under the same split, and then a carefully streamed dataset experiment after streaming support exists.

## 11. Implementation Plan For Codex

Use idea id:

```text
20260421_0436_move_delta_bottleneck
```

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0436_move_delta_bottleneck/idea.yaml` | Create | Machine-readable idea metadata copied from the `idea_yaml` block, plus status fields for benchmark results. |
| `ideas/20260421_0436_move_delta_bottleneck/math_thesis.md` | Create | Section 6 from this packet, with any implementation-discovered corrections noted explicitly. |
| `ideas/20260421_0436_move_delta_bottleneck/architecture.md` | Create | Section 7 from this packet, including tensor shapes, adapter rules, and generator constraints. |
| `ideas/20260421_0436_move_delta_bottleneck/implementation_notes.md` | Create | Detailed notes on deterministic pseudo-legal generation, channel adapter validation, sparsemax masking, and memory chunking. |
| `ideas/20260421_0436_move_delta_bottleneck/trainer_notes.md` | Create | Loss, optimizer, class weighting, determinism, fair-comparison constraints, and reporting requirements. |
| `ideas/20260421_0436_move_delta_bottleneck/ablations.md` | Create | Section 9 table plus exact commands/config names for each ablation. |
| `ideas/20260421_0436_move_delta_bottleneck/train.py` | Create | Thin wrapper or documented command entrypoint that calls the shared trainer with this idea's config; do not duplicate the trainer. |
| `ideas/20260421_0436_move_delta_bottleneck/config.yaml` | Create | Idea-local config mirroring `configs/counterfactual_delta_bottleneck_simple18.yaml`. |
| `ideas/20260421_0436_move_delta_bottleneck/report_template.md` | Create | Report skeleton requiring main metrics, `3x2` matrix, matched-FPR near-puzzle diagnostic, and ablation table. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to researched families after implementation, with anti-duplicate guidance for move-delta bottleneck variants if it fails or succeeds. Preserve all leakage and label rules. |
| `src/chess_nn_playground/models/counterfactual_delta_bottleneck.py` | Create | Model classes listed in Section 7; no engine calls; deterministic pseudo-legal generator; output logits `[batch, 2]`. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register `counterfactual_delta_bottleneck` and builder function. |
| `configs/counterfactual_delta_bottleneck_simple18.yaml` | Create | Main benchmark config using `simple_18`, coarse binary mode, seed `42`, balanced class weighting, and model fields from this packet. |
| `configs/counterfactual_delta_bottleneck_shuffle_ablation.yaml` | Create | Same config with degree-preserving cross-position move-delta shuffle enabled. |
| `configs/counterfactual_delta_bottleneck_static_only.yaml` | Create | Same context encoder/classifier capacity but move generator disabled. |
| `configs/counterfactual_delta_bottleneck_count_probe.yaml` | Create | Tiny diagnostic model or feature path using only pseudo-legal count/type counts; mark as leakage diagnostic, not a valid main model. |
| `configs/counterfactual_delta_bottleneck_mean_only.yaml` | Create | Replaces sparse bottleneck with mean pooling. |
| `tests/test_counterfactual_delta_bottleneck.py` | Create | Shape tests, determinism tests, fail-closed adapter tests, no-check/no-mate feature tests, sparsemax mask tests, and degree-preserving shuffle sanity tests. |
| `tests/test_pseudo_legal_move_delta_generator.py` | Create | Unit tests on handcrafted positions for pawns, sliders, captures, promotion, en-passant if enabled, own-piece blocking, and no king-safety filtering. |

Implementation warnings:

- Do not import Stockfish, python-chess engine analysis, or any verification metadata.
- If using `python-chess` only for rule-only move-generation tests, keep it out of the model forward path and do not ask it for engine scores, mate, or puzzle annotations. Prefer an internal deterministic generator for production.
- The generator must be differentiability-agnostic; gradients need not flow through discrete move enumeration.
- If sparsemax is not already available, implement a small masked sparsemax function with tests. Entmax is optional.
- If the model exceeds memory at batch size 512, chunk over `M` inside the move encoder and preserve exact masked pooling outputs.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0436_tuesday_los_angeles_move_delta_bottleneck.md
  generated_at: 2026-04-21T04:36:33-07:00
  weekday: tuesday
  timezone: los_angeles
  idea_slug: move_delta_bottleneck
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0436_move_delta_bottleneck
  name: Rule-Only Counterfactual Move-Delta Bottleneck
  slug: move_delta_bottleneck
  status: draft
  created_at: 2026-04-21T04:36:33-07:00
  author: ChatGPT Pro
  short_thesis: Puzzle-like boards have sparse, anisotropic one-ply pseudo-legal move-delta distributions that can be learned without engine scores or search.
  novelty_claim: Uses a deterministic current-board counterfactual intervention operator and sparse permutation-invariant bottleneck rather than static CNN, LC0-copy, square transformer, or attack-defense sheaf machinery.
  expected_advantage: Better near-puzzle detection and fewer material/source artifacts if the label depends on forcing move-consequence structure.
  central_falsification_ablation: Degree-preserving, moving-piece/captured-piece-histogram-preserving cross-position move-delta shuffle.
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112 and lc0_bt4_112 only with verified current-board channel maps and fail-closed adapters
  output_heads: binary_logits
  compute_notes: O(batch * max_moves * move_dim) memory; use max_moves 256 and chunking if needed; target under 750k parameters.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/counterfactual_delta_bottleneck_simple18.yaml
  model_path: src/chess_nn_playground/models/counterfactual_delta_bottleneck.py
  latest_result_path: null
  notes: Do not use full legal filtering, check/mate/stalemate oracles, engine metadata, source labels, or proposed labels as inputs; report central shuffle and count-only probes.
```

```yaml
config_yaml:
  run:
    name: counterfactual_delta_bottleneck_simple18
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
    name: counterfactual_delta_bottleneck
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
  model_name: counterfactual_delta_bottleneck
  file_path: src/chess_nn_playground/models/counterfactual_delta_bottleneck.py
  builder_function: build_counterfactual_delta_bottleneck
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18BoardAdapter
    - SideToMoveCanonicalizer
    - PseudoLegalMoveDeltaGenerator
    - BoardContextEncoder
    - MoveDeltaTupleEncoder
    - MaskedSparsemax
    - MoveConeBottleneck
    - CounterfactualDeltaClassifierHead
  required_config_fields:
    - model.name
    - model.input_channels
    - model.num_classes
    - model.hidden_channels
    - model.board_dim
    - model.move_dim
    - model.max_moves
    - model.sparse_attention
    - model.attention_temperature
    - model.canonicalize_side_to_move
    - model.include_pseudo_castling
    - model.include_en_passant
    - model.fail_closed_unknown_channels
  expected_parameter_count: 320k-550k for simple_18 default; keep below 750k in first experiment
  expected_memory_notes: Main tensor is [batch, max_moves, move_dim]; with 512x256x128 fp32 it is about 67 MB before MLP activations; implement move chunking if needed.
```

```yaml
research_continuity:
  idea_fingerprint: current-board occupancy and state -> pseudo-legal one-ply move-delta multiset -> sparse permutation-invariant consequence bottleneck -> binary puzzle-likeness; no engine/search/metadata inputs
  already_researched_family_overlap: Avoids imported static attack-defense graph, tactical sheaf, Hodge/Laplacian, curvature, and tension-energy families; overlap is only generic deterministic chess geometry.
  closest_duplicate_risk: Attention-based multiple-instance learning or Deep Sets over generated moves; the degree-preserving semantic shuffle is required to show this is not generic set attention.
  do_not_repeat_if_this_fails:
    - one-ply pseudo-legal move-delta bag classifiers
    - sparse attention over generated side-to-move interventions
    - mobility-normalized move-cone anisotropy statistics
    - deterministic from/to/piece delta tuple pooling as the central operator
    - count-probe-masked variants that still rely on pseudo-legal move cardinality
  suggested_next_search_directions:
    - label-safe uncertainty or abstention model that separates fine label 1 from fine label 2 without fabricating labels
    - causal invariance across encoding families, phase buckets, material buckets, or source-shift proxies if available
    - non-sheaf optimal transport between material-pressure distributions with semantics-destroying transport ablations
    - compression or information-bottleneck models that adversarially suppress material/source artifacts while preserving fine-label 1 recall
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Rule-Only Counterfactual Move-Delta Bottleneck` to the imported researched packets after implementation, including whether central shuffle passed or failed. | Prevents the next research pass from renaming the same one-ply move-delta bag idea. | `Imported Research Memory` |
| Add an anti-duplicate rule: do not propose another one-ply pseudo-legal move-delta set-pooling model unless it changes the falsifiable operator beyond more move types, larger hidden size, different sparse attention, or extra pooling statistics. | Keeps future ideas from drifting into shallow variants of this packet. | `Research Continuity` or `What Counts As Creative Enough` |
| Add a reminder that pseudo-legal move count is a possible nuisance and any rule-generated move-set model must include count-only and degree-preserving semantic-randomization ablations. | This is a reusable leakage lesson even if the current idea succeeds. | `Non-Negotiable Constraints` or `Required Markdown File Content / Ablation Plan` |
| Record whether `simple_18` channel order and LC0 current-board channel maps were verified during implementation. | Future packets need accurate adapter assumptions and should not repeatedly hedge about channel semantics. | `Project Context You Must Respect / Current available encodings` |
| If memory is problematic, add guidance that generated-move models must estimate `O(batch * max_moves * move_dim)` memory and provide a chunking plan. | Prevents infeasible future designs that fit mathematically but not in the existing trainer. | `Required Markdown File Content / Architecture Specification` |
| Preserve the existing leakage bans and explicitly keep engine scores, PVs, mate oracles, verification metadata, source labels, proposed labels, and unresolved candidate pools out of neural inputs. | The leakage rules are correct and should not be weakened. | `Non-Negotiable Constraints` |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0436_tuesday_los_angeles_move_delta_bottleneck.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes; this avoids static attack-defense sheaf, Hodge/Laplacian, curvature, and tension-energy variants
