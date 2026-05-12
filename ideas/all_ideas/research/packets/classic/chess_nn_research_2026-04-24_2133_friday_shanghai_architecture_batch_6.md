# Codex Research Batch: Additional Architecture Candidates 6

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`
- Generated at: 2026-04-24 21:33
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: draft architecture batch, not implemented

## Purpose

This batch adds six more chess neural architecture ideas. The search space is now crowded, so the emphasis is on sharply different computation objects:

- continuous state-space scans over chess lines
- pawn-skeleton barrier geometry
- square-color parity algebra
- run-length segment encodings
- local king-shelter microkernels
- low-rank material-phase adapters

These are not implementation commits. They are research candidates to promote into full packets or code later.

## Shared Data Contract

All candidates target the binary puzzle-likeness task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` remain diagnostics only. First implementations should use `simple_18`, the existing `crtk_sample_3class` splits, and the shared trainer.

Forbidden model inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, or future game outcomes.

Allowed model inputs:

- Current board occupancy, side-to-move, castling/en-passant planes, deterministic square coordinates, side-relative coordinates, material/count summaries, and deterministic transforms of current board tensors.

## Ranked Shortlist

| Rank | Candidate | Main object | Why expand it |
|---|---|---|---|
| 1 | Ray State-Space Scan Network | Continuous recurrent scans along ranks/files/diagonals | Efficient long-range line modeling without attention, automata, or linear solves. |
| 2 | Pawn Skeleton Barrier Network | Pawn-structure distance and barrier fields | Chess-specific abstraction for shelter, passed lanes, and blocked structures. |
| 3 | Square-Color Parity Mixer | Dark/light square subspace algebra | Uses a fundamental chess invariant ignored by most CNN baselines. |
| 4 | Occupancy Run-Length Segment Encoder | Empty/occupied segment statistics along lines | Compact board-line abstraction distinct from ray token automata. |
| 5 | King-Shelter Microkernel Network | Side-relative local king-zone filters and shelter residuals | Practical specialized module for many puzzle motifs. |
| 6 | Material-Phase Low-Rank Adapter Network | Low-rank adapters conditioned on material phase | Practical heterogeneity model with tight shortcut controls. |

Best next full packet from this batch: `Ray State-Space Scan Network`.

## Candidate 1: Ray State-Space Scan Network

### Thesis

Chess line motifs often require long-range context, but all-square attention and dynamic attack graphs are not the only way to get it. A state-space scan can process every rank, file, diagonal, and anti-diagonal as a short sequence with shared continuous recurrence parameters, giving efficient line memory without finite automata or dense square interactions.

### Fingerprint

```text
simple_18 board tensor
+ line extraction
+ learned state-space scan along each line
+ line-state pooling
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not the ray-language automaton: no discrete token grammar, no weighted finite automaton states, no string acceptance score.
- Not Schur-Ray: no line-incidence linear solve or Woodbury system.
- Not Bitboard Shift-Algebra: no fixed shift-polynomial bank.
- Not attention: no query-key matrix.

### Architecture Sketch

1. Project board tensor to square embeddings:

```text
x: (B, 18, 8, 8)
e: (B, 64, D)
```

2. Extract ordered line sequences:

```text
8 ranks
8 files
15 diagonals
15 anti-diagonals
```

3. For each line, run a small learned state-space scan:

```text
h_t = A_l h_{t-1} + B_l e_t
y_t = C_l h_t + D_l e_t
```

where `l` is a line-type bucket, not a unique line id.

4. Run both directions and concatenate:

```text
forward scan
backward scan
```

5. Pool line outputs:

```text
mean, max, endpoint states, topk response, king-line response
```

6. Fuse with a small CNN stem and classify.

### Efficient Variant

Because board lines have length at most 8, the recurrence can be simple. The point is not huge-sequence scaling; the point is using line memory with fewer parameters than attention.

Start with:

```text
D = 48
state_dim = 32
line_types = rank, file, diagonal, anti_diagonal
bidirectional = true
```

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `cnn_only` | Remove line scan branch | Tests line scan value | Should drop if long line context matters. |
| `bag_of_line_tokens` | Pool line tokens without recurrence | Tests ordered state | Recurrence should help pins/batteries. |
| `random_line_order` | Randomly permute order inside each line | Tests line order semantics | Should degrade if scanning matters. |
| `rank_file_only` | Remove diagonal scans | Tests diagonal motifs | Should hurt bishop/queen diagonal puzzles. |
| `attention_matched` | Replace scan with tiny line attention | Tests scan efficiency | Scan should be competitive with lower cost. |

### Diagnostics

- Endpoint hidden-state norms by label.
- Which line type contributes most.
- Random-order ablation gap.
- Diagonal-only and rank/file-only confusion shifts.
- Examples where scan branch fixes CNN false negatives.

### Failure Modes

- The scan can collapse to local smoothing.
- Line order may add little because lines are length 8.
- It may duplicate what a good dilated CNN already learns.

### Implementation Notes

This is the strongest practical candidate in the batch because it is easy to code and benchmark. Treat it as a lightweight line-memory baseline before larger attention or line-solve models.

## Candidate 2: Pawn Skeleton Barrier Network

### Thesis

Pawn structure is a slow, chess-specific skeleton that shapes king safety, open lines, promotion lanes, and tactical vulnerability. A model can compute deterministic pawn barrier and distance fields from the current board, then learn how these fields condition puzzle-likeness.

### Fingerprint

```text
current board tensor
+ pawn skeleton planes
+ fixed distance/barrier transforms
+ small conditioned CNN
+ binary head
```

### Core Fields

From current pawn planes only:

```text
own_pawns
opponent_pawns
side_relative_pawn_fronts
file_pawn_counts
isolated_pawn_candidate_planes
doubled_pawn_candidate_planes
passed_lane_candidate_planes
king_pawn_shelter_planes
```

These are not engine evaluations. They are deterministic current-board geometry.

Add fixed distance fields:

```text
distance_to_own_pawn
distance_to_opponent_pawn
distance_to_open_file
distance_to_pawn_frontier
king_to_pawn_shelter_distance
```

### Architecture Sketch

1. Build pawn skeleton feature stack `(B, Ps, 8, 8)`.
2. Project board tensor and pawn stack separately.
3. Use pawn stack to gate board channels:

```text
g = sigmoid(Conv1x1(pawn_features))
board_conditioned = board_features * (1 + g)
```

4. Run a small CNN/residual stack.
5. Pool king-zone, open-file, and global features.
6. Classify.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `no_pawn_stack` | Remove pawn skeleton fields | Tests core value | Should drop if pawn barriers matter. |
| `raw_pawns_only` | Use only raw pawn planes, no distances | Tests transform value | Distance fields should help. |
| `shuffled_pawn_files` | Preserve pawn counts but shuffle files | Tests file geometry | Should degrade if skeleton geometry matters. |
| `no_king_shelter_pool` | Remove king-zone pawn summaries | Tests king safety role | Should affect king-attack puzzles. |
| `material_bucket_eval` | Report inside material buckets | Tests material shortcut | Gains should survive buckets. |

### Diagnostics

- Pawn-stack feature importance.
- Metrics by pawn count bucket.
- King-shelter residual distributions.
- False positives with unusual pawn structures.
- Shuffled-file ablation gap.

### Failure Modes

- It may only learn phase/material information.
- Many tactics are piece-based and pawn skeleton adds little.
- Passed-pawn heuristics can become too close to human evaluation terms if overbuilt.

### Implementation Notes

Keep the first version conservative. Use simple deterministic pawn geometry and avoid writing a full evaluator.

## Candidate 3: Square-Color Parity Mixer

### Thesis

The chessboard is naturally bipartite by square color. Bishops stay on one color, knights alternate color, kings and queens mix colors locally, and pawn captures switch files and square color. A neural model can explicitly split dark/light square subspaces and learn within-parity and cross-parity mixing.

### Fingerprint

```text
board tensor
+ dark/light square partition
+ parity-preserving and parity-flipping mixers
+ square-color interaction diagnostics
+ binary head
```

### Architecture Sketch

1. Project board tensor to square embeddings.
2. Split square tokens:

```text
dark_tokens
light_tokens
```

3. Apply two mixer families:

```text
within_color_mixer: dark->dark, light->light
cross_color_mixer: dark<->light
```

4. Gate mixers by piece type:

```text
bishop-like channels prefer within-color
knight-like channels prefer cross-color
king/queen/pawn channels learn mixed gates
```

5. Reassemble board features and classify.

### Linear Algebra View

The square interaction matrix has block form:

```text
[ A_dark      C_cross ]
[ C_cross^T   A_light ]
```

The model constrains and summarizes these blocks instead of learning an arbitrary square interaction matrix.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `ordinary_token_mixer` | Remove parity block structure | Tests parity inductive bias | Parity should help if color geometry matters. |
| `within_only` | Remove cross-color mixer | Tests knight/pawn/king color transitions | Should hurt relevant motifs. |
| `cross_only` | Remove same-color mixer | Tests bishop/color-complex signal | Should hurt bishop motifs. |
| `random_bipartition` | Replace square colors with random 32/32 split | Tests true chess parity | Should degrade if square color matters. |
| `no_piece_gates` | Remove piece-conditioned mixer gates | Tests chess-aware routing | Should reduce interpretability and maybe accuracy. |

### Diagnostics

- Within/cross gate usage by piece type.
- Performance split by bishop material presence.
- Random-bipartition ablation gap.
- Square-color block norm ratios by label.

### Failure Modes

- Square color may be too weak for puzzle-likeness.
- A CNN may already encode parity through coordinates.
- Piece gates may become material shortcuts.

### Implementation Notes

This is small and cheap. It is a good candidate for a module inside a larger token/CNN hybrid.

## Candidate 4: Occupancy Run-Length Segment Encoder

### Thesis

Sliding tactics depend on contiguous empty and occupied segments along ranks, files, and diagonals. Instead of parsing full piece-token ray strings, encode run-length segment summaries: empty run lengths, blocker positions, endpoint piece types, and segment openness.

### Fingerprint

```text
rank/file/diagonal line extraction
+ deterministic run-length segment features
+ compact segment MLP
+ line and board pooling
+ binary head
```

### Why It Is Distinct

- Not ray-language automata: no ordered token grammar or learned automaton state transitions.
- Not Schur-Ray: no linear solve.
- Not Bitboard Shift-Algebra: no shift operator polynomial.
- More compressed than line scans: it summarizes structural segments, not every square token.

### Segment Features

For each line:

```text
empty_run_length
occupied_count
first_occupied_type
last_occupied_type
gap_between_king_and_slider_candidate
segment_touches_king_zone
segment_open_to_edge
side-relative direction bucket
```

Keep candidate names diagnostic only; they are computed from current occupancy and piece identity.

### Architecture Sketch

1. Extract all rank/file/diagonal lines.
2. Compute up to `Smax` segment feature rows per line.
3. Embed each segment with a shared MLP.
4. Pool by line and line type.
5. Fuse with small board CNN summary.
6. Classify.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `histogram_only` | Use only global run-length histograms | Tests line-local segment semantics | Full segment model should help. |
| `no_endpoint_types` | Remove piece types at segment ends | Tests piece identity | Should hurt pins/skewers/batteries. |
| `random_line_assignment` | Shuffle segments across line types | Tests chess line placement | Should degrade if placement matters. |
| `cnn_only` | Remove segment branch | Tests branch value | If equal, segment encoding unnecessary. |
| `run_lengths_only` | Remove king-zone and endpoint flags | Tests tactical conditioning | Full model should improve. |

### Diagnostics

- Segment types most associated with positives.
- Run-length distributions by label.
- Line-type contribution by label.
- False positives from open but non-tactical lines.

### Failure Modes

- Segment features may become handcrafted heuristics.
- It may overlap with ray-language ideas if made too token-order heavy.
- Compression may discard too much square-level context.

### Implementation Notes

Use it as a compact branch beside a CNN, not as the whole model in the first test.

## Candidate 5: King-Shelter Microkernel Network

### Thesis

Many puzzles are decided near the king. A specialized side-relative microkernel branch can examine king neighborhoods, escape rings, near sliders, and local blockers with high resolution while the main CNN handles global context.

### Fingerprint

```text
board tensor
+ crop around both kings
+ side-relative microkernels
+ shelter/escape residual features
+ global CNN fusion
```

### Architecture Sketch

1. Locate both kings from current board planes.
2. Extract fixed padded crops around each king:

```text
5x5 and 7x7 windows
```

3. Canonicalize each crop side-relative:

```text
own king view
opponent king view
```

4. Apply small microkernel CNNs with asymmetric filters:

```text
front shield filters
side escape filters
diagonal entry filters
rank backdoor filters
```

5. Compute residual between own/opponent king-zone representations.
6. Fuse with global encoder.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `global_only` | Remove king microkernel branch | Tests king-zone value | Should drop for king-attack puzzles. |
| `random_crop_center` | Crop around random occupied piece instead of king | Tests king specificity | Should degrade if king shelter matters. |
| `no_side_relative` | Use raw board orientation | Tests side-relative geometry | Should reduce generalization. |
| `single_crop_size` | Use only 5x5 or only 7x7 | Tests multi-radius context | Both may be useful. |
| `no_opponent_king_view` | Use only side-to-move king | Tests asymmetric king pressure | Full pair should help. |

### Diagnostics

- King-crop branch logit contribution.
- Matched-FPR fine-label `2` recall.
- Error cases with non-king tactical motifs.
- Crop activation maps.

### Failure Modes

- Too narrow for puzzles based on promotion, material traps, or zugzwang.
- Could overfit to check-like positions if dataset has bias.
- Requires careful king-plane extraction from `simple_18`.

### Implementation Notes

This is the most practical low-risk branch in the batch. It can be added to `Piece-Token CNN Hybrid` without disturbing the rest of the model.

## Candidate 6: Material-Phase Low-Rank Adapter Network

### Thesis

Chess positions vary greatly by material phase. Instead of one encoder for every position, condition low-rank adapter weights on material summaries while keeping a shared backbone. The architecture tests whether small material-conditioned rank updates improve without turning material into a shortcut.

### Fingerprint

```text
shared board encoder
+ material/phase summary
+ low-rank adapter generator
+ adapted hidden layers
+ shortcut-controlled binary head
```

### Architecture Sketch

1. Run a shared CNN or token mixer backbone.
2. Compute allowed material/phase summary:

```text
piece counts by side/type
side-to-move
castling/en-passant availability
```

3. Generate low-rank adapters for selected layers:

```text
Delta W = U(summary) V(summary)^T
```

or use LoRA-style:

```text
h = h + scale * B(summary) A(summary) h
```

4. Keep adapter rank tiny:

```text
rank = 2 or 4
```

5. Classify with shared head.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `shared_backbone_only` | Remove adapters | Tests conditional adaptation | Adapters should improve if phase heterogeneity matters. |
| `material_head_only` | Feed material summary only to classifier, no adapters | Tests whether adaptation is more than shortcut feature | Adapter should beat material-head-only. |
| `random_material_summary` | Shuffle summaries across batch within material-count buckets | Tests summary semantics | Should degrade if conditioning matters. |
| `rank0_adapter` | Zero adapter updates | Sanity baseline | Equals shared backbone. |
| `high_rank_adapter` | Increase rank to 16 | Tests capacity versus overfit | If only high rank works, shortcut risk rises. |

### Diagnostics

- Adapter norm by material phase.
- Metrics inside material buckets.
- Material-only probe accuracy.
- Whether adapter gains survive material-bucket evaluation.
- Rank sensitivity curve.

### Failure Modes

- The model may mostly learn material shortcuts.
- Generated adapters may destabilize training.
- If the backbone is already strong, adapters add little.

### Implementation Notes

This is not the most novel mathematically, but it is useful as a practical architecture control for heterogeneous chess data. Keep shortcut diagnostics mandatory.

## Recommended Promotion Order

1. `Ray State-Space Scan Network`
2. `King-Shelter Microkernel Network`
3. `Square-Color Parity Mixer`
4. `Pawn Skeleton Barrier Network`
5. `Occupancy Run-Length Segment Encoder`
6. `Material-Phase Low-Rank Adapter Network`

## Minimal Benchmark Plan

Use the same first-pass benchmark shape for all six:

```text
dataset: crtk_sample_3class
input: simple_18
target: binary puzzle-like
seeds: 3
metrics: accuracy, AUROC, PR-AUC, Brier, ECE
diagnostics: fine-label confusion, material-bucket metrics when relevant
```

Do not use source or provenance fields as model inputs.

## Duplicate Guardrails For Future Ideation

| Candidate | Do not repeat as |
|---|---|
| Ray State-Space Scan | Another line scanner that only swaps recurrence equations without changing random-order and bag-of-line-token falsifiers. |
| Pawn Skeleton Barrier | Another pawn-structure branch with only more hand-coded pawn heuristics. |
| Square-Color Parity Mixer | Another dark/light split model with only different token widths or block counts. |
| Occupancy Run-Length Segment | Another ray segment model that drifts into weighted finite automata or full ray-token grammars. |
| King-Shelter Microkernel | Another king-crop branch with only different crop sizes. |
| Material-Phase Low-Rank Adapter | Another material-conditioned adapter with only different adapter rank. |

## Best Full-Packet Candidate

`Ray State-Space Scan Network` is the best next full packet because it is:

- cheap to implement
- directly chess-related
- easy to falsify with line-order and bag-of-line-token controls
- more practical than another high-math bottleneck
- distinct from ray automata, line solves, shift algebra, attention, and attack graphs

