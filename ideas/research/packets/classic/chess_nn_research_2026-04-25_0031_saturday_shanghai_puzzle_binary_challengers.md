# Codex Research Packet: Puzzle-Binary Benchmark Challengers

## File Metadata

- Filename: `chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`
- Generated at: 2026-04-25 00:31
- Weekday: Saturday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: targeted architecture batch, not implemented

## Purpose

This packet adds new ideas after the corrected benchmark was established:

```text
model output: one puzzle logit
target 0: random/non-puzzle + near-puzzle
target 1: verified puzzle
diagnostic: 3x2 source-class matrix
```

The current best baseline is:

```text
LC0 BT4 tower
test F1: 0.7445
test PR AUC: 0.8068
near-puzzle -> puzzle false positive rate: 24.8%
```

The most useful new ideas should not merely raise ordinary binary accuracy. They should reduce near-puzzle false positives while keeping puzzle recall high.

## Shared Contract

First implementations should use:

```text
mode: puzzle_binary
loss: BCEWithLogitsLoss
primary ranking metric: test F1 and PR AUC
key diagnostic: near-puzzle -> puzzle false-positive rate
```

Allowed inputs:

- Current board tensor.
- Deterministic coordinates.
- Current-board occupancy, side-to-move, castling, en-passant.
- Training-only use of labels, source fine labels, and grouping metadata for auxiliary losses.

Forbidden inference inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, candidate status, or dataset provenance.
- Engine search outputs or future game outcomes.

## New Candidate Ranking

| Rank | Idea | Why it may beat BT4 | Risk |
|---:|---|---|---|
| 1 | Negative-Class Disentangled Puzzle Head | Directly models random-negative and near-puzzle-negative evidence separately. | Could overfit source artifacts. |
| 2 | Line-Piece Crossbar Network | Gives line geometry and occupied-piece structure without a heavy solve. | May duplicate Schur-Ray if too similar. |
| 3 | Near-Puzzle Margin Twin Network | Uses hard-negative pair/ranking pressure, exactly matching the benchmark failure mode. | Needs reliable pair/group metadata. |
| 4 | Stripe-Selective Mixer CNN | Practical line-aware CNN, cheaper and easier than Schur-Ray. | May only be a stronger CNN, not novel enough. |
| 5 | King-Zone Evidence Ledger | Focuses capacity on king danger, escape, blockers, and defender tension. | Could overfit checking/king-attack motifs. |
| 6 | Prototype-Margin Puzzle Network | Makes puzzle evidence compete against random and near-puzzle prototypes. | Prototype collapse unless ablated carefully. |
| 7 | Source-Rate Calibrated Objective | Training objective explicitly penalizes near-puzzle false positives. | More objective than architecture. |

Best immediate implementation:

```text
Negative-Class Disentangled Puzzle Head
```

Best architecture-only challenger:

```text
Line-Piece Crossbar Network
```

Best if group/pair metadata is reliable:

```text
Near-Puzzle Margin Twin Network
```

## Idea 1: Negative-Class Disentangled Puzzle Head

### Thesis

The target is binary, but the negative class has two very different sources:

```text
random non-puzzle
near-puzzle hard negative
```

A single negative representation encourages the model to learn an average negative concept. Instead, learn separate evidence channels:

```text
random-negative evidence
near-puzzle-negative evidence
puzzle-positive evidence
```

and expose only one final puzzle logit for inference.

### Architecture

Use any trunk:

```text
trunk = CNN, NNUE, BT4, or Piece-Token Hybrid
h = trunk_features(board)
```

Heads:

```text
e_random = head_random(h)
e_near   = head_near(h)
e_puzzle = head_puzzle(h)
```

Single inference logit:

```text
puzzle_logit = e_puzzle - logsumexp([e_random, e_near])
```

Training losses:

```text
main BCE on puzzle_logit:
  fine 0 -> 0
  fine 1 -> 0
  fine 2 -> 1

auxiliary 3-way CE on [e_random, e_near, e_puzzle]
  fine 0 -> random
  fine 1 -> near
  fine 2 -> puzzle
```

At inference, only `puzzle_logit` is used.

### Why It Could Beat BT4

The current BT4 tower is strong, but it still calls about a quarter of near-puzzles puzzles. This head directly gives the model a "near-puzzle but not puzzle" rejection channel.

### First Config

```yaml
model:
  name: disentangled_puzzle_head
  base: lc0_bt4_classifier
  input_channels: 112
  num_classes: 1
  evidence_dim: 128
  aux_weight: 0.25
```

Also test the same head on `simple_18` with a CNN or Piece-Token trunk.

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_aux_3way` | Tests whether the disentanglement is doing work. |
| `random_near_merged` | Collapses both negative heads into one; should lose near-puzzle discrimination. |
| `aux_only_no_logsumexp` | Tests whether the final evidence competition matters. |
| `shuffle_fine_negative_labels` | Ensures improvements are not just extra supervision capacity. |

### Success Criterion

Beat current BT4 on:

```text
test F1 > 0.7445
near-puzzle FP < 24.8%
```

without reducing puzzle recall below roughly `0.78`.

## Idea 2: Line-Piece Crossbar Network

### Thesis

Schur-Ray is mathematically powerful but more complex. A simpler line-aware architecture can create line tokens and piece tokens, then pass messages only through deterministic piece-line incidence.

This gives the network direct access to pins, skewers, batteries, and blockers without dense square attention.

### Representation

Extract:

```text
piece_tokens: up to 32 occupied pieces
line_tokens: 46 full lines = ranks + files + diagonals + anti-diagonals
optional segment_tokens: blocker-split line segments
```

Build incidence:

```text
I_piece_line[p, l] = 1 if piece p lies on line l
```

Message passing:

```text
line_l = pool_p I[p,l] * piece_proj(piece_p)
piece_p = piece_p + pool_l I[p,l] * line_proj(line_l)
```

Repeat `2-4` layers, then pool:

```text
global_piece_pool
global_line_pool
own/opponent line contrast
king-line summaries
```

Final output is one puzzle logit.

### Why It Could Beat BT4

BT4 can learn line interactions only indirectly through convolution depth. Crossbar line incidence gives global rank/file/diagonal interactions immediately and cheaply.

### First Config

```yaml
model:
  name: line_piece_crossbar
  input_channels: 18
  num_classes: 1
  piece_dim: 64
  line_dim: 64
  layers: 3
  include_segment_tokens: false
  include_king_line_pool: true
  dropout: 0.1
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_line_tokens` | Piece-token-only control. |
| `random_line_incidence` | Tests real line geometry. |
| `rank_file_only` | Tests whether diagonals matter. |
| `no_king_line_pool` | Tests king-specific line context. |
| `segment_tokens_on` | Tests blocker-sensitive upgrade. |

### Notes

This should be implemented before full Segment-Schur if we want a faster line-geometry smoke test.

## Idea 3: Near-Puzzle Margin Twin Network

### Thesis

The benchmark is fundamentally about ranking:

```text
real puzzle should score above near-puzzle
near-puzzle should score above or near random in some latent dimensions,
but below puzzle on final puzzle evidence
```

If `sister_group_id` or `split_group_id` connects similar positions, train with pairwise margins in addition to BCE.

### Architecture

Use a shared encoder:

```text
z = encoder(board)
logit = head(z)
```

Batch-level pair loss:

```text
for pairs (puzzle, near):
  loss_margin = relu(margin - logit_puzzle + logit_near)

for pairs (near, random):
  optional weak ordering or contrastive separation
```

Representation contrast:

```text
near-puzzle and puzzle may be close in ordinary-board latent
but separated in puzzle-evidence latent
```

This can use two latent projections:

```text
z_ordinary
z_tactical
```

### Why It Could Beat BT4

The benchmark's weak spot is near-puzzle false positives. Pairwise ranking forces the exact distinction we care about.

### Required Data Check

Before implementation, audit:

```text
sister_group_id
source_group_id
split_group_id
```

Confirm whether near-puzzles and puzzles are paired or grouped meaningfully. If not, fall back to in-batch supervised contrastive loss by fine label.

### Required Ablations

| Ablation | Purpose |
|---|---|
| `bce_only` | Baseline without pairwise loss. |
| `random_pairs` | Tests whether real pairing matters. |
| `label_only_contrast` | Tests group metadata vs generic fine-label contrast. |
| `ordinary_latent_only` | Tests tactical split. |

## Idea 4: Stripe-Selective Mixer CNN

### Thesis

A practical line-aware CNN may be enough to beat the current BT4 while staying simpler than Schur-Ray. Instead of ordinary `3x3` convolutions only, mix along chess stripes:

```text
ranks
files
diagonals
anti-diagonals
king-centered rays
```

### Architecture

Layer:

```text
x_local = Conv3x3(x)
x_rank  = rank_scan_mlp(x)
x_file  = file_scan_mlp(x)
x_diag  = diagonal_scan_mlp(x)
x_anti  = anti_diagonal_scan_mlp(x)
gate    = sigmoid(global_pool(x))
x_next  = x + Conv1x1([x_local, gate*x_rank, gate*x_file, gate*x_diag, gate*x_anti])
```

The scan can be implemented with simple sequence convs over each line, not recurrent machinery.

### Why It Could Beat BT4

It keeps the strengths of a CNN but adds exact long-range line paths. Near-puzzles often differ from puzzles by one blocker, one diagonal, one file, or one escape line.

### Required Ablations

| Ablation | Purpose |
|---|---|
| `local_only` | Ordinary CNN control. |
| `rank_file_only` | Tests rook-line contribution. |
| `diag_only` | Tests bishop/queen diagonal contribution. |
| `random_stripes` | Tests true stripe geometry. |
| `no_global_gate` | Tests context-dependent line selection. |

## Idea 5: King-Zone Evidence Ledger

### Thesis

Many real puzzles are ultimately about king safety or forcing geometry. A model can maintain a small set of learned evidence ledger slots around each king:

```text
attacker pressure
defender resources
escape space
blocker/pin pressure
tempo/initiative evidence
```

These are learned slots, not hand-coded tactical rules.

### Architecture

Inputs:

```text
CNN board map
piece tokens
king-centered coordinate features
```

Ledger slots:

```text
own_king_slots:  K x D
opp_king_slots:  K x D
global_slots:    K x D
```

Update:

```text
slot = slot + gated_pool(board_features, piece_features, king_relative_features)
```

Readout:

```text
puzzle_logit = MLP([
  own_king_ledger,
  opp_king_ledger,
  ledger_difference,
  ledger_product,
  global_board_pool
])
```

### Why It Could Beat BT4

BT4 has no explicit ledger of why a position is puzzle-like. The ledger creates a bottleneck that may learn compact king/tactic evidence and reduce spurious near-puzzle positives.

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_king_relative` | Tests king anchoring. |
| `random_king_anchor` | Tests real king semantics. |
| `global_slots_only` | Tests king-specific ledger value. |
| `slot_count_sweep` | Tests whether ledger is a bottleneck or just capacity. |

## Idea 6: Prototype-Margin Puzzle Network

### Thesis

The model should not merely say "puzzle-like." It should compare the board to separate learned prototypes:

```text
ordinary random positions
near-puzzle hard negatives
real puzzles
```

The final puzzle logit is a margin:

```text
puzzle_logit = sim(z, puzzle_proto) - max(sim(z, random_proto), sim(z, near_proto))
```

### Architecture

Use a trunk encoder and learn prototype banks:

```text
P_random: K x D
P_near:   K x D
P_puzzle: K x D
```

Similarity:

```text
sim_class = logsumexp_k cosine(z, P_class[k]) / temperature
```

Final:

```text
logit = sim_puzzle - logsumexp([sim_random, sim_near])
```

### Why It Could Beat BT4

Near-puzzles may live close to puzzles in raw representation. A prototype margin gives near-puzzles their own attractors rather than treating them as generic negatives.

### Required Ablations

| Ablation | Purpose |
|---|---|
| `single_negative_proto` | Tests separate near prototype. |
| `no_margin_logsumexp` | Tests prototype competition. |
| `random_proto_freeze` | Tests learned prototypes. |
| `prototype_count_sweep` | Detects collapse or overcapacity. |

## Idea 7: Source-Rate Calibrated Objective

### Thesis

The current benchmark's central question is not only "is F1 high?" It is:

```text
At useful puzzle recall, how many near-puzzles are falsely called puzzles?
```

Add a differentiable objective that penalizes near-puzzle false-positive rate at a target puzzle recall.

### Objective

Let:

```text
p = sigmoid(logit)
```

Soft rates:

```text
near_fp_soft = mean_{fine=1} sigmoid((p - tau) / temp)
puzzle_recall_soft = mean_{fine=2} sigmoid((p - tau) / temp)
```

Penalty:

```text
loss_rate =
  lambda_fp * relu(near_fp_soft - target_near_fp)^2
  + lambda_recall * relu(target_recall - puzzle_recall_soft)^2
```

Main loss:

```text
loss = BCEWithLogits + loss_rate
```

### Why It Could Beat BT4

It optimizes the failure mode directly. This may improve any architecture, especially BT4, NNUE, and the future token/line models.

### Required Ablations

| Ablation | Purpose |
|---|---|
| `bce_only` | Current baseline. |
| `random_label_rate_groups` | Ensures fine-source rates are meaningful. |
| `target_sweep` | Finds recall/FPR tradeoff. |
| `posthoc_threshold_only` | Tests whether training objective beats threshold tuning. |

## Recommended Implementation Sequence

1. Implement `Negative-Class Disentangled Puzzle Head` on top of the existing BT4 and NNUE trunks.
2. Implement `Line-Piece Crossbar Network` on `simple_18`.
3. Audit grouping metadata, then implement `Near-Puzzle Margin Twin Network` if groups are meaningful.
4. Add `Source-Rate Calibrated Objective` as a trainer option and test it across BT4, NNUE, Crossbar, and Piece-Token Hybrid.
5. Only then attempt heavier Segment-Schur and Semijoin-First relational models.

## Promotion Criteria

Promote an idea to an implemented `ideas/{idea_id}_{slug}` folder if it achieves at least one:

```text
test F1 > 0.755
test PR AUC > 0.820
near-puzzle false-positive rate < 20% with puzzle recall >= 78%
```

and has one central ablation proving that the new mechanism matters.

## Machine-Readable Summary

```yaml
packet:
  title: Puzzle-Binary Benchmark Challengers
  generated_at: 2026-04-25 00:31
  timezone: Asia/Shanghai
  status: not_implemented
  target_task: puzzle_binary
  current_best_baseline: LC0 BT4 tower
  current_best_test_f1: 0.7445
  current_best_near_puzzle_fp: 0.2477
ideas:
  - slug: negative_class_disentangled_puzzle_head
    priority: 1
    implementation_risk: low
    expected_impact: high
  - slug: line_piece_crossbar
    priority: 2
    implementation_risk: medium
    expected_impact: high
  - slug: near_puzzle_margin_twin
    priority: 3
    implementation_risk: medium
    expected_impact: high_if_group_metadata_is_reliable
  - slug: stripe_selective_mixer_cnn
    priority: 4
    implementation_risk: low
    expected_impact: medium
  - slug: king_zone_evidence_ledger
    priority: 5
    implementation_risk: medium
    expected_impact: medium_high
  - slug: prototype_margin_puzzle_network
    priority: 6
    implementation_risk: medium
    expected_impact: medium
  - slug: source_rate_calibrated_objective
    priority: 7
    implementation_risk: low
    expected_impact: medium_high
```
