# Codex Research Packet: Puzzle Architecture Batch 2

## File Metadata

- Filename: `chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`
- Generated at: 2026-04-25 00:37
- Weekday: Saturday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: targeted architecture batch, not implemented

## Purpose

This packet adds more architectures for the corrected puzzle-binary benchmark:

```text
input: chess position
output: one puzzle logit
target 0: random/non-puzzle + near-puzzle
target 1: verified puzzle
diagnostic: 3x2 source-class-to-binary-prediction matrix
```

The main benchmark pressure is not ordinary random-position rejection. It is rejecting near-puzzles while still detecting true puzzles.

Current strongest baseline to beat:

```text
LC0 BT4 tower
test F1: 0.7445
test PR AUC: 0.8068
near-puzzle -> puzzle false positive rate: 24.8%
puzzle recall: 79.4%
```

This batch prioritizes ideas that could reduce the near-puzzle false-positive row without collapsing puzzle recall.

## Design Principle

A true puzzle is often not just a sharp position. It usually has a missing extra property:

```text
forcing pressure + target vulnerability + insufficient defense + tempo alignment
```

Near-puzzles may have the first one or two pieces of that conjunction but not the complete chain. These architectures try to represent that missing chain directly.

## New Candidate Ranking

| Rank | Idea | Why it targets near-puzzles | First risk |
|---:|---|---|---|
| 1 | Forcing-Certificate Transformer | Forces the model to assemble a small tactical proof instead of scoring surface sharpness. | Certificate slots may collapse. |
| 2 | Defender-Exhaustion Cascade Network | Models whether defensive resources get over-committed. | Similar to Hall-defect unless made sequential. |
| 3 | Causal Piece-Derivative Network | Measures whether a few key pieces control the puzzle decision. | More expensive per batch. |
| 4 | Phase-Transition Pressure Network | Detects threshold-critical tactical pressure, not just high pressure. | Needs stable differentiable thresholding. |
| 5 | Disproof-Ledger Puzzle Network | Gives the model explicit ways to say "sharp but not a puzzle." | Close to negative-head ideas if too simple. |
| 6 | Motif Tensor Factorization Network | Uses multiplicative attacker-target-defender motifs with low-rank structure. | Could underperform a deep CNN if too rigid. |
| 7 | Tempo-Alignment Gate Network | Separates static danger from side-to-move forcing privilege. | May duplicate tempo-odd work unless tied to motifs. |
| 8 | Puzzle Boundary Twin Encoder | Learns the decision boundary between puzzle and near-puzzle as a margin surface. | Needs strong hard-negative sampling. |
| 9 | Critical-Square Budget Network | Forces the final logit through a tiny set of critical squares. | Can miss quiet or distributed tactics. |

Most promising first build:

```text
Forcing-Certificate Transformer
```

Most practical near-term build:

```text
Causal Piece-Derivative Network on top of an existing CNN or BT4 trunk
```

Most novel but still benchmark-relevant:

```text
Phase-Transition Pressure Network
```

## Idea 1: Forcing-Certificate Transformer

### Thesis

A real puzzle should admit a compact tactical certificate:

```text
attacker or forcing piece
target
defender or escape resource
blocker / pin / overload relation
tempo side
```

Near-puzzles often look tactical but fail because the certificate has a hole. This architecture classifies through a small set of learned certificate slots instead of only a global board embedding.

### Architecture

Build square and piece tokens:

```text
square_tokens: 64 tokens from CNN stem
piece_tokens: occupied-piece tokens with type, color, square, side-to-move features
relation_biases: fixed chess relations such as same line, knight reach, king zone, pawn attack, target value
```

Use `K` certificate slots:

```text
certificate_slots = learned queries, K = 4..8
```

Each slot attends to:

```text
attacker token
target token or square
defender token
line/blocker context
king-zone context
```

Slot update:

```text
slot_k <- cross_attention(slot_k, piece_tokens + square_tokens, relation_bias)
```

Readout:

```text
slot_score_k = MLP(slot_k)
puzzle_logit = logsumexp(slot_score_k) + global_residual_logit
```

Optional auxiliary losses:

- entropy penalty so not all slots attend to the same piece
- slot diversity penalty over attention maps
- weak near-puzzle penalty: near-puzzles should produce incomplete certificates, not simply low global confidence

### Why It Could Beat BT4

BT4 can learn tactical motifs, but its final evidence is diffuse. This model gives the final logit a small number of "because of these tactical objects" channels. That should help distinguish complete puzzles from sharp near-puzzles.

### First Config

```yaml
model:
  name: forcing_certificate_transformer
  input_channels: 18
  num_classes: 1
  cnn_channels: 64
  token_dim: 96
  slots: 6
  attention_layers: 3
  relation_biases: true
  diversity_weight: 0.02
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_relation_bias` | Tests whether chess relation priors matter. |
| `single_slot` | Tests whether multiple tactical certificates matter. |
| `no_global_residual` | Tests whether the slot bottleneck alone can classify. |
| `shuffled_near_labels` | Tests if hard-negative behavior is real. |

## Idea 2: Defender-Exhaustion Cascade Network

### Thesis

Many puzzles exist because one side cannot satisfy all defensive obligations at once:

```text
king escape
piece defense
mate threat
queen or rook attack
promotion stop
back-rank weakness
```

Near-puzzles may have threats, but the defense graph is still satisfiable. Instead of a one-shot matching statistic, model defense as a small recurrent cascade.

### Architecture

Construct typed obligation tokens:

```text
king_escape_obligations
high_value_piece_defense_obligations
line_block_obligations
promotion_or_mate_prevention_obligations
```

Construct resource tokens:

```text
legal defender pieces
king moves
blocker squares
capture resources
interposition resources
```

Run `T` recurrent allocation steps:

```text
demand_t = obligation_update(demand_{t-1}, threat_context)
allocation_t = softmax(resource_scores - demand_t)
residual_t = demand_t - allocated_resources_t
```

Pool:

```text
exhaustion_curve = [sum_positive(residual_t), entropy(allocation_t), max_deficit_t]
puzzle_logit = MLP(exhaustion_curve + board_context)
```

### Why It Could Beat BT4

Near-puzzles can have high raw attack pressure but enough defensive slack. A cascade readout can learn "does the defense eventually run out?" rather than "is there pressure?"

### Distinction From Existing Hall-Defect Ideas

Hall-defect packets use more direct set-system or matching bottlenecks. This architecture should be implemented as a sequential learned cascade with typed obligations and resource exhaustion curves, not only as a static Hall statistic.

### First Config

```yaml
model:
  name: defender_exhaustion_cascade
  input_channels: 18
  num_classes: 1
  token_dim: 80
  cascade_steps: 4
  obligation_types: 6
  resource_types: 6
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `one_step_only` | Tests whether cascade depth matters. |
| `no_obligation_types` | Tests whether typed defense structure matters. |
| `resource_shuffle` | Should damage near-puzzle rejection if allocation is meaningful. |

## Idea 3: Causal Piece-Derivative Network

### Thesis

In true puzzles, the puzzle signal often depends sharply on a few critical pieces or squares. In near-puzzles, the score may come from broad tactical texture without a decisive dependency.

This model computes a cheap neural approximation of "which pieces causally matter?"

### Architecture

Use a base trunk:

```text
base_logit, h = trunk(board)
```

Select top candidate pieces/squares:

```text
candidates = top_k(gating_head(h), k=8)
```

For each candidate, apply deterministic masked interventions:

```text
remove piece
hide square channel group
neutralize side-to-move ownership for that token
```

Run a lightweight shared delta encoder, not the full trunk:

```text
delta_i = delta_encoder(board, intervention_i)
sensitivity_i = base_logit - delta_logit_i
```

Readout:

```text
criticality_stats = [max, top2_gap, entropy, signed_sum, own_vs_enemy_split]
puzzle_logit = base_logit + MLP(criticality_stats)
```

### Why It Could Beat BT4

BT4 sees the board once. This model asks: "does removing one tactically important object destroy the puzzle signal?" Real puzzles should have sharper causal structure than many near-puzzles.

### Cost Control

Avoid a full forward pass for each intervention. The first version should:

- compute the trunk once
- compute local intervention embeddings with a small shared encoder
- only test `k=4` and `k=8`

### First Config

```yaml
model:
  name: causal_piece_derivative
  base: simple_cnn
  input_channels: 18
  num_classes: 1
  candidate_k: 8
  delta_channels: 32
  delta_layers: 2
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `random_candidates` | Should be worse than learned candidate pieces. |
| `no_delta_readout` | Gives plain trunk baseline. |
| `full_remove_only` | Tests whether multiple interventions matter. |
| `candidate_k_4_vs_8` | Measures cost/performance tradeoff. |

## Idea 4: Phase-Transition Pressure Network

### Thesis

The key difference between a true puzzle and a near-puzzle may be criticality. The board may sit near a threshold where small increases in pressure, line opening, or defender loss cause a tactical collapse.

Instead of measuring pressure magnitude, measure pressure phase transitions.

### Architecture

Create learned pressure fields:

```text
attack_pressure[square]
defense_pressure[square]
escape_pressure[square]
line_block_pressure[square]
target_value_pressure[square]
```

Sweep thresholds:

```text
tau in learned or fixed grid, e.g. 8 thresholds
binary-ish field_tau = sigmoid((pressure - tau) / temperature)
```

For each threshold, compute differentiable summaries:

```text
mass
connected king-zone mass
largest soft component approximation
boundary length
pressure surplus around king/queen/rook
```

Readout uses changes across thresholds:

```text
critical_curve = summary(tau_{i+1}) - summary(tau_i)
puzzle_logit = MLP(critical_curve)
```

### Why It Could Beat BT4

Near-puzzles may be high-pressure but not critical. A real tactic often has a sharp transition: one overloaded defender, one pinned piece, one escape square gone. The transition curve can expose that.

### First Config

```yaml
model:
  name: phase_transition_pressure
  input_channels: 18
  num_classes: 1
  field_channels: 32
  thresholds: 8
  temperature: 0.2
  cnn_backbone: shallow
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `single_threshold` | Tests whether transition curves matter. |
| `pressure_mean_only` | Tests criticality vs magnitude. |
| `no_king_zone_features` | Tests whether king-local transitions dominate. |

## Idea 5: Disproof-Ledger Puzzle Network

### Thesis

The model should not only collect evidence for "puzzle." It should collect explicit disproof evidence:

```text
king can escape
defender can recapture
line is blocked
threat is too slow
target is protected enough
side to move lacks tempo
```

Near-puzzles should light up disproof channels.

### Architecture

Use a shared trunk and a ledger head:

```text
h = trunk(board)
positive_evidence = pos_head(h)
disproof_entries = disproof_head(h)  # D channels
```

Final logit:

```text
disproof_strength = softplus(disproof_entries).sum()
puzzle_logit = positive_evidence - disproof_strength
```

Training:

- main binary BCE
- optional auxiliary source loss: near-puzzles should have at least one high disproof channel
- sparsity on disproof entries so one or two clear disproofs dominate

### Why It Could Beat BT4

The present failure row is "near-puzzle predicted puzzle." A disproof ledger gives the network internal language for "this looks tactical but one condition fails."

### Difference From Negative-Class Disentangled Head

Negative-class disentangling separates random and near-puzzle negatives. This idea separates positive evidence from named negative tactical disproof mechanisms. The final representation should be interpretable as evidence minus disproof.

### First Config

```yaml
model:
  name: disproof_ledger_puzzle_net
  base: lc0_bt4_classifier
  disproof_channels: 8
  disproof_sparsity: 0.01
  near_disproof_aux_weight: 0.1
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_disproof_subtraction` | Tests if ledger is more than extra parameters. |
| `dense_disproof_no_sparsity` | Tests whether sparse disproof helps. |
| `no_near_aux` | Tests source-label use. |

## Idea 6: Motif Tensor Factorization Network

### Thesis

Puzzle signal is often a multiplicative relation among typed roles:

```text
attacker type x target type x defender state x line relation x tempo
```

A plain CNN learns these interactions implicitly. A low-rank tensor factorization can model them directly and efficiently.

### Architecture

Create role embeddings:

```text
A_i = attacker candidate embeddings
T_j = target candidate embeddings
D_k = defender/context embeddings
R_ij = relation embedding between attacker and target
M_ijk = learned low-rank motif score
```

Use CP or Tucker factorization:

```text
score(i,j,k) = sum_r a_r(A_i) * t_r(T_j) * d_r(D_k) * rel_r(R_ij)
```

Pool:

```text
top_motif_scores
motif_entropy
own/opponent motif contrast
near-disproof motif scores
```

Output one puzzle logit.

### Why It Could Beat BT4

Multiplicative motif logic is natural for tactics: each part must be present. This can help reject near-puzzles that have some components but not the whole conjunction.

### First Config

```yaml
model:
  name: motif_tensor_factorization
  input_channels: 18
  num_classes: 1
  token_dim: 64
  rank: 24
  top_motifs: 16
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `additive_motif_score` | Tests whether multiplicative factors matter. |
| `rank_8_24_64` | Measures capacity. |
| `no_relation_embedding` | Tests chess geometry contribution. |

## Idea 7: Tempo-Alignment Gate Network

### Thesis

Many near-puzzles are tactical-looking for the wrong side or require a tempo that the side to move does not have. The model should explicitly gate static tactical danger by side-to-move tempo.

### Architecture

Compute two branches:

```text
static_tension = trunk_without_side_to_move(board)
tempo_privilege = side_to_move_branch(board)
```

Use a gated interaction:

```text
aligned_tension = static_tension * sigmoid(tempo_gate)
misaligned_tension = static_tension * sigmoid(-tempo_gate)
puzzle_logit = MLP([aligned_tension, misaligned_tension, aligned - misaligned])
```

Add an intervention view:

```text
flip side-to-move plane only
require meaningful logit change for true puzzles
```

### Why It Could Beat BT4

The benchmark target is puzzle-now, not "interesting position eventually." A tempo-alignment gate should reject positions where the board has tactical material but the side to move cannot force it.

### Difference From Prior Tempo-Odd Packets

Prior tempo-odd ideas split representations into side-to-move even/odd components. This one specifically uses side-to-move as a gate on tactical tension and requires motif-aligned tension, not just representation parity.

### First Config

```yaml
model:
  name: tempo_alignment_gate
  input_channels: 18
  num_classes: 1
  tension_channels: 96
  tempo_gate_dim: 64
  flip_consistency_weight: 0.05
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `concat_no_gate` | Tests multiplicative tempo alignment. |
| `no_side_flip_loss` | Tests intervention loss. |
| `side_to_move_removed` | Should hurt true puzzle recall. |

## Idea 8: Puzzle Boundary Twin Encoder

### Thesis

The hardest part is the boundary between verified puzzles and verified near-puzzles. Learn that boundary directly with a twin encoder and margin objective.

### Architecture

Use shared encoder:

```text
z = encoder(board)
puzzle_logit = head(z)
```

During training, form mini-batch pairs:

```text
positive puzzle positions
near-puzzle negatives
random negatives
```

Margin objective:

```text
logit(puzzle) >= logit(near) + margin_near
logit(near)   >= logit(random) + optional margin_random_surface
```

But final inference stays single-board single-logit.

### Why It Could Beat BT4

This turns the benchmark's hardest diagnostic into a training pressure. It should be especially useful if paired with a strong architecture like BT4, Piece-Token Hybrid, or Line-Piece Crossbar.

### First Config

```yaml
model:
  name: puzzle_boundary_twin
  base: piece_token_cnn_hybrid
  num_classes: 1
training:
  pair_mining: in_batch
  margin_near: 0.5
  margin_weight: 0.2
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `bce_only` | Base model without boundary pressure. |
| `random_pairs_only` | Should help less than near-puzzle pairs. |
| `hardest_in_batch_vs_random_near` | Tests pair mining. |

## Idea 9: Critical-Square Budget Network

### Thesis

Puzzles often hinge on a small number of critical squares: king escape squares, line intersections, pinned-piece squares, promotion squares, or overloaded defender squares.

Force the model to classify through a small square budget.

### Architecture

CNN stem:

```text
H: 64 square embeddings
```

Budget selector:

```text
g_s = selector(H_s)
selected_squares = differentiable top-k(g_s), k = 4..12
```

Readout only sees selected square features plus global lightweight context:

```text
z = pool(selected_squares * H_s)
puzzle_logit = MLP([z, global_context])
```

Add auxiliary diagnostics:

- selected-square entropy
- overlap with king zone
- stability under board symmetries

### Why It Could Beat BT4

This is a bottleneck against texture matching. Near-puzzles may look globally sharp, but if no small set of squares carries decisive evidence, the model should hesitate.

### First Config

```yaml
model:
  name: critical_square_budget
  input_channels: 18
  num_classes: 1
  channels: 96
  selected_squares: 8
  topk_temperature: 0.5
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `all_squares_readout` | Tests whether bottleneck helps near-puzzle rejection. |
| `random_square_budget` | Should underperform learned budget. |
| `k_4_8_12` | Tests how sparse the evidence can be. |

## Implementation Priority

Recommended order:

1. Implement `Causal Piece-Derivative Network` as a wrapper around an existing simple CNN. It should be fastest to test and can reuse most of the current trainer.
2. Implement `Forcing-Certificate Transformer` as the first genuinely new architecture.
3. Implement `Disproof-Ledger Puzzle Network` as a head on BT4 or Piece-Token Hybrid.
4. Implement `Phase-Transition Pressure Network` if the first three do not reduce the near-puzzle false-positive row enough.

## Benchmark Acceptance

Each model should report:

```text
test F1
test PR AUC
test accuracy
precision
recall
3x2 source-class confusion matrix
random -> puzzle false-positive rate
near-puzzle -> puzzle false-positive rate
puzzle recall
```

A model is interesting only if it beats or challenges:

```text
BT4 F1: 0.7445
BT4 PR AUC: 0.8068
BT4 near-puzzle FP: 24.8%
```

Best target for the next round:

```text
test F1 >= 0.76
test PR AUC >= 0.82
near-puzzle FP <= 20%
puzzle recall >= 78%
```

