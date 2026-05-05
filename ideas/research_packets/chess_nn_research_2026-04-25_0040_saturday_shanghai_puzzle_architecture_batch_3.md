# Codex Research Packet: Puzzle Architecture Batch 3

## File Metadata

- Filename: `chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`
- Generated at: 2026-04-25 00:40
- Weekday: Saturday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: targeted architecture batch, not implemented

## Purpose

This packet adds more candidate architectures for the single-logit puzzle benchmark:

```text
output: one puzzle logit
target 0: random/non-puzzle + verified near-puzzle
target 1: verified puzzle
key diagnostic: 3x2 source-class matrix
```

The central goal is still:

```text
reduce near-puzzle -> puzzle false positives
while keeping verified puzzle recall high
```

Current strongest baseline:

```text
LC0 BT4 tower
test F1: 0.7445
test PR AUC: 0.8068
near-puzzle -> puzzle FP: 24.8%
puzzle recall: 79.4%
```

This batch leans into response modeling, proof sketches, exchange soundness, and architectures that ask "what makes this tactic actually work?" instead of only "does this position look tactical?"

## New Candidate Ranking

| Rank | Idea | Core signal | First implementation difficulty |
|---:|---|---|---|
| 1 | Legal-Reaction Bottleneck Network | True puzzles collapse the opponent's good-looking replies. | Medium |
| 2 | Exchange-Soundness Graph Network | True tactics have favorable exchange or compensation structure. | Medium |
| 3 | Tactical Program Induction Network | Puzzle signal is a short latent program, not a texture. | High |
| 4 | Counterfactual Defender Dropout Network | True puzzles are highly sensitive to specific defenders. | Low-medium |
| 5 | Blocker-Pin Lattice Network | Puzzles often hinge on blocker order and pinned-piece constraints. | Medium |
| 6 | Safe-Reply Certificate Verifier | A puzzle is likely when no cheap safe reply certificate exists. | Medium-high |
| 7 | Latent Reply Entropy Network | True puzzles squeeze reply entropy more than near-puzzles. | Medium |
| 8 | Exchange-Then-King Dual Stream | Separates material tactics from king tactics before recombining. | Low |
| 9 | Tactical Symptom Bayesian Network | Uses differentiable noisy-AND/noisy-OR tactical symptoms. | Medium |
| 10 | Minimal-Edit Puzzle Distance Network | Measures how many small board edits separate a position from puzzlehood. | Medium |
| 11 | Source-Invariant Puzzle Bottleneck | Adversarially removes dataset-source shortcuts while keeping puzzle signal. | Low |
| 12 | Reply-Set Contrastive Transformer | Contrastively embeds puzzle positions against their own plausible replies. | High |

Most practical first build:

```text
Counterfactual Defender Dropout Network
```

Best likely benchmark challenger:

```text
Legal-Reaction Bottleneck Network
```

Most novel:

```text
Tactical Program Induction Network
```

## Idea 1: Legal-Reaction Bottleneck Network

### Thesis

A real puzzle is not merely a position with a threat. It is a position where normal-looking defensive reactions fail or are too few. Near-puzzles often contain pressure, but the opponent still has many valid ways to defuse it.

This model builds a compact representation of the opponent's legal reaction set without running engine search.

### Architecture

Input board goes through a trunk:

```text
H = board_trunk(board)
base_logit = head(H)
```

Generate pseudo-legal move/reply tokens for the side not to move, using deterministic chess rules:

```text
reply_tokens = [from, to, moving_piece, capture_flag, gives_check_flag, target_zone_flag]
```

Embed each reply with local before/after features:

```text
r_i = reply_encoder(reply_i, H[from], H[to], king_zone_context)
```

Compress the reply set:

```text
reply_summary = attention_pool(reply_tokens)
reply_entropy = entropy(score_reply_safe(r_i))
safe_reply_mass = sum(sigmoid(score_reply_safe(r_i)))
```

Final logit:

```text
puzzle_logit = base_logit + MLP([reply_summary, reply_entropy, safe_reply_mass])
```

### Why It Could Beat BT4

BT4 can learn attacks, but it does not explicitly ask whether the opponent has plausible reactions. The near-puzzle row should improve if the model learns that many sharp positions still have defensive replies.

### First Config

```yaml
model:
  name: legal_reaction_bottleneck
  input_channels: 18
  num_classes: 1
  trunk_channels: 96
  reply_dim: 96
  max_replies: 96
  reply_pool: attention
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_reply_tokens` | Plain trunk baseline. |
| `reply_count_only` | Tests whether learned reply content matters beyond mobility. |
| `random_reply_features` | Should break the near-puzzle gain if reply semantics matter. |
| `side_to_move_swapped_replies` | Tests whether reaction direction matters. |

## Idea 2: Exchange-Soundness Graph Network

### Thesis

Many false puzzle signals come from attacks that look strong but lose material or fail tactically after exchanges. A puzzle detector should know whether an apparent tactic is exchange-sound.

This architecture builds a learned static-exchange-style graph, but keeps it neural and differentiable.

### Architecture

Create target square tokens:

```text
target_squares = occupied opponent pieces + king-zone squares + promotion squares
```

For each target, collect attackers and defenders:

```text
A_t = side-to-move attackers of target t
D_t = opponent defenders or recapture resources of target t
```

Run exchange rounds:

```text
state_0 = target_value_embedding(t)
state_{r+1} = GRU(state_r, min_attacker_pool(A_t), min_defender_pool(D_t), line_context)
```

Pool exchange descriptors:

```text
exchange_gain_curve
uncertainty_of_exchange
sacrifice_compensation_features
```

Final:

```text
puzzle_logit = MLP([board_context, exchange_descriptors])
```

### Why It Could Beat BT4

Real puzzles often survive a basic "does the tactic actually win?" sanity check. Near-puzzles often fail there. A learned exchange graph gives the network a compact soundness test.

### First Config

```yaml
model:
  name: exchange_soundness_graph
  input_channels: 18
  num_classes: 1
  token_dim: 80
  exchange_rounds: 4
  target_top_k: 12
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `one_round_exchange` | Tests whether exchange depth matters. |
| `no_piece_values` | Tests whether material priors matter. |
| `attacker_only` | Should confuse near-puzzles with real puzzles if defenders matter. |

## Idea 3: Tactical Program Induction Network

### Thesis

A puzzle can be viewed as a tiny latent program:

```text
attack target
remove defender
force king move
exploit pin
win material or mate
```

Instead of learning only a vector representation, induce a short program sketch and classify from whether the sketch executes coherently.

### Architecture

Define a small library of learned-but-typed operations:

```text
OP_THREATEN
OP_PIN
OP_DEFLECT
OP_OVERLOAD
OP_FORK
OP_CLEAR_LINE
OP_TRAP_KING
OP_WIN_TARGET
```

Use program slots:

```text
program_steps = 3..5 learned operation slots
```

Each step selects:

```text
operation type
primary piece
target square or piece
relation context
precondition score
postcondition score
```

Execution update:

```text
latent_board_state_{t+1} = op_executor(op_t, latent_board_state_t)
```

Readout:

```text
coherence = product_or_sum_of_precondition_postcondition_scores
puzzle_logit = MLP([coherence, final_latent_state, global_context])
```

### Why It Could Beat BT4

Near-puzzles often contain tactic ingredients but no coherent forcing sequence. Program induction gives the model a way to reject incomplete tactical stories.

### First Config

```yaml
model:
  name: tactical_program_induction
  input_channels: 18
  num_classes: 1
  token_dim: 96
  program_steps: 4
  op_types: 8
  executor_layers: 2
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `bag_of_ops_no_order` | Tests whether ordered program structure matters. |
| `one_step_program` | Tests whether multi-step proof sketches matter. |
| `no_precondition_scores` | Tests whether coherence is real. |
| `random_op_labels` | Should not help if operation typing is meaningful. |

## Idea 4: Counterfactual Defender Dropout Network

### Thesis

If a near-puzzle is only superficially tactical, randomly removing defenders or attackers may not reveal a sharp causal structure. If a true puzzle hinges on overloaded defenders, pinning, or one critical escape square, dropout interventions should produce a distinctive sensitivity profile.

### Architecture

Run a normal trunk:

```text
h = trunk(board)
base_logit = head(h)
```

Build deterministic intervention masks:

```text
drop one opponent defender
drop one side-to-move attacker
drop king-zone escape square feature
drop blocker on main ray
```

For efficiency, do not rerun the whole trunk. Feed masks through a small intervention head:

```text
delta_j = intervention_head(h, mask_j)
```

Pool:

```text
defender_sensitivity = topk(delta_for_defenders)
attacker_sensitivity = topk(delta_for_attackers)
asymmetry = defender_sensitivity - attacker_sensitivity
puzzle_logit = base_logit + MLP([asymmetry, sensitivity_entropy])
```

### Why It Could Beat BT4

This is a cheaper cousin of causal intervention models. It is likely easier to implement quickly and might directly reduce near-puzzle false positives.

### First Config

```yaml
model:
  name: counterfactual_defender_dropout
  base: simple_cnn
  input_channels: 18
  num_classes: 1
  intervention_dim: 64
  max_masks: 16
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `random_masks` | Tests whether defender/attacker semantics matter. |
| `defenders_only` | Tests whether asymmetry matters. |
| `no_intervention_head` | Plain base trunk. |

## Idea 5: Blocker-Pin Lattice Network

### Thesis

Line tactics are not only about pieces sharing ranks, files, or diagonals. They depend on ordered blockers and pin constraints. A line can be almost tactical, but one blocker order or one unpinned defender changes everything.

This architecture builds a lattice of blocker states on each ray.

### Architecture

For each slider ray:

```text
ray = ordered squares from source piece
blocker_sequence = occupied squares on ray
pin_candidate = king or high-value target behind blocker
```

Build ray lattice states:

```text
state_0: current blocker order
state_remove_first: first blocker removed
state_remove_second: second blocker removed
state_swap_side: ownership role swap diagnostic
```

Run a small state-space model along each ray:

```text
lattice_state_{i+1} = gated_update(lattice_state_i, blocker_token_i, target_context)
```

Pool:

```text
pin_strength
discovered_attack_potential
blocked_tactic_residual
```

### Why It Could Beat BT4

Near-puzzles frequently contain a nearly open line or almost-pinned piece. The lattice can learn when "almost" is not enough.

### First Config

```yaml
model:
  name: blocker_pin_lattice
  input_channels: 18
  num_classes: 1
  ray_dim: 64
  lattice_states: 4
  layers: 3
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `unordered_blockers` | Tests ordered line structure. |
| `no_remove_states` | Tests counterfactual blocker states. |
| `only_rank_file` | Tests diagonal contribution. |

## Idea 6: Safe-Reply Certificate Verifier

### Thesis

Instead of proving that a position is a puzzle, try to prove that it is not a puzzle. If the model can find a cheap safe-reply certificate, the puzzle logit should go down.

This is a verifier-style architecture with a learned "non-puzzle witness."

### Architecture

Generate candidate safe-reply certificates:

```text
move away king
capture attacker
block line
defend target
counter-threat
trade down
```

Each certificate is a token:

```text
c_i = certificate_encoder(reply_or_resource_i, board_context)
```

Verifier:

```text
validity_i = sigmoid(validity_head(c_i))
strength_i = softplus(strength_head(c_i))
best_disproof = max_i(validity_i * strength_i)
```

Final:

```text
puzzle_logit = positive_puzzle_logit - best_disproof
```

### Why It Could Beat BT4

This directly targets the benchmark's hardest row: near-puzzles should often have a safe-reply witness, while true puzzles should not.

### Difference From Disproof-Ledger

The disproof ledger learns abstract negative evidence. This verifier explicitly proposes reply/resource certificates and selects the strongest one.

### First Config

```yaml
model:
  name: safe_reply_certificate_verifier
  input_channels: 18
  num_classes: 1
  certificate_dim: 96
  max_certificates: 128
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `mean_disproof_instead_of_max` | Tests witness-style max pooling. |
| `no_validity_gate` | Tests if validity and strength need separation. |
| `certificate_count_only` | Tests semantics beyond mobility. |

## Idea 7: Latent Reply Entropy Network

### Thesis

A forcing puzzle often reduces the opponent's viable reply distribution. A near-puzzle may have many replies that keep the position acceptable. The network can learn a reply entropy proxy without engine labels.

### Architecture

Generate pseudo-legal replies or response resources:

```text
reply_tokens = deterministic reply candidates
```

Score them with a learned safe-reply scorer:

```text
s_i = safe_reply_score(reply_i, board_context)
p_i = softmax(s_i / temperature)
```

Extract entropy and concentration features:

```text
H = -sum_i p_i log p_i
top1
top2_gap
effective_reply_count = exp(H)
```

Final:

```text
puzzle_logit = MLP([board_context, H, top1, top2_gap, effective_reply_count])
```

### Why It Could Beat BT4

This is a simpler and cheaper version of legal-reaction modeling. It tests whether reply-set compression alone helps separate puzzles from near-puzzles.

### First Config

```yaml
model:
  name: latent_reply_entropy
  input_channels: 18
  num_classes: 1
  reply_dim: 64
  temperature: 0.7
  max_replies: 96
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `reply_count_only` | Mobility baseline. |
| `fixed_uniform_scores` | Removes learned reply quality. |
| `no_entropy_features` | Tests whether entropy statistics matter. |

## Idea 8: Exchange-Then-King Dual Stream

### Thesis

Puzzle data likely mixes at least two broad families:

```text
material-winning tactics
king-safety or mate tactics
```

A single trunk may blur the difference. A dual-stream model can let one branch specialize in material exchange and the other in king danger.

### Architecture

Two branches:

```text
exchange_stream = piece/value/attacker/defender features
king_stream = king-zone/escape/check/line features
```

Gating:

```text
gate = sigmoid(phase_router(board_context))
puzzle_logit = gate * king_logit + (1 - gate) * exchange_logit + residual_logit
```

Training:

- main BCE only at first
- optional entropy penalty to prevent the gate from staying at 0.5
- optional source diagnostics by family once puzzle themes are available

### Why It Could Beat BT4

It is a practical architecture. It may improve quickly because it lets the model represent two different reasons a position is a puzzle.

### First Config

```yaml
model:
  name: exchange_then_king_dual_stream
  input_channels: 18
  num_classes: 1
  stream_channels: 64
  gate_dim: 32
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `shared_stream_only` | Tests value of specialization. |
| `fixed_half_gate` | Tests learned routing. |
| `king_only` | Tests mate/king subset bias. |
| `exchange_only` | Tests material-tactic subset bias. |

## Idea 9: Tactical Symptom Bayesian Network

### Thesis

Many tactical concepts behave like noisy logical symptoms:

```text
king exposed
defender overloaded
line opened
piece pinned
queen aligned
escape squares reduced
target under-defended
```

A differentiable Bayesian-style symptom network can force the classifier to combine these symptoms with noisy-AND/noisy-OR structure instead of uninterpretable dense pooling.

### Architecture

Learn symptom probabilities:

```text
s_k = sigmoid(symptom_head_k(board_features))
```

Combine symptoms through learned noisy gates:

```text
cause_j = 1 - product_k(1 - w_jk * s_k)
puzzle_prob = noisy_and_or(causes)
```

Output:

```text
puzzle_logit = logit(clamp(puzzle_prob)) + residual_logit
```

### Why It Could Beat BT4

Near-puzzles often activate some symptoms but miss the necessary conjunction. Noisy logical structure could reduce false positives by requiring compatible symptoms.

### First Config

```yaml
model:
  name: tactical_symptom_bayes
  input_channels: 18
  num_classes: 1
  symptoms: 24
  latent_causes: 8
  residual_weight_init: 0.1
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `linear_symptom_readout` | Tests noisy logical combination. |
| `no_residual_logit` | Tests pure symptom bottleneck. |
| `symptom_dropout` | Tests robustness of symptom decomposition. |

## Idea 10: Minimal-Edit Puzzle Distance Network

### Thesis

A near-puzzle may be one small edit away from being a true puzzle:

```text
remove defender
move king one square
open line
change side to move
```

The model should learn distance to puzzlehood, not just current puzzle score. A real puzzle should have distance near zero. A near-puzzle may have high "almost" signal but nonzero required edit distance.

### Architecture

Predict small edit costs:

```text
edit_remove_defender
edit_open_line
edit_target_unprotected
edit_tempo_flip
edit_king_escape_removed
```

Aggregate with softmin:

```text
min_edit_cost = -tau * logsumexp(-edit_costs / tau)
```

Final:

```text
puzzle_logit = positive_evidence - min_edit_cost_penalty
```

Training:

- BCE on final logit
- near-puzzles encouraged to have nonzero edit cost
- true puzzles encouraged to have at least one low-cost proof path

### Why It Could Beat BT4

This gives the model a place to put "almost puzzle" evidence without calling it puzzle.

### First Config

```yaml
model:
  name: minimal_edit_puzzle_distance
  input_channels: 18
  num_classes: 1
  edit_types: 10
  edit_dim: 64
  tau: 0.5
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_edit_penalty` | Tests distance mechanism. |
| `mean_edit_cost` | Tests softmin near-best edit. |
| `no_near_edit_aux` | Tests hard-negative source supervision. |

## Idea 11: Source-Invariant Puzzle Bottleneck

### Thesis

The dataset has three source groups. A model may accidentally learn source artifacts instead of puzzle structure. This architecture tries to preserve puzzle signal while removing source identity from the main representation.

### Architecture

Encoder:

```text
z = encoder(board)
puzzle_logit = puzzle_head(z)
source_logits = source_adversary(gradient_reverse(z))
```

Training:

```text
BCE(puzzle_logit, puzzle_binary)
- lambda * CE(source_logits, fine_source_label)
```

But preserve a small diagnostic branch:

```text
z_source = source_probe(stop_gradient(z))
```

This lets reports show whether the representation still leaks source identity.

### Why It Could Beat BT4

If some high accuracy comes from source artifacts, source-invariant training should generalize better and make near-puzzle false positives more meaningful.

### First Config

```yaml
model:
  name: source_invariant_puzzle_bottleneck
  base: simple_cnn
  input_channels: 18
  num_classes: 1
  bottleneck_dim: 128
  adversary_weight: 0.1
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_source_adversary` | Plain baseline. |
| `strong_adversary` | Tests if too much invariance hurts puzzle recall. |
| `source_probe_only` | Measures leakage without changing training. |

## Idea 12: Reply-Set Contrastive Transformer

### Thesis

A puzzle position should embed differently from its plausible reply positions. A near-puzzle may remain close to one or more safe replies. Use contrastive learning over current position and pseudo-reply positions.

### Architecture

For each board:

```text
z_current = encoder(board)
z_reply_i = encoder_light(apply_pseudo_reply_i(board))
```

Contrastive objective:

```text
true puzzles: push z_current away from safe-looking reply embeddings
near-puzzles: allow a close safe-reply neighbor
random negatives: no strong contrast pressure
```

Final logit:

```text
reply_gap = min_i distance(z_current, z_reply_i)
puzzle_logit = base_logit + gap_head(reply_gap, reply_pool)
```

### Why It Could Beat BT4

This makes "can the opponent move into a stable non-puzzle state?" a geometric property of the representation.

### First Config

```yaml
model:
  name: reply_set_contrastive_transformer
  input_channels: 18
  num_classes: 1
  embed_dim: 128
  max_replies: 32
  contrastive_weight: 0.1
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_reply_embeddings` | Plain encoder baseline. |
| `random_reply_positions` | Tests legal reply relevance. |
| `contrastive_only_no_gap_head` | Tests readout vs representation learning. |

## Implementation Priority

Recommended order:

1. Implement `Counterfactual Defender Dropout Network`; it can wrap an existing CNN and should be fast to benchmark.
2. Implement `Legal-Reaction Bottleneck Network`; it is likely the strongest idea in this packet for the near-puzzle row.
3. Implement `Exchange-Then-King Dual Stream`; it is practical and should give useful diagnostics even if it does not win.
4. Implement `Safe-Reply Certificate Verifier` if legal reaction modeling helps.
5. Implement `Tactical Program Induction Network` later; it is the most ambitious and highest-risk idea.

## Benchmark Acceptance

Every implementation should use:

```text
mode: puzzle_binary
num_classes: 1
loss: BCEWithLogitsLoss
```

Required metrics:

```text
test F1
test PR AUC
test accuracy
precision
recall
3x2 source-class confusion matrix
near-puzzle -> puzzle false-positive rate
puzzle recall
```

A model becomes a serious challenger if it reaches:

```text
test F1 >= 0.76
test PR AUC >= 0.82
near-puzzle FP <= 20%
puzzle recall >= 78%
```

