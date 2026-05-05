# Codex Research Packet: Tactical Bisimulation Puzzle Network

## File Metadata

- Filename: `chess_nn_research_2026-04-25_0113_saturday_shanghai_tactical_bisimulation.md`
- Generated at: 2026-04-25 01:13
- Weekday: Saturday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: single flagship research packet, not implemented, not benchmark results

## One-Sentence Thesis

Learn a chess-position representation where two positions are close only if their legal tactical consequences are behaviorally similar; classify puzzles from this tactical bisimulation quotient rather than raw board texture.

## Why This Is The Final Flagship Idea

Most proposed architectures ask one of these questions:

```text
Is there a proof?
Is there a defense?
Is there a near edit?
Is there attacker/defender equilibrium?
Does the model understand legal dynamics?
```

This idea asks a more general question:

```text
Which positions are the same chess situation up to tactical behavior?
```

If that quotient is learned well, it should help not just this benchmark but future chess classification tasks. A near-puzzle may look almost identical on the board, but if its legal continuations contain a defensive escape, it should land in a different tactical-bisimulation cell.

## Target

Use the corrected single-logit puzzle benchmark:

```text
source class 0: known non-puzzle / random position -> target 0
source class 1: verified near-puzzle / hard negative -> target 0
source class 2: verified puzzle -> target 1
```

The model emits:

```text
one puzzle logit
```

The 3x2 diagnostic matrix remains mandatory:

```text
rows    = random, near-puzzle, puzzle
columns = predicted non-puzzle, predicted puzzle
```

## Forbidden Inputs

Do not use these as inference inputs:

- Stockfish scores
- Stockfish PVs or best moves
- node counts
- mate scores
- verification metadata
- source labels
- source file identity
- future game outcomes

Legal move descriptors and rule-applied successor boards are allowed for self-supervised or architectural computation because they are deterministic consequences of the current board, not engine knowledge.

## Closest Existing Ideas

### Closest Registered Ideas

- `i007_neural_proof_number_search`
- `i009_tactical_equilibrium_network`
- `i010_rule_consistent_latent_dynamics`
- `i008_boundary_edit_lagrangian_network`

### Exact Overlap

```text
All use legal consequences or tactical structure beyond static board texture.
```

### Exact Difference

```text
Tactical Bisimulation learns a metric/quotient over positions from successor behavior. It does not build a proof tree, solve an attacker/defender matrix game, optimize edit distance, or merely add auxiliary transition prediction.
```

The key object is a learned behavior-equivalence metric:

```text
d(position A, position B) is small only if their tactical legal continuations are similar.
```

## Mathematical Thesis

### Definitions

Let:

```text
x = current chess position
z = E(x) = latent state
A(x) = deterministic capped legal/pseudo-legal action set
T(z, a) = learned latent successor for action a
g(z) = puzzle logit
d(z_i, z_j) = learned tactical bisimulation distance
```

For each position, construct a successor signature:

```text
mu_x = sum_a pi(a | x) delta_{T(E(x), a)}
```

where `pi(a | x)` is a learned rule-only action weighting network. It may use move descriptors, but not engine scores.

Define a bisimulation-style target relation:

```text
d(z_i, z_j)
  approx
  |g(z_i) - g(z_j)|
  + gamma * W(mu_i, mu_j)
```

where `W` is a small Sinkhorn/Wasserstein distance between successor signatures.

Intuition:

```text
two boards are behaviorally close if their current puzzle evidence and legal successor distributions are close
```

### Classifier

The final puzzle head uses:

```text
base_logit = g(E(x))
prototype_distances = d(E(x), learned_prototypes)
successor_spread = entropy / diameter of mu_x
bisim_residual = Bellman-style consistency residual
puzzle_logit = MLP([base_logit, prototype_distances, successor_spread, bisim_residual])
```

Learned prototypes should include:

```text
puzzle-like tactical cells
near-puzzle disproof cells
random quiet cells
```

These are learned from data, not used as input labels.

## Assumptions

- Puzzle-vs-near-puzzle distinction is often a behavioral distinction, not a visual one.
- Legal successor structure exposes whether a tactical-looking position actually has a defensive escape.
- A bisimulation metric can force the latent space to cluster by chess consequence structure.
- This quotient will generalize better than direct board texture classification.

## Claim

Hypothesis: a Tactical Bisimulation Puzzle Network should reduce near-puzzle false positives because near-puzzles that look like puzzles but have different legal continuation behavior will be pushed away from true puzzle cells in latent space.

## Mechanism

The model must learn three things together:

```text
1. direct puzzle evidence
2. legal successor behavior
3. a metric that says when two positions are tactically equivalent
```

This prevents the classifier from treating two positions as similar merely because they share piece layout, king pressure, material phase, or surface tension. Similarity must survive one-step consequence comparison.

## Why It Could Beat BT4

BT4 is a strong board-pattern learner, but it has no explicit requirement that latent distance reflect chess behavioral equivalence. A near-puzzle can sit close to a puzzle in board-texture space. Bisimulation training should separate them if their successor signatures differ.

The expected benchmark gain is not from more parameters. It is from a better latent geometry:

```text
near-puzzles become close to disproof/escape cells
true puzzles become close to forcing-proof cells
random positions become close to quiet low-consequence cells
```

## Architecture

### Components

```text
board_encoder E
move_encoder M
latent_transition T
successor_signature_pool mu
bisimulation_distance_head d
prototype_bank P
puzzle_head h
```

### Forward Pass

```text
z = E(board)
moves = deterministic_move_sampler(board)
move_tokens = M(moves)
z_next_pred = T(z, move_tokens)
mu = signature_pool(z_next_pred, move_tokens)
base_logit = h_base(z)
proto_dist = d(z, prototype_bank)
bisim_residual = consistency_residual(z, mu)
puzzle_logit = h_final([base_logit, proto_dist, signature_stats(mu), bisim_residual])
```

### Training Losses

Main:

```text
L_cls = BCEWithLogitsLoss(puzzle_logit, y)
```

Bisimulation consistency:

```text
L_bisim = | d(z_i,z_j) - stopgrad(|g_i-g_j| + gamma * W(mu_i,mu_j)) |
```

Transition consistency:

```text
L_next = || T(E(x),a) - stopgrad(E(apply(x,a))) ||^2
```

Metric separation:

```text
L_margin = triplet / supervised contrastive loss using binary labels
```

Optional use of verified near-puzzle source labels:

```text
Use fine source class only for diagnostic pair mining or an explicit ablation.
Do not use source class as an inference input.
```

## First Config

```yaml
model:
  name: tactical_bisimulation_puzzle_network
  input_channels: 18
  num_classes: 1
  latent_dim: 128
  max_moves: 32
  move_dim: 48
  prototype_count: 24
  sinkhorn_iters: 5
  gamma: 0.5
training:
  mode: puzzle_binary
  loss: bce_with_logits
  cls_weight: 1.0
  bisim_weight: 0.1
  next_latent_weight: 0.05
  margin_weight: 0.05
```

## Required Diagnostics

Report:

- test F1
- test PR AUC
- test accuracy
- 3x2 source-class confusion matrix
- near-puzzle false-positive rate
- puzzle recall
- mean distance to puzzle prototypes by source class
- mean distance to disproof prototypes by source class
- bisimulation residual by source class
- successor signature spread by source class

Expected diagnostic ordering:

```text
distance_to_puzzle_proto(puzzle) < distance_to_puzzle_proto(near) < distance_to_puzzle_proto(random)
distance_to_disproof_proto(near) < distance_to_disproof_proto(puzzle)
```

## Ablations

| Ablation | Purpose |
|---|---|
| `no_bisim_loss` | Tests whether quotient geometry matters. |
| `no_successor_signature` | Tests legal consequence structure. |
| `no_transition_consistency` | Tests learned dynamics contribution. |
| `euclidean_metric_only` | Tests learned metric head. |
| `random_move_sampler` | Tests legal/tactical move semantics. |
| `no_prototypes` | Tests prototype quotient readout. |
| `binary_margin_only` | Tests contrastive learning without bisimulation. |
| `fine_label_pair_mining_off` | Tests whether verified near labels are needed for mining. |

## Falsification Criteria

Reject or revise if:

```text
no_bisim_loss matches full model
or no_successor_signature matches full model
or random_move_sampler matches deterministic legal sampler
or prototype distances do not separate puzzle and near-puzzle rows
or near-puzzle FP is not below BT4's 24.8%
```

Ambitious target:

```text
test PR AUC >= 0.82
test F1 >= 0.76
near-puzzle FP <= 0.20
puzzle recall >= 0.78
```

## Counterexamples

This idea may fail on:

- puzzle labels that are dominated by static visual motifs
- positions where one-ply successor signatures are insufficient
- sparse or noisy legal move sampling
- datasets where near-puzzles are not behaviorally close to puzzles
- cases where the transition model learns weak successor predictions

## Why This Is Worth Trying

This is the cleanest "chess understanding" idea in the set:

```text
good chess representation = quotient by legal tactical behavior
```

It is broader than puzzle detection, more principled than another CNN variant, and cheaper than explicit proof-number search. If it works, it should give a reusable backbone for many future chess classification tasks.

## Implementation Priority

Build after `i010_rule_consistent_latent_dynamics` or as a direct extension of it:

1. Implement board encoder + legal move sampler + latent transition.
2. Add successor signature pooling.
3. Add prototype bank and learned metric.
4. Add bisimulation consistency loss.
5. Compare against the same encoder with only BCE and transition auxiliary loss.

Most important first comparison:

```text
i010-style latent dynamics vs tactical bisimulation quotient
```

If bisimulation wins, it should become a registered `i011` idea and a serious benchmark-track implementation.

