# Codex Research Packet: High-Upside Puzzle Architecture Batch 4

## File Metadata

- Filename: `chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md`
- Generated at: 2026-04-25 01:09
- Weekday: Saturday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: research packet, not implemented, not benchmark results

## Target

The benchmark target is still the corrected single-logit puzzle task:

```text
source class 0: known non-puzzle / random position -> target 0
source class 1: verified near-puzzle / hard negative -> target 0
source class 2: verified puzzle -> target 1
```

Inference input must not include:

- Stockfish scores
- PVs
- node counts
- mate scores
- verification metadata
- source labels
- source file identity
- engine best moves

The important diagnostic remains:

```text
3x2 source-class matrix:
rows    = random, near-puzzle, puzzle
columns = predicted non-puzzle, predicted puzzle
```

## Existing Strong Ideas To Avoid Duplicating

The active structured registry already contains:

- `i004_puzzle_obligation_flow_network`: obligation/resource flow residual.
- `i005_null_move_contrast_puzzle_network`: current-vs-null tempo contrast.
- `i006_proof_core_set_verifier`: sparse proof-core verifier.
- `i007_neural_proof_number_search`: bounded proof/disproof tree.
- `i008_boundary_edit_lagrangian_network`: minimum chess-edit boundary energy.
- `i009_tactical_equilibrium_network`: attacker/defender matrix-game equilibrium.
- `i010_rule_consistent_latent_dynamics`: latent legal-dynamics auxiliary training.

This packet avoids another direct copy of those mechanisms. The focus is on new falsifiable mechanisms that could become future registered ideas.

## New Candidate Ranking

| Rank | Idea | Why it might beat current baselines | Risk |
|---:|---|---|---|
| 1 | Barrier-Cut Puzzle Network | Measures whether defensive barriers actually separate threats from targets. | Could overlap with flow if implemented too broadly. |
| 2 | Tactical Hessian Spectrum Network | Measures local curvature of puzzle evidence under legal perturbations. | Hessian estimates can be noisy. |
| 3 | Absorbing Threat Markov Network | Models tactical collapse as absorption into proof/disproof states without full search. | Needs good state design. |
| 4 | Neural Clause-Resolution Puzzle Network | Tests whether puzzlehood follows from a small typed clause proof. | Clause library may be brittle. |
| 5 | Piece Liability Gradient Network | Identifies overloaded/pinned pieces by learned liability gradients. | Could become just a causal-piece variant. |
| 6 | Hierarchical Tactical Option Network | Learns reusable tactical options instead of raw move/reply tokens. | Needs careful option collapse prevention. |
| 7 | Cross-Defense Consistency Network | Checks whether independent defensive views agree that a tactic fails or survives. | Might be too close to agreement/fusion unless kept defense-specific. |

Best practical next implementation:

```text
Barrier-Cut Puzzle Network
```

Best high-math next implementation:

```text
Tactical Hessian Spectrum Network
```

Best compromise between novel and implementable:

```text
Absorbing Threat Markov Network
```

## Idea 1: Barrier-Cut Puzzle Network

### Thesis

A true puzzle often exists because the defender cannot maintain a barrier between attacking force and a valuable target: king, queen, promotion square, pinned defender, or mating square. A near-puzzle may contain pressure, but there is still a strong defensive cut.

### Closest Existing Idea

Closest overlap:

- `i004_puzzle_obligation_flow_network`
- king-escape percolation packets
- threat topology packets

Exact overlap:

```text
All model defensive sufficiency using chess-shaped graphs.
```

Exact difference:

```text
Barrier-Cut uses differentiable min-cut / cut-capacity features between attacker regions and target regions, not obligation-resource assignment, percolation, or Betti summaries.
```

### Mechanism

Build a typed graph:

```text
nodes: squares, occupied pieces, king-zone targets, high-value targets
edges: legal attacks, defenses, line-of-sight, blocker adjacency, escape adjacency
capacities: learned from piece type, side, target value, occupancy, line blocker status
```

For each target family:

```text
source set S = attacking force / pressure nodes
sink set T = target / king-zone / promotion nodes
```

Compute differentiable soft min-cut:

```text
cut_value(S,T) = soft_min_cut(capacities, S, T)
barrier_margin = defensive_cut_capacity - attack_pressure
puzzle_logit = MLP([cut_values, margins, target summaries, board_context])
```

### Why It Could Work

Near-puzzles can look tactical because there is pressure, but if the defensive barrier remains intact, the position is not actually a puzzle. The cut value gives the model a direct "can defense still separate threat from target?" statistic.

### First Config

```yaml
model:
  name: barrier_cut_puzzle_network
  input_channels: 18
  num_classes: 1
  node_dim: 96
  edge_dim: 32
  target_sets: 8
  cut_iterations: 6
```

### Ablations

| Ablation | Purpose |
|---|---|
| `no_cut_layer` | Replace cuts with graph pooling. |
| `random_edge_capacities` | Tests learned chess capacities. |
| `attack_only_no_barrier` | Tests pressure magnitude vs barrier. |
| `king_targets_only` | Tests mate/king dependence. |
| `material_targets_only` | Tests material-tactic dependence. |

### Falsification

Reject if:

```text
no_cut_layer matches full model
or random_edge_capacities match legal capacities
or near-puzzle FP does not improve over BT4/CNN baselines
```

## Idea 2: Tactical Hessian Spectrum Network

### Thesis

A real puzzle may be a sharp local maximum of tactical evidence under legal perturbations. Near-puzzles may have high raw evidence but flatter or less stable local geometry.

Instead of asking only "how high is the score?", ask:

```text
how curved is the score landscape around this position?
```

### Closest Existing Idea

Closest overlap:

- `i008_boundary_edit_lagrangian_network`
- variational board action packet
- causal derivative packets

Exact overlap:

```text
All use local perturbations or variational structure around a board position.
```

Exact difference:

```text
Hessian Spectrum estimates second-order curvature of puzzle evidence under legal perturbation directions; it does not optimize edit distance or compute Euler-Lagrange residuals.
```

### Mechanism

Define legal perturbation directions:

```text
v_i = latent delta for legal move, blocker edit, defender edit, tempo edit, target-protection edit
```

Given base evidence `s(z)`, estimate directional curvature:

```text
H_ij approx d^2 s(z + eps_i v_i + eps_j v_j) / d eps_i d eps_j
```

Use low-rank randomized Hessian probes:

```text
lambda_top
trace
condition estimate
positive/negative curvature split
curvature along defender directions
curvature along attacker directions
```

Final:

```text
puzzle_logit = MLP([base_logit, hessian_spectrum, legal_direction_stats])
```

### Why It Could Work

Puzzles should be structurally brittle: small legal changes around defenders, blockers, or tempo can sharply change tactical value. Near-puzzles may look high-pressure but lack the same decisive curvature.

### First Config

```yaml
model:
  name: tactical_hessian_spectrum
  input_channels: 18
  num_classes: 1
  latent_dim: 128
  probe_directions: 16
  hessian_rank: 8
```

### Ablations

| Ablation | Purpose |
|---|---|
| `first_order_only` | Tests curvature vs gradients. |
| `random_directions` | Tests legal perturbation semantics. |
| `no_defender_directions` | Tests defender curvature. |
| `trace_only` | Tests full spectrum vs scalar curvature. |

### Falsification

Reject if:

```text
first_order_only matches full model
or random directions match legal directions
or curvature diagnostics do not separate puzzle and near-puzzle rows
```

## Idea 3: Absorbing Threat Markov Network

### Thesis

Puzzle detection can be treated as a probabilistic process over tactical states:

```text
pressure -> threat -> forced response -> collapse/proof
pressure -> safe response -> disproof
```

A full proof tree is expensive. A compact absorbing Markov chain can approximate whether the position tends toward proof or disproof.

### Closest Existing Idea

Closest overlap:

- `i007_neural_proof_number_search`
- response-minimax
- tactical program induction packet

Exact overlap:

```text
All model tactical continuation structure.
```

Exact difference:

```text
Absorbing Threat Markov uses a learned finite-state transition matrix with proof/disproof absorbing states, not an explicit move tree or program slots.
```

### Mechanism

Create state tokens:

```text
attack_pressure
defender_available
line_open
king_constrained
target_hanging
counterplay
proof_absorb
disproof_absorb
```

Learn transition matrix:

```text
P = row_stochastic(state_transition(board_context, state_tokens))
```

Compute absorption probabilities:

```text
prob_proof = absorbing_probability(P, proof_absorb)
prob_disproof = absorbing_probability(P, disproof_absorb)
expected_steps = absorption_time(P)
```

Final:

```text
puzzle_logit = MLP([prob_proof, prob_disproof, expected_steps, board_context])
```

### Why It Could Work

This gives a continuation-shaped inductive bias without enumerating all moves. Near-puzzles should leak probability into disproof states. True puzzles should concentrate mass into proof states quickly.

### First Config

```yaml
model:
  name: absorbing_threat_markov_network
  input_channels: 18
  num_classes: 1
  state_count: 12
  state_dim: 96
  transition_steps: 4
```

### Ablations

| Ablation | Purpose |
|---|---|
| `no_absorbing_states` | Tests proof/disproof states. |
| `symmetric_transition` | Tests directed tactical process. |
| `one_step_only` | Tests iterative absorption. |
| `random_state_tokens` | Tests learned tactical states. |

### Falsification

Reject if:

```text
no_absorbing_states matches full model
or absorption probabilities do not separate source classes
or one_step_only matches iterative absorption
```

## Idea 4: Neural Clause-Resolution Puzzle Network

### Thesis

A puzzle often follows from a small proof made of typed facts:

```text
piece pinned
defender overloaded
king escape removed
target attacked
tempo belongs to attacker
```

Use a differentiable clause-resolution layer to derive puzzle evidence from typed facts.

### Closest Existing Idea

Closest overlap:

- tropical constraint circuit packet
- tactical program induction packet
- proof-core set verifier

Exact overlap:

```text
All use proof-like symbolic or semi-symbolic structure.
```

Exact difference:

```text
Clause-Resolution uses learned typed Horn clauses and differentiable unification over piece/square variables, not fixed tropical clauses, ordered program slots, or selected witness verification.
```

### Mechanism

Facts:

```text
Attack(piece, target)
Defends(piece, target)
Pinned(piece, king)
LineOpen(piece, target)
EscapeSquare(square)
Tempo(side)
```

Learn soft Horn clauses:

```text
PuzzleWitness(X,Y,Z) :-
  Attack(X,Y), Pinned(Z), Defends(Z,Y), Tempo(us)
```

Run `K` resolution rounds:

```text
fact_scores_{t+1} = soft_resolution(fact_scores_t, learned_clauses)
```

Pool derived facts for the puzzle logit.

### Why It Could Work

Near-puzzles may contain many raw facts but fail to derive a complete proof. Clause resolution tests the conjunction and variable binding structure more directly than a CNN.

### First Config

```yaml
model:
  name: neural_clause_resolution_puzzle_network
  input_channels: 18
  num_classes: 1
  predicate_dim: 64
  clause_count: 32
  resolution_rounds: 4
```

### Ablations

| Ablation | Purpose |
|---|---|
| `bag_of_facts` | Removes resolution. |
| `no_variable_binding` | Tests unification. |
| `one_round_only` | Tests multi-step derivation. |
| `random_clause_templates` | Tests clause semantics. |

### Falsification

Reject if:

```text
bag_of_facts matches full model
or no_variable_binding matches full model
or derived proof facts do not differ between puzzle and near-puzzle rows
```

## Idea 5: Piece Liability Gradient Network

### Thesis

In many puzzles, one piece is not merely attacked; it is liable. It cannot move, defend, capture, or stay without losing something. Near-puzzles may attack pieces, but the liability does not propagate.

### Closest Existing Idea

Closest overlap:

- causal piece derivative packets
- proof-core set verifier
- obligation flow

Exact overlap:

```text
All inspect important pieces or defensive stress.
```

Exact difference:

```text
Piece Liability Gradient computes per-piece liability by differentiating several learned tactical constraints with respect to piece token state, then propagates liability through defense/attack relations.
```

### Mechanism

For each piece token:

```text
liability_i = || grad_{piece_i} constraint_energy(board) ||
```

Constraint energies:

- king safety
- target defense
- line blocking
- recapture availability
- tempo pressure

Propagate liability:

```text
L_{t+1} = relation_message_passing(L_t, attack_defense_graph)
```

Classify from:

```text
top liability
liability concentration
own/opponent liability contrast
liability near king/queen/rook
```

### Why It Could Work

Puzzles often concentrate tactical necessity onto one overloaded or pinned piece. Near-puzzles may have pressure but no high-liability bottleneck.

### First Config

```yaml
model:
  name: piece_liability_gradient_network
  input_channels: 18
  num_classes: 1
  token_dim: 96
  liability_steps: 3
```

### Ablations

| Ablation | Purpose |
|---|---|
| `no_gradients` | Replace liability with learned scores. |
| `no_propagation` | Tests relation spread. |
| `random_piece_tokens` | Tests piece semantics. |
| `material_only_liability` | Tests tactical vs material shortcuts. |

### Falsification

Reject if:

```text
no_gradients matches full model
or liability concentration is not higher for puzzles
or near-puzzle false positives remain unchanged
```

## Idea 6: Hierarchical Tactical Option Network

### Thesis

Chess tactics are not just moves; they are options:

```text
check net
remove defender
open line
fork
promotion race
back-rank bind
```

A network should learn a small library of reusable tactical options and classify from option viability.

### Closest Existing Idea

Closest overlap:

- proof-number search
- response-minimax
- tactical program induction packet

Exact overlap:

```text
All model structured tactical continuations.
```

Exact difference:

```text
Option Network learns reusable option embeddings with initiation, continuation, and termination scores; it does not explicitly build a proof tree or fixed operation program.
```

### Mechanism

Option slots:

```text
o_k = learned tactical option query
```

For each option:

```text
initiation_k = can_this_option_start(board, o_k)
continuation_k = expected_option_progress(board, o_k)
termination_k = proof_or_disproof_termination(board, o_k)
```

Final:

```text
puzzle_logit = logsumexp_k(initiation + continuation - disproof_termination)
```

### Why It Could Work

It is more structured than a CNN but cheaper than search. If options specialize into real tactical families, this can detect puzzles robustly and expose which option fired.

### First Config

```yaml
model:
  name: hierarchical_tactical_option_network
  input_channels: 18
  num_classes: 1
  option_count: 12
  option_dim: 96
  refinement_steps: 3
```

### Ablations

| Ablation | Purpose |
|---|---|
| `single_option` | Tests option diversity. |
| `no_termination_head` | Tests proof/disproof completion. |
| `shared_option_no_specialization` | Tests learned option library. |
| `option_dropout` | Tests robust specialization. |

### Falsification

Reject if:

```text
single_option matches full model
or options collapse to identical attention maps
or no_termination_head matches full model
```

## Idea 7: Cross-Defense Consistency Network

### Thesis

A true puzzle should survive multiple independent defensive interpretations:

```text
material defense view
king safety defense view
line/blocker defense view
tempo defense view
counter-threat defense view
```

Near-puzzles often fail in one of those views.

### Closest Existing Idea

Closest overlap:

- factor-agreement classifier
- disproof ledger packet
- safe-reply verifier packet

Exact overlap:

```text
All compare multiple defensive/evidence views.
```

Exact difference:

```text
Cross-Defense Consistency restricts agreement to defense-specific falsifier heads and classifies from which independent defenses can still refute the tactic.
```

### Mechanism

Defense heads:

```text
d_material
d_king
d_line
d_tempo
d_counterplay
```

Each head outputs:

```text
refute_score_h
confidence_h
evidence_vector_h
```

Final:

```text
disproof_consensus = robust_pool(refute_scores, confidences)
puzzle_logit = positive_evidence - disproof_consensus
```

### Why It Could Work

Instead of asking several views to agree positively, this asks whether any strong independent defensive refutation exists. That directly targets near-puzzle false positives.

### First Config

```yaml
model:
  name: cross_defense_consistency_network
  input_channels: 18
  num_classes: 1
  defense_heads: 5
  head_dim: 64
```

### Ablations

| Ablation | Purpose |
|---|---|
| `positive_agreement_only` | Tests defense-specific design. |
| `remove_each_defense_head` | Identifies useful defenses. |
| `mean_pool_refutations` | Tests robust pooling. |
| `no_confidence` | Tests calibrated refutation. |

### Falsification

Reject if:

```text
positive_agreement_only matches full model
or no defense head changes near-puzzle FP
or refutation scores do not rise on near-puzzles
```

## Implementation Priority

Recommended next order:

1. `Barrier-Cut Puzzle Network`: strongest combination of novelty, implementability, and direct near-puzzle pressure.
2. `Tactical Hessian Spectrum Network`: highest math interest and could expose boundary sharpness.
3. `Absorbing Threat Markov Network`: useful middle ground before proof-number search.
4. `Piece Liability Gradient Network`: practical diagnostic model if candidate generation is easier than graph cuts.
5. `Neural Clause-Resolution Puzzle Network`: promising but more brittle.
6. `Hierarchical Tactical Option Network`: good if we want interpretable option specialization.
7. `Cross-Defense Consistency Network`: implement only if factor-agreement and disproof-ledger style results look promising.

## Benchmark Acceptance

Every implementation should use:

```text
mode: puzzle_binary
num_classes: 1
loss: BCEWithLogitsLoss
```

Minimum serious-challenger criteria:

```text
test PR AUC > 0.8068
test F1 > 0.7445
near-puzzle -> puzzle FP < 0.2477
```

Ambitious target:

```text
test PR AUC >= 0.82
test F1 >= 0.76
near-puzzle FP <= 0.20
puzzle recall >= 0.78
```

