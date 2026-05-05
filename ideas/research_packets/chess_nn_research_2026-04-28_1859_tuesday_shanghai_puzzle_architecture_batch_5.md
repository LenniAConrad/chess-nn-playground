# Codex Research Packet: Puzzle Architecture Batch 5

## File Metadata

- Filename: `chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md`
- Generated at: 2026-04-28 18:59
- Weekday: Tuesday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: targeted architecture batch, not implemented, not benchmark results

## Target

The benchmark target remains the corrected single-logit puzzle task:

```text
source class 0: known non-puzzle / random position -> target 0
source class 1: verified near-puzzle / hard negative -> target 0
source class 2: verified puzzle -> target 1
```

The main failure mode to attack is still:

```text
near-puzzle -> predicted puzzle false positives
```

Allowed inference inputs:

- Current board tensor.
- Rule-derived current-board geometry.
- Side to move, castling, en-passant, legal occupancy, deterministic coordinates.
- Pseudo-legal or legal-current-position features computed without engine scores or search.

Forbidden inference inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, source file identity, engine best moves, or dataset provenance.

## Existing Directions To Avoid Duplicating

The registered ideas already cover:

- operator-basis relation mixing
- response-minimax bottlenecks
- factor agreement
- obligation/resource flow
- null-move contrast
- sparse proof-core verification
- bounded proof-number search
- boundary-edit energy
- attacker/defender equilibrium
- rule-consistent latent dynamics

Recent packets also cover Hall defects, king-cage path DP, threat topology, sheaf/Hodge variants, move-delta landscapes, optimal transport, orbit/tempo bottlenecks, masked-codec surprise, non-puzzle score fields, non-backtracking walks, barrier cuts, tactical Hessians, absorbing Markov chains, clause resolution, liability gradients, tactical options, and tactical bisimulation.

This packet tries to stay away from exact repeats by focusing on timing, opportunity cost, hidden-line activation, role counterfactuals, and phase-specialized calibration.

## New Candidate Ranking

| Rank | Idea | Core bottleneck | Why it might help near-puzzle separation | Main risk |
|---:|---|---|---|---|
| 1 | Defender Timing Schedule Network | Differentiable lateness over defensive deadlines | Near-puzzles may have enough resources but wrong move order is not forced. | Can blur into proof search if overexpanded. |
| 2 | Discovered-Ray Switchboard Network | Latent switches for blocker vacation and hidden sliders | Many real tactics depend on one piece moving and uncovering a line. | Too narrow if puzzles are mostly non-line tactics. |
| 3 | Counterplay Insolvency Ledger | Own forcing assets minus opponent counterplay liabilities | Filters positions where the defender has a stronger counterthreat. | Could become ordinary feature accounting. |
| 4 | Pinned Mobility Nullspace Network | Defender mobility projected through pin and x-ray constraints | Distinguishes apparent defenders from immobilized defenders. | Pins are only one tactical family. |
| 5 | Tactical Effective Resistance Network | Current-flow resistance between threats, defenders, and targets | Complements min-cut by measuring redundancy and alternate defensive paths. | Close to harmonic/cut ideas unless ablated tightly. |
| 6 | Defender Opportunity-Cost Auction Network | Auction prices for what each defensive resource gives up | Near-puzzles often have a defense, but only by abandoning something else. | Overlaps obligation flow if it ignores opportunity costs. |
| 7 | Role-Counterfactual Necessity Network | Contrast actual board with safe role-swapped/material-preserved boards | Tests whether the exact piece-role geometry is necessary, not just material. | Synthetic swaps can create unrealistic positions. |
| 8 | Phase-Specialist Calibration Mixture | Calibrated expert mixture for mate, material, promotion, endgame, and attack phases | Prevents one global threshold from overcalling near-puzzles in special phases. | More engineering than mathematical novelty. |
| 9 | Forced-Target Funnel Network | Entropy collapse of target sets after candidate forcing actions | A true puzzle should funnel many lines toward the same tactical target. | Must avoid becoming move-delta pooling. |
| 10 | Tactical Subgoal Automaton Network | Learned finite automata over tactical subgoal predicates | Tests whether puzzle evidence is a short ordered tactical script. | Could duplicate program induction if too broad. |

Best immediate implementation:

```text
Defender Timing Schedule Network
```

Best cheap line-tactics implementation:

```text
Discovered-Ray Switchboard Network
```

Best practical calibration implementation:

```text
Phase-Specialist Calibration Mixture
```

## Idea 1: Defender Timing Schedule Network

### Thesis

A true puzzle is not only a position where the defender has too few resources. It is often a position where the defender cannot schedule resources before tactical deadlines expire.

A near-puzzle may have high pressure, but the defender can answer obligations in a feasible order.

### Difference From Existing Ideas

Closest overlap:

- `i004_puzzle_obligation_flow_network`
- `i007_neural_proof_number_search`
- absorbing threat Markov packets

Exact difference:

```text
Obligation flow asks whether resources can cover obligations at all.
Defender timing asks whether obligations can be covered before learned deadlines.
It uses lateness, slack, and precedence, not a search tree or static assignment residual.
```

### Mechanism

Build typed obligations from current-board facts:

```text
king escape square must remain safe
pinned defender must not move
piece under attack must be recaptured
promotion square must be stopped
mate square must be covered
line blocker must remain present
```

For each obligation `o` and defensive resource `r`, predict:

```text
duration(r, o)
deadline(o)
precedence(o_i before o_j)
coverage_score(r, o)
```

Use a soft scheduling layer:

```text
start_time(o) = softmin over feasible resource assignments and precedence chains
lateness(o) = softplus(start_time(o) + duration(o) - deadline(o))
schedule_defect = topk_pool(lateness, k=8)
puzzle_logit = MLP([board_context, schedule_defect, slack_stats])
```

### First Config

```yaml
model:
  name: defender_timing_schedule_network
  input_channels: 18
  num_classes: 1
  hidden_dim: 128
  max_obligations: 32
  max_resources: 32
  schedule_iterations: 6
  topk_lateness: 8
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_timing_static_assignment` | Tests deadlines/slack against plain coverage. |
| `random_deadlines` | Tests whether learned chess deadlines matter. |
| `no_precedence` | Tests ordering constraints. |
| `count_only_obligations` | Tests whether obligation count alone explains gains. |
| `resource_shuffle_same_piece_counts` | Tests semantic resource identity. |

### Falsification

Reject if:

```text
no_timing_static_assignment matches the full model
or random_deadlines match learned deadlines
or near-puzzle false positives do not improve at matched puzzle recall
```

## Idea 2: Discovered-Ray Switchboard Network

### Thesis

Many tactics are not visible in the current attack map because the critical line appears only after a blocker moves. A discovered attack, skewer, deflection, or back-rank tactic can be modeled as a switchboard:

```text
blocker state -> hidden ray activation -> target exposure
```

### Difference From Existing Ideas

Closest overlap:

- line-piece crossbar packets
- Schur-Ray line algebra
- bitboard shift algebra
- causal piece derivative packets

Exact difference:

```text
This model does not score all line geometry equally.
It specializes in conditional ray activation caused by moving, capturing, or deflecting blockers.
```

### Mechanism

For each rank, file, diagonal, and anti-diagonal, extract ordered line tokens:

```text
[piece_type, side, square, is_king_zone, is_high_value_target, blocker_role]
```

Predict switch gates:

```text
g_open(piece p, line l) = probability that p can vacate or be deflected
g_target(line l, target t) = hidden ray exposes target t
```

Compute discovered-ray evidence:

```text
open_ray_energy(l, t) = sum_p g_open(p,l) * g_target(l,t) * line_context(l)
```

Final readout:

```text
puzzle_logit = MLP([ordinary_trunk, topk_open_ray_energy, target_exposure_stats])
```

### First Config

```yaml
model:
  name: discovered_ray_switchboard
  input_channels: 18
  num_classes: 1
  line_dim: 96
  switch_dim: 64
  layers: 3
  topk_rays: 12
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `visible_rays_only` | Tests hidden/discovered lines. |
| `random_blocker_gates` | Tests switch semantics. |
| `no_target_exposure` | Tests target-specific line evidence. |
| `rank_file_only` | Tests diagonal discovered attacks. |
| `line_order_shuffle` | Destroys line geometry while preserving tokens. |

### Falsification

Reject if:

```text
visible_rays_only matches the full model
or line_order_shuffle does not hurt
or gains appear only on line-heavy validation slices but not overall puzzle_binary
```

## Idea 3: Counterplay Insolvency Ledger

### Thesis

Near-puzzles often fail because the defender has counterplay. A model that only measures side-to-move pressure may overcall these. Puzzlehood should depend on whether the opponent's counterthreats remain solvent after the side-to-move begins forcing play.

### Mechanism

Compute two ledgers:

```text
forcing_assets: checks, captures, threats, promotion races, overload pressure
counterplay_liabilities: opponent checks, recaptures, passed-pawn races, queen threats, perpetual-risk motifs
```

Use signed accounting:

```text
net_initiative = assets - discounted_counterplay
insolvency_gap = softplus(counterplay_liabilities - forcing_assets)
puzzle_logit = MLP([net_initiative, insolvency_gap, phase_context])
```

The bottleneck is the exported ledger, not hidden pooling.

### First Config

```yaml
model:
  name: counterplay_insolvency_ledger
  input_channels: 18
  num_classes: 1
  ledger_dim: 96
  buckets: 12
  discount_heads: 4
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `assets_only` | Tests whether counterplay rejection matters. |
| `liabilities_only` | Tests negative evidence alone. |
| `unsigned_ledger` | Tests signed accounting. |
| `phase_context_off` | Tests whether phase-specific discounting matters. |
| `random_counterplay_side` | Tests side semantics. |

### Falsification

Reject if:

```text
assets_only matches the full ledger
or random_counterplay_side does not hurt
or calibration worsens for near-puzzles at the same puzzle recall
```

## Idea 4: Pinned Mobility Nullspace Network

### Thesis

Many near-puzzles contain apparent defenders that are actually mobile. Many true puzzles contain apparent defenders whose legal or pseudo-legal mobility lies in a nullspace because moving them exposes a king, queen, mate square, or promotion stop.

### Mechanism

Build a defender mobility matrix:

```text
M[piece, defensive_square_or_target]
```

Build pin and x-ray constraints:

```text
C[piece, constraint] = moving this piece violates this line, king safety, or target defense
```

Project mobility through constraints:

```text
M_effective = M - project_C(M)
null_mobility = norm(M - M_effective)
```

Classify from:

```text
pinned_defender_count
null_mobility_by_target
remaining_effective_defense
```

### First Config

```yaml
model:
  name: pinned_mobility_nullspace
  input_channels: 18
  num_classes: 1
  piece_dim: 80
  constraint_dim: 64
  projection_rank: 12
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_projection` | Tests nullspace projection. |
| `random_constraints` | Tests pin/x-ray semantics. |
| `mobility_count_only` | Tests whether legal mobility count is enough. |
| `king_constraints_only` | Tests king-pin dependence. |
| `nonking_targets_only` | Tests queen/promotion/critical-square pins. |

### Falsification

Reject if:

```text
mobility_count_only matches full model
or random_constraints match real constraints
or improvements only occur on positions with obvious absolute pins
```

## Idea 5: Tactical Effective Resistance Network

### Thesis

A cut measures the weakest separation between attacker and target. Effective resistance measures how many redundant routes exist and how well defenders can dissipate pressure across the whole tactical graph.

True puzzles should often have low attacker-to-target resistance and high defender-to-target resistance.

### Mechanism

Build a small typed conductance graph:

```text
nodes: pieces, squares, target zones, blocker states
edges: attacks, defenses, x-rays, adjacency, promotion paths
conductance: learned nonnegative function of current-board facts
```

For target pairs, solve a damped Laplacian system:

```text
R_eff(a, t) = (e_a - e_t)^T (L + eps I)^-1 (e_a - e_t)
```

Pool:

```text
attacker_low_resistance
defender_high_resistance
resistance_gap
target current concentration
```

### First Config

```yaml
model:
  name: tactical_effective_resistance
  input_channels: 18
  num_classes: 1
  node_dim: 96
  edge_dim: 32
  target_pairs: 16
  laplacian_jitter: 0.001
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_laplacian_solve` | Tests resistance bottleneck vs graph pooling. |
| `random_conductance` | Tests learned conductance semantics. |
| `degree_only_graph` | Tests topology vs chess relation types. |
| `attacker_only_resistance` | Tests defender contrast. |
| `cut_value_control` | Separates resistance from min-cut behavior. |

### Falsification

Reject if:

```text
no_laplacian_solve matches full model
or cut_value_control fully explains gains
or degree_only_graph is competitive
```

## Idea 6: Defender Opportunity-Cost Auction Network

### Thesis

A defender can often answer one threat only by abandoning another duty. Static coverage says a resource exists. Opportunity-cost pricing asks what the resource gives up when assigned.

### Mechanism

Create bids:

```text
bid(resource r, obligation o)
opportunity_cost(resource r, o) = value_lost_elsewhere(r, o)
net_bid = bid - opportunity_cost
```

Run a differentiable auction:

```text
price(o), assignment(r -> o), unsold_obligation(o)
```

Classify from:

```text
auction_prices
unsold obligations
high-cost assignments
price spread by target family
```

### First Config

```yaml
model:
  name: defender_opportunity_cost_auction
  input_channels: 18
  num_classes: 1
  resource_dim: 96
  obligation_dim: 96
  auction_steps: 8
  max_resources: 32
  max_obligations: 32
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `zero_opportunity_cost` | Tests cost pricing beyond coverage. |
| `greedy_assignment` | Tests auction dynamics. |
| `random_costs_same_scale` | Tests semantic opportunity costs. |
| `price_pooling_only` | Tests whether assignments matter. |

### Falsification

Reject if:

```text
zero_opportunity_cost matches full model
or random costs perform similarly
or exported prices do not correlate with near-puzzle rejection errors
```

## Idea 7: Role-Counterfactual Necessity Network

### Thesis

Some false positives come from shortcut features: material imbalance, king exposure, or generic pressure. A real tactic should depend on exact role geometry. If safe role-preserving counterfactuals destroy the evidence, the puzzle signal is more credible.

### Mechanism

Generate safe synthetic board views at training and inference:

```text
swap two same-color non-king piece roles when legal-enough for tensor encoding
swap same-type pieces across files
move a non-critical defender to a matched safe square
shuffle low-value own pieces within phase-compatible zones
```

These are not labels. They are contrastive probes.

Compute:

```text
actual_logit
counterfactual_logits
necessity_gap = actual_logit - max(counterfactual_logits)
stability_gap = std(counterfactual_logits)
puzzle_logit = MLP([actual_features, necessity_gap, stability_gap])
```

### First Config

```yaml
model:
  name: role_counterfactual_necessity
  input_channels: 18
  num_classes: 1
  base: simple_cnn
  counterfactual_views: 4
  contrast_weight: 0.2
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `no_counterfactuals` | Tests necessity contrast. |
| `random_illegal_swaps` | Ensures realistic swaps matter. |
| `material_only_swaps` | Tests material shortcut removal. |
| `train_only_no_inference_views` | Tests whether views improve representation or readout. |

### Falsification

Reject if:

```text
no_counterfactuals matches full model
or illegal/random swaps improve equally
or inference-time views make the model too slow for benchmark use
```

## Idea 8: Phase-Specialist Calibration Mixture

### Thesis

The boundary between near-puzzle and true puzzle differs across opening tactics, mating attacks, material tactics, promotion races, and simplified endings. A single global head may overcall near-puzzles in one phase to preserve recall in another.

### Mechanism

Use a shared trunk plus calibrated specialists:

```text
mate_attack_head
material_tactic_head
promotion_race_head
endgame_table_like_head
opening_development_trap_head
quiet_positional_head
```

Gate from safe current-board phase features:

```text
phase_gate = softmax(MLP(material, king_safety, pawn_structure, move_number_plane_if_available))
raw_logit = sum_k phase_gate[k] * head_k(h)
calibrated_logit = temperature_by_phase(raw_logit, phase_gate)
```

Add an auxiliary penalty:

```text
near-puzzle FP by phase should not exceed a configured slack over global near-puzzle FP
```

### First Config

```yaml
model:
  name: phase_specialist_calibration_mixture
  input_channels: 18
  num_classes: 1
  experts: 6
  trunk_channels: 96
  calibration_weight: 0.1
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `single_head` | Tests specialist mixture. |
| `random_phase_gate` | Tests phase semantics. |
| `no_calibration_penalty` | Tests phase FP balancing. |
| `hard_gate` | Tests whether soft mixtures are needed. |
| `expert_dropout` | Tests redundancy and expert collapse. |

### Falsification

Reject if:

```text
single_head matches full mixture
or random_phase_gate is competitive
or gains come only from threshold tuning rather than PR AUC/F1 improvement
```

## Idea 9: Forced-Target Funnel Network

### Thesis

A true puzzle often funnels different tactical symptoms toward the same target: king, queen, pinned defender, promotion square, or overloaded defender. A near-puzzle may have many threats, but they point in different directions.

### Mechanism

Generate a bounded set of current-position candidate tactical actions:

```text
checks
captures
threat-making moves
promotion pushes
blocker removals
deflections
```

For each action, predict a distribution over target identities:

```text
q(target | action, board)
```

Compute funnel statistics:

```text
target_entropy = H(mean_action q)
within_action_confidence = mean max q
target_consensus = max_target mean_action q[target]
funnel_gap = target_consensus - target_entropy
```

### First Config

```yaml
model:
  name: forced_target_funnel
  input_channels: 18
  num_classes: 1
  action_dim: 96
  target_dim: 96
  max_actions: 64
  max_targets: 24
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `action_count_only` | Tests funnel statistics vs number of actions. |
| `random_targets` | Tests target semantics. |
| `no_entropy_features` | Tests collapse/consensus. |
| `captures_only` | Tests whether quiet forcing actions matter. |
| `shuffle_action_target_pairs` | Destroys action-target alignment. |

### Falsification

Reject if:

```text
action_count_only matches full model
or random targets do not hurt
or the model becomes a generic one-ply move-delta pool without target funnel diagnostics
```

## Idea 10: Tactical Subgoal Automaton Network

### Thesis

Many puzzles are short scripts:

```text
force king movement -> overload defender -> expose target
deflect queen -> win pinned piece
remove blocker -> activate mating line
```

Instead of using a full proof tree, learn a small finite automaton over typed tactical subgoal predicates.

### Mechanism

Build subgoal predicate streams:

```text
check_pressure
king_escape_reduced
defender_overloaded
blocker_removable
target_unprotected
promotion_stop_missing
counterplay_suppressed
```

Run differentiable automata:

```text
state_{t+1} = soft_transition(state_t, predicate_vector_t)
accept_score = final_accepting_mass
```

The time index is not game-tree ply. It is a fixed learned ordering over subgoal tests.

### First Config

```yaml
model:
  name: tactical_subgoal_automaton
  input_channels: 18
  num_classes: 1
  predicate_dim: 64
  automata: 8
  states_per_automaton: 6
  steps: 5
```

### Required Ablations

| Ablation | Purpose |
|---|---|
| `bag_of_predicates` | Tests ordered automaton structure. |
| `random_transition_matrix` | Tests learned transitions. |
| `single_automaton` | Tests motif diversity. |
| `predicate_shuffle` | Tests subgoal semantics. |
| `accept_score_only` | Tests whether automaton diagnostics suffice. |

### Falsification

Reject if:

```text
bag_of_predicates matches full automata
or predicate shuffling is harmless
or exported accepting states cannot be mapped to stable tactical motifs
```

## Suggested Promotion Order

Promote only one or two candidates into full registered idea folders at a time:

1. `Defender Timing Schedule Network`: best complement to existing obligation and proof ideas.
2. `Discovered-Ray Switchboard Network`: narrow, cheap, line-tactic focused.
3. `Phase-Specialist Calibration Mixture`: practical fallback if the goal is benchmark improvement over theoretical novelty.

## Packet-Level Falsification

This batch is not useful if all successful candidates reduce to one of these controls:

```text
plain bigger CNN
line-token pooling without the proposed bottleneck
obligation count or material count only
phase/threshold calibration only
generic one-ply move list pooling
```

For any promoted candidate, require the standard `3x2` diagnostic:

```text
rows: source fine labels 0, 1, 2
columns: predicted 0, predicted 1
```

and report near-puzzle false positives at matched source-class-0 false-positive rate where feasible.
