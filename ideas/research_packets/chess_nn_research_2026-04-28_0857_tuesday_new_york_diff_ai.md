# Codex Handoff Packet: Differentiable Chess Fact Lattice

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0857_tuesday_new_york_diff_ai.md`
- **Generated:** 2026-04-28 08:57 new_york
- **Project:** `chess-nn-playground`
- **Research item:** 19. Differentiable Abstract Interpretation
- **Short idea name:** Differentiable Chess Fact Lattice, abbreviated **DCFL**
- **Primary goal:** classify puzzlehood from the current board only.
- **Interface:** `board_tensor -> puzzle_logit`
- **Target construction:** `fine_label in {0, 1} -> y = 0`; `fine_label == 2 -> y = 1`.
- **Required diagnostic:** one 3x2 matrix over original fine labels by predicted binary class.
- **Hard exclusions:** no proof trees, no clause resolution, no program induction, no subgoal automata, no move-delta pooling, no legal continuation search, no engine analysis.
- **Forbidden inference inputs:** Stockfish scores, principal variations, node counts, mate scores, best moves, verification metadata, source labels, source file IDs.

## 2. Executive Selection

Build **DCFL**, a neural classifier with an explicit differentiable abstract interpretation bottleneck. The model converts a board tensor into interval-valued chess facts, applies a small fixed number of monotone transfer passes over those facts, then reads out a single puzzle logit from the final abstract state.

The central bet is that many puzzle positions have a static-board signature: compressed tactical tension, unstable king-zone facts, high-value exposure, overloaded local control, or sharp attacker/defender asymmetry. The classifier is not allowed to prove a tactic, search continuations, ask an engine, or inspect best moves. It should instead learn whether the current position has the kind of abstract fact pattern that commonly precedes a puzzle.

The selected implementation is deliberately not a generic CNN with a chess-themed name. It must expose a concrete abstract domain, differentiable join/meet operators, transfer operators that propagate current-board facts, and a widening operator that controls iterative fact growth. The required control is a matched ordinary-pooling model that removes the abstract domain and replaces it with conventional convolution plus global pooling.

## 3. Data Contract

### Inputs

`x: float32[B, C, 8, 8]`

Minimum expected semantic channels:

- 12 piece planes: `{white, black} x {pawn, knight, bishop, rook, queen, king}`.
- Side-to-move plane or two one-hot side-to-move planes, if available in the existing board tensor.

Recommended restriction: the DCFL experiment should consume only piece placement and side-to-move. Drop or mask all metadata channels unless they are already part of the ordinary board representation and are clearly current-board facts. Do not feed source labels, source file IDs, verification fields, engine annotations, best-move fields, PV text, node counts, mate scores, or any fields derived from post-position analysis.

### Outputs

`logit: float32[B, 1]`

No policy head, value head, move head, legal-move tensor, continuation tensor, or auxiliary engine target should be exposed as an output for this experiment.

### Labels

Use only the fine label to form the binary target:

```text
fine_label 0 -> target 0
fine_label 1 -> target 0
fine_label 2 -> target 1
```

Fine labels 0 and 1 are collapsed for the training target but must remain separate for diagnostics.

### Required 3x2 Diagnostic

Report one matrix per split:

```text
rows:    fine_label in [0, 1, 2]
columns: predicted_class in [0, 1]
cell:    count of examples
```

Prediction threshold should be selected on the validation split only, normally `sigmoid(logit) >= 0.5` unless the existing evaluation harness already uses a fixed validation-calibrated threshold.

Also report, outside the 3x2 matrix if desired, mean logit and mean probability per fine-label row. The required matrix itself should remain a simple 3x2 count table.

### Split Hygiene

If duplicate or near-duplicate boards exist, group them by a hash derived only from the board tensor and side-to-move before splitting. Do not group, stratify, weight, or infer from source file ID or source label. Those fields are forbidden as inference inputs and should not become hidden leakage through split construction.

## 4. Abstract Interpretation Research Map

Classical abstract interpretation replaces concrete program states with elements of an abstract domain, then propagates approximate facts through transfer functions. DCFL adapts that idea to a chess board without treating chess as a program to be searched.

The concrete object is a single current board. The abstract state is a product lattice of interval-valued chess facts: piece occupancy, attack pressure, defense pressure, line clearance, king-zone contact, material exposure, and local tension. The transfer functions propagate these facts along board topology: leaper masks, pawn directions, slider rays, ownership channels, and king-zone dilation. Join combines evidence from multiple pieces or rays. Meet computes intersections such as “occupied by a valuable friendly piece” and “attacked by the opponent” and “not well defended.” Widening prevents repeated transfer passes from becoming an unbounded soft message-passing system.

The key distinction from forbidden approaches:

- It does not build proof trees.
- It does not resolve clauses.
- It does not induce programs.
- It does not create subgoals.
- It does not pool move deltas.
- It does not search legal continuations.
- It does not call an engine or consume engine-derived fields.

The only permitted question is: given this board tensor, what abstract current-board facts can be propagated, joined, met, widened, pooled, and mapped to one puzzle logit?

## 5. Candidate Search Trace

Candidate ideas considered inside this packet:

1. **Plain attack-map CNN.** Rejected as too close to ordinary feature engineering. It computes chess facts but does not create an abstract domain with join/meet/widening.
2. **Neural theorem prover over tactical motifs.** Rejected because it drifts toward proof trees and clause-style reasoning.
3. **Future-move vulnerability analyzer.** Rejected because it risks legal continuation search and move-delta pooling.
4. **Differentiable Chess Fact Lattice.** Selected because it is concrete, current-board-only, differentiable, and directly testable against an ordinary-pooling control.

The final handoff is DCFL only.

## 6. Rejected Common Approaches

Do not implement any of the following as the main model or as hidden preprocessing:

- Engine evaluation, Stockfish probing, mate-score filtering, node-count features, PV parsing, or best-move extraction.
- Legal move generation followed by continuation scoring, even if shallow.
- Making candidate moves, undoing moves, and pooling before/after board differences.
- Proof trees, AND/OR tactical trees, SAT-style clause resolution, or symbolic theorem proving.
- Program induction or learned rule programs over board states.
- Subgoal automata such as “first win material, then checkmate.”
- Source-aware shortcuts: file IDs, source labels, verification labels, puzzle collection names, or metadata that identifies how a board entered the dataset.
- A model that predicts fine label 2 by reading any non-board artifact.

A generic CNN baseline is allowed only as the required ordinary-pooling control and must use the same allowed board tensor.

## 7. Mathematical Thesis

Let `B` be the current board tensor and `y in {0,1}` the puzzlehood target. DCFL assumes there exists an abstraction map `alpha` into a lattice `A` such that puzzlehood is more separable in abstract fact space than in raw board space:

```text
alpha(B) = a_0 in A
a_{n+1} = widen(a_n, T(a_n) join a_n)
logit = h(a_K)
```

where:

- `A` is a product of bounded interval domains over current-board chess facts.
- `T` is a monotone differentiable transfer operator over board topology.
- `join` aggregates alternative evidence sources.
- `meet` constructs conjunctive tension facts.
- `widen` stabilizes the finite transfer sequence.
- `h` is a small readout network.

The thesis is not that the model proves a tactic. The thesis is that puzzlehood is correlated with **static abstract tension**: narrow intervals for high-confidence attacks, wider intervals for unresolved defense/control facts, and strong meets between opponent pressure and valuable/king-adjacent occupancy.

A position should receive a high puzzle logit when the final abstract state contains multiple locally consistent signals such as:

```text
opponent controls valuable square
and friendly defense redundancy is low
and king-zone or high-value exposure is high
and line/ray pressure is structurally sharp
```

These are current-board facts only. They do not assert that a tactic is sound after legal play.

## 8. Abstract Domain

### Concrete Fact Universe

Define a concrete fact universe `F` over the current board:

```text
piece(c, p, s)          piece of color c and type p occupies square s
occupied(c, s)          color c occupies square s
empty(s)                no piece occupies square s
attacks(c, p, s)        square s is attacked by color c piece type p under static attack geometry
defends(c, p, s)        square s occupied by c is attacked by another piece of c
ray_clear(d, s, k)      along direction d from origin s, first k squares are clear
king_zone(c, s)         square s is in or near color c king zone
line_exposed(c, s)      color c has a king/valuable-piece line exposure through s
value_mass(c, s)        normalized material value on square s for color c
tension(s)              both colors exert relevant pressure on s
vulnerable(c, s)        color c has valuable occupancy on s under opponent pressure
```

Static attack geometry is allowed. Legal continuation search is not. Slider attack facts are computed from current occupancy and board rays only.

### Abstract Elements

Each fact has a bounded interval truth value:

```text
I = {[l, u] | 0 <= l <= u <= 1}
A = product over f in F of I_f
```

Interpretation:

- `l_f` is lower confidence that fact `f` is true.
- `u_f` is upper confidence that fact `f` may be true.
- Exact deterministic facts can start as `[0,0]` or `[1,1]`.
- Learned relaxed facts may start wider, for example `[sigmoid(q - r), sigmoid(q + r)]` with `r >= 0`.

Partial order uses interval precision:

```text
[l1, u1] <= [l2, u2] iff l2 <= l1 and u1 <= u2
```

So a narrower interval is more precise.

### Domain Components

Use the following abstract tensor groups:

1. **Occupancy intervals**
   - Shape: `[B, 2, 6, 2, 8, 8]`
   - Color, piece type, interval endpoint, rank, file.

2. **Attack intervals**
   - Shape: `[B, 2, 6, 2, 8, 8]`
   - Static attacked squares by color and piece type.

3. **Defense intervals**
   - Shape: `[B, 2, 6, 2, 8, 8]`
   - Friendly pressure on occupied or potentially occupied squares.

4. **Ray-clearance intervals**
   - Shape: `[B, 8, 7, 2, 8, 8]`
   - Direction, distance, interval endpoint, origin square.
   - Can be computed lazily to save memory.

5. **King-zone intervals**
   - Shape: `[B, 2, 2, 8, 8]`
   - Color, endpoint, square.

6. **Exposure/tension intervals**
   - Shape: `[B, F_tension, 2, 8, 8]`
   - Suggested channels: opponent attack mass, friendly defense mass, attack-defense imbalance, value-at-risk, king-zone pressure, line exposure, contested-square pressure, loose-piece pressure.

The final abstract state is the concatenation of these interval groups after transfer and widening. The readout may see both endpoints plus derived width `u - l`.

## 9. Transfer/Join/Widening Operators

### Smooth Join

For probability-like fact values, use noisy-OR as the default differentiable join across independent evidence sources:

```text
join(z1, ..., zn) = 1 - product_i(1 - zi)
```

For interval endpoints:

```text
[l, u] join [l', u'] = [softmin_tau(l, l'), softmax_tau(u, u')]
```

`tau` should be small, for example annealed from `0.25` to `0.05`. In the exact limit this recovers interval union.

### Smooth Meet

Use product t-norm or soft-min for conjunction:

```text
meet_prob(z1, ..., zn) = product_i zi
```

For intervals:

```text
[l, u] meet [l', u'] = [softmax_tau(l, l'), softmin_tau(u, u')]
```

If the soft lower endpoint exceeds the soft upper endpoint, record the positive gap as a separate **conflict/tension channel** before clamping:

```text
conflict = relu(l_meet - u_meet)
l_meet = min(l_meet, u_meet)
```

This preserves information about incompatible abstract facts without breaking tensor shape.

### Base Transfer: Occupancy

Initialize exact occupancy intervals from board piece planes:

```text
occ_l = occ_u = x_piece
empty_l = empty_u = product over piece planes (1 - x_piece)
```

If the existing tensor is not perfectly one-hot, clamp to `[0,1]` and add a consistency penalty during training:

```text
L_board_consistency = mean(abs(sum_piece_planes_per_square - {0 or 1 relaxed target}))
```

Do not repair the board with a chess engine or move generator.

### Leaper Transfer

For kings, knights, and pawns, attacks are fixed masked shifts:

```text
attack_{c,p} = fixed_mask_conv(occ_{c,p}, mask_{p,c})
```

Use fixed kernels for attack geometry. The kernels are not learned. A learned nonnegative calibration gate may rescale each piece-type contribution after the fixed transfer:

```text
gated_attack = sigmoid(beta_{p} + softplus(w_p) * attack_{p})
```

The nonnegative parameterization keeps the transfer monotone.

### Slider Transfer

For bishops, rooks, and queens, compute line-of-sight over current occupancy.

For origin square `s`, direction `d`, and distance `k`:

```text
clear(s,d,k) = product_{j=1 to k-1} (1 - occupied_any(s + j*d))
slider_hit(s,d,k) = occ_slider(s) * clear(s,d,k)
```

The target square receives joined evidence from all origins and valid board directions by static ray geometry only. This is not move search; it does not create a moved board or inspect future legality.

Interval arithmetic:

```text
clear_l = product_j (1 - occ_u_j)
clear_u = product_j (1 - occ_l_j)
hit_l   = occ_l_slider * clear_l
hit_u   = occ_u_slider * clear_u
```

Use log-space products for numerical stability.

### Defense Transfer

Defense is same-color attack pressure over own occupied squares:

```text
defended(c, s) = meet(occupied(c, s), attacks(c, s))
```

Keep piece-type-specific attack facts as well as aggregate attack mass:

```text
attack_mass(c, s) = join_p attacks(c, p, s)
defense_mass(c, s) = meet(occupied(c, s), attack_mass(c, s))
```

### King-Zone Transfer

Find king occupancy from the king planes. Expand by fixed dilation masks:

```text
king_zone(c) = dilate_1(king_occ(c)) join dilate_2(king_occ(c)) * gamma_2
```

`gamma_2` can be a small learned nonnegative scalar. The transfer remains current-board-only.

### Exposure and Tension Transfers

Construct tension channels with differentiable meets:

```text
value_at_risk(c, s) = meet(
  occupied(c, s),
  value_mass(c, s),
  attack_mass(opponent(c), s),
  1 - defense_mass(c, s)
)

king_zone_pressure(c, s) = meet(
  king_zone(c, s),
  attack_mass(opponent(c), s)
)

contested_square(s) = meet(
  attack_mass(white, s),
  attack_mass(black, s)
)

line_exposure(c, s) = meet(
  ray_clear_between_king_and_square(c, s),
  opponent_slider_pressure_on_line(c, s),
  occupied(c, s) or value_mass(c, s)
)
```

Use “line exposure” rather than “pin proof.” It is a static ray relationship, not a legal-move claim.

### Iterative Transfer Schedule

Use a small fixed number of transfer passes, for example `K=3`:

```text
a_0 = alpha(board_tensor)
for t in 0..K-1:
    proposal = T_t(a_t)
    joined   = a_t join proposal
    a_{t+1}  = widen(a_t, joined)
```

The passes allow facts such as attack mass, defense mass, and exposure to build from each other. They are not plies and must not depend on legal move generation.

### Widening

For interval `[l, u]`, define differentiable widening:

```text
widen_l = clamp(softmin_tau(l_old, l_new) - epsilon_t, 0, 1)
widen_u = clamp(softmax_tau(u_old, u_new) + epsilon_t, 0, 1)
```

Recommended schedule:

```text
epsilon_0 = 0.02
epsilon_1 = 0.01
epsilon_2 = 0.00
```

Purpose:

- Prevent the abstract state from pretending to be exact after multiple soft transfers.
- Preserve uncertainty width as a useful signal.
- Keep transfer passes from becoming unconstrained generic message passing.

Optional final narrowing:

```text
a_final = meet(a_K, board_consistency_constraints)
```

Only use constraints derived from the board tensor itself.

## 10. Architecture Tensor Contract

### Model Skeleton

```text
Board tensor x
  -> Factizer alpha
  -> Abstract interval state a0
  -> K differentiable transfer/join/widen passes
  -> Abstract readout tensor [endpoints, widths, conflicts]
  -> small spatial readout
  -> global summary
  -> one puzzle logit
```

### Factizer

Input:

```text
x: [B, C, 8, 8]
```

Output exact or relaxed base intervals:

```text
occ_lu:      [B, 2, 6, 2, 8, 8]
side_to_move:[B, 2]
```

Piece planes should be reshaped deterministically from the input. If there are additional non-forbidden board channels, either ignore them or pass them through a small allowed-board-channel projection that is documented and ablated.

### Abstract Interpreter Module

Inputs:

```text
occ_lu
side_to_move
```

Internal tensors:

```text
attack_lu:       [B, 2, 6, 2, 8, 8]
defense_lu:      [B, 2, 6, 2, 8, 8]
king_zone_lu:    [B, 2, 2, 8, 8]
ray_clear_lu:    [B, 8, 7, 2, 8, 8] or lazy equivalent
tension_lu:      [B, F_tension, 2, 8, 8]
conflict:        [B, F_conflict, 8, 8]
```

Output:

```text
abstract_features: [B, F_abs, 8, 8]
```

where `F_abs` concatenates lower endpoints, upper endpoints, interval widths, and conflict channels.

### Readout

Use a deliberately small readout so the abstract domain does the work:

```text
1x1 conv / monotone projection
GELU or SiLU
3x3 conv, width <= 64
global average pooling + global max pooling
MLP hidden width <= 128
logit [B, 1]
```

The readout may condition on side-to-move by concatenating a broadcast side-to-move plane before pooling.

### Required Ordinary-Pooling Control

Create a matched control named **Pool-Control**:

```text
board tensor x
  -> input projection conv
  -> same number of spatial blocks as DCFL transfer passes
  -> no interval endpoints
  -> no join
  -> no meet
  -> no widening
  -> global average pooling + global max pooling
  -> MLP
  -> one puzzle logit
```

Rules for the control:

- Same input channels as DCFL.
- Same target labels and split.
- Same training objective.
- Parameter count matched within approximately 5 percent by adjusting width.
- No attack maps, no ray facts, no abstract intervals.
- No move generation or engine fields.

The comparison answers the core research question: does the explicit abstract domain improve puzzlehood classification beyond ordinary pooling?

## 11. Training Objective

Primary loss:

```text
L_cls = BCEWithLogitsLoss(logit, target)
```

Use class weighting or balanced sampling if the binary target is imbalanced. Keep the target exactly as specified: fine 0/1 are negative, fine 2 is positive.

Recommended full objective:

```text
L = L_cls
  + lambda_width * mean(width_over_abstract_facts)
  + lambda_conflict * mean(non_tension_conflicts)
  + lambda_mono * monotonicity_penalty
```

Guidance:

- `lambda_width` should be small. Width is a useful signal, not merely an error.
- `lambda_conflict` should not suppress designated tactical tension conflict channels.
- `lambda_mono` penalizes learned transfer weights that violate nonnegative monotone parameterization.

Do not add an engine-value auxiliary loss. Do not predict best move. Do not predict Stockfish score buckets. Do not use mate-score labels.

### Evaluation Metrics

Minimum:

- Binary accuracy.
- AUROC or AUPRC if already used by the project.
- Calibration curve or expected calibration error if cheap.
- Required 3x2 diagnostic matrix.

Required diagnostic implementation:

```python
# Pseudocode only.
prob = sigmoid(logit)
pred = (prob >= threshold).long()
diag = zeros((3, 2), dtype=int)
for f, p in zip(fine_label, pred):
    diag[int(f), int(p)] += 1
```

Report the diagnostic for train, validation, and test if all splits exist. Validation and test are most important.

## 12. Ablations

Run at least these ablations:

| Ablation | Change | Expected interpretation |
|---|---|---|
| Pool-Control | Replace abstract domain with ordinary conv blocks and pooling | Tests whether DCFL is better than ordinary pooling |
| No intervals | Use single scalar fact values instead of `[l,u]` | Tests whether uncertainty width matters |
| No meet channels | Remove vulnerability/tension meets | Tests whether conjunctive abstract facts matter |
| No widening | Replace widening with direct residual update | Tests whether widening stabilizes or regularizes |
| No ray transfer | Remove slider/ray-clearance facts | Tests contribution of line pressure |
| No king-zone transfer | Remove king-zone dilation and pressure | Tests king-adjacent static tension |
| Fixed-only transfers | Disable learned gates and use only deterministic transfers | Tests whether learning is in readout or abstract calibration |
| Shuffled abstract channels | Shuffle abstract channels after transfer during evaluation | Sanity check for learned semantic use |
| Material-only | Keep only occupancy and value_mass channels | Ensures model is not just material imbalance |
| Side-to-move masked | Remove side-to-move channel | Tests whether side-to-move is essential or leaking through distribution |

The mandatory control is **Pool-Control**. Other ablations are secondary but valuable.

## 13. Falsification Criteria

Treat the idea as falsified or not worth prioritizing if any of the following hold after reasonable tuning:

1. **No advantage over ordinary pooling.** DCFL does not beat Pool-Control by a meaningful margin across multiple seeds, for example less than 0.01 AUROC improvement with overlapping confidence intervals.
2. **Fine-label diagnostic fails.** The 3x2 matrix shows that fine label 2 is not separated from fine labels 0 and 1, or that all gains come from miscalibrating fine label 1.
3. **Material-only matches DCFL.** A material/occupancy-only ablation performs essentially the same as the full domain.
4. **Abstract shuffling does not hurt.** Shuffling abstract channels at evaluation leaves performance unchanged, implying the readout is not using fact semantics.
5. **Widening irrelevant and intervals collapse.** All interval widths collapse to zero or one constant and no interval ablation changes performance.
6. **Leakage suspected.** Performance drops sharply after removing metadata, source fields, or split artifacts, or improves when forbidden metadata is accidentally included.
7. **Transfer visualizations are nonsensical.** Attack, defense, ray, and king-zone maps do not correspond to recognizable current-board facts in hand-checked positions.
8. **Generalization failure.** The model performs well only on one source distribution while failing on board-hash-separated or source-blind splits.

A negative result is still useful if it cleanly shows that ordinary pooling is sufficient under the same input constraints.

## 14. Implementation Notes

### Suggested Modules

Use project-appropriate paths after checking the actual repository layout. Suggested names:

```text
models/dcfl.py
models/pool_control.py
features/chess_fact_lattice.py
training/train_puzzlehood_dcfl.py
evaluation/diagnostics.py
```

Do not assume these paths already exist. Adapt to the repository structure without changing the experiment contract.

### Implementation Checklist

1. Add a loader path that returns:

```text
board_tensor, fine_label, binary_target
```

2. Ensure forbidden fields are not included in the batch object passed to the model.
3. Implement fixed chess topology tensors:

```text
knight masks
king masks
pawn attack masks by color
rook directions
bishop directions
queen as rook+bishop directions
board-edge masks
ray index tables
```

4. Implement interval operations as pure tensor functions:

```text
smooth_join_interval
smooth_meet_interval
noisy_or_join
product_meet
widen_interval
```

5. Implement leaper and slider transfers without legal move generation.
6. Add hand tests for static facts:

```text
single knight attacks expected squares
single rook ray stops at blocker
bishop ray stops at blocker
king-zone dilation centered on king
value_at_risk activates for attacked high-value occupancy
```

7. Add grep-style guardrails in tests or CI where possible:

```text
Stockfish
engine
pv
nodes
mate_score
best_move
legal_moves
push(
make_move
source_id
source_label
verification
```

The exact grep terms should be adapted to project naming, but the goal is to catch accidental forbidden inference paths.

### Numerical Stability

- Clamp probabilities to `[1e-5, 1 - 1e-5]` before log-space products.
- Use log-space cumulative sums for slider clearance products.
- Keep transfer pass count small, normally `K=2` or `K=3`.
- Anneal soft join/meet temperature rather than starting near hard max/min.
- Track mean interval width by fact group during training.

### Interpretability Artifacts

For a small validation sample, save or print compact summaries rather than images if the project does not already have visualization support:

```text
top vulnerable squares by value_at_risk
top king_zone_pressure squares
top line_exposure squares
mean attack-defense imbalance by side
final logit and probability
```

These summaries must be derived only from current-board abstract facts.

## 15. Prompt Maintenance

Preserve the following invariants in future prompts and code edits:

- The experiment is `board_tensor -> one puzzle_logit`.
- Fine labels 0 and 1 map to target 0; fine label 2 maps to target 1.
- Always report the 3x2 diagnostic over original fine labels by predicted binary class.
- The model may compute current-board static facts but must not search legal continuations.
- No engine-derived fields are allowed at inference or training input time.
- No source labels, source file IDs, verification metadata, or collection identifiers are allowed as features.
- Do not introduce proof trees, clause resolution, program induction, subgoal automata, move-delta pooling, or engine analysis while “improving” the method.
- Keep the ordinary-pooling control. It is not optional; it is the main sanity check.
- If a future edit adds richer chess facts, classify them as one of:
  - allowed current-board static fact;
  - suspicious metadata/leakage;
  - forbidden continuation or engine-derived fact.
- When uncertain, prefer removing a feature over risking leakage.

