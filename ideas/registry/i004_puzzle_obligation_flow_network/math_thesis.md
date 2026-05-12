# Mathematical Thesis

## Actual Task And Labels

The first target is the corrected puzzle-binary benchmark:

```text
source class 0: known non-puzzle / random position -> y = 0
source class 1: verified near-puzzle / hard negative -> y = 0
source class 2: verified puzzle -> y = 1
```

The model emits one puzzle logit. The three source classes are used for reporting the 3x2 diagnostic matrix, not as inference inputs.

Allowed inputs:

- current-board tensor
- legal rule-derived current-board facts
- deterministic candidate obligations and resources derived from the current board

Forbidden inputs:

- Stockfish scores
- principal variations
- node counts
- mate scores
- verification metadata
- source labels
- source file identity
- unresolved candidate status
- future game outcomes

## Baseline And Weakness

Closest registered ideas:

- `i001_chess_operator_basis_classifier`: exposes chess relation operators.
- `i002_response_minimax_classifier`: uses a learned action/reply minimax bottleneck.
- `i003_factor_agreement_classifier`: requires multiple views to agree.

Closest research-packet ideas:

- Hall-defect / defender-exhaustion style packets.
- Safe-reply and disproof-ledger packets.

Overlap: this idea also models defensive failure and resource insufficiency.

Difference: this idea makes the main representation a typed differentiable allocation problem between obligations and defensive resources. The logit is driven by primal residuals and dual prices of unsatisfied obligations, not by explicit move-reply scoring, static Hall summaries, generic agreement, or plain negative evidence heads.

## Definitions

Let `x` be a chess position. Build two finite sets from the current board:

```text
O(x) = {o_i}: tactical obligations the defending side may need to satisfy
R(x) = {r_j}: defensive resources available to satisfy obligations
```

Examples of obligation candidates:

- protect king escape square
- answer check or latent checking line
- defend high-value target
- block or capture slider line
- stop promotion or back-rank threat
- preserve pinned defender

Examples of resource candidates:

- king move
- capture attacker
- interpose blocker
- recapture target
- move defender
- counter-threat resource

Each obligation and resource receives an embedding:

```text
o_i = obligation_encoder(board_context, candidate_i)
r_j = resource_encoder(board_context, candidate_j)
```

Learn demand, capacity, and compatibility:

```text
d_i = softplus(w_d^T o_i)
c_j = softplus(w_c^T r_j)
a_ij = compatibility(o_i, r_j)
```

Define a soft allocation matrix:

```text
P_ij >= 0
sum_j P_ij <= d_i
sum_i P_ij <= c_j
```

Use a differentiable Sinkhorn, auction, or unrolled primal-dual solver to estimate allocation. Residual obligation:

```text
u_i = relu(d_i - sum_j P_ij)
```

Final puzzle evidence:

```text
flow_residual = pool_i [u_i, dual_price_i, obligation_type_i]
puzzle_logit = MLP([board_context, flow_residual])
```

## Assumptions

- Verified puzzles more often contain over-constrained defensive obligations than verified near-puzzles.
- Near-puzzles can have high pressure but still enough defensive resources.
- Deterministic current-board candidates include enough obligations/resources to expose that difference.
- The learned compatibility function can discover resource-obligation matching from binary supervision.

## Claim

Hypothesis: a typed obligation-resource flow bottleneck should reduce the near-puzzle false-positive rate relative to a similarly sized CNN or BT4-style trunk, because it separates "there is pressure" from "the pressure cannot be defended."

## Mechanism

The model cannot classify only from diffuse tactical texture. It must estimate:

```text
what needs to be defended
what can defend it
which resources can cover which obligations
what remains uncovered
```

True puzzles should tend to create high residual obligation or high dual prices. Near-puzzles should often have lower residual because at least one defensive resource covers the apparent threat.

## Proof Sketch

What can be reasoned about:

- The allocation bottleneck is permutation-invariant over obligation and resource candidates.
- If every obligation has enough compatible resource capacity, residuals are low.
- If demands exceed compatible capacities, residuals or dual prices become high.
- Removing compatibility or capacity constraints gives explicit falsification ablations.

This is not a proof that puzzle labels equal infeasible defensive flow. It is a testable mathematical hypothesis about one important source of puzzle signal.

## Not Proven

- That all puzzle positions require detectable over-constrained defensive resources.
- That the candidate generator captures the correct obligations.
- That binary labels alone are enough to learn valid compatibility.
- That the model will not overfit to material or king-safety shortcuts.

## Counterexamples

- Quiet puzzle-like positions where the tactic is a long strategic zugzwang.
- Endgame puzzles with underpromotion or tablebase-like precision not captured by current candidates.
- Sacrificial attacks where defensive resources exist locally but fail after deeper continuation.
- Labels containing source artifacts unrelated to tactical obligation flow.

## Falsification Test

Run against a size-matched CNN and the current BT4 baseline on the same `puzzle_binary` split.

Revise or reject if:

```text
near-puzzle -> puzzle false-positive rate is not at least 0.03 absolute lower than size-matched CNN
and test PR AUC does not improve by at least 0.015
```

Reject the flow hypothesis if:

```text
shuffled compatibility edges perform the same as legal compatibility edges
or removing capacity constraints performs the same as full flow
or residual obligation is not higher on false positives than true negatives
```

