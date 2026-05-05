# Mathematical Thesis

## Actual Task And Labels

First benchmark:

```text
source class 0: known non-puzzle / random position -> y = 0
source class 1: verified near-puzzle / hard negative -> y = 0
source class 2: verified puzzle -> y = 1
```

The model emits one puzzle logit. Source classes are used only for diagnostics and the 3x2 source-class matrix.

Allowed inputs:

- current-board tensor
- deterministic attacker candidate tokens
- deterministic defender candidate tokens
- rule-derived relation features between attacker and defender candidates

Forbidden inputs:

- Stockfish scores, PVs, node counts, mate scores
- engine best moves
- verification metadata
- source labels or source file identity
- future game outcomes

## Baseline And Weakness

Closest registered ideas:

- `i002_response_minimax_classifier`: explicit one-ply action/reply bottleneck.
- `i004_puzzle_obligation_flow_network`: defensive obligation/resource allocation.
- `i007_neural_proof_number_search`: bounded multi-ply proof/disproof tree.

Overlap: this idea also models attacker/defender interaction.

Difference: this idea uses a simultaneous entropy-regularized matrix game over tactical candidates in the current position. It does not enumerate a tree, and it does not solve coverage/flow. The core output is an equilibrium value and exploitability profile.

## Definitions

Let `A(x) = {a_i}` be attacker candidates for the side to move:

- checking motifs
- capture threats
- attacks on high-value targets
- line-opening threats
- pins/skewers/forks
- promotion threats
- mate-net pressure tokens

Let `D(x) = {d_j}` be defender candidates:

- king escape resources
- captures of attacker
- interpositions
- recaptures
- defender moves
- counter-threats
- target reinforcement

Encode:

```text
a_i = attacker_encoder(candidate_i, board_context)
d_j = defender_encoder(candidate_j, board_context)
r_ij = relation_features(a_i, d_j)
```

Learn a payoff matrix:

```text
P_ij = payoff(a_i, d_j, r_ij)
```

Positive payoff means attacker candidate `i` remains tactically valuable after defender candidate `j`. Negative payoff means the defender neutralizes it.

Solve an entropy-regularized zero-sum game:

```text
max_p min_q p^T P q + tau_A H(p) - tau_D H(q)
```

where `p` is the attacker's mixed strategy and `q` is the defender's mixed response.

Diagnostics:

```text
V = p^T P q
attacker_entropy = H(p)
defender_entropy = H(q)
exploitability = best_response_gap(P, p, q)
```

Final logit:

```text
f(x) = MLP([board_context, V, attacker_entropy, defender_entropy, exploitability, top_payoff_stats])
```

## Assumptions

- True puzzles often have attack candidates whose payoff stays high even under good defensive mixing.
- Near-puzzles often have attractive threats that one or more defender candidates neutralize.
- A small current-board candidate set is enough to expose many puzzle/non-puzzle differences.
- Entropy-regularized equilibrium is a better inductive bias than averaging threats or taking only max threat score.

## Claim

Hypothesis: a tactical equilibrium bottleneck should outperform plain CNN/BT4-style static scoring on near-puzzle false positives, because it asks whether threats survive the defender's best available responses rather than whether threats merely exist.

## Mechanism

A near-puzzle can contain a high raw attack score:

```text
max_i attack_score(a_i) is high
```

but still have:

```text
min_j payoff(a_i, d_j) is low
```

The equilibrium layer pressures the model to find robust attacker value. A true puzzle should have a higher equilibrium value or a more favorable exploitability profile.

## Proof Sketch

What can be reasoned about:

- The equilibrium layer is permutation-invariant over candidate ordering.
- It represents attacker/defender asymmetry without engine search.
- It gives clean diagnostics: value, entropy, best responses, exploitability.
- Replacing equilibrium with mean or max pooling directly tests the mechanism.

This is not a proof that all puzzle labels equal high tactical game value. It is a falsifiable model of one important hard-negative distinction.

## Not Proven

- That candidate generation captures the right threats and defenses.
- That binary labels can train meaningful payoff entries.
- That entropy-regularized game value aligns with actual chess tactics.
- That current-board equilibrium is enough for multi-move puzzles.

## Counterexamples

- Long forcing sequences where the key reply appears only after several moves.
- Quiet puzzles not expressible as immediate attacker/defender tension.
- Endgame tablebase-like puzzles.
- Positions where all candidate threats are low-level but search reveals a tactic.

## Falsification Test

Compare against:

- BT4 baseline
- size-matched CNN trunk
- max-threat pooling without defender candidates
- mean payoff pooling without equilibrium
- random defender candidates
- no exploitability diagnostics

Revise or reject if:

```text
test PR AUC <= 0.82
or near-puzzle -> puzzle false-positive rate is not below 0.20
or equilibrium pooling does not beat max/mean pooling
or random defender candidates match legal defender candidates
```

Reject the equilibrium hypothesis if learned game value does not order source classes in the expected direction:

```text
V(puzzle) > V(near-puzzle) > V(random)
```

