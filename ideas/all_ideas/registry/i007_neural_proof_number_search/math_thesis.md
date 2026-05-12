# Mathematical Thesis

## Actual Task And Labels

The first target is the corrected puzzle-binary benchmark:

```text
source class 0: known non-puzzle / random position -> y = 0
source class 1: verified near-puzzle / hard negative -> y = 0
source class 2: verified puzzle -> y = 1
```

The model emits one puzzle logit. Fine source labels are used only for diagnostics, especially the 3x2 matrix.

Allowed inputs:

- current-board tensor
- deterministic pseudo-legal move tree generated from the current board
- rule-derived move descriptors
- latent board states produced by the model

Forbidden inputs:

- Stockfish scores
- Stockfish PVs or best moves
- node counts
- mate scores
- verification metadata
- source labels
- source file identity
- future game outcomes

## Baseline And Weakness

Closest registered ideas:

- `i002_response_minimax_classifier`: one-ply action/reply bottleneck.
- `i004_puzzle_obligation_flow_network`: static defensive obligation allocation.
- `i006_proof_core_set_verifier`: sparse current-position proof core.

Overlap: this idea also uses legal move structure and proof-like evidence.

Difference: this idea builds a bounded multi-ply AND/OR proof tree and learns differentiable proof/disproof numbers. The central hypothesis is that true puzzlehood is better detected by existence of a short forcing continuation than by one-ply response summaries or static current-board witnesses.

## Definitions

Let `x` be a chess position. Construct a deterministic bounded game tree:

```text
T_B,D(x): pseudo-legal tree with beam width B and depth D
```

Nodes alternate:

```text
OR nodes: side-to-move can choose a candidate forcing move
AND nodes: opponent can choose a candidate defensive reply
```

Each node has a latent state:

```text
z_v = latent_state(v)
```

A move transition is learned from parent state and rule move descriptor:

```text
z_child = transition(z_parent, move_descriptor, local_board_delta)
```

Each leaf predicts proof and disproof costs:

```text
p_leaf = softplus(proof_head(z_leaf))
d_leaf = softplus(disproof_head(z_leaf))
```

Internal nodes aggregate with differentiable proof-number rules:

```text
OR proof:      p_v = softmin_child(p_child)
OR disproof:   d_v = softsum_child(d_child)
AND proof:     p_v = softsum_child(p_child)
AND disproof:  d_v = softmin_child(d_child)
```

Final puzzle logit:

```text
f(x) = w_1 * (-p_root) + w_2 * d_root + w_3 * proof_disproof_gap + residual_context
```

The residual context must be bounded so the search path cannot be ignored.

## Assumptions

- Many verified puzzles have short forcing continuations.
- Verified near-puzzles often fail because at least one defensive reply breaks the proof.
- A learned proof/disproof cost can be trained from binary labels even without engine move supervision.
- A small beam can include enough forcing candidates if move descriptors are tactically biased by legal rules, not engine scores.

## Claim

Hypothesis: a neural proof-number search network should beat static and one-ply baselines on puzzle-binary PR AUC and near-puzzle false-positive rate, because it directly tests the existence of a short forcing proof and the absence of cheap disproof replies.

## Mechanism

A true puzzle should have:

```text
low proof cost at at least one OR branch
high disproof cost across opponent AND replies
large proof-disproof gap
```

A near-puzzle may have an attractive first move, but at an AND node at least one defensive reply should produce low disproof cost. The architecture encodes this difference as a structural aggregation rule, not only as learned pooling.

## Proof Sketch

What can be reasoned about:

- The aggregation is permutation-invariant over move order after candidate scoring.
- OR/AND proof-number rules represent the logical asymmetry of "there exists a move" versus "all replies fail."
- The model is forbidden from using engine outputs, so improvements must come from learned structure and labels.
- Ablations can isolate depth, beam, move legality, and proof-number aggregation.

This does not prove the benchmark labels are exactly short tactical proofs. It makes that hypothesis directly testable.

## Not Proven

- That binary puzzle labels are sufficient to learn useful proof/disproof costs.
- That depth 2-3 is enough.
- That pseudo-legal beams include the key forcing move.
- That the transition model accurately represents board consequences.
- That the model will not learn shallow move-count shortcuts.

## Counterexamples

- Quiet puzzles requiring long maneuvering.
- Defensive resources that require deeper calculation than the tree depth.
- Positions where a legal-looking pseudo-move is illegal due to check constraints if pseudo-legal generation is too loose.
- Puzzles whose signal is theme metadata rather than position structure.

## Falsification Test

Compare against:

- current BT4 benchmark
- `i002` response-minimax one-ply version
- same trunk with no tree
- same tree with mean pooling instead of proof-number aggregation

Revise or reject if:

```text
test PR AUC <= 0.82
or near-puzzle -> puzzle false-positive rate is not below 0.20
or proof-number aggregation does not beat mean tree pooling
or depth 1 matches depth 3
```

This is intentionally ambitious. If it cannot beat the BT4 baseline meaningfully, the project should probably prioritize cheaper static architectures first.

