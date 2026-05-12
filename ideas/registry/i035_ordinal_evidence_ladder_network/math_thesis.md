# Math Thesis

Ordinal Evidence Ladder Network (OEL-Net).

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0711_tuesday_los_angeles_ordinal_evidence_ladder.md`.

## Thesis

The available fine labels carry an ordinal structure: `0 = known
non-puzzle`, `1 = verified near-puzzle`, `2 = verified puzzle`. OEL-Net
treats this as a supervised ladder and forces every prediction through a
one-dimensional **puzzle-potential score** combined with two **ordered
cumulative thresholds**. The same scalar then defines a Dirichlet
**evidence concentration** for selective diagnostics. Under the i035
puzzle-binary contract, fine labels `0` and `1` map to non-puzzle and
fine label `2` maps to puzzle, so the binary event is `Y >= 2` and the
benchmark logit is `q_2(x) = P(Y >= 2 | x)`.

## Setup

Let `x in R^{C x 8 x 8}` be a current-board encoding (`simple_18`, so
`C = 18`). A board trunk produces an embedding `h_theta(x) in R^d` with
`d = embedding_dim`. The head computes a scalar score and a positive
evidence concentration:

```text
s_theta(x)     = w_s . h_theta(x) + b_s
kappa_theta(x) = kappa_min + softplus(w_k . h_theta(x) + b_k)
```

It also keeps three global learned scalars: a center `a_center`, a
positive gap `gap = softplus(a_gap) + epsilon`, and a positive slope
`rho = softplus(a_rho) + epsilon`. The two ordered thresholds are

```text
tau_0 = a_center - gap / 2,   tau_1 = a_center + gap / 2,
```

so `tau_0 < tau_1` by construction. Cumulative survival probabilities
are

```text
ell_1 = rho * (s_theta(x) - tau_0)
ell_2 = rho * (s_theta(x) - tau_1)
q_1   = sigmoid(ell_1) = P_theta(Y >= 1 | x)
q_2   = sigmoid(ell_2) = P_theta(Y >= 2 | x).
```

Fine-label probabilities follow from the ladder:

```text
p_0 = 1 - q_1,   p_1 = q_1 - q_2,   p_2 = q_2.
```

Dirichlet evidence parameters `alpha_j(x) = 1 + kappa_theta(x) * p_j(x)`
give total evidence `S(x) = sum_j alpha_j(x)` and `vacuity = 3 / S(x)`.

## i035 binary contract

The i035 puzzle-binary task is `B = 1[Y == 2]`, so the benchmark posterior
is `P_theta(B = 1 | x) = q_2(x)` and the trainer logit is `ell_2`. The
cumulative parameterization still produces the auxiliary `q_1` head used
for the near-puzzle ordinal supervision and for matched-FPR class-`1`
diagnostics in the report template.

## Proposition: rank-consistent fine-label distribution

For every `x`, `tau_0 < tau_1` and `rho > 0`, so `ell_1 >= ell_2` and the
sigmoid monotonicity gives `q_1 >= q_2`. Therefore

```text
p_0 = 1 - q_1 >= 0,   p_1 = q_1 - q_2 >= 0,   p_2 = q_2 >= 0,
p_0 + p_1 + p_2 = 1.
```

The ordered ladder always yields a valid categorical distribution over
fine labels with no inconsistent rank probabilities (CORN/CORAL-style
rank consistency).

## Proposition: Bayes optimality under scalar-threshold realizability

If the true conditional cumulative probabilities admit a shared
representation

```text
P(Y >= j | X = x) = sigmoid(rho* * (s*(x) - tau*_j)),  j in {1, 2},
                     tau*_0 < tau*_1, rho* > 0,
```

then minimizing the expected cumulative binary cross-entropy loss on
`(Y >= 1)` and `(Y >= 2)` over a sufficiently expressive backbone and
the shared ladder head recovers the Bayes-optimal `q_1` and `q_2` (BCE is
strictly proper). Because the binary benchmark target `B = 1[Y == 2]`
equals the event `(Y >= 2)`, the ladder's `q_2` then equals the
Bayes-optimal binary posterior under the realizability assumption.

## Optimization objective

The trainer optimises the ordinal ladder via cumulative BCE on both
events using the available fine labels as auxiliary targets, plus the
shared puzzle-binary BCE on the benchmark logit:

```text
L = binary_weight * BCE(ell_2, B)
  + ordinal_weight * (BCE(ell_1, 1[Y >= 1]) + lambda2 * BCE(ell_2, 1[Y >= 2]))
  + fine_nll_weight * NLL(p_fine, Y)
  + evidential_weight * EvidentialExpectedCE(alpha, Y).
```

Here `BCE(ell_2, B)` and the second term of the ordinal loss share the
same event under the i035 contract; the duplication is intentional so
that the puzzle-binary trainer's standard `bce_with_logits` loss remains
compatible. Class weighting is `balanced` and is computed from the train
split only.

## Counterexamples where OEL-Net should fail

- Fine label `1` is not a middle band but a separate source-specific
  bucket whose visual statistics are unrelated to `0` and `2`.
- Verified puzzles in the dataset have several disconnected tactical
  modes that no scalar puzzle-potential can simultaneously rank.
- Class `2` differs from class `1` mainly through verification metadata
  not visible in the board tensor.
- The dataset has near-duplicate leakage across splits, so any scoring
  rule looks Bayes-optimal for the wrong reason.

## Self-critique

The strongest objection is that OEL-Net might look like "just a better
loss head" on a CNN trunk. The ablation plan answers this: the
unconstrained 3-class softmax with the same backbone, the binary-only
ablation, the order-permutation ablation, and the fixed-threshold
ablation must each be reported. OEL-Net survives only if rank-consistent
ordinal supervision specifically improves class-`1` matched-FPR
diagnostics without sacrificing class-`2` recall.
