# Math Thesis

Credal Near-Puzzle Evidence Network (idea i045).

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0750_tuesday_los_angeles_credal_evidence.md`.

## Working thesis

Train a binary puzzle-likeness classifier whose output is a Dirichlet
evidence distribution. Verified near-puzzles (fine label `1`) are
treated as *interval-valued* positive targets with deliberately limited
evidence, instead of being collapsed onto the same hard positive target
as verified puzzles (fine label `2`). Hard non-puzzles (fine label `0`)
remain singleton-target negatives.

## Spaces

- Input: `x in R^{C x 8 x 8}` (`C = 18` for `simple_18`; `C = 112` is
  available behind the fail-closed adapter).
- Observed fine label `Z in {0, 1, 2}` and binary task target
  `Y = 1[Z in {1, 2}]`.

## Predictive object

The encoder produces nonnegative evidence
`e_theta(x) in R_+^2`. The Dirichlet predictive over the positive
probability `pi_1 in [0, 1]` is

```
alpha_theta(x) = 1 + e_theta(x)
S_theta(x)     = alpha_0(x) + alpha_1(x)
mu_theta(x)    = alpha_1(x) / S_theta(x)
Pi_theta(.|x)  = Dirichlet(alpha_0(x), alpha_1(x))
```

The predictive Dirichlet *mean* is `(1 - mu, mu)`. The single binary
logit reported to the puzzle-binary trainer is

```
puzzle_logit(x) = log(alpha_1(x) + eps) - log(alpha_0(x) + eps),
sigma(puzzle_logit) = mu_theta(x)
```

so the BCE-with-logits trainer's predicted positive probability equals
the Dirichlet predictive mean. When the head is configured with
`num_classes = 2`, the model returns `log(alpha + eps)` directly so that
`softmax` of those logits equals the Dirichlet mean (this is the form
used by the markdown ablations).

## Credal target sets

For each fine label define a feasible set over `q = (q_0, q_1) in
Delta^1`:

```
C_0      = { (1, 0) }
C_2      = { (0, 1) }
C_1(tau) = { q in Delta^1 : q_1 >= tau }   with default tau = 0.55.
```

The credal projection loss on the Dirichlet mean `m = (1 - mu, mu)` is

```
L_set(z, m) = min_{q in C_z} KL(q || m).
```

For `z in {0, 2}` this reduces to ordinary hard-label NLL up to a
constant. For `z = 1`,

```
L_set(1, m) = 0                                                       if mu >= tau,
L_set(1, m) = tau * log(tau / mu) + (1 - tau) * log((1 - tau)/(1 - mu)) if mu <  tau.
```

Proof sketch: `KL(q||m)` is strictly convex on the binary simplex; if
`m in C_1(tau)` the minimum is at `q = m`, otherwise it lies on the
boundary `q_1 = tau`, yielding the displayed Bernoulli KL.

## Evidence-shaping term

```
L_ev(1, S) = lambda_near * [ max(0, log(S) - log(S_near_max)) ]^2
L_ev(z, S) = lambda_kl   * KL( Dir(alpha_tilde_z) || Dir(1, 1) )   for z in {0, 2}
```

with `lambda_kl` annealed from `0` to `lambda_dirichlet_kl` over the
first `kl_anneal_epochs`. The fine-label-`1` cap forbids the model from
becoming as concentrated on near-puzzles as it does on verified
puzzles; the wrong-evidence KL on hard labels prevents inflated
unsupported alpha.

## Total loss

```
L(theta) = E_{(X, Z)} [ w_Z * ( L_set(Z, m_theta(X)) + L_ev(Z, S_theta(X)) ) ]
```

with balanced fine-label weights `w_Z` by default. The forward model
returns `alpha`, `evidence`, `S`, `mu_pos`, `uncertainty = 2/S` so the
trainer (or a custom CredalEvidenceLoss outside `forward`) can compute
`L_set` and `L_ev` from those auxiliary tensors.

## Hypotheses and falsifiers

What is proven mathematically: the loss geometry; specifically that
`L_set(1, m)` is one-sided in `mu` and zero on the credal interval.

What remains hypothesised: that fine label `1` is genuinely an
uncertainty band in the data distribution; that lower evidence on
near-puzzles improves binary classification.

Central falsification ablation: same backbone, same optimizer,
same balanced weighting, but replace fine-label-`1` credal interval and
evidence cap with ordinary hard-positive BCE.
