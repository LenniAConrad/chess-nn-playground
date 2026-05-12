# Math Thesis

Counterfactual Defender Dropout Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `4`.

Working thesis: A near-puzzle is *superficially tactical* — there
is some pressure on the position, but no single role is critical.
A true puzzle hinges on a small set of *causally critical*
participants: an overloaded defender, a pinning slider, the one
escape square the king has, the single blocker on a discovered-
attack line. Concretely, write the side-to-move's puzzle predictor
as a function of the board

```
f(board) ≈ p(puzzle | board).
```

For each *typed deletion* operator `do(remove role R, square s)` —
removing the defender at `s`, removing the attacker at `s`,
plugging the king-escape square `s`, removing the single blocker
on a slider-to-king ray at `s` — the **counterfactual delta** is

```
δ(R, s) = f(board) − f(do(remove R, s)(board)).
```

The thesis says: on a true puzzle, the distribution of `|δ|` over
typed roles is *sharply asymmetric*: a few defender / blocker
deletions move the prediction strongly, while attacker deletions
and king-escape pluggings do not — i.e. removing a defender that
participates in a pin or overload makes the puzzle no longer a
puzzle, while removing an attacker or any random escape square
does not change the answer the same way. On a near-puzzle, the
typed deletions produce a flat or symmetric `|δ|` profile because
there is no single overloaded participant. Formally, the
puzzle-relevant signal is

```
asymmetry(board)
  = topk_mean_{s in defender candidates}|δ(defender, s)|
  − topk_mean_{s in attacker candidates}|δ(attacker, s)|.
```

with related contrasts

```
defender − king_escape, defender − blocker,
H(|δ| / sum |δ|), max|δ|, mean|δ|
```

and per-kind counts of valid candidates. Asymmetry, the per-kind
top-k means, the entropy of `|δ|` over valid candidates, and the
mask-count features form a 13-dimensional **counterfactual
evidence vector** that supplements the trunk's pooled board
features.

## Differentiable Implementation

The model approximates `f(do(remove R, s)(board))` without
re-running a full forward pass per square. Instead:

1. The trunk produces per-square features `h_s ∈ R^{channels}`.
2. For each candidate mask `m_k ∈ {0, 1}^{64}` selecting one role
   on a small set of squares (top-k from the deterministic
   defender / attacker / king-escape / blocker score fields), the
   `InterventionHead` reads the **deletion footprint** features

   ```
   local_mean   = sum_{s in m_k} h_s / |m_k|,
   retained_mean = sum_{s not in m_k} h_s / (64 − |m_k|),
   abs_gap       = |local_mean − retained_mean|,
   product       = local_mean ⊙ retained_mean,
   type_emb(role(m_k)), score(m_k), valid(m_k),
   global context.
   ```

   and predicts a counterfactual delta `δ_k ∈ R`. The dependence on
   `local_mean` and `retained_mean` is the model's smooth
   surrogate for "how much would `f` change if I deleted these
   squares from the position?"
3. The 13 counterfactual evidence scalars are pooled from
   `δ_k` and fed to a small `correction` MLP that produces an
   additive correction to the baseline logit:

   ```
   logits = base_head(context) + correction(eviden(δ)).
   ```

This factorisation lets the markdown thesis enter the head: only
true puzzles can produce a strongly positive `asymmetry` and a
non-uniform `|δ|` distribution, so only true puzzles can move the
correction term substantially away from zero.

## What This Buys

- **Typed counterfactual signal.** Every intervention is grounded
  in a specific role label (defender / attacker / king-escape /
  ray-blocker) computed from the closed-form board geometry, not
  in arbitrary pixel masking.
- **Asymmetry as a discriminant.** The defender − attacker contrast
  is the explicit puzzle vs. near-puzzle discriminant, exactly as
  the working thesis predicts.
- **Ablation control.** The `random_masks` ablation replaces typed
  scores with a fixed random permutation; the asymmetry signal
  should collapse. The `no_intervention_head` ablation forces
  `δ = 0`, so the correction head must collapse to its base bias;
  the network falls back to the trunk-only baseline. The
  `defenders_only` ablation tests whether defender deletions
  alone carry the signal.
- **Reporting.** Per-kind sensitivities, the asymmetry, the entropy
  of `|δ|`, and per-kind valid-mask counts are exposed as
  diagnostics so prediction artifacts record *which* dropout role
  produced the puzzle vote, not just whether the logit was high.
