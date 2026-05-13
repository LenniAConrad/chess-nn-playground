# Math Thesis

Source: `ideas/research/primitives/codex_04_tail_copula_concordance.md`
(Tail Copula Concordance Primitive, TCC).

## Working thesis

For a position `x`, let `X(x) in R^{N x C}` be a per-site evidence
field with `N = 64` (squares) and `C` learned evidence channels.
TCC converts each channel to soft ranks (Sklar copula representation):

```
u_{n,c} = soft_rank_uniform(X_{n,c} over n) in (0, 1]
m_{n,c} = sigmoid((u_{n,c} - q) / tau_tail)
```

`m_{n,c}` is a soft tail-membership indicator. The directional
tail-dependence estimate is:

```
lambda_{c -> d} = sum_n m_{n,c} m_{n,d} / (eps + sum_n m_{n,c})
```

Symmetric concordance is:

```
Lambda_{c,d} = sqrt(lambda_{c -> d} * lambda_{d -> c})
```

The primitive returns the concordance matrix, pooled tail mean / max,
per-site tail mass (the soft tactical hotspot map), and per-channel
tail mass.

The architecture-level claim is additive:

```
final_logit(x) = i193_trunk(x) + gate(x) * delta(x)
```

with `delta(x), gate(x)` MLPs over the flattened concordance matrix,
channel tail mass, trunk pool feature, and pooled tail mean / max.

## Why this matters

The per-class puzzle_binary benchmark exposes a stable failure on hard
and very-hard slices. Two boards can have identical per-channel sorted
values and identical quantiles but very different cross-site
alignment: a real tactic typically has king pressure, exchange swing,
defender overload, and reply danger all spike on the same critical
square, while a near-puzzle of similar marginal texture has those
spikes scattered. Marginal pooling (`i095`-style) cannot see this;
TCC does.

## Falsifier

- Primitive-level: shuffle squares per channel
  (ablation `square_shuffle`) — the marginals are preserved but the
  cross-site alignment dies. Channel-shuffle (`channel_shuffle`)
  destroys the cross-channel structure. The `rank_quantile_only`
  ablation reduces the output to a channel-independent rank pool —
  the matched control versus `i095_rank_quantile_evidence_field_network`.
- Architecture-level: p004 must improve matched-recall near-puzzle FP
  at recall 0.80 by at least 2% over `i095`-style baselines on the
  same parent, without regressing aggregate PR AUC by more than 0.005.

## Composition with other Codex reply primitives

TCC is the only Codex reply primitive that operates over per-square
evidence instead of candidate/reply tokens. It composes naturally
with the rest of the batch by feeding the tail-hotspot map to a later
candidate compiler.
