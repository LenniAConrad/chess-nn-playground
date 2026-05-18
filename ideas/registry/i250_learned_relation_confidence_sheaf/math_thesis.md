# Math Thesis

Learned Relation Confidence Sheaf -- i250.

Source packet:
`ideas/research/packets/classic/i250_learned_relation_confidence_sheaf.md`.

Working thesis: i018's typed cellular sheaf already proves that real chess
relation topology is doing real work on `puzzle_binary` (`-0.0424` PR-AUC
under degree-preserving topology scrambling). The remaining degree of
freedom that i018 cannot express is *which specific edges inside each
relation family matter on this board*. i250 adds that degree of freedom by
attaching a board-only, normalized edge-confidence multiplier to the
already-active i018 edges, and proves the rest of the sheaf math
unchanged.

## Object

i250 keeps i018's cell complex `K(x)`:

- 64 square 0-cells with stalk `F(v) = R^s` (default `s = 8`);
- 12 typed tactical relations as 1-cells;
- the optional triad 2-cell pool.

The relation edge weights are now

```text
w^{(l)}_{b,r,u,v}(x) = M_{b,r,u,v}(x) * g_r^{(l)} * alpha_hat_{b,r,u,v}(x)
```

where `M` is i018's exact deterministic relation mask, `g_r^{(l)}` is
i018's existing relation-level scalar gate at layer `l`, and `alpha_hat`
is the new normalized per-edge confidence.

## Confidence head

For every active edge `(u, v, r)` with `M(u, v, r) = 1`, a deterministic
board-only feature vector `phi_r(u, v; x)` is built from `piece_state`,
`occupancy`, `pin_mask`, and the relation in-/out-degrees. A small grouped
MLP `f_{g(r)}` per semantic confidence group `g(r) in {0, ..., 4}` maps
those features (plus a learned relation embedding and low-rank node
context) to a scalar logit. Raw confidence is

```text
raw_{b,r,u,v} = floor + (1 - floor) * sigmoid(f_{g(r)}(phi) + bias_r)
```

with `floor in (0, 0.05]`. Normalization is applied within each relation:

```text
alpha_hat_{b,r,u,v} = raw_{b,r,u,v} / mean_active_{b,r}(raw),
```

where the mean is over the active edges of `M_{b,r,.,.}`. By construction
`mean_active_{b,r}(alpha_hat) = 1`, so the confidence head redistributes
mass inside each relation rather than absorbing it into the global gate.

## Sheaf Laplacian (unchanged form)

The signed typed coboundary is

```text
(delta_{rho, w}^{(l)} z)_{b,r,u,v}
  = sqrt(w^{(l)}_{b,r,u,v}) *
    (rho_dst^{(l)}[r] z_v^{(l)} - sigma_r * rho_src^{(l)}[r] z_u^{(l)}).
```

Since `alpha_hat >= floor > 0` on active edges and `M, g_r >= 0`, the
weights `w^{(l)}_{b,r,u,v}` are nonnegative. The sheaf Laplacian

```text
L_{rho, w}^{(l)}(x) = (delta_{rho, w}^{(l)})^T delta_{rho, w}^{(l)}
```

is therefore symmetric positive semidefinite, exactly as in i018. The
bounded heat step

```text
z^{(l+1)} = z^{(l)} - eta_l D_l^{-1} L_{rho, w}^{(l)}(x) z^{(l)}
```

uses the same per-relation degree normalization and learned but clipped
`eta_l` as i018.

## Identity at zero-init

The output linear of every group MLP is zero-initialized, and the relation
embedding is zero-initialized too. So at init, `f_{g(r)}(phi) + bias_r = 0`
everywhere, `raw = floor + (1 - floor) * 0.5 = (floor + 1) / 2` is
constant, and therefore

```text
alpha_hat = raw / raw = 1
```

on every active edge. Substituting `alpha_hat = 1` gives back i018's
exact sheaf weight `w = M * g_r`. The forward and gradient of i250 at
init match i018 up to floating-point reduction order on shared weights;
the observed max logit difference is about `6e-8` on a 4-sample CPU
batch.

## Hypothesis

If i018's typed topology is the right object but its uniform-within-
relation edge weighting is suboptimal, then a small grouped confidence
head with relation-wise normalization will improve mean test PR-AUC or
reduce matched-recall near-puzzle false positives without changing the
parent topology, the trainer, or the benchmark contract.

## Falsifiers

The required falsifiers, in order of priority:

1. **Degree-preserving topology scramble.** Inherited from i018; reuse
   `scramble_relations: true`. If the drop is small, reject the family.
2. **Flat confidence.** Set `flat_confidence: true` to force
   `alpha_hat = M`. If the matched-seed test PR-AUC is within seed noise
   of full i250, the confidence head is unnecessary.
3. **Confidence permutation.** Keep the raw confidence values but shuffle
   them across active edges within each `(batch, relation)` plane. If the
   matched-seed test PR-AUC is within seed noise of full i250, the learned
   weights are not edge-specific in a useful way.
4. **Normalization off.** Set
   `normalize_confidence_within_relation: false`. If results improve only
   when normalization is off, the head is duplicating the global gate.
5. **Confidence in readout only.** Diffuse with raw i018 masks but feed
   the confidence summaries into the head. If equal to full i250, the
   sheaf weighting claim is weak.

## Decision rule

Treat i250 as a meaningful improvement over i018 only if at least one of
the following holds across seeds 42, 43, 44 at base scale, with all other
hyperparameters matched:

- `+0.003` absolute mean test PR-AUC over i018, or
- a `>=1%` absolute reduction in near-puzzle false positives at
  validation-derived recall `0.80` or `0.85`, without compensating
  regressions on puzzle recall or precision.

Falsifier 1 should still drop test PR-AUC by `>= 0.02` on i250; if the
drop is below `0.01`, the typed-topology claim has decayed and the family
must be re-examined.
