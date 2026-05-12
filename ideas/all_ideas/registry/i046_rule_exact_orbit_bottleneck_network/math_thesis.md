# Math Thesis

Rule-Exact Orbit Bottleneck Network (idea i046).

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0750_tuesday_los_angeles_orbit_bottleneck.md`.

## Working thesis

A chess position should remain equally puzzle-like under the *exact*
color-flip automorphism of chess rules. A puzzle classifier whose
predictions are obtained by averaging a shared neural encoder over
that rule-exact orbit must therefore be invariant to color-perspective
artifacts by construction, while preserving the tactical geometry of
the position.

## Spaces and group action

- Input: `x in R^{C x 8 x 8}` (`C = 18` for `simple_18`).
- Binary target `Y in {0, 1}` from the project puzzle-binary contract
  (fine labels `{1, 2} -> 1`, fine label `0 -> 0`).
- Group `G = {e, kappa}` of order two acting on `simple_18` boards
  through the deterministic chess color-flip:

  ```
  kappa(x) = (rank_mirror . swap(white_pieces, black_pieces) .
              complement(side_to_move) . swap(KQkq, kqKQ) .
              rank_mirror(en_passant)) (x).
  ```

  This is the standard chess "color flipping" operation; it is exact
  in the sense that it leaves the rules of chess (and therefore the
  ground-truth label `Y`) invariant. The implementation lives in
  `Simple18ColorFlipAdapter.color_flip`.

## Predictive object

For any per-view classifier
`f_theta : R^{C x 8 x 8} -> R^{num_classes}`, the rule-exact orbit
predictor is the Reynolds projection

```
R_G[f_theta](x) = (1 / |G|) * sum_{g in G} f_theta(g . x).
```

The model emits the binary puzzle logit through `pool_mode`:

```
probability_mean (default):
    p_g(x)     = sigmoid(f_theta(g . x))         for g in G,
    p_bar(x)   = (1/|G|) * sum_g p_g(x),
    logits(x)  = logit(p_bar(x)).

logit_mean:
    logits(x) = (1/|G|) * sum_g f_theta(g . x).

latent_mean (latent Reynolds):
    z_bar(x)  = (1/|G|) * sum_g h_theta(g . x),
    logits(x) = classifier(z_bar(x)).
```

Probability-mean pooling is the canonical Reynolds projection for the
binary contract; logit-mean and latent-mean are diagnostic ablations
the model exposes through its config without changing parameter count.

## Invariance theorem

For `pool_mode in {probability_mean, logit_mean, latent_mean}` the
pooled logit satisfies

```
logits(g . x) = logits(x)        for every g in G.
```

Proof sketch: each pooling is a symmetric average over a group action,
so re-indexing `g' = g . g_0` for any fixed `g_0 in G` permutes the
sum without changing it; the identical argument applies to the
latent-mean variant.

## Hypotheses and falsifiers

What is proven mathematically: the model's pooled output is exactly
invariant under `kappa` for every choice of `pool_mode`.

What remains hypothesised: that the binary puzzle-likeness target is
itself approximately invariant under `kappa` and that exposing this
invariance to the architecture improves generalization.

Central falsification ablation (specified by the research packet):
replace `kappa` with the rank-flip-only transform
`rank_flip_no_color`, which preserves shape, material counts,
side-to-move marginal, and compute, but does *not* swap colors,
side-to-move, castling rights, or en-passant rank. This is wired into
the same model class as `orbit_group="rank_flip_no_color"` and shares
all parameters with `orbit_group="color_flip"`. If the falsifier
matches the rule-exact orbit pooler on the puzzle-binary metric, the
gain is not evidence for chess-rule invariance.

## Falsification observable

The model exposes the per-sample symmetry residual

```
symmetry_residual(x) = | sigmoid(f_theta(x)) - sigmoid(f_theta(kappa . x)) |.
```

This is `0` exactly when the per-view classifier is itself invariant
and quantifies how much the orbit averaging is doing on each input.
