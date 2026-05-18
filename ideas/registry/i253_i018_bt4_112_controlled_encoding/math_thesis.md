# Math Thesis

i253 keeps the i018 oriented-tactical-sheaf object exactly and asks one
controlled question about the input encoding: does the current repo's
112-channel BT4-style encoding add enough usable board-state signal to
strengthen i018, after the exact 12 mover-oriented tactical relation masks
are kept fixed.

## Inherited Object

The cell complex, stalks, sheaf restriction maps, signs, gates, heat step,
triad-defect pool, and readout are inherited from i018:

- 64 square 0-cells;
- 12 typed tactical relations `M_r` (attacker/defender, king-zone, slider
  rays, knight, oriented pawn, pin candidate);
- per-relation source/target restriction matrices `rho_src[r]`,
  `rho_dst[r]` of size `(stalk_dim, stalk_dim)`;
- fixed signs `sigma_r`;
- bounded gates `g_r` and heat step `eta`;
- the same triad-defect and readout diagnostics.

## Controlled Encoding Pathway

Both encodings are routed through a shared 112-channel raw pathway:

```text
square_raw = pad_to_112(simple_18_canonical)   if encoding == simple_18
square_raw = lc0_bt4_112                       if encoding == lc0_bt4_112
piece_state = exact mover-relative pieces      for both encodings
```

The piece-state path is exact for both encodings, so relation construction
is encoding-aware but never depends on a learned probe. The padded
simple_18 raw branch is a control trick: with `pad_to_112` the input
projection matrix is the same size on both encodings and the parameter
count is matched.

## Three Relation Modes

Let `M_r` be the exact i018 relation mask for relation `r`, `T_r` the
relation-specific geometric template superset (e.g. the full rook ray
template for the rook relation, or the bishop+rook template for queen),
`C_r(square_raw)` the learned per-edge confidence logit, and
`A_r(square_raw)` the learned augmentation logit.

```text
exact:        W_r = M_r
confidence:   W_r = M_r * sigmoid(C_r)
hybrid:       W_r = clamp(M_r * sigmoid(C_r) + lambda * T_r * sigmoid(A_r), 0, 1)
```

with `lambda = 0.25` by default. The exact-only mode is the hard control;
confidence learns edge importance on exact support; hybrid adds a bounded
residual on a fixed geometric superset and cannot invent arbitrary edges.

The relation-confidence head is intentionally low-rank: per-square pre-
projection `f_i in R^16`, then per-relation source code `S_r[i] in R^8`
and target code `D_r[j] in R^8`, with logit
`<S_r[i], D_r[j]> + bias_r`. This is weaker than a free dense graph by
construction, which is exactly what the research markdown requires.

## Falsifiers

- **Relation scramble.** `M_r -> permute_columns(M_r)` per
  (batch, relation), preserving per-source out-degree. In hybrid mode
  `T_r` is scrambled the same way so the comparison is apples-to-apples.
  If the scrambled run matches the intact run, the chess geometry is not
  load-bearing for the score - that contradicts the i018 thesis.
- **Augmentation-only.** Hybrid mode with `M_r` dropped:
  `W_r = clamp(lambda * T_r * sigmoid(A_r), 0, 1)`. If this row matches
  the intact hybrid, the learned augmentation is doing the work that the
  exact masks were supposed to do, and the present thesis is weakened.

## Equivalence Claim with i018

When `relation_mode = exact`, `encoding = simple_18`, and the raw input
projection in i018 is widened to 112 channels (zero-padded weights), the
forward computation of i253 should reduce to i018's forward computation
up to floating-point reduction order. In practice the test suite spot-
checks shape and finiteness; full bit-equivalence is not the contract
here because the input projection has different weight initialisation
(112-channel Linear vs 18-channel Linear).

## Expected Effect Size

Following the research markdown, the most likely outcome is a small BT4
effect, not a large one. The decision rule mirrors the repo's promotion
logic: BT4 earns a real encoding win only if, against the matched
simple18 row of the same variant, it improves mean PR-AUC by at least
about `+0.003` or reduces near-puzzle false positives at matched recall
by at least `1%`, without causing obvious regressions on hard, equal,
endgame, mate-in-1, promotion, or underpromotion slices.

## Failure Modes That Drop i253

- The relation scramble does *not* collapse PR-AUC in any encoding -
  i018's load-bearing-geometry result has not transferred to this
  controlled setup; investigate before publishing.
- The augmentation-only hybrid row beats the intact hybrid - the
  augmentation template is doing too much of the work and the exact
  masks are decorative; tighten `lambda` or shrink `T_r`.
- A BT4 row beats simple18 only because the relation head is consuming
  metadata-like content of the BT4 aux planes (e.g. reserved zeros that
  encode something the exporter did not advertise); confirm by zeroing
  out the dead BT4 planes (dead-plane control row) and re-running.
