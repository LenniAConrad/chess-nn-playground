# Ablations — a016 BT4 Primitive Mixer (legal_edge_compile_scatter)

The BT4 tower shell is held constant across this entire family
(`a001..a040`). The single variable is `model.mixer`, so the ablations
for any individual a### idea are the *cross-idea comparisons* against
the conv and attention baselines plus a small set of mixer-level
switches.

## Cross-idea baselines (mandatory)

| Comparator | Where |
|---|---|
| `mixer: conv` | sibling BT4 baseline (lc0_bt4-style 3x3 conv block) |
| `mixer: attention` | sibling BT4 baseline (multi-head self-attention) |

Both run under an identical config (tower depth/width, optimizer, data,
seeds). Promote `legal_edge_compile_scatter` as a spatial mixer only if
it improves aggregate puzzle_binary metrics or a targeted CRTK tactical
slice over both baselines.

## Mixer-level switches

The geometric typed-edge adjacency is the load-bearing rule signal.
`scripts/ideas/scaffold_bt4_primitive_mixers.py` exercises these
hypotheses by editing the mixer module rather than the idea folder:

| Switch | What it tests |
|---|---|
| `random_typed_edges` | Replace the move-pattern adjacency with a random mask of identical density. Tests whether the chess-rule geometry is load-bearing or whether per-edge gating alone matters. |
| `no_edge_gate` | Hold `g_{r,i,j} = A_r(i,j)`, dropping the σ-gate. Tests whether the feature-conditioned gate carries lift beyond the rule mask. |
| `shared_type_weight` | Collapse the six per-type `W_r` into one shared linear. Tests whether per-type weight tying matters in the mixer setting. |

## Falsification criteria

Drop `legal_edge_compile_scatter` as a BT4 mixer if any of the
following hold under the canonical config:

- `mixer: conv` matches or beats it on aggregate puzzle_binary metrics
  *and* on the CRTK tactical / high-difficulty slices (see
  `report_template.md`).
- `random_typed_edges` matches the default (geometry was noise).
- `no_edge_gate` matches the default (gate was noise; equivalent to a
  cheaper typed scatter).
- Wall-clock per epoch exceeds 2x the `mixer: conv` baseline at equal
  paper-grade reliability tier.
