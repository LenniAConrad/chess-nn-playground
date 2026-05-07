# Math Thesis

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0813_tuesday_los_angeles_hall_defect.md`.

The thesis is that puzzle-like boards can contain a static overload certificate: a small defender set is responsible for too many attacked assets or king-zone obligations. The model tests this by computing exact Hall-deficiency profiles over a rule-derived defender-obligation relation and letting a small neural head calibrate those frozen certificates.

## Formal Object

For the supported `simple_18` encoding, the adapter decodes current pieces:

```text
P(x) in {0,1}^{2 x 6 x 8 x 8}
```

and the side-to-move scalar. From the current board only, it computes pseudo-legal controls `C_c(p, s)` for each color, local piece slot, and square. Sliding controls stop at blockers. No legal move generation, checkmate oracle, engine score, future board, or source metadata is used.

For each role `r` and stratum `g`, the model builds obligations `O_{r,g}` and selected defenders `D_{r,g}`. An obligation is either an attacked non-king asset or a contested square in a defender king ring. A defender is an own piece that pseudo-legally controls the obligation square, excluding the defended asset itself as its own defender. After deterministic truncation to `D_max` selected defenders, every obligation has a bitmask neighborhood:

```text
N(o) subseteq D_{r,g}
```

## Hall Defect

For each graph, the cardinal Hall defect is:

```text
H_{r,g}(x) = max_{T subseteq D_{r,g}} |{o in O_{r,g}: N(o) subseteq T}| - |T|
```

The weighted profile uses deterministic obligation weights `w(o)`:

```text
H^w_{r,g,lambda}(x)
  = max_T sum_{o: N(o) subseteq T} w(o) - lambda |T|
```

The zeta layer computes these maxima exactly over the truncated defender universe by histogramming obligation neighborhoods and applying a subset zeta transform.

## Proposition

For an unweighted, untruncated defender-obligation bipartite graph, the cardinal quantity above equals the Hall deficiency of the obligation side, equivalently the number of obligations that cannot be matched to distinct defenders in the corresponding transversal matroid.

Proof sketch: for any obligation subset `S`, let `T = N(S)`. Every `o in S` has `N(o) subseteq T`, so `|S| - |N(S)|` is bounded by the displayed maximum. Conversely, for any `T`, let `S_T = {o: N(o) subseteq T}`. Then `N(S_T) subseteq T`, so `|S_T| - |N(S_T)| >= |S_T| - |T|`. Maximizing both sides gives the Hall obstruction.

## Hypothesis Under Test

The classifier should improve when overload structure is visible in the current board: many valuable obligations concentrate on a too-small defender neighborhood. The central falsifier is a degree-matched edge rewire or count-only tokenization. If those controls match the main model, the Hall set-system semantics did not add information beyond counts, material, and attack density.

The repository task contract remains `puzzle_binary`: fine labels `0` and `1` are non-puzzle, and fine label `2` is puzzle. Fine labels and source metadata are diagnostics only and never neural inputs.
