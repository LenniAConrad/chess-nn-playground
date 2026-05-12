# Math Thesis

Soft Majorization Line Sorter

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.

Batch candidate rank: `3`.

Working thesis: On a tactical line, the exact order and dominance of
pieces often matters more than a bag of line pieces. Instead of a ray
automaton or line language model, compute differentiable sorted salience
profiles along ranks, files, diagonals, and anti-diagonals, then
classify from majorization-style inequalities, gaps, and concentration
of those sorted profiles.

## Mathematical objects

- **Salience field.** A learned map `s : (B, 18, 8, 8) -> (B, K, 8, 8)`
  composed of a small CNN trunk and a 1x1 projection to `K` scalar
  salience-head fields. The trunk also exposes a pooled board-context
  vector `c \in R^{2C}` for the head.
- **Lines.** For board size `N = 8` we work with the 46 standard chess
  lines: 8 ranks, 8 files, 15 diagonals (`r - c = const`), and
  15 anti-diagonals (`r + c = const`). Each line is a multiset of squares
  of length `L \in {1, ..., 8}`; we encode them with a flat-square index
  buffer of shape `(46, 8)` and a Boolean validity mask of shape
  `(46, 8)`.
- **Per-line salience profiles.** For salience head `k` and line `\ell`
  with squares `q_1, ..., q_L`, the line profile is
  `s_{k, \ell} = (s_k(q_1), ..., s_k(q_L)) \in R^L`. We pad with `-\inf`
  to length 8 so the soft-sort pushes padded slots to the right.
- **SoftSort operator.** With temperature `\tau > 0` we form the
  doubly-stochastic permutation matrix
  `P_{ij} = softmax_j(-|\hat{s}_i - s_j| / \tau)`,
  where `\hat{s} = sort(s, descending=True)` is the hard sort used as a
  no-grad reference (Prillo & Eisenschlos, 2020). The differentiable
  sorted profile is `\tilde{s} = P s`, which converges to the hard
  descending sort as `\tau \to 0` and to a uniform mix as
  `\tau \to \infty`.
- **Majorization descriptors.** For each `(k, \ell)` we compute, on
  `\tilde{s}_{k, \ell}`:
  - top values `\tilde{s}_{[1]}, \tilde{s}_{[2]}, \tilde{s}_{[3]}`
  - dominance gaps `g_1 = \tilde{s}_{[1]} - \tilde{s}_{[2]}`,
    `g_2 = \tilde{s}_{[2]} - \tilde{s}_{[3]}`
  - line shape moments `mean`, `sum`, `max - mean`
  - majorization concentration ratios
    `c_1 = |\tilde{s}_{[1]}| / \sum_j |\tilde{s}_{[j]}|`,
    `c_2 = (|\tilde{s}_{[1]}| + |\tilde{s}_{[2]}|) / \sum_j |\tilde{s}_{[j]}|`
  - normalized softmax entropy
    `H = -\sum_j p_j \log p_j / \log L`
    where `p = softmax(\tilde{s})` over the *valid* (non-padded) slots
    only and `L \geq 2`.
  Padded slots are zeroed before any of these reductions.
- **Bucket pool.** Lines of the same type (rank, file, diagonal,
  anti-diagonal) are pooled within each `(k, line_type)` bucket by
  `mean` and `max` over the 11 descriptors, yielding a fixed-size
  `K \cdot 4 \cdot (2 \cdot 11)`-dimensional summary regardless of how
  many lines fall in each bucket.
- **Classifier.** A LayerNorm + GELU MLP over
  `[c, \mathrm{vec}(\mathrm{bucket\_pool})]` returns one puzzle logit.

## Why majorization is the right inequality

Majorization compares vectors by their sorted partial sums:
`a \succ b` iff `\sum_{j \leq m} a_{[j]} \geq \sum_{j \leq m} b_{[j]}` for
every `m` (with equality on the full sum). Several puzzle-relevant
phenomena — overloaded pieces, dominant attackers on a line, single-piece
king-line pressure — are *concentration on a few squares of a line*.
Concentration is exactly what the cumulative sums `\sum_{j \leq m}
\tilde{s}_{[j]}` and the ratios `c_1, c_2` track, and the adjacent gaps
`g_1, g_2` capture how steeply that concentration falls off. Bag-of-line
statistics (mean / sum / max alone) cannot distinguish a single dominant
attacker from a uniform line of the same total mass, but the sorted
profile and its gaps can.

## Why a differentiable sort

Hard sorting blocks gradient flow through ranks, so a model with a hard
sort can only learn the salience field. With SoftSort the *ordering
itself* is differentiable in `\tau`: gradients reach both the salience
heads (through `\tilde{s} = P s`) and, weakly, the relative magnitudes
that pin down the soft permutation. This is what the source packet
proposes ("differentiable sorting per line + sorted salience gaps and
majorization curves"), and it is what distinguishes this idea from a
ray-grammar automaton, a recurrent line scan, or a line-mixing
attention.

## Implementation pointer

The bespoke implementation of this thesis lives in
`src/chess_nn_playground/models/trunk/soft_majorization_line_sorter.py`
(`SoftMajorizationLineSorter`); the idea-local wrapper at
`ideas/registry/i139_soft_majorization_line_sorter/model.py` delegates to its
builder. The shared `ResearchPacketProbe` scaffold is no longer used by
this folder.
