# Math Thesis

Differentiable Bitboard Boolean Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.

Batch candidate rank: `2`.

Working thesis: Chess rules are often written as bitboard Boolean
algebra: masks, shifts, intersections, unions, and complements. A
neural model can learn soft bitboard predicates and combine them with
differentiable Boolean operations, producing an efficient
symbolic-neural model whose primitives mirror how engines reason with
bitboards.

The model treats predicates as real-valued maps
``p in (0, 1)^{8 \times 8}``, i.e. soft bitboards. The Boolean
operations are realised as differentiable t-norms / t-conorms over
those maps:

- **Soft NOT** ``not(p) = 1 - p``.
- **Soft AND** ``and(a, b) = a \cdot b``, with the per-clause variant
  ``\prod_l g(lit_l, s_l)`` where ``g(lit, s) = s \cdot lit + (1 - s)``
  is the standard noisy-AND gate parametrised by a learnable selector
  ``s in (0, 1)``. When ``s = 0`` the literal is a no-op factor of 1;
  when ``s = 1`` it contributes the literal itself.
- **Soft OR** is the De Morgan dual,
  ``or(a, b) = 1 - (1 - a)(1 - b)`` and, with selectors
  ``v in (0, 1)``,
  ``or_v(\{c\}) = 1 - \prod_c (1 - v_c \cdot c)``.

Chess-shape **shifts** appear as deterministic permutation operators on
the spatial dimension (king and knight directions; out-of-board
destinations clamp to zero). Combined with NOT, AND and OR over the
shifted predicate bank, the network can express the same
masks-and-shifts identities a hand-written bitboard engine would, while
remaining fully differentiable.

The whole composition stays in disjunctive-normal form: a learned soft
predicate bank produces literals, a soft-AND layer aggregates clauses,
a soft-OR layer aggregates disjuncts, and a small MLP reads spatial
mean/max pools of the disjuncts into a single ``puzzle_binary`` logit.
