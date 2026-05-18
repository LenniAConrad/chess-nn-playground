# Math Thesis

Source: `ideas/research/primitives/external_47_promotion_underpromotion_primitive.md`
(the "PUGP" primitive recommended there as a board-only,
side-to-move-canonicalised, rule-exact promotion-geometry primitive).

## Working thesis

1. Canonicalise the side to move so the own side always moves toward
   canonical row 0 (the promotion rank). Concretely, for white-to-move
   samples ``C`` is the identity; for black-to-move samples ``C``
   vertically mirrors the board AND swaps the white/black piece
   planes (and castling-plane pairs). After ``C``:

     - own pawns live at canonical plane 0, opp pawns at plane 6;
     - near-promotion own pawns are at canonical row 1;
     - the promotion rank is canonical row 0;
     - the STM plane is forced to 1.0.

2. **Candidate promotion moves**. For an own pawn at canonical
   ``(1, f)`` the legal arrival templates are:

     - quiet promotion to ``(0, f)`` if that square is unoccupied;
     - capture promotion to ``(0, f-1)`` or ``(0, f+1)`` if the file
       exists and the square holds an enemy piece.

   These are the only geometrically possible promotion arrivals under
   FIDE Article 3.7. PUGP exposes the **set** of per-file candidate
   indicators, separating the quiet (`push`) candidate from the two
   capture (`capL`, `capR`) candidates.

3. **Per-arrival-square per-type features**. For each arrival square
   ``u = (0, g)`` on canonical row 0 and each promotion piece type
   ``t in {Q, R, B, N}`` the head computes the post-promotion
   feature triple ``(c_t, z_t, s_t)``:

     - ``c_t(u, B^c) = 1[enemy_king in A_t(u; B^c)]``
       (gives-check indicator).
     - ``z_t(u, B^c) = |A_t(u; B^c) cap Z(enemy_king)|``
       (king-zone overlap).
     - ``s_t(u, B^c) = clip(d(u) - a(u), -4, 4) / 4`` where
       ``d`` and ``a`` are the own-defender / opp-attacker counts on
       ``u``. The current implementation uses a piece-agnostic safety
       score per arrival square (no source-pawn correction); the
       per-type variants are exposed via the delta-to-queen channels
       so the model can still learn underpromotion-arrival
       preferences if they exist.

4. **Sliding attack masks via the shared ray geometry**. ``A_Q(u)``,
   ``A_R(u)`` and ``A_B(u)`` are computed by gathering ``occupancy``
   along the 8 fixed ray directions from ``u`` and applying a
   cumulative-blocker scan: the attack set includes every on-board
   square up to and including the first blocker.

5. **Knight and king attack patterns**. ``A_N(u)`` is read from a
   precomputed ``(64, 64)`` knight template. The enemy king zone
   ``Z(k_{opp})`` is read from a precomputed ``(64, 64)`` Chebyshev-
   distance-1 template.

6. **Underpromotion encoding**. For each non-queen type ``t in {R, B,
   N}`` the per-arrival-square delta-to-queen triple is

       Delta_t(u) = (c_t(u) - c_Q(u), z_t(u) - z_Q(u), s_t(u) - s_Q(u)).

   Knight gets the additional ``kappa_N(u)`` scalar -- the weighted
   sum over enemy high-value targets (Q, R, B, N, king) that a knight
   from ``u`` attacks. ``kappa_N`` is the load-bearing feature that
   queen-only collapse cannot recover, since queen does not subsume
   knight geometry.

7. **Per-candidate-kind tokens**. For each candidate kind ``c in
   {push, capL, capR}`` the head builds the per-arrival-file token

       phi_c(u) = [m_c(u), capture_flag, edge_file,
                   c_Q(u), z_Q(u), s_Q(u),
                   Delta_R(u), Delta_B(u), Delta_N(u),
                   kappa_N(u)],

   masked elementwise by ``m_c(u)``. Quiet pushes set
   ``capture_flag = 0`` and capture promotions set it to 1.

8. **Pooling and head**. Sum-and-max pool tokens over the 8 arrival
   files per candidate kind, concatenate with the global pawn-distance
   summary and the candidate counts, LayerNorm, concatenate with the
   i193 trunk joint pool, project through a 2-layer MLP to
   ``primitive_delta_raw``. The gate MLP consumes the joint pool plus
   ``(total_count, n_own_r1, n_opp_r1, has_capture)``; initial bias
   is strongly negative. Output:

       final_logit = base_logit + primitive_gate * primitive_delta_raw.

## Why this matters

The repo's strongest published i193 group reports ~0.876 aggregate
test PR AUC but only ~0.652 on both the merged
``promotion / underpromotion`` motif slices. PFCT (i246) is a strong
primitive but its current i246 implementation collapses the
promoted arrival to the same file (no diagonal capture promotions),
allows the substituted piece to land on a square that need not
correspond to a legal move, and lacks an explicit underpromotion-as-
delta-to-queen encoding. PUGP is the geometry-first complement: it
exactly enumerates quiet+capture promotion candidates, scores arrival
square safety and king-zone reach for each promotion piece type, and
exposes a typed underpromotion / knight-fork channel that queen-only
collapse cannot recover.

## What is actually proven

- The canonicalisation map is an involutive equivariance: applying
  ``C`` twice recovers the original board on a symmetric
  representation of own-vs-opp pieces (modulo the en-passant plane
  which is not load-bearing for promotion arrival geometry).
- The per-file push/capL/capR mask construction matches the FIDE
  promotion rule template: a pawn at ``(1, f)`` may quiet-promote to
  ``(0, f)`` iff that square is unoccupied; may capture-promote to
  ``(0, f-1)`` or ``(0, f+1)`` iff the diagonal arrival square holds
  an enemy piece.
- The sliding attack mask from ``u`` exactly matches the chess attack
  set of a queen / rook / bishop on ``u`` over the **current**
  occupancy (no post-promotion correction for the source pawn or the
  captured piece). For practical safety scoring this is a small bias
  whose direction is documented in the implementation notes.
- The cumulative blocker scan over the (B, 8, 8, 7) ray-gather is
  arithmetic-only (``cummax`` over a {0, 1} indicator) and is
  differentiable through `pytorch`'s autograd.

## What is only hypothesised

- That the explicit per-type delta-to-queen channels and the
  dedicated knight-fork hint ``kappa_N`` carry promotion-slice
  signal not already absorbed by i193 or i246.
- That the additive gated head can use this signal without leaking
  into aggregate behaviour on the bulk of (non-promotion) data; the
  gate's negative initialisation is the load-bearing inductive bias.

## Failure cases

1. *Pseudo-only collapse*: tested by the ``pseudo_only`` ablation
   (drop legality filtering on candidates).
2. *Capture-promotion redundancy*: tested by ``no_capture``.
3. *Queen subsumes underpromotion*: tested by ``queen_only`` (zero
   delta-to-queen + zero ``kappa_N``).
4. *Arrival safety irrelevant*: tested by ``no_attack_defense``.

## Falsifier

- ``pseudo_only`` -- primary geometry falsifier (matched legality vs.
  raw geometric candidates).
- ``no_capture`` -- capture-geometry falsifier.
- ``queen_only`` -- underpromotion-hint falsifier.
- ``no_attack_defense`` -- arrival-safety falsifier.
- ``zero_delta`` / ``trunk_only`` -- structural i193 baseline.
- ``disable_gate`` -- gate load-bearing check.
