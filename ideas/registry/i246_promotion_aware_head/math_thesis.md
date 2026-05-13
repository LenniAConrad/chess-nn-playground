# Mathematical Thesis — PFCT / Promotion-Aware Head (i246)

- Mathematical motivation:
  - The four legal promotion piece types {Q, R, B, N} are the only chess-rule-legal
    type transformation in the game. Static encoders collapse a pre-promotion
    pawn to type `P`, hiding the latent piece identity until the move is played.
  - The Promotion-Fanout Counterfactual Tensor (PFCT) primitive enumerates this
    transformation explicitly: for each own pawn `p` on its 7th rank, it
    substitutes `p` with each promoted piece `T ∈ {Q, R, B, N}` on the
    promotion square and records the corresponding feature vector
    `phi_theta(x → (p → T at s_p + 1)) ∈ R^d`. The resulting per-pawn fanout
    `F_theta(p, x) ∈ R^{4×d}` is the canonical PFCT signature for that pawn.

- Assumptions:
  - `phi_theta` is the i193 trunk's joint pool feature (concatenated exchange
    pool, king pool, and deterministic summary planes). It is piece-type
    sensitive because the trunk's geometric attack tables are piece-typed.
  - Promotion legality conditions (no own piece on the promotion square; pawn
    must be on rank 7 for white / rank 2 for black) are enforced by the
    substitution geometry: the source pawn is always removed and any piece
    currently on the promotion square is cleared before the promoted piece
    is placed, so the output is a valid one-hot piece encoding regardless of
    whether the chess move would be legal.
  - The simple_18 encoding contains enough state to compute promotions
    purely from piece planes (no python-chess call required inside forward).

- Claimed advantage:
  - Concentrated lift on the `crtk_tactic_motifs = promotion` slice, currently
    the largest per-slice PR-AUC gap in the benchmark (best 0.667 vs
    aggregate 0.876, ≈0.21 absolute gap, far larger than the `equal` bucket's
    ≈0.06 gap). Underpromotion shares the same slice in CRTK.
  - Zero overhead on positions without near-promotion pawns; the gate is
    structurally clamped to zero in that case.
  - Orthogonal to the other two implemented primitives (TSDP rule features,
    TDCD cross-derivative) — the primitives target disjoint failure modes.

- Proof sketch:
  - For any encoder that is sensitive to piece type, `F_theta(p, x)` rows must
    differ across {Q, R, B, N} whenever the geometric features of the four
    promoted pieces differ on the destination square. The i193 trunk's
    deterministic feature builder treats Q, R, B, N as distinct piece types
    with different attack geometries, so the rows are guaranteed to vary
    unless the destination square is dominated by features that all four
    piece types contribute equally to.
  - The cross-attention head learns to weight the four rows. Under the
    softmax + linear-readout composition, the per-pawn delta is
    `delta(p) = w_T · MLP(alpha · F)` where `alpha ∈ Δ^3` is the attention
    distribution; for a position where promotion to a non-queen is best,
    the gradient w.r.t. `alpha_N` (knight) becomes positive whenever the
    knight row encodes a tactical motif (e.g. a fork) and the queen row
    does not.
  - The empirical prototype (`prototypes/pfct_prototype.py`) demonstrates
    this exactly: in a planted knight-fork motif scenario, the fanout's
    `argmax_T |F[T]|` flips from Q to N purely from the substitution, with
    no architectural retraining.

- What is actually proven:
  - The substitution is a deterministic, rule-derived function of the
    simple_18 piece planes (verified by unit tests).
  - The gate is structurally zero on positions without own near-promotion
    pawns (verified by unit tests).
  - Gradients flow through the trunk, the cross-attention head, and the
    delta MLP under BCE loss (verified by unit tests).

- What is only hypothesized:
  - Whether the learned attention reliably picks up the "correct" promotion
    on real puzzle data at scout scale. This is what the falsifier
    (matched `copy_baseline_fanout` ablation on the promotion slice) is
    designed to test.
  - Whether the i193 trunk's piece-type features are sharp enough to make
    the substitution informative. The trunk's `feats.exchange` and
    `feats.king` already differentiate Q, R, B, N attacks via the
    `geom_attacks` table, so this is at least plausible.

- Failure cases:
  - If `phi_theta` produces near-identical features for Q vs R vs B vs N on
    the same square (e.g. when no piece-type-sensitive head exists
    downstream), the fanout collapses and the primitive degrades to a
    fancy no-op. The shuffled-fanout ablation tests for this.
  - If the promotion square is occupied by an enemy piece, the
    substitution still produces a valid output (capture-promotion), but
    the trunk may "see" a feature that wasn't reachable by a legal move
    — this could mislead the head. Diagnostic `promotion_dominant_type`
    can be sliced by capture-promotion vs quiet-promotion positions to
    audit this.
  - Promotion is only ~6% of positions, so even a perfect target-slice
    fix yields ≈+0.5pp aggregate. The architecture is designed as a
    *complement* to broader primitives (TSDP, TDCD, DHPE), not a
    standalone aggregate-PR-AUC fix.
