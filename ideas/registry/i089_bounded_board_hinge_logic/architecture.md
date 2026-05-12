# Architecture

`Bounded Board Hinge Logic` (BBHL) is a bespoke PyTorch implementation of
the source packet's differentiable PSL-style logic classifier over the
`puzzle_binary` contract. It compiles a fixed library of typed, shallow
formulas over deterministic closed-world board facts into tensor
operations and emits one BCE logit per board through a probabilistic
soft-logic energy-gap head.

## Implementation Binding

- Registered model name: `bounded_board_hinge_logic`
- Source implementation file: `src/chess_nn_playground/models/trunk/bounded_board_hinge_logic.py`
- Idea-local wrapper: `ideas/registry/i089_bounded_board_hinge_logic/model.py`

## Modules

`CurrentBoardFactExtractor` is the deterministic closed-world fact layer.
It consumes only the current-board planes (12 piece planes plus the
side-to-move plane) and emits 32 unary predicate truths per square plus
18 binary relation truths per square pair. Geometry tensors (king-zone,
between-square clearance, per-piece pseudo-legal attack masks, knight
and pawn step masks) are precomputed once at construction and reused for
every batch. Sliding-piece relations (`stm_ray_attacks`,
`enemy_ray_attacks`) are gated by between-square occupancy so a pinned
ray collapses to its blocker. The layer also emits `attacked_by_*`,
`occupied_and_attacked_by_*`, and `*_valuable` unary derivatives
together with `same_rank/file/diag`, `knight_step`, `king_step`, and
`rays_align` purely-static relation truths.

`FuzzyPredicateBank` projects the 32 raw unary facts into 24 latent
typed concepts and the 18 relation channels into 16 typed roles via a
softmax-mixed convex combination. The mixture weights are warm-started
to a near-one-hot initialization so each concept and role starts close
to a single raw predicate but is free to drift during training. The
bank also exposes a per-batch concept and role mixture entropy.

`BoundedFormulaEvaluator` compiles a fixed library of typed shallow
formulas. The library has three families:

```text
F1: exists_x. C(x)                              (24 unary words)
F2: exists_x exists_y. C_left(x) AND R(x, y) AND C_right(y)   (96 binary words)
F4: exists_x exists_y. C_left(x) AND R(x, y) AND (C_right(y) AND enemy_king_zone(y))   (48 king-zone words)
```

Each formula references a deterministic concept/role index into the
bank, so the library shape is fixed at construction. The body of each
binary formula is evaluated as a Lukasiewicz-tnorm conjunction
`max(0, a + r + b - 2)` of the three truth tensors over `(64, 64)`
square pairs. The `exists` quantifier is the temperature-controlled
softmax-pooled max with a positive learnable temperature `tau`, so the
formula truth stays in `[0, 1]` and is differentiable. Formulas are
evaluated in chunks to bound peak activation memory.

`PSLEnergyGapHead` is a probabilistic soft-logic energy gap head. Each
formula has two non-negative weights, `w_pos = softplus(pos_raw)` and
`w_neg = softplus(neg_raw)`, that scale its hinged truth `a^p` (with
`p in {1, 2}`) on the puzzle-positive and puzzle-negative side. The
puzzle logit is

```text
logit = bias + tau * ( a^p . w_pos  -  a^p . w_neg ).
```

This is the exact PSL energy difference between `y = 0` and `y = 1`
under a hinge potential, so the architecture stays a differentiable
logic classifier and not a generic MLP. The head exposes the per-rule
weights, the energy difference, the L1 weight mass, and the rule
overlap diagnostic.

`BoundedBoardHingeLogicNet` glues the four modules. The forward path
computes:

1. `unary, relations, extras = fact_extractor(board)`.
2. `concepts, roles = predicate_bank(unary, relations)`.
3. `formula_truths, formula_diag = formula_evaluator(concepts, roles, extras["enemy_king_zone"])`.
4. `formula_truths = dropout(formula_truths)`.
5. `logits, head_diag = head(formula_truths)`.

There is no convolutional trunk and no separate embedding mixer: the
logit is a hinge function of the formula truths only.

## Diagnostics

`forward(x)` returns a dict containing:

- `logits`: shape `(B,)`, BCE-compatible for the one-logit
  puzzle_binary head.
- `prob`: sigmoid of the puzzle logit.
- `formula_truths`: shape `(B, F)` with `F = 24 + 96 + 48 = 168`.
- `top_formula_idx`: shape `(B, k)` of the highest-truth formulas.
- `formula_truth_mean`, `formula_truth_max`,
  `unary_formula_truth`, `binary_formula_truth`,
  `kingzone_formula_truth`: scalar means over the formula library.
- `concept_truth_mean`, `role_truth_mean`,
  `concept_mixture_entropy`, `role_mixture_entropy`: predicate-bank
  diagnostics.
- `psl_energy_y0`, `psl_energy_y1`, `logic_energy_gap`,
  `positive_rule_weights`, `negative_rule_weights`,
  `psl_weight_l1`, `psl_weight_overlap`, `psl_tau`: PSL head
  diagnostics.
- `exists_tau`, `formula_count`, `unary_fact_count`,
  `relation_fact_count`: structural counters preserved for
  reporting parity.
- `mechanism_energy`, `proposal_profile_strength`,
  `proposal_keyword_count`: reporting fields preserved for
  compatibility with the project's research-packet diagnostic
  schema.

`forward(x, return_diag=True)` additionally returns `concept_bank`,
`role_bank_mean`, `unary_facts`, and `binary_relations` for
interpretability harnesses.

## Contract

- Input: `(B, C, 8, 8)` board tensor only with `C >= 13`. CRTK /
  verification / source / engine metadata is reporting-only and is not
  consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  puzzle_binary BCE-with-logits trainer, plus the diagnostics listed
  above.
- Target mapping: fine labels `0` and `1` map to binary target `0`;
  fine label `2` maps to binary target `1`.
- Side-to-move: the layer-12 plane is averaged per board to recover the
  scalar `stm` selector that switches the `us`/`them` perspective in
  every fact and every formula.
- Formula library: a fixed shallow PSL-style vocabulary of 168 typed
  formulas (24 unary, 96 binary, 48 king-zone). The library is not
  learned; only the predicate-bank concept/role mixtures, the
  `exists`-temperature, the head's per-rule positive/negative weights,
  the head bias, and the head temperature are trainable.
