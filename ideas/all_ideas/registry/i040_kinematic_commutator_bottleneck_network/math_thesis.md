# Math Thesis

Kinematic Commutator Bottleneck Network (`KCBN`).

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0728_tuesday_local_kinematic_commutator.md`.

Thesis: chess puzzle-likeness is enriched for non-commuting interactions
between rule-only piece motion operators. Generic CNNs may fit
second-order interactions only at the cost of depth and parameters,
because they cannot natively distinguish ordered operator products
`K_iK_j` from `K_jK_i`. KCBN exposes the antisymmetric correction
directly through degree-two Lie commutators
`B_ij(x) h = (K_i(x) K_j(x) - K_j(x) K_i(x)) h`
over learned square features `h_theta(x)` and pseudo-legal motion
operators `{K_m(x)}` derived from the current board.

## Operator

Let `S = {1,...,64}` be chessboard squares. Each `K_m(x): R^S -> R^S`
is a sparse linear operator built from current-board occupancy, board
boundaries, side-aware pawn directions, and line-of-sight blockers
for sliders. Concretely, for slider direction `d` with one-step
matrix `M_d`,

```
K_d(x) = sum_{k=0}^{T_max} (M_d D_E(x))^k M_d
```

where `D_E(x) = diag(empty(x))` is the diagonal of empty squares for
position `x`. Knight, king, and side-aware pawn-attack operators are
static masks. The Lie-bracket field on a learned feature
`h_theta(x): S -> R^d` is

```
B_ij(x) h_theta(x) = K_i(x) K_j(x) h_theta(x) - K_j(x) K_i(x) h_theta(x)
```

The classifier pools `|B_ij|` together with a learned per-pair vector
`w_{ij}`, the first-order summaries, and safe board metadata, then
maps the concatenation through a small MLP to one puzzle logit for
the BCE-with-logits puzzle_binary trainer.

## Falsification

The packet's central falsifier replaces every Lie bracket with the
symmetric product `K_iK_jH + K_jK_iH` while preserving operator count,
pair count, tensor shape, parameter count, and pooling head. The
bespoke implementation is hyperparameter-clean: a one-line change in
the bracket block converts the model to its symmetric-product control.
If the symmetric control matches KCBN, ordered non-commutativity is
not driving the signal and the idea must be abandoned rather than
tuned.
