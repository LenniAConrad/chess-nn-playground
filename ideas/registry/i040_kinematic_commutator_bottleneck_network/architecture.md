# Architecture

`Kinematic Commutator Bottleneck Network` (`KCBN`) is a board-only
`puzzle_binary` classifier whose central operator is a family of
degree-two Lie commutators over rule-only chess kinematic motion
operators. The implementation replaces the shared research-packet
probe with a materially distinct bespoke model so the markdown thesis
is exercised by trainable code rather than a generic mechanism profile.

## Forward Pipeline

1. **Encoding semantic adapter.** A `1x1` conv `Conv1x1(C -> d)`
   projects the simple_18 board tensor `(B, 18, 8, 8)` to learned
   square features `H0` of shape `(B, d, 8, 8)`. Deterministic
   per-square empty-square mask `E in {0,1}^{B x 64}` is extracted
   from the 12 piece planes and the side-to-move scalar is read from
   plane 12.
2. **Rule motion operator bank.** Twelve sparse, current-board motion
   operators `{K_m(x)}` are constructed: four orthogonal slider
   directions (N, S, E, W), four diagonal slider directions (NE, NW,
   SE, SW), the knight leaper, the king one-step adjacency, and two
   pawn-attack flavours (side-to-move pawns, opponent pawns). Sliders
   apply the one-step adjacency matrix `M_d` repeatedly with a
   line-of-sight gate so that `K_d h = sum_{k=0..6} (M_d D_E)^k M_d h`
   is the pseudo-legal reach of `h` along `d` with current-board
   blockers; leapers and pawn attacks are static (pawn flavour is
   selected per batch element from the side-to-move scalar). No legal
   moves, mate flags, engine metadata, CRTK source labels, or
   verification metadata are consumed as input.
3. **Lie bracket pair block.** A deterministic ordered list of `P`
   operator pairs `(i, j)` (lexicographic with `i < j`) is processed
   in chunks. For each pair the model evaluates
   `C_ij = K_i(K_j H) - K_j(K_i H)` of shape `(B, d, 64)`. The
   absolute commutator map is multiplied by a per-pair learned vector
   `w_{ij} in R^d` and accumulated into a single commutator field
   `H_c in R^{B x d x 64}`. Per-pair mean and max of `|C_ij|` are kept
   as diagnostics.
4. **First-order control branch.** Optionally, mean and max pool over
   the first-order maps `K_m H` for all twelve operators are
   concatenated and compressed by a small linear layer to a fixed
   summary vector. This is the architecture's first-order ablation
   complement, kept on by default for the central comparison.
5. **Pooling head.** The MLP head consumes `[mean(H), max(H),
   mean(H_c), max(H_c), pair_stats, first_order_summary, side_to_move]`
   and returns a single puzzle logit with hidden_dim `96`, GELU
   activations, and dropout.

## Output Contract

Forward returns a `dict` whose `"logits"` entry has shape `(B,)` so the
shared `puzzle_binary` BCE-with-logits trainer can consume it directly.
Diagnostics include `commutator_field`, `pair_stats`, `bracket_energy`,
`bracket_max`, and `commutator_field_energy`. All diagnostic tensors
are finite by construction.

## Implementation Binding

- Registered model name: `kinematic_commutator_bottleneck_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/kinematic_commutator_bottleneck.py`
- Idea-local wrapper: `ideas/registry/i040_kinematic_commutator_bottleneck_network/model.py`
