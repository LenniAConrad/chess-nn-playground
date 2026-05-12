# Architecture

`Hall-Defect Zeta Operator` (HDZ) is a deterministic finite-algebraic operator paired with a small two-branch convolutional classifier. The deterministic operator computes Hall defects over a Boolean lattice of local tactical obligations, evaluated under a pin/king-exposure-filtered defense relation. The neural classifier consumes the current-board planes and the HDZ tensor and returns a single puzzle logit.

## Implementation Binding

- Registered model name: `hall_defect_zeta_operator`
- Source implementation file: `src/chess_nn_playground/models/hall_defect_zeta.py`
- Idea-local wrapper: `ideas/registry/i085_hall_defect_zeta_operator/model.py`

## Input Contract

The model accepts the repo `simple_18` board tensor with shape `(B, 18, 8, 8)`:

- channels `0..5` are white pawn, knight, bishop, rook, queen, king planes;
- channels `6..11` are the corresponding black planes;
- channel `12` is the side-to-move broadcast plane (white-to-move = 1);
- channels `13..17` are repo metadata planes that the deterministic HDZ branch ignores; only the first 13 channels are interpreted as current-board state.

`HallDefectZetaConvLite` inherits the repo-wide `BoardTensorSpec` shape check via `require_board_tensor`. CRTK and source metadata are never used as input.

## Deterministic HDZ Tensor

`HallDefectZetaBuilder` constructs `H = HDZ(X) ∈ R^{8x8x40}` per board, with 20 channels per side in the order `[white, black]`:

| Channel offset | Description |
|---:|---|
| `0..3` | `maxdef_r1`, `meandef_r1`, `mindefenders_r1`, `pinshare_r1` |
| `4..7` | `maxdef_r2`, `meandef_r2`, `mindefenders_r2`, `pinshare_r2` |
| `8..11` | `maxdef_r3`, `meandef_r3`, `mindefenders_r3`, `pinshare_r3` |
| `12..15` | `maxdef_r4`, `meandef_r4`, `mindefenders_r4`, `pinshare_r4` |
| `16` | `raw_attackers_on_t / 16` |
| `17` | `effective_defenders_on_t / 16` |
| `18` | `pinned_defenders_on_t / 16` |
| `19` | `loose_target_flag` |

For each anchor square `t` and color `c`, the builder:

1. parses the current-board planes into pieces, occupancy, and king squares;
2. computes raw contact via `_piece_contacts_square` for pawns, knights, kings, and slider rays with blocker-aware path checks;
3. detects pins by finding opponent sliders aligned with the friendly king through exactly one friendly blocker, and stores the allowed pin line for each pinned piece;
4. derives the effective contact relation by restricting pinned pieces to their pin line and removing king moves into squares occupied by friends or controlled by the opponent;
5. orders the obligation universe `Ω_{c,t}` deterministically (anchor, king ring within Chebyshev 2, high-value friendly pieces within distance 2, attacked friendly pieces within distance 3, line-interposition squares for opponent rook/bishop/queen lines reaching key targets through at most one blocker, then remaining Chebyshev rings) and trims to 12 atoms;
6. encodes each atom's defender support as a 16-bit integer over the side's pieces and enumerates all subsets of order `r ∈ {1,2,3,4}` up to the configured maximum;
7. computes per-order `maxdef`, `meandef`, `mindefenders`, and `pinshare` from `δ(U) = max(0, |U| − |D(U)|)` with `D(U) = ∪_{o∈U} D_o`, normalising defects by `r` and counts by 16;
8. records the four scalar channels (`raw_attackers_on_t`, `effective_defenders_on_t`, `pinned_defenders_on_t`, `loose_target_flag`) at the anchor.

The tensor is detached and returned to the requested device/dtype before being consumed by the trainable branch. The deterministic algebra is not differentiated through.

## Neural Branches

`HallDefectZetaConvLite` follows the HDZ-ConvLite design from the packet:

- **Raw board branch:** `Conv2d(13 → channels, 3x3) → GELU` repeated `depth` times over the current-board planes.
- **Algebraic branch:** `Conv2d(40 → channels, 1x1) → GELU` repeated `depth` times over `H`.
- **Fusion branch:** concatenation along channels followed by `Conv2d(2·channels → hidden_dim, 3x3) → GELU` twice, global mean pooling over the 64 squares, a side-to-move scalar gate computed from the broadcast plane mean, and a linear head `hidden_dim → 64 → 1` with optional dropout.

A small auxiliary `hdz_head` reads the spatially pooled HDZ tensor as `hdz_only_logits` for ablation diagnostics. No absolute square embeddings, no rank/file one-hots, and no engine, source, or verification side channels are consumed.

## Algebra Modes

`algebra_mode` selects which deterministic tensor is fed into the algebraic branch while keeping the trainable parameter count fixed:

- `hdz` (default): real Hall-defect zeta tensor with pin/exposure filter.
- `atom_scramble_hdz`: same enumeration and channel layout, but every obligation atom `o` is replaced by `(37·o + 11) mod 64` when querying the effective defense relation. This is the packet's semantics-destroying ablation (AtomScramble-HDZ).
- `neural_synth_40`: the deterministic builder is bypassed and the 13 current-board planes are tiled into 40 channels via a fixed channel schedule. This is the packet's same-parameter NeuralSynth-40 control.

`use_pin_filter=False` reproduces the packet's pin-filter ablation; `max_subset_order` and `max_atoms` reproduce the subset-order and obligation-universe ablations.

## Output

`forward(x)` returns a dictionary with at least the puzzle logit and HDZ diagnostics:

- `logits`: BCE-compatible puzzle logits, shape `(B,)`;
- `hdz_only_logits`: linear head over the spatially pooled HDZ tensor;
- `hdz_tensor`: the raw `H ∈ R^{8x8x40}` tensor;
- `zeta_defect_spectrum`, `mean_hall_defect`, `hall_defect_energy`, `max_hall_defect`;
- `effective_defense_density`, `pinned_defender_density`, `loose_target_density`;
- `loose_target_count`, `pinned_piece_count`, `effective_defense_total`;
- `mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count` reporting fields used by repo-wide diagnostic plots.

Calling with `return_aux=True` additionally exposes the active algebra mode code for ablation logging.
