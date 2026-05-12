# Architecture

`Chess Hypercut Polynomial Network` (CHPNet) is a board-only `puzzle_binary`
classifier whose decisive non-linearity is a masked high-order cut polynomial
over deterministic chess-rule hyperedges, with exclusive-product cut
derivatives scattered back to per-square states. It follows the markdown
thesis from
`ideas/research/packets/classic/chess_nn_research_2026-04-28_0733_tuesday_new_york_hypercut_poly.md`.

## Mechanism

1. **Side-to-move canonicalisation.** Each `simple_18` board tensor is rotated
   180 degrees and has its piece planes swapped when it is black to move so
   the model sees a white-to-move view. The side-to-move plane is then forced
   to one and the four castling-right planes are reordered to match the
   canonical view. The forward pass uses
   `normalize_side_to_move_tensor(board)` for the convolutional stem.
2. **Deterministic chess-rule hypergraph builder.** `ChessHyperedgeBuilder`
   converts each (uncanonicalised) `simple_18` sample into a set of vertex
   subsets of the 64 board squares using only the chess rules:
   - **Sliding ray edges** for every white/black bishop, rook, and queen,
     extending until the first occupied square (inclusive).
   - **Piece stencil edges** for every white/black knight (knight halo),
     king (8-neighbourhood shell), and pawn (forward push, double-push from
     the start row, and two diagonal captures), centred on the piece.
   - **Occupied line windows** along ranks, files, and both diagonals for
     window sizes `3..max_edge_size`, kept only when at least two squares in
     the window are occupied.
   - **King shell edges** repeated as a hard alias of the king
     8-neighbourhood so king-anchored hyperedges are always present.
   The set is deduplicated, capped to `max_edge_size=9`, and the top
   `max_edges=1024` edges are kept by a deterministic priority that favours
   king-touching, occupancy-dense, larger edges. The builder caches per-board
   `HyperedgeBatch(edge_index, edge_mask, edge_active, edge_size)` tensors so
   identical positions reuse the same hypergraph between epochs.
3. **Board-to-square stem.** A two-layer `Conv2d -> GELU -> Conv2d` stem
   maps `(B, input_channels, 8, 8)` to a `(B, hidden_dim, 8, 8)` field, which
   is flattened to 64 square tokens and added to a learned per-square
   positional embedding.
4. **Hypercut blocks.** `hypercut_blocks` `HypercutBlock` layers each:
   - Apply `tanh(Linear)` to produce `cut_probes` learned probe states `s_v`
     per square.
   - Gather probes onto every hyperedge `e` and form per-slot factors
     `(1 + s_v) / 2` and `(1 - s_v) / 2`, masking unused slots to one and
     inactive edges to zero so padding does not leak gradients.
   - Compute the masked **cut polynomial**
     `c_e = 1 - prod_v (1 + s_v)/2 - prod_v (1 - s_v)/2`, which is exactly
     one when the probe signs split across the edge and zero when they
     agree. Per-block mean/max/std summaries of `c_e` over active edges
     feed the readout.
   - Compute the **exclusive-product derivative** of `c_e` w.r.t. each
     vertex slot (`-1/2 prod_{w!=v} (1+s_w)/2 + 1/2 prod_{w!=v} (1-s_w)/2`),
     mix it through a learned `out_weight` matrix, and scatter-add the
     resulting per-vertex residual back to square states with
     `1 / sqrt(1 + incidence)` normalisation. A residual `LayerNorm` and a
     small `Linear -> GELU -> Linear` feedforward update the square states.
5. **Readout and head.** Square states are mean- and max-pooled over the 64
   squares and concatenated with the per-block cut moments. The head is
   `LayerNorm -> Linear -> GELU -> Dropout -> Linear -> 1` and emits the
   `puzzle_binary` logit `(B,)`.

## Output Contract

Forward returns a dict whose `"logits"` entry is `(B,)` for the repository
`puzzle_binary` BCE-with-logits trainer. Diagnostic tensors saved to
prediction artefacts include:

- `hyperedge_count`: `(B,)` active hyperedge count per sample.
- `hyperedge_size_mean`: `(B,)` mean active hyperedge size.
- `hypercut_energy`: `(B,)` mean-square cut summary energy.
- `hypercut_mean`, `hypercut_max`, `hypercut_std`: `(B,)` final-block cut
  moments averaged across cut probes.
- `higher_order_residual_energy`: `(B,)` mean-square energy of the final
  square-state field after the cut-derivative residuals.
- `mechanism_energy`: `(B,)` packet-family compatible energy proxy
  (`hypercut_energy`).
- `proposal_profile_strength`: `(B,)` normalised active-edge fraction.
- `proposal_keyword_count`: `(B,)` constant marker (`5.0`) for legacy
  reporting parity.

## Leakage Guards

The forward pass consumes only the board tensor and deterministic
chess-rule hyperedges derived from that tensor. The packet's forbidden
inputs (engine scores, PVs, best moves, mate scores, source IDs,
verification flags, fine labels) are never passed to the model. CRTK
metadata is reporting-only.

## Implementation Binding

- Registered model name: `chess_hypercut_polynomial_network`.
- Source implementation file: `src/chess_nn_playground/models/trunk/chess_hypercut_polynomial.py`.
- Idea-local wrapper: `ideas/registry/i082_chess_hypercut_polynomial_network/model.py`.
