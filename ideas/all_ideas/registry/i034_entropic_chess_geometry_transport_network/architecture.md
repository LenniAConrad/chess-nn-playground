# Architecture

`Entropic Chess Geometry Transport Network` (ECGT-Net) is a bespoke
board-only puzzle classifier. The forward pass parses the
`simple_18` tensor into side-to-move-canonical piece planes, builds
side-to-move source atoms and opponent target atoms, computes a
deterministic chess-distance cost matrix between them, and solves a
log-domain entropic Sinkhorn problem. Plan-derived flow features and
square pressure maps are fused with the raw board through a compact
CNN trunk before a binary classifier.

## Modules

1. **EncodingSemanticAdapter** (`simple_18` only). Extracts the 12
   piece planes, side-to-move plane, castling planes, and en-passant
   plane from the 18-channel tensor. Mirrors ranks for black-to-move
   so that `us` always sits at the bottom of the board. Unknown
   deterministic channel maps fail closed with a `ValueError`
   mentioning `simple_18`.
2. **TransportAtomBuilder**. Selects the top `max_sources=16`
   side-to-move pieces by occupancy and prior weight, and constructs
   target atoms in the order `(king_square, king_ring,
   heavy_piece, minor_piece, pawn, promotion_anchor)` up to
   `max_targets=40`. Source and target marginals are formed from
   softplus type/role priors masked by atom existence and normalized
   onto the simplex.
3. **ChessDistanceCost**. Builds the `(B, S, T)` cost tensor by
   gathering precomputed empty-board chess-distance tables (knight
   BFS, rook line, bishop colour-aware, queen, king Chebyshev,
   directional pawn) plus a Manhattan correction. Learned non-negative
   per-(piece-type, target-role) scales `alpha`, `gamma` and additive
   `beta` shape the cost. The module supports `cost_ablation_mode in
   {none, uniform, random_cost_histogram_preserving}` for the central
   falsification.
4. **LogSinkhornTransport**. Fixed-iteration log-domain Sinkhorn
   producing the `(B, S, T)` plan with `epsilon=0.25` and
   `sinkhorn_iters=8` by default. Masked invalid pairs are clamped to
   zero and the plan is renormalized.
5. **TransportFeatureProjector**. Projects the plan into:
   - the flattened type-role flow `(B, 6 * R)`,
   - six scalar plan summaries (expected cost, normalized entropy,
     max row mass, max column mass, valid source/target ratios),
   - a source pressure map `(B, 1, 8, 8)` and per-role target
     pressure maps `(B, R, 8, 8)` produced by `scatter_add` onto
     square indices,
   plus diagnostic tensors (`transport_cost`, `transport_entropy`,
   `transport_source_concentration`, `transport_target_concentration`,
   `transport_king_flow`, `transport_role_pressure`).
6. **TransportAugmentedCNN**. Concatenates the raw board tensor with
   the transport pressure maps and runs a `depth`-block
   Conv -> BatchNorm -> GELU stack with hidden width
   `channels`/`hidden_width`. Mean and max pooled features form the
   board embedding.
7. **Transport MLP and classifier**. The flow + scalar vector passes
   through a `LayerNorm -> Linear -> GELU -> Dropout -> Linear -> GELU`
   MLP. The board and transport embeddings are concatenated and a
   two-layer head produces `num_classes` logits. With
   `num_classes == 1` the output is squeezed to shape `(batch,)`.

## Outputs

The forward pass returns a dictionary that includes:

- `logits`: `(batch,)` puzzle logit consumed by the puzzle-binary
  trainer.
- `transport_cost`: per-batch expected transport cost.
- `transport_entropy`: normalized plan entropy in `[0, 1]`.
- `transport_source_concentration` and
  `transport_target_concentration`: row and column L2 mass.
- `transport_king_flow`: total flow into the opponent king square.
- `transport_role_pressure`: max role-pooled inflow.

## Encoding Adapter Policy

- `simple_18` (default): supported. The deterministic atom builder
  reads the 12 piece planes and the side-to-move plane.
- `lc0_static_112` and `lc0_bt4_112`: not supported. Constructing
  `EncodingSemanticAdapter` with anything other than `simple_18` /
  `input_channels=18` raises `ValueError`, matching the leakage and
  fail-closed policy in the research packet.

## Implementation Binding

- Registered model name: `entropic_chess_geometry_transport_network`.
- Source implementation:
  `src/chess_nn_playground/models/chess_geometry_transport.py`.
- Idea-local wrapper:
  `ideas/all_ideas/registry/i034_entropic_chess_geometry_transport_network/model.py`,
  which exposes `build_model_from_config(config)` and delegates to
  `build_entropic_chess_geometry_transport_network_from_config`.
- Registry wiring: `src/chess_nn_playground/models/registry.py`
  registers the bespoke builder, taking precedence over the
  auto-registered `ResearchPacketProbe` fallback that would otherwise
  resolve the slug from `RESEARCH_PACKET_MODEL_NAMES`.
