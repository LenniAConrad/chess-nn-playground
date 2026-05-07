# Architecture

`Piece-Target Entropic Transport Bottleneck` (PT-ETB) is a bespoke
board-only puzzle classifier. The forward pass canonicalizes the position
to a side-to-move-relative orientation, then runs entropy-regularized
optimal transport in two directions between learned piece-source and
piece-target measures over the 64 board squares. Plan statistics and
projected transport maps are fused with a shallow CNN adapter through a
compact bottleneck that emits one puzzle logit and per-direction transport
diagnostics.

## Forward Pass

1. **Piece-plane adapter** (`PiecePlaneAdapter`). Validates the
   `simple_18` channel contract (12 piece planes, side-to-move plane,
   castling and en-passant planes). Unknown channel maps fail closed with
   a `ValueError` mentioning `simple_18`.
2. **Side-to-move canonicalization** (`RelativeBoardCanonicalizer`).
   Swaps colors and flips ranks for black-to-move so that "friendly"
   always denotes the side to move with the friendly back rank fixed at
   row 7 of the tensor.
3. **Shallow board adapter** (`BoardAdapter`). Conv -> BatchNorm -> GELU
   stack producing `(B, adapter_width, 8, 8)` board features. Used only as
   the learned adapter `A` in the packet's pseudocode.
4. **Type-aware marginals** (`TransportMarginals`). Per-head softplus
   piece-type weights produce normalized source measures `mu` and target
   measures `nu` from the canonical friendly/enemy occupancy. `epsilon_mass`
   keeps every entry strictly positive so the transport problem is
   well-posed (Proposition 1).
5. **Type-aware transport cost** (`TypeAwareTransportCost`). For each
   (source square, target square) pair the cost MLP combines the
   square-mixed piece-type embeddings, deterministic geometry features
   (file/rank deltas, Chebyshev/Manhattan distances, same-file/rank/diag
   indicators, knight vector, forward relation, source/target centrality),
   and a forward/reverse direction embedding. The cost has shape
   `(B, H, 64, 64)` with `softplus` non-negativity and a small floor to
   keep Sinkhorn stable.
6. **Log-domain Sinkhorn** (`LogSinkhornTransport`). Fixed-iteration log
   stabilized solver returning the entropic plan
   `(B, H, 64, 64)` for both the friendly→enemy (forward) and
   enemy→friendly (reverse) directions.
7. **Transport summary projection** (`TransportSummaryProjector`). Each
   plan/cost pair contributes the nine global statistics required by the
   packet (`expected_cost`, plan entropy, plan L2 concentration,
   expected `|df|/|dr|`, expected manhattan, same-line mass, knight-vector
   mass, low-cost soft mass) and four projected board maps
   (`source_cost_map`, `target_cost_map`, `source_conc_map`,
   `target_conc_map`).
8. **Fusion and bottleneck** (`PieceTargetEntropicTransportBottleneck`).
   Concatenates `A` with all `2 * H * 4 = 8H` projected transport maps,
   passes them through a 1x1/3x3 fusion stack, mean+max pools, then
   concatenates the layer-normalized global statistics. A two-layer MLP
   bottleneck and a `Linear(num_classes)` classifier produce the puzzle
   logit. With `num_classes == 1` the output is squeezed to shape
   `(batch,)`.

## Outputs

The forward pass returns a dictionary that includes:

- `logits`: `(batch,)` puzzle logit consumed by the puzzle-binary trainer.
- `transport_cost_forward`, `transport_cost_reverse`: per-batch mean
  expected transport cost in each direction.
- `transport_entropy_forward`, `transport_entropy_reverse`: normalized
  plan entropy (divided by `log(64*64)`).
- `transport_asymmetry`: signed difference between reverse and forward
  costs.
- `transport_low_cost_mass`: forward-direction soft mass on low-cost
  pairs.
- `transport_bottleneck_norm`: scaled L2 norm of the latent bottleneck for
  diagnostic logging.

## Encoding Adapter Policy

- `simple_18` (default): supported. The deterministic transport branch
  extracts the 12 piece planes and the side-to-move plane.
- `lc0_static_112` and `lc0_bt4_112`: not yet supported by this bespoke
  implementation. The adapter raises `ValueError` so unknown deterministic
  channel semantics fail closed, matching the leakage policy in the
  research packet.

## Implementation Binding

- Registered model name: `piece_target_entropic_transport_bottleneck`.
- Source implementation: `src/chess_nn_playground/models/piece_target_transport.py`.
- Idea-local wrapper: `ideas/i033_piece_target_entropic_transport_bottleneck/model.py`,
  which exposes `build_model_from_config(config)` and delegates to
  `build_piece_target_entropic_transport_bottleneck_from_config`.
- Registry wiring: `src/chess_nn_playground/models/registry.py` registers
  the bespoke builder; the slug is excluded from
  `RESEARCH_PACKET_MODEL_NAMES` so the auto-registered `ResearchPacketProbe`
  fallback no longer applies.
