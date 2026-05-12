# Architecture

`Local Neighborhood Geometry Network` is a bespoke implementation of idea
`i124`. It encodes ``V`` deterministic, board-only perturbations of the
input through a single shared encoder and reads tactical content off the
*geometry* of the resulting embedding cloud. The local-sharpness thesis
predicts that puzzle-like positions sit in sharper local representation
basins than quiet non-puzzle positions: removing one piece plane, masking
one square neighborhood, or zeroing the coordinate planes should move a
puzzle-like board's representation farther than the same perturbations
applied to a quiet position.

## Pipeline

- Input: board tensor `(B, 18, 8, 8)`. CRTK / source metadata is
  reporting-only and never used as model input.
- Deterministic perturbation set (V = 6, board-only):
  1. `identity`
  2. `horizontal_mirror` — file-axis flip (kept as a diagnostic; the model
     is not told that labels are invariant under this perturbation, only
     that local response is informative)
  3. `mask_corner_quadrant` — zero a fixed 4x4 corner quadrant
  4. `zero_coordinate_planes` — zero metadata / coordinate planes from
     index `coordinate_plane_start` onward
  5. `mask_king_neighborhood_ring` — zero the 3x3 centre region of the
     board (a deterministic, board-agnostic stand-in for "mask one square
     neighborhood")
  6. `piece_type_dropout_group` — zero one piece-type group (white and
     black planes of `piece_dropout_group_index`)
- Shared encoder: a small CNN stem followed by mean+max pooling and a
  linear projection to `embed_dim`.  Weights are shared across all V
  views by construction.
- Local geometry statistics, computed per-sample:
  - `lng_center_embedding` — embedding of the identity view
  - `lng_view_deltas` — per-view deltas to the centre embedding
  - `lng_delta_norms` — L2 norms of those deltas
  - `lng_cosine_delta_offdiag` — strict upper-triangular cosines between
    pairs of deltas
  - `lng_local_covariance_spectrum` — top-K eigenvalues of the centred
    `V x V` Gram matrix of all view embeddings (proxy for the local
    embedding covariance spectrum)
  - `lng_pairwise_distances`, `lng_mean_pairwise_distance`,
    `lng_max_pairwise_distance`
  - `lng_mean_delta_norm`
  - `lng_anisotropy_ratio` — top eigenvalue / sum of eigenvalues of the
    local Gram spectrum
- Head: a LayerNorm + GELU MLP over `[center_embedding, geometry_stats]`
  returning one puzzle logit.

## Implementation Binding

- Registered model name: `local_neighborhood_geometry_network` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Source implementation file:
  `src/chess_nn_playground/models/trunk/local_neighborhood_geometry_network.py`
  (`LocalNeighborhoodGeometryNetwork` and
  `build_local_neighborhood_geometry_network_from_config`).
- Idea-local wrapper:
  `ideas/registry/i124_local_neighborhood_geometry_network/model.py` calls
  `build_local_neighborhood_geometry_network_from_config`.
- The shared `ResearchPacketProbe` scaffold is no longer used by this idea.
