# Architecture

`Chess-Mode Tucker Relation Certificate` (`CMTRC`) realizes the source packet's
chess-constrained Tucker contraction over a fixed relation-moment tensor as a
bespoke PyTorch model for the repo's `puzzle_binary` task.

## Implementation Binding

- Registered model name: `chess_mode_tucker_relation_certificate`
- Source implementation file: `src/chess_nn_playground/models/chess_mode_tucker_relation_certificate.py`
- Idea-local wrapper: `ideas/i090_chess_mode_tucker_relation_certificate/model.py`

## Modules

`RelationMaskBuilder` precomputes the fixed legal-chess relation tensor
`M in R^{12 x 8 x 64 x 64}` and the board-region mask `A in R^{10 x 64}`.
The 12 relation families are 8 sliding ray directions, signed knight jumps,
king-adjacent neighbours, and the white / black pawn-attack geometries. The
8-slot depth axis stores the 1..7 ray distances (with depth 7 padded for ray
families) and the 8 signed jump variants for knights and king moves; pawn
families occupy two depth slots and pad the rest. Region masks include the
full board, light/dark squares, the four-square center, the extended sixteen-
square center, the corners, edges, back ranks, side files, and the promotion
bands; each region is normalized to sum to one over its active squares.
Buffers are registered as non-persistent so they ride the module to GPU
without being saved into checkpoints.

`ChessModeTuckerRelationCertificate` runs the four-mode pipeline from
section 8 of the source packet:

1. A `1x1` channel lift `Conv2d(C, K=32)` followed by `GroupNorm` and `SiLU`
   produces the latent board embedding `E in R^{B x K x 64}`.
2. The fixed relation scan
   `N_{b,k,rho,delta,s} = sum_t M_{rho,delta,s,t} E_{b,k,t}` is computed by
   einsum and divided by `sqrt(deg)` to keep ray and jump scales comparable.
3. The relation moment tensor
   `T_{b,k,rho,delta,gamma} = sum_s A_{gamma,s} E_{b,k,s} tanh(N_{b,k,rho,delta,s})`
   has shape `(B, K, R, D, G) = (B, 32, 12, 8, 10)`.
4. The Tucker mode projection
   `S = T x_K U_K^T x_R U_R^T x_D U_D^T x_G U_G^T` contracts each mode against
   a learnable rectangular factor and produces
   `S in R^{B x rK x rR x rD x rG} = R^{B x 8 x 6 x 4 x 5}`.
5. The core contraction `z_{b,h} = <S_b, Omega_{:,:,:,:,h}>` collapses the
   four mode axes against the learnable core
   `Omega in R^{rK x rR x rD x rG x H}` with `H = 24`.
6. A `Linear(24, 32) -> SiLU -> Dropout -> Linear(32, 1)` head produces the
   single BCE-compatible puzzle logit.

The tensor object is fixed-geometry: there are no learnable square-pair
attention weights, no token mixing across the 64 squares, and no ingestion of
engine, principal-variation, mate-score, or source metadata.

`rank_certificate` unfolds the projected tensor `S` along each Tucker mode
and computes the per-example effective rank
`(sum sigma_i) / (sum sigma_i^2)^{1/2})^2`. The four mode-effective ranks are
exposed as diagnostics. The nuclear bottleneck term sums the mode nuclear
norms and divides by `||S||_F`, supplying an optional rank-regularisation
signal.

`orthogonality_penalty` returns
`sum_m ||U_m^T U_m - I||_F^2`, which the trainer can mix into the loss with a
small coefficient to stabilise the rectangular mode factors.

`FlatProjectedMLPControl` lives next to the main module and provides the
same-parameter non-tensor control mandated by the source packet. It shares
the channel lift, GroupNorm, fixed `M`, fixed `A`, and the relation moment
construction, then flattens `T` through a deterministic CountSketch
projection (signed-bucket scatter, no trainable parameters) into 112 features
and feeds them to `Linear(112, 213) -> SiLU -> Linear(213, 1)`. The trainable
head parameter count is matched exactly to the Tucker head.

## Diagnostics

`forward` returns a dict containing:

- `logits`: shape `(B,)`, BCE-compatible for the one-logit puzzle_binary head.
- `prob`: sigmoid of the logit.
- `tucker_features`: the `H = 24` dimensional core contraction.
- `projected_tensor_energy`, `relation_tensor_energy`: mean squared activation
  of `S` and `T` respectively.
- `nuclear_bottleneck`: pooled mode-nuclear norm divided by `||S||_F`,
  available for the optional Tucker bottleneck regulariser.
- `orthogonality_penalty`: scalar broadcast across the batch, ready for the
  `R_orth` term in the loss.
- `rank_certificate`: per-example `(B, 4)` effective ranks, plus
  `K_mode_eff_rank`, `R_mode_eff_rank`, `D_mode_eff_rank`, and
  `G_mode_eff_rank` as separate scalars per example.
- `fixed_relation_density`, `region_mass_error`: structural sanity checks on
  the fixed masks.
- `mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count`:
  scalar reporting fields preserved for compatibility with the project's
  research-packet diagnostic schema.

`forward_with_aux` additionally returns the full `projected_tensor` and
`relation_tensor` for offline analysis and ablations. `fine_label_diagnostic_3x2`
provides the mandatory `3 x 2` diagnostic counts and row-normalised rates.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. CRTK / verification / source /
  engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit puzzle_binary
  BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine
  label `2` maps to binary target `1`.
- Symmetry: side-to-move and pawn direction are not assumed equivariant; the
  white and black pawn-attack relations are kept as separate families to
  preserve direction.
