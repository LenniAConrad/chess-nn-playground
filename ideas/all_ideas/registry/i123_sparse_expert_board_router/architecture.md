# Architecture

`Sparse Expert Board Router` realises the heterogeneous mixture-of-experts thesis from `math_thesis.md`. A cheap routing summary picks a sparse subset of small board encoders, fuses their outputs with the router's gating weights, and emits a single puzzle logit plus rich routing diagnostics.

## Routing Summary

The router consumes a 12-dimensional deterministic summary computed from the `simple_18` planes:

- `white_material / 39` and `black_material / 39` (normalised material totals)
- `white_to_move` field reduced to a scalar
- white and black king rank/file means
- four coarse-quadrant occupancy means (TL, TR, BL, BR)
- the combined material total

A small CNN stem produces a `(B, channels, 8, 8)` summary; its mean and max pools are concatenated with the deterministic summary, layer-normalised, and passed through a 2-layer MLP that outputs `E=6` router logits.

## Sparse Top-k Gating

The router selects the top `k=2` experts per example. Selection is implemented by masking non-top-k logits to `-inf` and re-applying softmax so the gate weights for the selected experts sum to one and unselected experts get zero gate weight. The selected gate weights drive both expert fusion and the load-balance diagnostics.

## Expert Pool

Six small experts cover distinct inductive biases, matching the markdown's expert list:

1. `local_cnn` – stacked 3×3 convolutions with mean+max pool head.
2. `dilated_cnn` – multi-dilation 3×3 convolutions merged with 1×1.
3. `token_mixer` – per-square token MLP over 64 board tokens.
4. `rank_file_mixer` – separate rank/file aggregate MLPs.
5. `morphology_lite` – soft-dilation/erosion via max-pool stencils with morphological gradient.
6. `compact_mlp_mixer` – pixel-wise 1×1 conv MLP plus pooling.

Each expert outputs a hidden vector of width `hidden_dim`. A per-expert binary head produces an expert logit, and a fused classifier reads the gate-weighted hidden vector. The final puzzle logit blends the router-weighted mixture logit with the fused classifier logit using a learned sigmoid gate.

## Diagnostics And Auxiliary Losses

The forward pass returns:

- `logits` (binary puzzle logit)
- `mixture_logit`, `fused_logit`, `expert_logits`
- `router_logits`, `router_probs`, `router_gate`
- `router_entropy`, `sparse_gate_entropy`
- `top1_gate_mass`, `top2_gate_mass`, `dominant_expert`
- `expert_usage` (mean gate per expert), `expert_selection_counts`
- `pairwise_expert_disagreement`
- `load_balance_loss`, `switch_aux_loss`, `router_entropy_loss`, `auxiliary_loss`

`auxiliary_loss = load_balance_weight * (load_balance_loss + switch_aux_loss) - router_entropy_weight * mean_router_entropy` so the trainer can add it as a regulariser to the binary cross-entropy puzzle loss without recomputing the router.

## Implementation Binding

- Registered model name: `sparse_expert_board_router`
- Source implementation file: `src/chess_nn_playground/models/sparse_expert_board_router.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i123_sparse_expert_board_router/model.py`

The wrapper imports `SparseExpertBoardRouter` and `build_sparse_expert_board_router_from_config` and delegates `build_model_from_config` to that builder. The shared `ResearchPacketProbe` scaffold is no longer used.
