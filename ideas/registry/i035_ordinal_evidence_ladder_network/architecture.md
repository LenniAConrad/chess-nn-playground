# Architecture

`Ordinal Evidence Ladder Network` (OEL-Net) is a bespoke board-only
puzzle classifier. The forward pass runs the `simple_18` tensor through
a compact convolutional trunk, projects to a scalar puzzle-potential
score and a positive evidence concentration, and applies two learned
ordered cumulative thresholds. The benchmark logit is the survival
probability for `Y >= 2`, matching the i035 puzzle-binary contract that
treats fine labels `0` and `1` as non-puzzle and fine label `2` as
puzzle.

## Modules

1. **EncodingSafeStem**. Validates that the input tensor matches the
   configured `input_channels` (default `18` for `simple_18`) and runs a
   `1x1 Conv -> BatchNorm -> GELU` projection from `C` to
   `stem_width = 32`. Mismatched channel counts raise a `ValueError`
   that mentions `Expected board tensor`.
2. **TinyBoardBackbone**. A `3x3 Conv -> BatchNorm -> GELU` entry from
   `stem_width` to `backbone_width = 64`, followed by
   `residual_blocks = 2` residual blocks (two `3x3` convs each) at width
   `64`, then a `3x3 Conv -> BatchNorm -> GELU` projection to
   `embedding_dim = 96`. Global average pooling over the `8x8` board
   produces a `(B, embedding_dim)` board embedding `h`.
3. **OrdinalLadderHead**. Two linear projections produce the scalar
   puzzle-potential `s = w_s . h + b_s` and the evidence concentration
   `kappa = kappa_min + softplus(w_k . h + b_k)`. Three global learned
   scalars define `tau_0 = center - gap / 2`, `tau_1 = center + gap / 2`
   with `gap = softplus(raw_gap) + eps`, and a positive
   `rho = softplus(raw_slope) + eps`. The cumulative logits are
   `ell_1 = rho * (s - tau_0)` and `ell_2 = rho * (s - tau_1)`.
4. **Fine-label ladder**. From `q_1 = sigmoid(ell_1)` and
   `q_2 = sigmoid(ell_2)` the head builds the rank-consistent
   distribution `p_fine = (1 - q_1, q_1 - q_2, q_2)`, the Dirichlet
   evidence parameters `alpha = 1 + kappa * p_fine`, the total evidence
   `S = sum(alpha)`, and `vacuity = 3 / S`.
5. **Binary readout**. The default `binary_event = "ge2"` selects
   `ell_2` as the benchmark logit, matching the i035 puzzle-binary
   target `B = 1[Y == 2]`. With `num_classes = 1` the head returns
   `logits` of shape `(batch,)`. With `num_classes = 2` the head returns
   `(batch, 2)` logits formed as `[zeros, binary_logit]` so that the
   binary cross-entropy reduces to `BCE(ell_2, B)`.

## Outputs

`forward(x)` returns a dictionary that always contains:

- `logits`: puzzle-binary logit consumed by the puzzle-binary trainer.
- `ordinal_logits`: `(batch, 2)` containing `[ell_1, ell_2]`.
- `fine_probs`: rank-consistent `(batch, 3)` distribution over
  `{0, 1, 2}`.
- `alpha`: Dirichlet evidence parameters `(batch, 3)`.
- `q_ge1`, `q_ge2`: cumulative survival probabilities.
- `near_or_puzzle_logit`, `puzzle_logit`: cumulative logits exposed for
  auxiliary losses.
- `score`, `evidence_concentration`, `thresholds`, `vacuity`,
  `threshold_gap`, `slope`: diagnostics for selective evaluation and
  ablation reports.

## Encoding Adapter Policy

- `simple_18` (default): supported. The stem validates
  `input_channels = 18` and treats every channel as a learned tensor
  input.
- `lc0_static_112` and `lc0_bt4_112`: supported only as opaque learned
  tensors when `input_channels = 112`. No deterministic geometry is
  derived from history channels and the leakage policy of the source
  packet is preserved.

CRTK / source / verification metadata is never consumed as input.

## Implementation Binding

- Registered model name: `ordinal_evidence_ladder_network`.
- Source implementation:
  `src/chess_nn_playground/models/trunk/ordinal_evidence_ladder.py`.
- Idea-local wrapper:
  `ideas/registry/i035_ordinal_evidence_ladder_network/model.py`,
  which exposes `build_model_from_config(config)` and delegates to
  `build_ordinal_evidence_ladder_network_from_config`.
- Registry wiring: `src/chess_nn_playground/models/registry.py`
  registers the bespoke builder, taking precedence over the
  auto-registered `ResearchPacketProbe` fallback that would otherwise
  resolve the slug from `RESEARCH_PACKET_MODEL_NAMES`.
