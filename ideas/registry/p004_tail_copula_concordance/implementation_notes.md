# Implementation Notes

- Central model code:
  `src/chess_nn_playground/models/primitives/tail_copula_concordance_network.py`.
- Shared candidate/reply utilities:
  `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Idea-local wrapper:
  `ideas/registry/p004_tail_copula_concordance/model.py`.
- Registry key: `tail_copula_concordance_network`.
- Source primitive:
  `ideas/research/primitives/codex_04_tail_copula_concordance.md`.

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. The 64-site evidence field is computed inside the model by a
1x1 conv on the i193 trunk's spatial features.

## Stop-gradient and ablation contract

The trunk supplies `base_logit` and the joint pool feature; both are
gradient-connected. The TCC operator uses sigmoid, log, and softmax
primitives — fully differentiable.

The pairwise soft-rank step is `O(N^2 * C)` quadratic in sites. For
chess `N = 64` this is fine; for larger fields a fused differentiable
sorting kernel would be a future optimisation.

## Output dict contract

The model output is a `dict[str, Tensor]` following the i193 contract,
extended with:

- `logits` (rebound to `base_logit + primitive_delta`)
- `base_logit`           (i193 logit, retained for diagnostics)
- `primitive_delta`      (`primitive_gate * primitive_delta_raw`)
- `primitive_delta_raw`  (head MLP output)
- `primitive_gate`       (sigmoid scalar gate)
- `primitive_gate_logit`
- `tcc_tail_mean`
- `tcc_tail_max`
- `tcc_channel_mass_mean`
- `tcc_channel_mass_max`
- `tcc_concordance_trace`
- `tcc_site_mass_max`

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` that wraps the bespoke i193
`ExchangeThenKingDualStreamNetwork` and adds: a per-square evidence
1x1 conv, the differentiable soft-rank + tail-membership +
concordance reducer, and two new MLP heads. It does not call
`build_research_packet_probe_from_config`, does not delegate to a
shared CNN / MLP / NNUE / LC0 baseline builder, and has its own
forward pass.
