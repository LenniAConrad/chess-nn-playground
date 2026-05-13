# Implementation Notes

- Central model code:
  `src/chess_nn_playground/models/primitives/reply_channel_capacity_network.py`.
- Shared candidate/reply utilities:
  `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Idea-local wrapper:
  `ideas/registry/p003_reply_channel_capacity/model.py`.
- Registry key: `reply_channel_capacity_network`.
- Source primitive:
  `ideas/research/primitives/codex_03_reply_channel_capacity.md`.

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Candidate and reply tokens are compiled inside the model by
two independent `BoardTokenAttention` pools.

## Stop-gradient and ablation contract

The trunk supplies `base_logit` and the joint pool feature; both are
gradient-connected. The RCC solver uses softmax + Blahut-Arimoto
fixed-point iterations and is fully differentiable.

The bilinear logit scale `1 + tanh(logit_scale)` starts at 1 so the
solver receives a well-conditioned softmax distribution from the
start.

## Output dict contract

The model output is a `dict[str, Tensor]` following the i193 contract,
extended with:

- `logits` (rebound to `base_logit + primitive_delta`)
- `base_logit`           (i193 logit, retained for diagnostics)
- `primitive_delta`      (`primitive_gate * primitive_delta_raw`)
- `primitive_delta_raw`  (head MLP output)
- `primitive_gate`       (sigmoid scalar gate)
- `primitive_gate_logit`
- `rcc_capacity_nats`
- `rcc_capacity_bits`
- `rcc_conditional_entropy`
- `rcc_output_entropy`
- `rcc_capacity_gap`
- `rcc_prior_entropy`
- `rcc_marginal_entropy`

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` that wraps the bespoke i193
`ExchangeThenKingDualStreamNetwork` and adds: two learnable
`BoardTokenAttention` token compilers, a bilinear logit projection,
the differentiable Blahut-Arimoto channel-capacity solver, and two
new MLP heads. It does not call
`build_research_packet_probe_from_config`, does not delegate to a
shared CNN / MLP / NNUE / LC0 baseline builder, and has its own
forward pass.
