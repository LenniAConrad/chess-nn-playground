# Implementation Notes

- Central model code:
  `src/chess_nn_playground/models/primitives/regret_saddlepoint_network.py`.
- Shared candidate/reply utilities:
  `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Idea-local wrapper:
  `ideas/registry/p002_regret_saddlepoint/model.py`.
- Registry key: `regret_saddlepoint` (the legacy alias
  `regret_saddlepoint_network` resolves to the same builder so existing
  smoke tests stay green).
- Source primitive:
  `ideas/research/primitives/codex_02_regret_saddlepoint.md`.

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Candidate and reply tokens are compiled inside the model by
two independent `BoardTokenAttention` pools over the i193 trunk's
spatial features. No `python-chess` call is required inside the
forward pass.

## Stop-gradient and ablation contract

The trunk supplies `base_logit` and the joint pool feature; both are
gradient-connected. The RSP solver is unrolled with damped updates,
so the backward pass is fully differentiable, matching the documented
"controlled unrolled backward" path in the source packet. Numerical
stability follows from the entropy temperatures `tau_p`, `tau_q` and
the damping factor (defaults 0.45 / 0.35, matching the packet).

The payoff bilinear scale `scale = tanh(payoff_scale)` is initialised
at zero so `A` starts at the batch-broadcast bias and the saddle is
well-defined.

## Output dict contract

The model output is a `dict[str, Tensor]` following the i193 contract,
extended with:

- `logits` (rebound to `base_logit + primitive_delta`)
- `base_logit`           (i193 logit, retained for diagnostics)
- `primitive_delta`      (`primitive_gate * primitive_delta_raw`)
- `primitive_delta_raw`  (head MLP output)
- `primitive_gate`       (sigmoid scalar gate)
- `primitive_gate_logit`
- `rsp_saddle_value`
- `rsp_attacker_regret`
- `rsp_defender_regret`
- `rsp_exploitability`
- `rsp_attacker_entropy`
- `rsp_defender_entropy`
- `rsp_best_witness_index` (argmax of `p`)
- `rsp_best_reply_index`   (argmax of `q`)

All per-sample scalar tensors are emitted in the standard
one-column-per-key shape so the shared trainer copies them into
`predictions_<split>.parquet`.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` that wraps the bespoke i193
`ExchangeThenKingDualStreamNetwork` and adds: two learnable
`BoardTokenAttention` token compilers, a bilinear payoff projection,
the differentiable entropy-regularized saddle solver, and two new MLP
heads. It does not call `build_research_packet_probe_from_config`,
does not delegate to a shared CNN / MLP / NNUE / LC0 baseline builder,
and has its own forward pass.
