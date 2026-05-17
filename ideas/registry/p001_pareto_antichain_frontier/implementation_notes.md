# Implementation Notes

- Central model code:
  `src/chess_nn_playground/models/primitives/pareto_antichain_frontier_network.py`.
- Shared candidate/reply utilities:
  `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Idea-local wrapper:
  `ideas/registry/p001_pareto_antichain_frontier/model.py`.
- Registry key: `pareto_antichain_frontier` (matches the idea slug).
  The legacy alias `pareto_antichain_frontier_network` is retained in
  the registry manifest so older tests resolve to the same builder.
- Source primitive:
  `ideas/research/primitives/codex_01_pareto_antichain_frontier.md`.

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Candidate tokens are compiled inside the model by the
`BoardTokenAttention` pool from the i193 trunk's spatial features.
No `python-chess` call is required inside the forward pass.

## Stop-gradient and ablation contract

The trunk supplies `base_logit` and the joint pool feature; both are
gradient-connected so the trunk continues to learn. The PAFR operator
itself is differentiable (sigmoid, log1p, softmax composition). The
ablations either:

- collapse the operator's partial order (scalar_max, single_channel,
  uniform_frontier), or
- decouple channels from candidates (shuffle_channels), or
- bypass the primitive entirely (zero_delta, trunk_only,
  disable_gate).

`shuffle_channels` is the primary falsifier — see `ablations.md`.

## Output dict contract

The model output is a `dict[str, Tensor]` following the i193 contract,
extended with:

- `logits` (rebound to `base_logit + primitive_delta`)
- `base_logit`           (i193 logit, retained for diagnostics)
- `primitive_delta`      (`primitive_gate * primitive_delta_raw`)
- `primitive_delta_raw`  (head MLP output)
- `primitive_gate`       (sigmoid scalar gate)
- `primitive_gate_logit`
- `pafr_frontier_width`
- `pafr_frontier_entropy`
- `pafr_max_nondominated_prob`
- `pafr_summary_norm`
- `pafr_utility_mean`
- `pafr_utility_max`

All per-sample scalar tensors are emitted in the standard
one-column-per-key shape so the shared trainer copies them into
`predictions_<split>.parquet`.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` that wraps the bespoke i193
`ExchangeThenKingDualStreamNetwork` and adds: a learnable
`BoardTokenAttention` candidate compiler, a bilinear utility table, the
PAFR pairwise-dominance reducer, and two new MLP heads. It does not
call `build_research_packet_probe_from_config`, does not delegate to a
shared CNN / MLP / NNUE / LC0 baseline builder, and has its own forward
pass.

The `implementation_kind: bespoke_model` declaration is consistent with
the `audit_implementation_kinds.py` heuristics.
