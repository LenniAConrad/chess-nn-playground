# Implementation Notes

- Central model code:
  `src/chess_nn_playground/models/primitives/witness_counterwitness_quantifier_network.py`.
- Shared candidate/reply utilities:
  `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Idea-local wrapper:
  `ideas/registry/p005_witness_counterwitness_quantifier/model.py`.
- Registry key: `witness_counterwitness_quantifier_network`.
- Source primitive:
  `ideas/research/primitives/codex_05_witness_counterwitness_quantifier.md`.

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Candidate and reply tokens are compiled inside the model by
two independent `BoardTokenAttention` pools.

## Stop-gradient and ablation contract

The trunk supplies `base_logit` and the joint pool feature; both are
gradient-connected. The WCQ operator uses logsumexp / softmax
primitives — fully differentiable. Candidate / reply soft assignments
are saved in the backward pass via PyTorch's autograd graph, matching
the "saved witness and counterwitness soft assignments in the
backward pass" wording from the source packet.

The counter-envelope soft fallback (zero instead of `-inf` when a
candidate has no surviving counterwitness) is implemented in the
operator and exercised under masked-input tests.

## Output dict contract

The model output is a `dict[str, Tensor]` following the i193 contract,
extended with:

- `logits` (rebound to `base_logit + primitive_delta`)
- `base_logit`           (i193 logit, retained for diagnostics)
- `primitive_delta`      (`primitive_gate * primitive_delta_raw`)
- `primitive_delta_raw`  (head MLP output)
- `primitive_gate`       (sigmoid scalar gate)
- `primitive_gate_logit`
- `wcq_value`
- `wcq_max_margin`
- `wcq_min_margin`
- `wcq_counter_envelope_max`
- `wcq_witness_entropy`
- `wcq_best_witness_index` (argmax of margin)
- `wcq_best_counter_index` (argmax of counter-weights flat)
- `wcq_claim_max`

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` that wraps the bespoke i193
`ExchangeThenKingDualStreamNetwork` and adds: two learnable
`BoardTokenAttention` token compilers, a claim MLP, a bilinear
counter MLP, the nested logsumexp quantifier with independent
temperatures, and two new MLP heads. It does not call
`build_research_packet_probe_from_config`, does not delegate to a
shared CNN / MLP / NNUE / LC0 baseline builder, and has its own
forward pass.
