# Implementation Notes

- Central code:
  `src/chess_nn_playground/models/trunk/defender_exhaustion_cascade_network.py`.
- Registry key: `defender_exhaustion_cascade_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.
- Board-only architecture; engine, verification, source, and CRTK
  metadata are reporting-only and never consumed as model input.
- Typed obligation and resource tokens are produced by softmax
  pooling the per-square trunk features through learnable per-type
  spatial queries (`obligation_type_queries`,
  `resource_type_queries`); identity embeddings
  (`obligation_type_embed`, `resource_type_embed`) are added on top.
- Resource capacity is bounded in `(0, capacity_init]` via a per-token
  MLP and `sigmoid * capacity_init`; allocations consume that
  capacity through the running `pressure` accumulator, so resources
  saturate across cascade steps.
- The "obligation_update" of the source packet is implemented as a
  shared `nn.GRUCell` over the per-obligation `demand_state` driven
  by the threat context. `demand_t` is `softplus(demand_head(state))`
  and is non-negative.
- `demand_pressure` controls how strongly accumulated allocation +
  modulator suppresses future allocation; `cascade_steps` controls
  the cascade depth and is the natural ablation knob for the
  `one_step_only` ablation in the source packet.
- The shared `ResearchPacketProbe` scaffold is no longer used; the
  bespoke `DefenderExhaustionCascadeNetwork` is registered directly
  in `chess_nn_playground/models/registry.py`.
