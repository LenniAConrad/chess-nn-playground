# Implementation Notes

- Central code:
  `src/chess_nn_playground/models/trunk/relation_masked_attention_i018.py`
  (`RelationMaskedAttentionI018Net`, `RelationMaskedAttentionGraft`,
  `build_relation_masked_attention_i018_from_config`).
- Idea-local wrapper:
  `ideas/registry/i258_relation_masked_attention_i018/model.py`
  (`build_model_from_config`).
- Registry key: `relation_masked_attention_i018`
  (added in `src/chess_nn_playground/models/_registry_manifest.py`).
- Source research packet:
  `ideas/research/packets/classic/i258_relation_masked_attention_i018.md`.

## What is implemented

- The full i018 trunk (`BoardStateAdapter`, `TacticalIncidenceBuilder`,
  `SquareTokenEncoder`, `SheafDiffusionBlock` x depth, `TriadDefectPool`)
  is reused unchanged from
  `chess_nn_playground.models.trunk.oriented_tactical_sheaf`.
- One `RelationMaskedAttentionGraft` is inserted between the last
  `SheafDiffusionBlock` and the readout. The graft computes a sparse
  attention residual over a top-K relation-derived neighbor list,
  applies a low-rank relation-conditioned bias, and adds the result via
  a sigmoid gate. The output projection is zero-initialised and the
  gate bias is negative so the graft starts as approximate identity.
- The four research-packet neighborhood modes (`relation`, `global`,
  `king_zone`, `candidate`) are exposed as a single config flag
  (`model.relation_attention.neighborhood`). The mode flips only the
  edge score and the structural mask; the rest of the math is shared.
- The i018 `scramble_relations` falsifier path is reused. Setting
  `model.scramble_relations: true` randomises the typed relation masks
  before either the diffusion or the attention block sees them, so the
  random-mask falsifier costs zero extra wiring.
- The readout `hidden_dim` defaults to `76` (reduced from i018's `96`)
  so the parameter budget remains within ~40 parameters of the
  matched-budget i018 baseline.
- Forward output extends the i018 diagnostics with
  `attention_entropy`, `attention_king_share`, `attention_gate_mean`,
  `attention_delta_norm`, `attention_neighbor_count`, and
  `attention_relation_bias_norm`. The shared trainer's prediction
  writer picks these up automatically.

## What is intentionally not implemented yet

- A multi-block attention stack. The packet recommends a single
  post-sheaf block as the conservative first deployment, and that is
  what the implementation here exposes. A two-block variant is a
  one-line constructor change but is not promoted by default; the
  matched-budget claim depends on the single-block design.
- The packet's extended loss
  `L = L_BCE + lambda_gate * sum_k E[gate_k] + lambda_slice * L_slice
   + lambda_near * L_near + lambda_kd * KL(...)`. The architecture
  already exports per-sample gate and delta values, so adding this is a
  trainer extension rather than an architecture change. The default
  trainer uses BCE-with-logits to keep the comparison against i018
  honest at the architecture layer.
- Exact pseudo-legal move enumeration for the `candidate` mode. The
  current candidate mask is the deterministic own-piece outer product
  (an own piece on `u` can plausibly act on any square `v`, gated by
  the structural mask). The exact pseudo-legal alternative requires
  the i248 TSDP rule cache and is deferred to a follow-up tied to
  that cache pipeline.
- Hard-concrete / L0-style gates. The current gate is a plain sigmoid
  with a structural mask. Swapping to a stochastic hard-concrete gate
  with an L0 expectation penalty is one class swap inside the graft
  and is documented as a planned follow-up.
- An i249-fast trunk swap. i249 documents the same numerics as i018
  with a faster execution path; swapping the encoder stack in is a
  straightforward replacement once the `forward_features` refactor
  proposed in the packet lands.

## Why a graft instead of a new tower

The packet is explicit that the repo's prior negative attention
evidence (the full transformer benchmark and i242) is most consistent
with "unconstrained global attention under this budget and recipe is
hard" rather than "attention is a dead end for chess evaluation". The
graft design follows from that: attach a small chess-constrained
attention block to the strongest existing relation-aware trunk, with
zero-init and a structural mask, and run the matched-recipe
falsifiers. This is the same operational scale the repo's successful
p007 / p008 sparse-attention primitives use, scaled into a post-sheaf
residual rather than a fresh tower.

## Numerical guards

- Output projection `W_O` is zero-initialised, so before training the
  attention residual is exactly zero and the model recovers i018.
- Gate weight is zero-initialised and bias is `-2.0`, so on
  construction the gate value is `sigmoid(-2.0) ~ 0.119`. Combined
  with zero `W_O` this means `h_attn = h0` to machine precision at
  initialisation.
- Softmax falls back to the uniform distribution over `N(u)` when the
  structural mask sums to zero so entropy and message expectations
  stay finite. This is the same numerical contract used by the
  promotion specialist (i257) and the rejection specialist (i256).
- `top_k` is clamped to `min(top_k, num_squares = 64)` per call.
- Self-edge is always included in the structural mask, so the
  attention residual cannot silently lose its own square's value.

## Output contract

`forward(x: (B, 18, 8, 8)) -> dict` containing the i018 diagnostic
bundle plus the six attention-specific keys. `logits` has shape `(B,)`
and is compatible with the shared trainer's BCE-with-logits objective.

## Default constructor settings

| Field | Default | Source |
|---|---|---|
| `channels` | 64 | i018 base config |
| `depth` | 2 | i018 base config |
| `stalk_dim` | 8 | i018 base config |
| `hidden_dim` | 76 | reduced from i018 96 to match parameter budget |
| `dropout` | 0.1 | i018 base config |
| `attention_num_heads` | 2 | research packet |
| `attention_dim` | 24 | research packet |
| `attention_top_k` | 8 | research packet |
| `attention_relation_rank` | 4 | research packet |
| `attention_king_boost` | 0.5 | research packet |
| `attention_gate_init_bias` | -2.0 | research packet |
| `attention_zero_init_out` | True | research packet |
| `attention_neighborhood` | "relation" | research packet primary variant |
