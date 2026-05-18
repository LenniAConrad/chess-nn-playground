# Architecture

`Relation-Masked Attention Graft over i018` (i258) keeps the i018 oriented
tactical sheaf trunk intact and inserts a single small sparse-attention
residual block between the last sheaf diffusion block and the readout head.
The attention block attends only over a fixed top-K neighbor list per source
square selected from i018's typed relation masks, adds a low-rank
relation-conditioned bias to the attention logits, and applies the update
through a gated residual whose output projection is zero-initialised so the
parent i018 trunk is recovered when the gate sits at zero.

The source research packet is
`ideas/research/packets/classic/i258_relation_masked_attention_i018.md`. The
implementation realises the packet's "small post-sheaf relation-masked
attention graft" first-deployment target. The full transformer alternative
flagged as risky in the packet is intentionally not implemented.

## Implementation Binding

- Registered model name: `relation_masked_attention_i018`
- Source implementation:
  `src/chess_nn_playground/models/trunk/relation_masked_attention_i018.py`
  (`RelationMaskedAttentionI018Net`,
  `RelationMaskedAttentionGraft`,
  `build_relation_masked_attention_i018_from_config`)
- Idea-local wrapper:
  `ideas/registry/i258_relation_masked_attention_i018/model.py`
  (`build_model_from_config`)
- Registry manifest key: `relation_masked_attention_i018` in
  `src/chess_nn_playground/models/_registry_manifest.py`

## Dataflow

```text
simple_18 board
   |
   |-- BoardStateAdapter (reused from i018)
   |
   |-- TacticalIncidenceBuilder (reused from i018)
   |      |
   |      `-- relation_masks  (B, 12, 64, 64)
   |
   |-- (optional) scramble_relations: degree-preserving random rewiring
   |      reused from i018's falsifier path
   |
   |-- SquareTokenEncoder (reused from i018) -> h: (B, 64, channels)
   |
   |-- SheafDiffusionBlock x depth (reused from i018) -> h: (B, 64, channels)
   |
   |-- RelationMaskedAttentionGraft (new, one block)
   |      |
   |      |-- top-K neighborhood from union(relation_masks) + king_boost
   |      |-- low-rank relation-conditioned bias (rank=4)
   |      |-- multi-head sparse attention (heads=2, attn_dim=24)
   |      |-- zero-init output projection
   |      `-- sigmoid gate * delta (gate bias = -2.0)
   |
   |-- existing i018 readout head and diagnostics
   |
   `-- one puzzle logit + extended diagnostics
```

The block-level shapes (`B, 64, channels`) and the readout contract match
i018 exactly. The graft consumes the same `relation_masks` tensor the sheaf
diffusion already uses, so the `scramble_relations: true` falsifier feeds
both code paths uniformly.

## Equations

Let `h \in R^{B x 64 x d}` be the post-sheaf square states, and let
`M \in [0, 1]^{B x 12 x 64 x 64}` be i018's typed relation masks. The
union and king-zone-emphasised edge score are

```text
u_{uv}     = max_r M_{uvr}
k_{uv}     = sum_{r in {king_attack_us, king_attack_them, pin}} M_{uvr}
edge_uv    = u_{uv} + king_boost * k_{uv} + I[u = v]
```

The top-K neighbor index per source square `u` is `topk_K(edge_{u .})`. With
`Q, K, V = hW_Q, hW_K, hW_V` and per-head dimension `d_h = attn_dim /
num_heads`, the attention is

```text
alpha_{h,uv} = softmax_{v in N(u)} (q_{h,u}^T k_{h,v} / sqrt(d_h) + b_{h,uv})
z_u          = concat_h sum_{v in N(u)} alpha_{h,uv} v_{h,v}
delta h_u    = W_O z_u
```

The relation-conditioned bias is low-rank:

```text
phi_{uv}     = tanh(U r_{uv})              # U: 12 -> relation_rank=4
b^{type}_{h,uv} = a_h^T phi_{uv}           # per-head learned weight
b^{log}_{h,uv}  = c_h * log(1 + edge_{uv} / eps)
b_{h,uv}     = b^{type}_{h,uv} + b^{log}_{h,uv}
```

The residual update is gated and structurally masked:

```text
gate_u   = sigmoid(w_g^T [h_u || rho_u || kappa_u || pi_u])
out_u    = h_u + gate_u * delta h_u
```

with `rho_u = max_v u_{uv}`, `kappa_u = mean_v k_{uv}`, and `pi_u =
mean_v M_{uv, pin}`. The output projection `W_O` is zero-initialised and
the gate bias is `-2.0`, so a freshly built graft starts as approximate
identity over `h`.

## Neighborhood Modes

`relation_attention.neighborhood` (config flag) controls the neighborhood
selector. Four modes match the research packet's ablation grid:

| Mode | Edge score | Notes |
|---|---|---|
| `relation` (default) | `union + king_boost * king_zone` | Primary i258 design. |
| `global` | constant ones | Generic global attention falsifier. |
| `king_zone` | only king-zone / pin relations | High-precision tactical / mate check. |
| `candidate` | own-piece-by-own-piece outer-product mask | Move-targeted reweighting. |

The mode flips only the edge score (and structural mask). The low-rank bias,
gate, dropout, and zero-init contract are unchanged.

## Cost

At default scale (`channels=64`, `depth=2`, `stalk_dim=8`,
`hidden_dim=76`, attention: `num_heads=2`, `attn_dim=24`, `top_k=8`,
`relation_rank=4`):

- Attention QKV + output: 64 * 24 * 3 + 24 * 64 ~= 4,608 + 1,536 = 6,144
- Relation low-rank projector: 12 * 4 = 48
- Per-head relation vectors / log-scale: 2 * 4 + 2 = 10
- Gate (67 -> 1) + LayerNorm: 68 + 128 = 196
- Reducing readout `hidden_dim` from 96 to 76 removes ~6,360 parameters

The net delta versus matched-budget i018 is on the order of tens of
parameters (`+6,398 - 6,360 ~ +38`). The attention forward adds a single
gather-and-attention pass on a 64-square graph with `K = 8`, so the
arithmetic overhead is ~5% of the i018 sheaf step.

## Inputs and Contract

- Input: simple_18 current-board tensor `(B, 18, 8, 8)`.
- Output: dict with `logits` of shape `(B,)` plus per-sample diagnostic
  scalars. Compatible with the repo's shared trainer artifact pipeline.
- The model never reads CRTK metadata, source labels, verification flags,
  PVs, or engine evaluations.

## Diagnostics

The forward output contains all i018 diagnostics plus the attention
extras:

| Key | Meaning |
|---|---|
| `attention_entropy` | Mean attention entropy across the K neighbors and heads. |
| `attention_king_share` | Fraction of attention mass landing on king-zone relations. |
| `attention_gate_mean` | Mean residual gate value per sample. |
| `attention_delta_norm` | L2 norm of the attention residual `delta h`. |
| `attention_neighbor_count` | Mean neighbor count actually used (structural mask). |
| `attention_relation_bias_norm` | L2 norm of the relation-conditioned bias. |

These let the matched-recall report attribute lifts (or non-lifts) to the
attention block rather than to general trunk drift.

## Scope Notes

- The post-sheaf "one attention block before readout" placement is the
  conservative deployment target from the packet. A multi-block variant is
  documented as a planned follow-up rather than promoted here.
- Exact pseudo-legal move enumeration for the `candidate` mode is *not*
  implemented; the candidate mask is the deterministic own-piece outer
  product (any own piece can act on any square, gated by the structural
  mask). The exact pseudo-legal alternative requires the i248 TSDP rule
  cache and is deferred.
- The extended packet loss (matched-recall / slice-restricted ranking /
  KD distillation) is *not* bundled with this architecture promotion;
  the default trainer uses BCE-with-logits on the puzzle logit so the
  comparison against i018 stays honest at the architecture layer.
