# Math Thesis

`Relation-Masked Attention Graft over i018` -- i258.

The benchmark contract is binary puzzle classification on the canonical
tagged split (`crtk_sample_3class_unique_crtk_tags`). Let `y in {0, 1}`
be the binary label and `s` be the per-board puzzle logit. The deployment
metric set is aggregate PR-AUC and matched-recall near-puzzle FP at
`puzzle_recall in {0.80, 0.85}`, with slice PR-AUC on `hard`, `equal`,
`endgame`, `promotion`, `underpromotion`, and `mate_in_1` as required
checks (no slice regression).

## Trunk Reuse Identity

Let `T_i018` be the i018 forward map from a simple_18 board `x` to the
post-sheaf square states `h \in R^{B x 64 x d}` and let `R(x) \in
[0, 1]^{B x 12 x 64 x 64}` be the typed relation tensor. The i258
forward is

```text
h0(x)          = T_i018(x)
M(x)           = scramble(R(x))               # identity unless scramble_relations
h_attn(x)      = AttentionGraft(h0(x), M(x))
final_logit(x) = i018_readout(h_attn(x), incidence(x))
```

where `AttentionGraft` is the small relation-masked attention block
defined below. When the attention output projection is zero or when the
gate is zero, `h_attn(x) = h0(x)` exactly, and `final_logit(x)` recovers
the i018 logit identically. That is the i258 base identity: the graft
cannot make i018 worse without first opening the gate.

## Edge Score and Neighborhood

The edge score per ordered pair of squares `(u, v)` is

```text
u_uv     = max_r M(x)_{uvr}
k_uv     = sum_{r in {king_attack_us, king_attack_them, pin}} M(x)_{uvr}
edge_uv  = u_uv + king_boost * k_uv + I[u = v]
```

with `king_boost = 0.5`. The neighborhood selector is

```text
N(u) = TopK(edge_{u, .}, K)
```

with `K = 8` (default). The four neighborhood modes from the research
packet only change the edge score, not the rest of the math:

| Mode | Edge score | Structural mask |
|---|---|---|
| `relation` | `u + king_boost * k + I` | nonzero where any of those is nonzero |
| `global`   | constant ones (+ tiny self bias) | all ones |
| `king_zone` | `sum_{king-zone rels} M + I` | nonzero where king-zone rels are nonzero |
| `candidate` | own_piece_u * own_piece_v + I | nonzero where own-piece outer product is nonzero |

## Attention with Low-Rank Relation Bias

For per-head dimension `d_h = attn_dim / num_heads` (default
`attn_dim = 24`, `num_heads = 2`, so `d_h = 12`),

```text
Q = h0 W_Q,    K = h0 W_K,    V = h0 W_V   (bias-free)
phi_{uv}      = tanh(U r_{uv})              (U: R^{12 -> 4})
b^type_{h,uv} = a_h^T phi_{uv}              (a_h: R^4)
b^log_{h,uv}  = c_h * log(1 + edge_{uv} / eps)
b_{h,uv}      = b^type_{h,uv} + b^log_{h,uv}
alpha_{h,uv}  = softmax_{v in N(u)}(q_{h,u}^T k_{h,v} / sqrt(d_h) + b_{h,uv})
z_u           = concat_h sum_{v in N(u)} alpha_{h,uv} v_{h,v}
delta h_u     = W_O z_u                      (W_O zero-init)
```

Softmax is restricted to `N(u)` by setting masked logits to `-1e9`. Rows
whose structural mask sums to zero fall back to the uniform distribution
over `N(u)` so entropy and message expectations stay finite.

## Gated Residual

The per-square scalar features

```text
rho_u   = max_v u_{uv}
kappa_u = mean_v k_{uv}
pi_u    = mean_v M_{uv, pin}
```

feed a sigmoid gate

```text
gate_u  = sigmoid(w_g^T [h0_u || rho_u || kappa_u || pi_u] + b_g)
h_attn_u = h0_u + gate_u * delta h_u
```

with `b_g = -2.0` (default gate ~0.12 at initialisation). The gate weight
`w_g` is zero-initialised so the gate value at construction is exactly
`sigmoid(b_g)`. The attention output `W_O` is also zero-initialised, so
`h_attn_u = h0_u` to machine precision before any training step.

## Falsifier Identity

The i018 `scramble_relations: true` path applies a per-(batch, relation)
degree-preserving random column permutation to `R(x)` before any of the
trunk math runs:

```text
M_scrambled(x) = gather(R(x), random_permutation_per_relation)
```

i258 reuses this path for both the sheaf diffusion and the attention
graft. The relation-density per relation plane is preserved; only the
spatial pattern of edges is randomized. The random-mask ablation is then
the single config flip `scramble_relations: true` -- no extra falsifier
wiring is required.

## Parameter Budget

For `channels = 64`, `attn_dim = 24`, `num_heads = 2`, `relation_rank = 4`,
`top_k = 8`, and `hidden_dim = 76` (reduced from i018's 96):

```text
QKV projection (64 -> 72)                4,608 params
Output projection (24 -> 64)             1,536 params
Relation low-rank projector (12 -> 4)       48 params
Per-head relation vectors                    8 params
Per-head log scale                           2 params
Gate (67 -> 1) + LayerNorm                 196 params
i018 readout reduction (96 -> 76)       -6,360 params (saved)
-----------------------------------------------
Net delta vs matched-budget i018:           ~+38 params
```

The arithmetic cost is dominated by the K=8 gather-and-attention pass on
the 64-square graph -- about 0.4 to 0.5M extra multiply-adds versus
i018's ~9M baseline, well under a 10% overhead in typical batch sizes.

## Falsifiers

- `attention_disabled` (gate forced to 0) closely matches the matched-
  budget i018 baseline (within seed noise on the validation split). If
  the disabled version meaningfully under-performs i018, the graft has
  leaked through the readout reduction rather than through the gated
  delta.
- `relation` beats `global` by `>= 0.003` test PR-AUC under the matched
  recipe. If global ties relation, the chess constraint is not
  load-bearing -- prefer scaling a graph-transformer trunk instead.
- `relation` beats `scramble_relations: true` by `>= 0.010` test PR-AUC.
  If random masks tie, the graft is just an extra content mixer; do not
  scale.
- `king_zone` and `candidate` modes do not regress vs `relation` by more
  than `~0.005` on the matched-recall hard slices. If they do, the
  neighborhood selector is over-fit to the specific relation profile and
  needs a softer selector.

If the relation-masked variant beats `global` and `scramble_relations`
controls but does not beat the matched i018 baseline by the C1 target
(`+0.003 to +0.008` mean test PR-AUC), report as "mechanism-direction
positive, magnitude inconclusive" rather than promoting i258 over i018.
