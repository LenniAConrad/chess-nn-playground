# Math Thesis

Tactical State Bottleneck Inference

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-28_0901_tuesday_new_york_tactical_latent.md`.

Working thesis: Select **Tactical State Bottleneck Inference**.

## Variables

```text
x in R^{C x 8 x 8}                board tensor (current position only)
y in {0, 1}                       puzzle target, y = 1[fine_label == 2]
z = (m, a, t, r, v, h)            typed tactical-state tuple
```

The latent cardinalities are fixed by the packet:

```text
m  motif         in {0..9}      (10)
a  anchor        in {0..63, null}   (65)
t  target        in {0..63, null}   (65)
r  relation      in {0..7}      (8)
v  vulnerability in {0..7}      (8)
h  tempo         in {0..3}      (4)
```

## Model

`H_omega(x)` is the residual board trunk producing both the spatial map
`h in R^{B x D x 8 x 8}` and the pooled vector `pooled in R^{B x D}`.

The prior network `p_psi(z | x)` factorises across the six groups:

```text
p_psi(z_j | x) = softmax(prior_head_j(h, pooled))
```

For the `anchor` and `target` groups the logits are
`[B, 65] = [conv1x1(h).flatten(); null_logit]`. For the remaining groups
the logits are `[B, K_j] = Linear_j(pooled)`.

The latent embedding for a single forward pass is

```text
e_j(z_j)        = z_j @ E_j      with E_j the per-group embedding table
z_emb           = concat_j e_j(z_j)
latent_logit    = latent_head([pooled, z_emb])
direct_logit    = direct_head(pooled)
logit           = latent_logit + alpha_direct * direct_logit
```

The predictive model marginalises through the soft latent assignment:

```text
p(y = 1 | x) = sigmoid(logit)
```

The posterior network `q_phi(z | x, y)` uses the same group factorisation
with a label embedding fused into both the spatial map and the pooled
features. It is training-only; inference uses the prior path exclusively.

## Objective

The full TSBI objective combines five terms at the levels recommended by
the packet:

```text
L = L_pred                                             (posterior-path BCE)
  + lambda_prior * L_prior_pred                        (prior-path BCE)
  + beta_kl      * sum_j max(KL(q_j || p_j), tau_j)   (free bits)
  + lambda_usage * sum_j KL(mean_batch q_j || U_j)    (anti-collapse)
  + lambda_entropy * sum_j relu(H_min_j - H(q_j))     (entropy floor)
```

The implementation in
`src/chess_nn_playground/models/tactical_state_bottleneck.py` exposes the
full bundle through `forward_train` and `tactical_state_loss_components`.
The shared puzzle_binary trainer in this repository wires only the
`L_prior_pred` term, which the packet labels mandatory because it closes
the train/inference gap by training the prior latents `p_psi(z | x)` that
inference uses.

## Thesis

A puzzle position is more likely when the board supports a compact,
localised tactical explanation. The bottleneck is the categorical
tactical-state tuple `z`. If the bottleneck is real, conditioning the
puzzle logit on `z` should improve generalisation and produce
non-collapsed, board-sensitive latent assignments. The repository tests
this prediction under the canonical `puzzle_binary` benchmark contract
using the bespoke `TacticalStateBottleneckModel`.
