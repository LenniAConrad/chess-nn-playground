# i242 — Chess-Decomposed Attention Network

A direct synthesis of three independently-validated chess-evaluation priors:

| prior | source | role in this architecture |
|---|---|---|
| King-conditioned input features | Stockfish NNUE (HalfKA) | Every per-square embedding is derived from precomputed king-relative geometry (own/enemy king zone, check rays, escape squares, attacker/defender pressure). |
| Exchange + king-safety dual-stream decomposition | i193 (scout winner) | Two parallel sub-trunks specialize: an *exchange* sub-trunk sees attacker/defender bias, a *king* sub-trunk sees king-zone bias. |
| Global multi-head self-attention over square tokens | LC0 BT4 | A third parallel sub-trunk with vanilla self-attention catches long-range piece relationships that conv-only trunks need many layers to model. |

Fused via a learned softmax phase router over the three stream pools.

## Sketch

```
Input: simple_18 board tensor [B, 18, 8, 8]
   │
   ▼
DualStreamFeatureBuilder (reused from i193, deterministic, no learning):
   - exchange planes (own/enemy piece, value, attack counts, defender/attacker pressure)
   - king planes     (own/enemy king zone, check, escape, line-to-zone pressure)
   - precomputed attack tables (used for attention bias below)

   ┌────────────────────────────────┐ ┌────────────────────────────┐ ┌─────────────────┐
   │ Exchange tower (2 blocks)      │ │ King tower (2 blocks)      │ │ Global (2 blks) │
   │ Input: simple_18 + exchange    │ │ Input: simple_18 + king    │ │ Input: simple_18│
   │ Bias:  attacker/defender pairs │ │ Bias:  king-zone pairs     │ │ Bias:  none     │
   └───────────────┬────────────────┘ └─────────────┬──────────────┘ └────────┬────────┘
                   │ mean pool                       │ mean pool               │ mean pool
                   ▼                                 ▼                         ▼
              ex_pool [B, d]                    kg_pool [B, d]            gl_pool [B, d]

   Phase router (MLP):  softmax over (alpha_ex, alpha_kg, alpha_gl)

   Final puzzle logit  =  alpha_ex * ex_head(ex_pool)
                       +  alpha_kg * kg_head(kg_pool)
                       +  alpha_gl * gl_head(gl_pool)
                       +  residual_head(concat(ex_pool, kg_pool, gl_pool))
```

## Key equations

**Per-token chess-aware embedding (king-conditioned, NNUE-style):**

$$
e^{(s)}_E \;=\; \mathrm{Proj}_E\bigl([\,x^{(s)} \,\|\, \phi_E^{(s)}(x)\,]\bigr), \qquad
e^{(s)}_K \;=\; \mathrm{Proj}_K\bigl([\,x^{(s)} \,\|\, \phi_K^{(s)}(x)\,]\bigr)
$$

where $x^{(s)}$ is the simple_18 channels at square $s$ and $\phi_E, \phi_K$ are the deterministic exchange / king feature builders. The dependence on king positions makes this **HalfKA-equivalent**.

**Chess-aware attention bias (i193 prior, lifted into attention):**

$$
\mathrm{Attn}_E(Q, K, V) \;=\; \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_h}} + B_E(x)\right) V
$$

where $B_E(x)_{s,t}$ is large iff square $s$ attacks $t$ or vice versa (computed from the precomputed attacker tables); $B_K(x)_{s,t}$ is large iff $s$ or $t$ sits in either king's 8-ring.

**Phase fusion (i193 prior, generalized to 3 streams):**

$$
\hat{y}(x) \;=\; \alpha_E\,h_E(e_E) \,+\, \alpha_K\,h_K(e_K) \,+\, \alpha_G\,h_G(e_G) \,+\, h_R(e_E \oplus e_K \oplus e_G),
\quad \alpha = \mathrm{softmax}(\mathrm{Router}(e_E \oplus e_K \oplus e_G))
$$

## Sizing

At base scout scale (`embed_dim=64, num_heads=4, 2 blocks per stream`) the model has **~271k parameters** — directly comparable to i193 (157k) and the rule-symmetry family (~180k each).

## What this architecture has that BT4 doesn't

| property | BT4 | i242 |
|---|---|---|
| Global attention over 64 squares | ✓ | ✓ (global stream) |
| King-conditioned input features | ✗ | ✓ (i193 builder reused) |
| Exchange/king dual-stream decomposition | ✗ | ✓ (two specialised streams) |
| Chess-aware attention bias | ✗ | ✓ (exchange + king biases) |
| Per-stream interpretability (alpha weights, per-stream logits) | ✗ | ✓ |

## What this architecture gives up vs i193

- Higher parameter count (~270k vs i193's 157k) — adds an attention tower.
- Higher FLOPs/position — attention costs O(n²·d) which at n=64 is small but non-zero.
- Higher per-batch inference time at batch 1 (small absolute, but a 3-stream + attention setup is slightly heavier than i193's conv-only design).

## Predicted ranking (before training)

- Beats i193 (~0.876 test PR AUC) by a small margin (~0.005–0.015) because the global attention recovers the long-range information the conv-only dual-stream loses.
- Roughly ties or slightly beats the rule-symmetry family (~0.86) because both have strong chess priors but i242's are more directly aligned with how chess is evaluated.
- The actual measurement is the scout run.
