# Math Thesis

Piece-Conditioned Hypernetwork CNN

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `2`.

Working thesis: The best local filters may depend on material and piece
inventory. A lightweight hypernetwork can condition CNN channel gates
*and* depthwise kernels on safe current-board summaries, adapting the
feature extractor per sample without using engine metadata.

## Conditioning signal

Let `B(x) ∈ R^{18 × 8 × 8}` be the simple_18 board tensor. Define a
deterministic piece-inventory summary `s(x) ∈ R^{27}`:

- `c_w ∈ R^6` -- white piece-type counts (planes 0..5), divided by 64.
- `c_b ∈ R^6` -- black piece-type counts (planes 6..11), divided by 64.
- `Δ_p = c_w − c_b ∈ R^6` -- per-type material deltas.
- `μ_s ∈ R^6` -- means of state planes (12..17), e.g. side-to-move,
  castling rights, en-passant indicators.
- `o ∈ R` -- total occupancy `Σ(c_w + c_b)`.
- `Δ ∈ R` -- net material `Σ c_w − Σ c_b`.
- `Δ_minor ∈ R` -- knight+bishop imbalance, scaled.

`s(x)` is invariant to absolute board scale and contains no engine,
verification, source, or CRTK metadata. It is the *only* signal the
hypernetwork sees.

## Hypernetwork

A shared encoder `E : R^{27} → R^{H_h}` produces an embedding
`e(x) = E(s(x))`. For each of `D` blocks, two heads produce
per-sample weights from `e(x)`:

- `g_d(x) = σ(W_g^{(d)} e(x) + b_g^{(d)}) ∈ R^C` -- channel gates.
- `k_d(x) = W_k^{(d)} e(x) + b_k^{(d)} ∈ R^{C × K × K}` -- depthwise
  kernel weights for a `K × K` convolution (default `K = 3`).

Both heads are initialized so the untrained network behaves like a
residual conv net: `g_d(x) ≈ σ(2.0) ≈ 0.88` per channel and `k_d(x)`
starts at a centered identity-like 3x3 stencil.

## Per-sample feature flow

Let `f_0(x) = E_0 B(x)` where `E_0` is a 1x1 piece-plane embedding to
`C` channels. For block `d ∈ {1, …, D}`:

```
y_d = BN(f_{d-1})
y_d = DW3x3(y_d ; k_d(x))      # depthwise per-sample conv
y_d = W_p^{(d)} y_d            # pointwise 1x1 (static)
y_d = GELU(y_d)
y_d = y_d ⊙ g_d(x)             # broadcast per-channel sigmoid gate
f_d = f_{d-1} + y_d            # residual
```

Mean-pool `f_D` over the spatial axes, LayerNorm the pooled feature
`z(x) ∈ R^C`, and apply a small MLP head to obtain a single puzzle
logit `ŷ(x) ∈ R`.

## Diagnostics

The forward returns the inventory summary, per-block gate means and
binary entropies, per-block kernel L2 norms, per-block gated
post-activation energies, and the scalar material delta. The
hypernetwork thus exposes both *what* it conditioned on and *how
strongly* it adapted its filters per sample.
