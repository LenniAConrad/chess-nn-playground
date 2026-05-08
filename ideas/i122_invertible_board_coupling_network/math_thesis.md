# Math Thesis

Invertible Board Coupling Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.

Batch candidate rank: `4`.

Working thesis: Standard encoders can discard information early, which
makes it hard to know whether a model learned legitimate current-board
structure or fragile shortcuts. A reversible board encoder preserves
information by construction and classifies from latent distortions
created by invertible coupling blocks. We project the `simple_18` board
into a wider feature space `D` and apply a stack of invertible blocks of
the form

```text
z₀ = pad(x)            # zero-padded reversible projection (B, D, 8, 8)
z_{ℓ+1} = AC_ℓ ∘ M_ℓ (z_ℓ)
```

where each block is the composition of:

- an invertible 1x1 channel mixing `M_ℓ(z) = W_ℓ z` with full-rank
  `W_ℓ ∈ ℝ^{D×D}` initialised to a random orthogonal matrix; the inverse
  is `M_ℓ⁻¹(y) = W_ℓ⁻¹ y` and is exact under `torch.linalg.inv`;
- an affine coupling

  ```text
  y_a = x_a
  y_b = x_b ⊙ exp(s(x_a)) + t(x_a)
  ```

  with the active half alternating across blocks (channel-split swap).

To keep the affine transform bounded — preventing the scale explosion the
source packet flags as a failure mode — we clamp the raw scale through

```text
s = scale_clamp · tanh(raw_s),   |s| ≤ scale_clamp.
```

The inverse `x_b = (y_b − t(x_a)) / exp(s(x_a))` is therefore numerically
stable, which is exactly the property the `frozen_inverse_check` ablation
checks for. The whole encoder `E = M_L ∘ AC_{L−1} ∘ M_{L−1} ∘ ⋯ ∘ AC_0 ∘ M_0`
is invertible by construction, so

```text
inverse_reconstruction_error = mean | z₀ − E⁻¹(E(z₀)) |
```

is the natural diagnostic for invertibility quality. At init, with the
last conv of each `s,t` predictor zero-initialised, every block is the
identity and the reconstruction error is machine-precision-small.

Classification reads out from the final latent together with per-block
scale statistics, latent energy, latent peak, and the reconstruction
error — i.e. from latent distortions induced by the invertible
transformation, exactly as the markdown candidate prescribes.
