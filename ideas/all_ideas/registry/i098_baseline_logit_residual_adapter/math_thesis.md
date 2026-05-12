# Math Thesis

Baseline Logit Residual Adapter

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.

Batch candidate rank: `2`.

Working thesis: The existing simple CNN likely has systematic errors. A small residual adapter can test what information remains after the baseline logit and latent representation are known.

Decomposition. Let `f_b(x) = (z_b, s_b)` be a `simple_cnn`-shaped baseline branch with pooled latent `z_b ∈ R^{c}` and scalar logit `s_b ∈ R`. Let `m(x) ∈ R^{26}` be a deterministic board summary (material, occupancy, central / king-ring pressure, rank-file imbalance). The adapter is a learned function `r_θ : (x, z_b, s_b, m) → (s_r, g)` with residual logit `s_r ∈ R` and gate `g ∈ (0, 1)`. The classifier is

```
s(x) = s_b(x) + α · g(x) · s_r(x)
```

with `α = residual_scale` (default `1`). Training uses BCE-with-logits on `s(x)`, and `s_b` is detached when feeding the adapter's conditioning vector `c = [z_b, s_b, m]` so the residual branch carries no gradient back into the baseline through its conditioning input.

Identifiability. By construction, any predictor of the form `s_b + α · g · s_r` reduces to the baseline at `α = 0` or `g ≡ 0`, so `‖α · g · s_r‖` is the *baseline-residual signal* the adapter has extracted from the board representation `(x, m)` after `s_b` and `z_b` are fixed. The detached condition prevents `r_θ` from cancelling baseline error through `s_b` itself; instead it must encode signal that is genuinely orthogonal to the baseline's pooled latent and logit in a function-space sense.

Diagnostic outputs. The forward pass exposes `baseline_logit`, `residual_logit`, `adapter_correction = α · g · s_r`, `residual_gate = g`, `residual_to_baseline_ratio = |α · g · s_r| / max(|s_b|, ε)`, the `baseline_latent_norm`, and the `adapter_feature_norm` and `adapter_field_energy` of the residual branch's pooled and spatial features. These are sufficient to test the packet's central claim: *what fraction of puzzle-binary signal remains after the baseline logit and latent representation are known?*
