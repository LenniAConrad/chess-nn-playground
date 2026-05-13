# Architecture

- Architecture description: i193 dual-stream trunk (`exchange_then_king`)
  evaluated once on the original simple_18 board to produce the base
  puzzle logit and trunk diagnostics, plus an additive TDCD primitive head
  that computes a mixed-partial cross-derivative spectrum over the same
  board through a separate, compact `TDCDEncoder`. A learned per-square
  saliency head selects the top-K critical defenders (K=3 by default), and
  the 2*(K+1) perturbation grid (baseline + per-defender removal, each in
  both tempo phases) is batched through the encoder in a single forward
  pass. A cross-derivative fingerprint (8 spectrum scalars + K sorted
  DeltaDelta values + saliency entropy/concentration + main-effect norms)
  drives a small discriminator MLP and a sigmoid gate; the gated primitive
  delta is added to the i193 base logit to form the final puzzle logit.

- Input format: `(batch, 18, 8, 8)` simple_18 tensor. The tempo involution
  `sigma_T` flips channel 12 only; `delta_k` zeros the enemy-coloured piece
  planes at a single saliency-selected square. Castling planes 13-16 and
  en-passant plane 17 are untouched by `sigma_T` and `delta_k`. No FEN,
  CRTK, tactic, source, Stockfish, or PV metadata is read at any point.

- Forward pass:
  1. `trunk_output = ExchangeThenKingDualStreamNetwork(x)` produces
     `base_logit` plus i193 trunk diagnostics.
  2. `saliency = SaliencyHead(x)` produces a per-square logit, masked to
     enemy-occupied squares; top-K indices and a softmax distribution for
     entropy/concentration diagnostics are returned.
  3. `grid` is a `(batch, 2*(K+1), 18, 8, 8)` tensor with layout
     `[B0+, B0-, B1+, B1-, B2+, B2-, B3+, B3-]` where `+`/`-` is the
     tempo phase (channel 12 = 1 or 0) and slot index identifies which
     defender (if any) was removed.
  4. `features = TDCDEncoder(grid.flatten(0,1)).view(batch, 2*(K+1), F)` is
     the pooled feature stack from a compact GroupNorm + GELU CNN.
  5. Spectrum: `g_T = B0+ - B0-`, `g_Dk = B0+ - Bk+`, `tau_k = Bk+ - Bk-`,
     `DeltaDelta_k = ||tau_k|| - ||g_T||`. Invalid slots (when fewer than K
     enemy pieces exist) are zeroed via a validity mask before reduction.
  6. `fingerprint` concatenates 8 spectrum scalars (`||g_T||`, max/mean/std
     `DeltaDelta`, `topk_dd_ratio`, baseline norm, mean `||g_Dk||`, valid
     fraction), K sorted DeltaDelta values, saliency entropy, saliency
     concentration, and three additional repeated norms used by the head
     LayerNorm as scale references.
  7. `primitive_delta = HeadMLP(fingerprint)` and `gate = sigmoid(
     GateMLP(fingerprint))` produce the gated primitive contribution.
  8. `logits = base_logit + gate * primitive_delta`.

- Tensor shapes:
  - `x`: `(B, 18, 8, 8)`
  - `trunk_output["logits"]`: `(B,)`
  - `saliency.top_indices`: `(B, K)`
  - `grid`: `(B, 2*(K+1), 18, 8, 8)`
  - `features`: `(B, 2*(K+1), 2 * tdcd_channels)`
  - `fingerprint`: `(B, 8 + K + 5)`
  - `primitive_delta`, `gate`, `logits`: `(B,)`

- Output heads: one puzzle logit. Diagnostics added to the output dict for
  reporting and slice analysis include `base_logit`, `primitive_delta`,
  `primitive_gate`, `primitive_gate_logit`, `primitive_gate_entropy`,
  `primitive_logit_contribution`, `saliency_entropy`,
  `saliency_concentration`, `saliency_top_valid_count`, `g_T_norm`,
  `max_dd`, `mean_dd`, `std_dd`, `topk_dd_ratio`, `baseline_feature_norm`,
  `g_D_mean_norm`, `valid_slot_count`, `mechanism_energy`,
  `proposal_profile_strength`, `proposal_keyword_count`, and the i193
  trunk diagnostics re-exported under `trunk_*` keys.

- Parameter estimate (default config, K=3, tdcd_channels=48,
  trunk_channels=64, hidden_dim=96):
  - i193 trunk: ~145k parameters (two stream encoders + heads + router).
  - Saliency head: ~5k parameters (one conv + 1x1 conv).
  - TDCD encoder: ~45k parameters (two conv layers).
  - Discriminator MLP + gate: ~6k parameters.
  - Total: ~200k parameters, an additive head over the ~145k i193 trunk.

- FLOP estimate (per sample, K=3):
  - i193 trunk: one full forward pass at i193's nominal FLOP.
  - TDCD encoder: 2*(K+1) = 8 forward passes through a compact 2-layer
    CNN at roughly 1/4 the FLOP of the i193 trunk. Effective added cost
    is therefore ~2x i193 alone, total ~3x i193 per sample.
  - Discriminator MLP/gate: negligible at the fingerprint scale.

## Implementation Binding

- Registered model name: `tempo_defender_cross_derivative_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/tempo_defender_cross_derivative_network.py`
- Idea-local wrapper: `ideas/registry/i244_tempo_defender_cross_derivative_network/model.py`
