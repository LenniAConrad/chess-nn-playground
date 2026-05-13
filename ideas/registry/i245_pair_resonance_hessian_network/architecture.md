# Architecture

- Architecture description: i193 dual-stream trunk + DHPE additive primitive
  head, fused through a learned sigmoid gate. The trunk produces the base
  puzzle logit unchanged from i193; the primitive head computes a 6-d
  signed-Hessian fingerprint and a delta logit; the gate decides how much
  of the delta to apply.

- Input format: simple_18 board tensor `x` of shape `(B, 18, 8, 8)`. CRTK
  metadata is reporting-only and not threaded through the model. The model
  consumes the canonical `batch["x"]` tensor produced by the shared
  trainer.

- Forward pass:
  1. `trunk_out = ExchangeThenKingDualStream(x)` → `base_logit`,
     plus the i193 diagnostics (`gate`, `gate_entropy`, `mechanism_energy`,
     `stream_disagreement`, `exchange_logit`, `king_logit`, …).
  2. `top_indices, valid = select_top_k_positions(piece_planes(x), top_k)`
     using deterministic piece-value priors as saliency.
  3. `variants = assemble_variant_boards(x, top_indices, valid, top_k,
     pair_count)` → `(B, 1 + top_k + C(top_k, 2), 18, 8, 8)`.
  4. `phi = PhiScorer(variants)` → per-variant scalar score.
  5. Compute `H_ij = phi(P) - phi(P\\i) - phi(P\\j) + phi(P\\{i, j})` for
     each `(i, j) in combinations(range(top_k), 2)`.
  6. Aggregate to the 6-d fingerprint
     `feat = [base_phi, z_pos, z_neg, z_total, z_ratio, z_top1]`.
  7. `delta = head_mlp(feat)` → primitive delta logit per position.
  8. `gate = sigmoid(gate_mlp([trunk_diag, dhpe_fingerprint]))` (initialised
     near zero so the head starts as a small additive correction).
  9. `logits = base_logit + gate * delta`.

- Tensor shapes:
  - `board`: `(B, 18, 8, 8)`.
  - `top_indices`: `(B, top_k)` long; `valid`: `(B, top_k)` float in {0,1}.
  - `variants`: `(B, V, 18, 8, 8)` with `V = 1 + top_k + C(top_k, 2)` (default
    `V = 11` for `top_k = 4`).
  - `phi_grid`: `(B, V)`.
  - `H`: `(B, C(top_k, 2))`.
  - `fingerprint`: `(B, 6)`.
  - `logits`: `(B,)`.

- Output heads: a single puzzle logit `logits` plus the full diagnostic dict
  enumerated in `idea.yaml > output_heads`.

- Parameter estimate (default config): i193 trunk ≈ 70k parameters at
  `trunk_channels=64`; PhiScorer ≈ 30k at `phi_channels=32, depth=3`;
  the gate / delta MLPs are < 5k parameters combined. The DHPE additive
  overhead is therefore roughly **+35–40k** parameters on top of i193.

- FLOP estimate: dominated by the PhiScorer running on `V = 11` variants
  per position; for `top_k = 4` and the default config this is roughly
  **1.5x the trunk FLOPs**, putting the model in the ~2.5x i193 wall-clock
  envelope.

## Implementation Binding

- Registered model name: `pair_resonance_hessian_network`.
- Source implementation: `src/chess_nn_playground/models/primitives/pair_resonance_hessian_network.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`
  (the bespoke i193 `ExchangeThenKingDualStreamNetwork` is wrapped, not
  reimplemented).
- Idea-local wrapper: `ideas/registry/i245_pair_resonance_hessian_network/model.py`.
- Training config: `ideas/registry/i245_pair_resonance_hessian_network/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["pair_resonance_hessian_network"] = build_pair_resonance_hessian_network_from_config`.

- Ablation modes (string `ablation` arg):
  | mode | what it does | falsifies |
  |---|---|---|
  | `none` | full DHPE primitive | (control) |
  | `unsigned` | replace `H` with `|H|` | sign is load-bearing |
  | `no_dhpe` | zero DHPE fingerprint and gate | primitive adds signal at all |
  | `shuffled_pairs` | permute pair indices in `H` before aggregation | pair-identity matters |
  | `shuffle_singles` | permute the per-piece singles before Hessian formation | single-piece ordering matters |
  | `zero_gate` | force `gate -> 0` while keeping the primitive | gate is doing the work |
  | `trunk_only` | return the i193 logit verbatim | DHPE adds anything |
