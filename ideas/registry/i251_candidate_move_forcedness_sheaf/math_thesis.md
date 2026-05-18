# Math Thesis

Candidate Move Forcedness Sheaf -- i251.

Source packet:
`ideas/research/packets/classic/i251_candidate_move_forcedness_sheaf.md`.

Working thesis: i018's typed cellular sheaf already proves that real chess
relation topology is doing real work on `puzzle_binary`. What it does not
contain is an explicit bottleneck over *which move, among the available
moves, is forcing*. i251 wraps the i018 trunk with a deterministic move
bottleneck whose pooled summary feeds a gated additive logit delta. The
delta head and gate are zero-initialized so the network is i018 at init
and only deviates if the move branch finds real, falsifiable signal.

## Object

Let i018 produce final per-square states `h_v in R^d`, board context
scalars `r`, and the sheaf diagnostics bundle `u`. Let the move builder
return a bounded candidate set

```text
M(x) = {m_j = (s_j, t_j, kind_j, flags_j)}_{j=1}^{n_x},
```

with `n_x <= max_candidates`. For each move, the shared encoder
produces

```text
z_j = phi([h_{s_j}, h_{t_j}, h_{t_j} - h_{s_j}, flags_j, onehot(kind_j),
          psi_{src/dst}(M)]) in R^E,
```

where `psi_{src/dst}(M)` is the local sheaf summary defined below.

## Move-local sheaf summary

For each relation `r`, define the per-square in-degree
`d_in[b, r, v] = sum_u M[b, r, u, v]` and out-degree
`d_out[b, r, u] = sum_v M[b, r, u, v]`. For each candidate move
`(s_j, t_j)`, the per-move sheaf summary is

```text
psi_j = (1/8) * [d_in[:, :, s_j], d_out[:, :, s_j],
                 d_in[:, :, t_j], d_out[:, :, t_j]] in R^{4R}.
```

This gives the encoder a direct read of i018's local tactical pressure at
both endpoints of the candidate without recomputing anything.

## Top-k softmax pool

Per-move scores `a_j = w^T z_j + b` are computed by a zero-initialized
linear head, so every valid move starts with the same score. We keep the
top `top_k` (default 8) scored moves, mask out padding, and apply a
learned-temperature softmax to obtain pool weights

```text
alpha = softmax(a_S / tau),       a_S = top_k(a),
```

with `tau` in a clamped range to prevent collapse. The pooled embedding
is

```text
m_pool = sum_{j in S} alpha_j z_j in R^E,
```

with derived scalar forcedness diagnostics

```text
H        = -sum alpha_j log alpha_j,
top1     = max alpha_j,
gap      = a_(1) - a_(2),
check_m  = sum alpha_j * flags_check_j,
promo_m  = sum alpha_j * flags_promo_j,
upromo_m = sum alpha_j * flags_underpromo_j,
pin_m    = sum alpha_j * (flags_src_pinned_j + flags_pin_aligned_j),
cap_m    = sum alpha_j * flags_capture_j,
kz_m     = sum alpha_j * flags_king_zone_j,
overflow = (n_x >= top_k),
n_x_norm = n_x / max_candidates.
```

## Gated fusion

The final puzzle logit is

```text
final_logit = base_logit + sigmoid(gate(features)) * delta(features),
features    = [m_pool, forced_scalars, trunk_scalars, top_move_kind],
```

where `trunk_scalars` are taken directly from the i018 diagnostic bundle:
`sheaf_tension`, `ray_language_energy`, `triad_defect_energy`,
`pin_pressure`, `king_ring_pressure`, `transport_imbalance`,
`defense_gap`, `reply_pressure`. Both `gate` and `delta` are small MLPs
whose final linear layer is zero-initialized:

```text
gate(features)  = 0  at init    -> sigmoid(0) = 0.5,
delta(features) = 0  at init    -> 0.5 * 0 = 0,
```

so `final_logit = base_logit` at init. With shared weights this is
exact (max abs diff `0.0` on a 4-sample CPU batch).

## Hypothesis

If i018's typed topology is right but the gap is forcedness (single
dominating move under local sheaf evidence), then a tiny gated delta
fed by the top-k move pool should improve mean test PR-AUC or reduce
matched-recall near-puzzle false positives without changing the parent
topology, the trainer, or the benchmark contract.

## Falsifiers

The required falsifiers, in order of priority:

1. **Degree-preserving topology scramble.** Inherited from i018; reuse
   `scramble_relations: true`. If the drop is small, reject the family.
2. **Disable move branch.** `disable_move_branch: true` forces
   `final_logit = base_logit`; with shared weights this must reproduce
   i018 exactly and bounds the maximum possible regression.
3. **Flat move pool.** `flat_move_pool: true` forces uniform pool
   weights over valid candidates. If the matched-seed test PR-AUC is
   within seed noise of the full i251, the top-k bottleneck is not
   load-bearing.
4. **Random move set.** Replace the deterministic candidate scoring
   with random selection of `max_candidates` valid edges. Requires a
   one-line forward edit; not exposed as a config flag.
5. **Feature ablations on flags.** Zero out check, promotion, or pin
   flag channels in `CandidateMoveBuilder`. The slice impact should
   localize to the matching CRTK slices
   (`mate_in_1`, `promotion` / `underpromotion`, overload-like motifs).
6. **No sheaf summary.** Replace `psi_j` with zeros. If results are
   unchanged, the move branch is not really i018-compatible -- it is a
   side module.

## Decision rule

Treat i251 as a meaningful improvement over i018 only if at least one of
the following holds across seeds 42, 43, 44 at base scale, with all
other hyperparameters matched:

- `+0.003` absolute mean test PR-AUC over i018, or
- a `>=1%` absolute reduction in near-puzzle false positives at
  validation-derived recall `0.80` or `0.85`, without compensating
  regressions on puzzle recall or precision.

Falsifier 1 should still drop test PR-AUC by `>= 0.02` on i251; if the
drop is below `0.01`, the typed-topology claim has decayed and the
family must be re-examined first.
