# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/near_puzzle_hard_negative.py`.
- Shared helpers:
  - `trunk_joint_features` from
    `src/chess_nn_playground/models/primitives/trunk_features.py`.
  - `BoardTokenAttention` from
    `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Idea-local wrapper: `ideas/registry/p055_near_puzzle_hard_negative/model.py`.
- Registry key: `near_puzzle_hard_negative`.
- Source primitive: `ideas/research/primitives/external_50_near_puzzle_hard_negative_primitive.md`.

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
Per-square candidate and reply attention pools are computed inline
from the i193 trunk's spatial features. The bounded king-pressure
reductions (`KEP`, `DOA`, `d_bal`, `Counter`) read only from the 12
piece-presence planes and the side-to-move plane.

CRTK metadata, source labels, fine labels, verification flags, engine
scores, and principal variations are **not** consulted.

## Stop-gradient contract

- All operators (attention pool, bilinear, MLPs, `logsumexp`,
  `softmax`) are autograd-friendly.
- Trunk diagnostics fed to the gate are detached.
- The joint pool feature used in the veto head is *not* detached so
  the head can co-train the trunk pool path.

## Output dict contract

The output dict follows the i193 contract, extended with:

- `logits` (rebound to `base_logit + gate * (-veto_raw)`)
- `base_logit`
- `primitive_delta` / `primitive_delta_raw` (sign: `<= 0`)
- `primitive_gate` / `primitive_gate_applied` / `primitive_gate_logit`
  / `primitive_gate_entropy`
- `primitive_contribution`
- `nphn_veto_pressure` -- raw softplus output of the veto MLP
- `nphn_forcedness_gap` -- `FG*`
- `nphn_forcedness_at_mstar` -- `FG(m*)`
- `nphn_legality_discount` -- `Disc(m*)`
- `nphn_candidate_concentration` -- `Conc`
- `nphn_candidate_gap` -- `Gap12`
- `nphn_reply_availability` -- `Avail`
- `nphn_reply_channel_information` -- `RCI`
- `nphn_attack_defense_balance` -- `d_bal`
- `nphn_king_escape_pressure` -- `KEP`
- `nphn_defender_overload_asymmetry` -- `DOA`
- `nphn_counterpressure` -- `Counter`
- `trunk_<name>` for every diagnostic the i193 trunk produced
- `mechanism_energy` augmented with `veto_raw.detach()`
- `proposal_profile_strength` = `|delta| * gate_entropy`

## Ablation modes

See `ALLOWED_ABLATIONS`. Primary falsifiers are `no_replies` (zeros
the reply-aware z entries) and `no_legality_discount` (collapses
`Disc(m*)`). Order/shuffle falsifier is `shuffle_replies`.
Single-feature drops are `no_overload` and `no_king_escape`.

## Numerical notes

- The veto head uses `softplus` so its raw output is non-negative.
  The primitive delta is the negation of this, ensuring the head can
  only lower the puzzle logit (rejection-only).
- Concentration is normalized by `log(num_candidates)` so it lives in
  `[0, 1]`.
- RCI is clipped to `[0, log(num_candidates)]` before normalization;
  the MI is otherwise bounded by that quantity but autograd through
  the softmax can produce tiny numerical drift.
- King-pressure reductions are normalized by zone size and rescaled
  from `[-1, 1]` to `[0, 1]` to keep the diagnostic vector in a small
  bounded range that the head's LayerNorm can stabilize.

## Production upgrade path

- Replace the learned candidate/reply attention pools with a true
  `python-chess`-driven candidate/reply compiler (with cached precom-
  puted parquet attack maps).
- Add a sampler-level near-puzzle hard-negative mining replay buffer
  (out of scope for the primitive itself; would land in the trainer).
- Optional FiLM-style conditioning of the trunk's final residual
  block by `z(x)` for a tighter trunk/primitive coupling.

All deferred until a keep-decision is in.
