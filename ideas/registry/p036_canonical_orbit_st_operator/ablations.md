# Ablations

The model exposes five named ablations via the `model.ablation` config
field. The primary falsifier is `shuffle_canonical`.

| ID | `ablation` | Effect on `forward()` | What it falsifies |
|---|---|---|---|
| A0 | `none` | Full operator | Baseline (unablated) |
| A1 | `shuffle_canonical` | In-batch permutation of the canonical representative | Whether the canonical representative carries usable signal. Primary falsifier. |
| A2 | `identity_only` | `chosen = e` regardless of input | Whether orbit search adds value beyond a trivial branch. |
| A3 | `fixed_choice` | `chosen = F` (file mirror) regardless of input | Whether the orbit decision needs to be input-dependent. |
| A4 | `zero_delta` | `primitive_delta = 0` | Recovers i193 base logit. |
| A5 | `trunk_only` | `primitive_delta = 0` (alias of `zero_delta`) | Strongest baseline control. |

## Deferred extensions

The deferred ablations below are *not* in `ALLOWED_ABLATIONS` of the
implementation. They are documented here for the next implementation
pass.

| ID | Proposed ablation | Reason |
|---|---|---|
| D1 | `colour_swap_group` -- extend `G` to include colour-swap + rank-flip + logit sign | The current `G = C2 x C2` does not touch channels. Promoting colour swap requires a channel permutation buffer and a sign-on-logit flag; it is intentionally deferred until the basic file/rank group passes the falsifier. |
| D2 | `dihedral_group` -- add 90- and 270-deg rotations | These are non-involutive permutations on 8x8 squares; the inverse must be applied explicitly in backward. Deferred until C2 x C2 passes. |
| D3 | `soft_canonicalisation` -- replace hard argmin with softmin over keys | Reintroduces gradient through the discrete branch but loses the speed advantage; the source packet flags it as a soft variant only. |

## Decision rule

Keep p036 only if:

- The unablated run improves PR AUC or near-puzzle FP rate on
  `cost_orbit_gap > median` slice versus i193 baseline at matched
  recall.
- `shuffle_canonical` loses >=70% of that slice lift.
- `identity_only` loses at least 50% of that slice lift.
- Aggregate PR AUC is no worse than 1.0% below i193 baseline.

If any of these fails, document the decision and drop p036.
