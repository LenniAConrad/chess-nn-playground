# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite
  blocks -> value head) fixed and replaces only the per-block spatial-
  mixing operator with the `event_symmetric_interaction_accumulator`
  primitive from `p024_event_symmetric_interaction_accumulator`.
  Source primitive math:
  `ideas/registry/p024_event_symmetric_interaction_accumulator/math_thesis.md`.
  For per-square tokens `u_s in R^d`, the elementary symmetric
  polynomial states under the Hadamard product are

  ```
  E^{(0)} = 1                                            (in R^d)
  E^{(r)} = sum_{s_1 < ... < s_r} u_{s_1} (.) ... (.) u_{s_r}
  ```

  computed exactly with the streaming Newton-style recurrence (one
  pass over the 64 squares, total cost `O(R |S| d)`, no pair or triple
  enumeration):

  ```
  for s = 0 .. 63:
      for r = R, R-1, ..., 1:
          E^{(r)} <- E^{(r)} + u_s (.) E^{(r-1)}
  ```

  `E^{(1)}` is the plain EmbeddingBag-equivalent sum; `E^{(2)}` and
  `E^{(3)}` carry all 2nd- and 3rd-order Hadamard interactions across
  the 64 squares as a single permutation-invariant readout. For the
  default `R = 2` the closed-form FM-style identity
  `E^{(2)} = (1/2) ((sum_s u_s) (.) (sum_s u_s) - sum_s u_s (.) u_s)`
  recovers exactly the p022 pair-term sum with `U_s = V_s = u_s`; for
  `R = 3` the streaming recurrence is the most numerically stable way
  to assemble the third-order symmetric without Newton's identity
  divisions.

- Assumptions:
  1. The `event_symmetric_interaction_accumulator` primitive is well-
     defined as a shape-preserving operator
     `(B, C, 8, 8) -> (B, C, 8, 8)` under the
     `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract.
  2. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and
     across the `conv` and `attention` baselines.
  3. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is
     the mixer.
  4. The mixer cannot read the piece planes directly (the BT4 block
     hands it a generic `(B, C, 8, 8)` channel tensor), so the
     occupancy indicator is derived inside the operator as
     `O_s = sigmoid(w . x_s + b)` from the per-square channel vector
     and multiplied into the token vector `u_s` before the
     elementary-symmetric recurrence. The occupancy mask is therefore
     *learned* from features, but it is still generated *inside* the
     operator and never supplied externally, which preserves the
     source thesis's defining property (the symmetric recurrence is
     computed over a sparse active set).

- Claimed advantage: If the
  `event_symmetric_interaction_accumulator` primitive carries
  second- and third-order multiplicative interaction signal across
  active pieces that conv and attention do not, dropping it into the
  BT4 block must lift held-out PR AUC (aggregate or on a slice that
  depends on multi-piece interactions, e.g. fork / discovered-attack
  / double-attack / battery / x-ray patterns and the upper
  `crtk_difficulty` tail where three-piece coordination dominates)
  versus the two baselines under the same tower, optimizer, and data.
  This is a controlled architecture-level test of "is
  `event_symmetric_interaction_accumulator` a better spatial mixer
  than conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim. The streaming recurrence cost is
  `O(R * 64 * d + C * d * R)` per block where `d = token_dim` and `C`
  is the channel width, so it is asymptotically cheaper than the
  `O(64^R d)` naive enumeration and competitive with a full 64x64
  attention map at small `R`.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for
  `event_symmetric_interaction_accumulator` itself (the
  add/remove-reversible elementary-symmetric recurrence) is proven
  in the source primitive's math thesis and falsified by its own
  ablation grid (`first_order_only`, `second_order_only`,
  `shuffle_higher_orders`, `trunk_only`). This folder inherits that
  math and tests whether the resulting operator, used as a token
  mixer rather than as an additive head, transfers its signal
  through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The streaming recurrence
  `E^{(r)} <- E^{(r)} + u_s (.) E^{(r-1)}` is implemented exactly,
  with each state normalised by the soft active count raised to the
  matching order so the readout is scale-invariant in `|S|`.

- What is only hypothesized: That replacing the conv mixer with the
  `event_symmetric_interaction_accumulator` mixer lifts PR AUC on at
  least one CRTK slice (most likely slices where higher-order multi-
  piece interactions are load-bearing -- fork / discovered-attack /
  double-attack / battery / x-ray -- and the upper `crtk_difficulty`
  tail where three-piece coordination dominates) without regressing
  aggregate PR AUC by more than the matched-baseline tolerance.

- Failure cases:
  - The learned soft occupancy `O_s = sigmoid(w . x_s + b)` fails to
    recover the piece-plane occupancy that the source primitive uses,
    so the higher-order states are contaminated by empty-square
    contributions; the in-mixer `zero_occupancy`-style ablation
    matches this idea on its declared target slice.
  - The `event_symmetric_interaction_accumulator` mixer collapses
    inside the BT4 shell because the residual + SqueezeExcite path
    dominates the mixer output; the `conv` baseline matches the
    variant within noise.
  - The broadcast-back fusion `y_s = MLP([u_s; E^{(1)}; ...; E^{(R)}])`
    carries only the per-square own-token signal `u_s` and ignores
    the broadcast context, meaning the higher-order states are
    decorative; the `first_order_only` ablation (keep only `E^{(1)}`)
    closes the gap.
  - The higher-order states `E^{(>=2)}` collapse numerically because
    the per-square tokens have near-zero magnitude after the inner
    LayerNorm, leaving `E^{(2)}` dominated by noise; report block-
    level `||E^{(r)}||` statistics across the residual stack.
