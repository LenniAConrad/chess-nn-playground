# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `occlusion_semiring_delta_bilinear_hyperedge`
  primitive from `p023_occlusion_semiring_delta_bilinear_hyperedge`.
  Source primitive math:
  `ideas/registry/p023_occlusion_semiring_delta_bilinear_hyperedge/math_thesis.md`.
  For each square `s`, each of the 8 queen-ray directions `r`, ordered
  ray cells `c_{r,1..L}` (`L <= 7`), per-square soft occupancy `O`, and
  a learned value projection `V`, the primitive runs a *backward*
  occlusion-semiring recurrence outward from the source square:

  ```
  h_{r,L} = 0
  h_{r,t} = (1 - O_{c_{r,t+1}}) * h_{r,t+1} + V x_{c_{r,t+1}}
  ```

  so `h_{r,0}` aggregates ray contributions weighted by the
  transmittance product of all unoccupied cells encountered. The
  hidden state at depth `t` therefore depends on the *deeper* steps of
  the ray, not the prior ones.

  After the recurrence, the operator forms a **bilinear hyperedge**
  over each of the 4 opposite-direction pairs
  `(N,S), (NE,SW), (E,W), (SE,NW)`:

  ```
  edge_{s,p} = (W_L h_{left_p,s}) (.) (W_R h_{right_p,s})
  ```

  The hyperedge embedding encodes the "attacker -- own piece --
  defender along one line" motif: a non-trivial Hadamard product
  requires both halves of the line to carry information through the
  transmittance gate.

- Assumptions:
  1. The `occlusion_semiring_delta_bilinear_hyperedge` primitive is
     well-defined as a shape-preserving operator
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
     hands it a generic `(B, C, 8, 8)` channel tensor), so occupancy
     is derived inside the operator as `O_s = sigmoid(w . x_s + b)`
     from the per-square channel vector and used as the
     transmittance gate `(1 - O)`. The occupancy mask is therefore
     *learned* from features, but it is still generated *inside* the
     operator and never supplied externally, which preserves the
     source thesis's defining property (the backward recurrence is
     gated by an occlusion signal that comes from the board, not
     from outside the operator).

- Claimed advantage: If the
  `occlusion_semiring_delta_bilinear_hyperedge` primitive carries a
  through-the-square line signal that conv and attention do not,
  dropping it into the BT4 block must lift held-out PR AUC (aggregate
  or on a slice that depends on through-line interactions, e.g.
  `pin`, `skewer`, `xRayAttack`, `discoveredAttack`, `battery`, and
  long-line `mate_in_*` patterns where the gating of a blocker on the
  line is load-bearing) versus the two baselines under the same
  tower, optimizer, and data. This is a controlled architecture-level
  test of "is occlusion_semiring_delta_bilinear_hyperedge a better
  spatial mixer than conv or attention inside a fixed BT4 tower
  shell?", not a new primitive claim. The per-block cost is
  `O(64 * 7 * C)` for the value projection, `O(64 * 7 * C)` for the
  backward recurrence sweep, and `O(64 * 4 * d)` for the bilinear
  hyperedge fusion (`d = bilinear_dim`), so it is asymptotically
  cheaper than a full `64 x 64` softmax attention map and comparable
  to a 3x3 conv stack at small ray lengths.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for
  `occlusion_semiring_delta_bilinear_hyperedge` itself (the backward
  occlusion-semiring recurrence and the opposite-direction bilinear
  hyperedge) is proven in the source primitive's math thesis and
  falsified by its own ablation grid (`zero_occupancy`,
  `uniform_occupancy`, `disable_bilinear`). This folder inherits that
  math and tests whether the resulting operator, used as a token
  mixer rather than as an additive head, transfers its signal
  through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The backward recurrence
  `h_{r,t} = (1 - O_{c_{r,t+1}}) h_{r,t+1} + V x_{c_{r,t+1}}` is
  implemented exactly along the 8 queen-ray direction tables, with
  the off-board mask applied per-step, and the bilinear hyperedge
  `edge_{s,p} = (W_L h_{left_p,s}) (.) (W_R h_{right_p,s})` is
  computed exactly for the 4 opposite-direction pairs before the
  concatenate-and-project fusion `out = W_O [edge_1; ...; edge_4]`.

- What is only hypothesized: That replacing the conv mixer with the
  `occlusion_semiring_delta_bilinear_hyperedge` mixer lifts PR AUC on
  at least one CRTK slice (most likely slices where through-the-line
  interactions are load-bearing -- `pin`, `skewer`, `xRayAttack`,
  `discoveredAttack`, `battery`, long-line `mate_in_*` -- and slices
  where the position has a clear blocker on a line) without
  regressing aggregate PR AUC by more than the matched-baseline
  tolerance.

- Failure cases:
  - The learned soft occupancy `O_s = sigmoid(w . x_s + b)` fails to
    recover the piece-plane occupancy that the source primitive uses,
    so the transmittance product is dominated by noise from empty
    squares; the in-mixer `zero_occupancy`/`uniform_occupancy`
    ablation matches this idea on its declared target slice.
  - The `occlusion_semiring_delta_bilinear_hyperedge` mixer collapses
    inside the BT4 shell because the residual + SqueezeExcite path
    dominates the mixer output; the `conv` baseline matches the
    variant within noise.
  - The bilinear hyperedge `edge_{s,p}` collapses because the
    `W_L` and `W_R` projections learn aligned subspaces so
    `left ~ right` and the Hadamard product carries no more signal
    than `left + right`; report the in-mixer `disable_bilinear`
    ablation alongside the headline number.
  - The backward recurrence saturates because `(1 - O)` is near 1
    everywhere (the learned occupancy stays near zero), so the
    transmittance gate does not differentiate blocked from unblocked
    rays and the operator degenerates into a directional unweighted
    sum of `V x` along each ray.
