# Ablations

- Ablation switches:
  - `none`: full mixer (per-square `(a_j, b_j)` projections + global
    SEBM pair state `p = s (.) u - sum_j a_j (.) b_j` + FiLM broadcast +
    per-square readout).
  - `mixer=conv` (baseline): swap to the BT4 conv-pair mixer; isolates the
    benefit of the SEBM pair-state mixer over the original BT4 block.
  - `mixer=attention` (baseline): swap to the attention mixer; isolates the
    benefit over a generic global token mixer.
  - `shuffle_pair_state`: randomly permute the entries of the global
    `(s, u, p)` memory across the batch before they enter the FiLM layer --
    removes the position-specific pair-state signal while keeping parameter
    count and per-square projections intact.
  - `drop_pair_term`: replace the FM cross-term `p = s (.) u - sum_j a_j (.) b_j`
    with `p = s (.) u` (i.e. drop the diagonal subtraction). Tests whether
    the load-bearing pair identity actually relies on the FM correction.
  - `disable_film`: hold the FiLM `gamma` at 0 and `beta` at 0, leaving
    only the per-square readout of the raw `(a_j, b_j)` projections --
    proves the global pair memory is the operative spatial-mixing channel.
  - `bilinear_rank=8` / `bilinear_rank=128`: vary the rank of the
    bilinear memory; the source primitive treats this as a load-bearing
    capacity hyperparameter.

- What each ablation tests:
  - `mixer=conv` / `mixer=attention`: the headline comparison -- does the
    SEBM mixer beat the per-block mixers used by the existing BT4 family on
    puzzle_binary, especially on slices where pair interactions matter?
  - `shuffle_pair_state`: does the lift come from the *content* of the
    position-conditioned pair memory rather than from extra parameters?
  - `drop_pair_term`: is the FM cross-term identity the load-bearing
    algebra, or does a plain rank-r outer-product mixer suffice?
  - `disable_film`: confirms that the global broadcast (not just the
    per-square projections) is the operative spatial-mixing mechanism.
  - `bilinear_rank` sweep: maps the capacity / generalisation trade-off the
    source primitive documents.

- Falsification criteria:
  - Headline: aggregate puzzle_binary PR AUC must be within 0.5 percentage
    points of the conv baseline (no large regression) AND must improve at
    least one declared target slice (see `report_template.md`) by at least
    +0.02 PR AUC over the conv baseline. If both fail, drop.
  - Mechanism: `shuffle_pair_state` must lose at least 50 percent of the
    target-slice lift; otherwise the lift is not driven by the
    position-conditioned pair memory.
  - Identity: `drop_pair_term` must lose at least 30 percent of the
    target-slice lift; otherwise the FM cross-term identity is not the
    load-bearing algebra and any rank-r outer-product mixer would suffice.
  - Stability: `disable_film` must regress to the conv baseline within 0.5
    PR AUC points on the target slice; otherwise the global broadcast is
    not the operative mechanism.
  - Throughput: average step time must stay within 25 percent of the conv
    baseline; otherwise the mixer pays for itself only at extreme tower
    sizes and the comparison is not apples-to-apples.
