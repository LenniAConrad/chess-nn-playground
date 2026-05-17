# Ablations

- Ablation switches:
  - `none`: full mixer (BoardTokenAttention pool over the 64 squares,
    utility-head MLP, soft product partial-order dominance, frontier
    softmax, scatter-back via the compiling attention).
  - `mixer=conv` (baseline): swap to the BT4 conv-pair mixer; isolates the
    benefit of the partial-order reducer over the original BT4 block.
  - `mixer=attention` (baseline): swap to the attention mixer; isolates the
    benefit over a generic global token mixer with no partial-order
    structure.
  - `scalar_max`: collapse the utility table to per-candidate scalar max
    before the dominance product -- collapses the product partial order to
    a total order.
  - `single_channel`: use only utility channel 0 in the dominance product
    -- collapses the product partial order to a 1-D partial order.
  - `shuffle_channels`: permute utility channels across candidates
    in-batch before the dominance product -- decouples channels from
    candidates while keeping marginal channel distributions.
  - `uniform_frontier`: replace `alpha = softmax(...)` with a uniform
    distribution over candidates -- removes the frontier-softmax signal
    while keeping the spatial scatter-back path intact.

- What each ablation tests:
  - `mixer=conv` / `mixer=attention`: the headline comparison -- does the
    Pareto-antichain frontier mixer beat the per-block mixers used by the
    existing BT4 family on puzzle_binary, especially on near-puzzle false
    positives at matched recall?
  - `scalar_max` / `single_channel`: do we genuinely need the
    multi-channel product partial order, or does a scalar / 1-D
    surrogate match the headline?
  - `shuffle_channels`: does the lift come from the *channel-to-candidate*
    binding rather than from extra parameters or marginal channel
    distributions?
  - `uniform_frontier`: does the lift come from the frontier-softmax
    selection rather than from the candidate-token compilation alone?

- Falsification criteria:
  - Headline: aggregate puzzle_binary PR AUC must be within 0.5 percentage
    points of the conv baseline (no large regression) AND must improve the
    `near_puzzle` false-positive rate at recall 0.80 by at least 3 percent
    over the conv baseline. If both fail, drop.
  - Mechanism: `scalar_max` and `shuffle_channels` must each lose at least
    50 percent of the near-puzzle FP rate lift; otherwise the lift is not
    driven by the partial-order structure and the mixer should be replaced
    by a simpler scalar reducer.
  - Sanity: `single_channel` must underperform `none` by at least 30
    percent of the near-puzzle FP rate lift; otherwise the multi-channel
    `C_u > 1` setting is not load-bearing.
  - Throughput: average step time must stay within 25 percent of the conv
    baseline; otherwise the mixer pays for itself only at extreme tower
    sizes and the comparison is not apples-to-apples.
