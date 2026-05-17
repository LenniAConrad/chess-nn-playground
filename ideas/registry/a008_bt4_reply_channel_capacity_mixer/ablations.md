# Ablations

- Ablation switches:
  - `none`: full mixer (candidate + reply BoardTokenAttention pools over the
    64 squares, bilinear reply-logit table `L_{kr}`, 24-step Blahut-Arimoto
    iteration on the soft conditional reply distribution
    `P_{kr} = softmax_r(L_{kr} / tau)`, capacity-achieving prior `q*`
    scattered back via the candidate-compiling attention).
  - `mixer=conv` (baseline): swap to the BT4 conv-pair mixer; isolates the
    benefit of the channel-capacity solver over the original BT4 block.
  - `mixer=attention` (baseline): swap to the attention mixer; isolates
    the benefit over a generic global token mixer with no game structure.
  - `row_shuffle_channel`: permute the reply axis independently within
    each candidate row of `P` before the Blahut-Arimoto loop -- destroys
    the candidate-to-reply correlation that makes capacity meaningful
    while preserving each row's entropy.
  - `duplicate_rows`: replace every row of `P` with the first row before
    the solver -- forces zero capacity by construction (all candidates
    induce the same reply distribution) while preserving per-row entropy
    and the bilinear compute path.
  - `entropy_only`: bypass the Blahut-Arimoto solver and use a uniform
    `q* = 1/K` so the scatter-back is driven purely by per-row reply
    entropy rather than by the capacity-achieving prior -- tests whether
    the capacity argmax is load-bearing over an `i192`-style entropy
    aggregate.
  - `uniform_q_init_only`: run a single Blahut-Arimoto step instead of 24
    -- tests whether the unrolled fixed-point convergence is load-bearing
    over a single softmax step.
  - `low_tau` / `high_tau`: scale `tau` by 0.25 / 4.0 before the row
    softmax to probe the hard-vs-soft conditional regime; the headline
    `tau = 1.0` should beat both extremes.

- What each ablation tests:
  - `mixer=conv` / `mixer=attention`: the headline comparison -- does the
    reply-channel-capacity mixer beat the per-block mixers used by the
    existing BT4 family on puzzle_binary, especially on near-puzzle false
    positives at matched recall?
  - `row_shuffle_channel`: do we genuinely need the row-correlated
    candidate-to-reply structure of the transition table, or does a
    row-entropy-matched but reply-shuffled table match the headline?
  - `duplicate_rows`: does the lift come from the candidate -> reply
    channel at all, or from the extra compute of two attention pools plus
    a scatter-back path?
  - `entropy_only`: does the lift come from the capacity-achieving
    candidate prior `q*`, or does a uniform candidate weighting plus
    per-row reply entropy suffice?
  - `uniform_q_init_only`: does the lift come from the unrolled
    Blahut-Arimoto convergence, or does a single softmax step suffice?
  - `low_tau` / `high_tau`: is the headline temperature near a sweet spot
    that balances row sharpness against capacity saturation, or is the
    operator robust across the temperature regime?

- Falsification criteria:
  - Headline: aggregate puzzle_binary PR AUC must be within 0.5 percentage
    points of the conv baseline (no large regression) AND must improve the
    `near_puzzle` false-positive rate at recall 0.80 by at least 3 percent
    over the conv baseline. If both fail, drop.
  - Mechanism: `row_shuffle_channel` must lose at least 50 percent of the
    near-puzzle FP rate lift; otherwise the lift is not driven by the
    row-correlated channel structure and the mixer should be replaced by
    a simpler bilinear pool.
  - Sanity: `duplicate_rows` and `entropy_only` must each lose at least 80
    percent of the near-puzzle FP rate lift; otherwise the gain is driven
    by extra parameters, the scatter-back path, or per-row entropy, not
    by the capacity argmax `q*`.
  - Solver depth: `uniform_q_init_only` must lose at least 30 percent of
    the near-puzzle FP rate lift; otherwise the 24-step Blahut-Arimoto
    iteration is not load-bearing and a single softmax step suffices.
  - Throughput: average step time must stay within 25 percent of the conv
    baseline; otherwise the mixer pays for itself only at extreme tower
    sizes and the comparison is not apples-to-apples.
