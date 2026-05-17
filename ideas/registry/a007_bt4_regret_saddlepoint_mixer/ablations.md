# Ablations

- Ablation switches:
  - `none`: full mixer (candidate + reply BoardTokenAttention pools over the
    64 squares, bilinear payoff table, 24-step damped entropy-regularized
    saddle solver, attacker-equilibrium scatter-back via the candidate-
    compiling attention).
  - `mixer=conv` (baseline): swap to the BT4 conv-pair mixer; isolates the
    benefit of the saddle solver over the original BT4 block.
  - `mixer=attention` (baseline): swap to the attention mixer; isolates the
    benefit over a generic global token mixer with no game structure.
  - `row_shuffle_payoff`: permute payoff rows along the candidate axis
    before the saddle solver -- destroys candidate-side game structure
    but preserves the entry distribution.
  - `col_shuffle_payoff`: permute payoff columns along the reply axis
    before the saddle solver -- destroys reply-side game structure.
  - `uniform_payoff`: replace `A` with its per-batch mean; completely
    removes game structure while preserving the saddle-solver compute.
  - `pure_max_min`: bypass the solver and use the raw `max_i min_j A_ij`
    saddle (via straight-through softmaxes at `tau -> 0`); tests whether
    the entropy-regularized solver is load-bearing over a hard saddle.
  - `single_iter`: run the saddle solver for one iteration instead of 24
    -- tests whether the unrolled fixed-point convergence is load-bearing
    over a single softmax step.

- What each ablation tests:
  - `mixer=conv` / `mixer=attention`: the headline comparison -- does the
    regret-saddlepoint mixer beat the per-block mixers used by the
    existing BT4 family on puzzle_binary, especially on near-puzzle false
    positives at matched recall?
  - `row_shuffle_payoff` / `col_shuffle_payoff`: do we genuinely need the
    candidate-to-reply binding of the payoff table, or does an entry-
    matched but shuffled table match the headline?
  - `uniform_payoff`: does the lift come from the game-structured payoff
    table at all, or from the extra compute of two attention pools plus
    a scatter-back path?
  - `pure_max_min` / `single_iter`: does the lift come from the entropy-
    regularized fixed-point iteration, or does a hard saddle / single
    softmax step suffice?

- Falsification criteria:
  - Headline: aggregate puzzle_binary PR AUC must be within 0.5 percentage
    points of the conv baseline (no large regression) AND must improve the
    `near_puzzle` false-positive rate at recall 0.80 by at least 3 percent
    over the conv baseline. If both fail, drop.
  - Mechanism: `row_shuffle_payoff` and `col_shuffle_payoff` must each
    lose at least 50 percent of the near-puzzle FP rate lift; otherwise
    the lift is not driven by the candidate / reply game structure and the
    mixer should be replaced by a simpler bilinear pool.
  - Sanity: `uniform_payoff` must lose at least 80 percent of the
    near-puzzle FP rate lift; otherwise the gain is driven by extra
    parameters or by the scatter-back path, not by the saddle solver.
  - Solver depth: `pure_max_min` and `single_iter` must each lose at
    least 30 percent of the near-puzzle FP rate lift; otherwise the
    24-step damped entropy-regularized iteration is not load-bearing and
    a cheaper saddle suffices.
  - Throughput: average step time must stay within 25 percent of the conv
    baseline; otherwise the mixer pays for itself only at extreme tower
    sizes and the comparison is not apples-to-apples.
