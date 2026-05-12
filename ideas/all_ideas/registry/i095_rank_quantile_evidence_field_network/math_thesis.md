# Math Thesis

Rank-Quantile Evidence Field Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`.

Batch candidate rank: `4`.

## Working thesis

Puzzle-likeness may be driven by *extreme*, sparse evidence on a few squares rather than by the average board evidence. Mean-pooling washes out those tails. A sparse witness mask in turn throws away most of the board. Differentiable rank and quantile pooling sit in between: the classifier still sees the full board but reads it through order statistics that emphasise tail behaviour.

## Setup

The trunk produces a bank of `E` learned scalar evidence fields `f_e: B x 8 x 8 -> R`, indexed by `e in {1, ..., E}`. For each field we sort the 64 square activations to obtain order statistics

$$
f_e^{(1)} \le f_e^{(2)} \le \dots \le f_e^{(64)}.
$$

For a probability level `tau in [0, 1]` we define the linearly-interpolated empirical quantile

$$
Q_\tau(f_e) = (1 - w)\, f_e^{(\lfloor p \rfloor + 1)} + w\, f_e^{(\lceil p \rceil + 1)}, \qquad p = \tau \cdot 63, \quad w = p - \lfloor p \rfloor.
$$

The board readout is the concatenation of `Q_\tau(f_e)` over a configured grid of `\tau`s (default `0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99`), the four tail gaps `Q_{0.99} - Q_{0.95}`, `Q_{0.95} - Q_{0.50}`, `Q_{0.50} - Q_{0.05}`, `Q_{0.05} - Q_{0.01}`, the per-field mean and std, top-`k` / bottom-`k` order means, a normalised softmax entropy of the field, the robust range `Q_{max} - Q_{min}`, and two soft tail-mass scalars `\bar{\sigma}(8 (f - Q_{max}))`, `\bar{\sigma}(8 (Q_{min} - f))`. All slots are differentiable — sorting is implemented through `torch.sort`, which has well-defined sub-gradients away from ties.

## Why this is informative

- A *mean-pool* baseline collapses each field to `\bar f_e = (1/64)\sum_s f_e[s]`. Two boards with identical mean evidence but different tails are indistinguishable to a mean head. The rank readout separates them whenever the tails differ.
- A *sparse-witness* baseline picks a small subset `S` of squares and discards the rest. Order statistics keep contributions from every square (in the soft-tail-mass and entropy slots) while still emphasising the tails.
- Tail gaps `Q_{0.99} - Q_{0.95}` operationalise the packet's hypothesis that the *spike* — not the average — carries the puzzle signal. They go to zero when the field is uniform and grow when a few squares dominate.

## Ablations

- `mean_pool_only` replaces every quantile slot by the per-field mean and tests whether a mean head is enough.
- `topk_only` keeps only the top-`k`/bottom-`k` order means and tests whether a narrow witness set captures the rank signal.
- `random_field_encoder` swaps the learned trunk for fixed random `Conv2d` evidence fields and tests whether *learned* fields matter.
- `square_shuffle` applies a deterministic permutation before encoding and tests whether the spatial layout — not just the marginal distribution — is what the rank readout exploits.

The five modes are cheap to swap because the rank/quantile head is shared; only the field source changes between learned, random, and shuffled.

## Contract

- Input: board tensor only; CRTK / source metadata is reporting-only.
- Output: one BCE logit per board for `puzzle_binary`, plus rank-pool diagnostics for prediction artefacts.
