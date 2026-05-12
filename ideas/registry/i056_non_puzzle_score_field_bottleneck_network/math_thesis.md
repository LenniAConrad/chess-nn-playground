# Math Thesis

Non-Puzzle Score-Field Bottleneck Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0922_tuesday_local_nonpuzzle_score_field.md`.

## Working Thesis

Train a rule-safe denoising score prior only on verified non-puzzle boards,
then classify puzzle-likeness from the current board plus a bottlenecked
vector field estimating how the non-puzzle manifold would locally repair
that board.

## Tweedie / Denoising-Score Identity

For `X ~ P_0`, `epsilon ~ N(0, I)`, and `U = X + sigma * epsilon`, the
optimal squared-error denoiser satisfies

```text
D^*(u, sigma) = E[X | U = u] = u + sigma^2 * grad_u log p_{0,sigma}(u),
```

so

```text
s_{0,sigma}(u) = grad_u log p_{0,sigma}(u) = (D^*(u, sigma) - u) / sigma^2.
```

The `OrdinaryScoreDenoiser` module estimates `D^*` by minimizing

```text
L_DSM(theta) = E_{x ~ P_0, sigma ~ pi, epsilon ~ N(0,I)}
               || D_theta(x + sigma * epsilon, sigma) - x ||^2 / (2 sigma^2)
```

over class-0 (verified non-puzzle) training rows. The implementation
exposes this objective via
`NonPuzzleScoreFieldBottleneckNetwork.denoising_score_matching_loss` with a
`score_prior_train_on_binary_zero_only=True` filter.

## Non-Puzzle Repair Score Stack

The introduced object is the bottlenecked stack

```text
S_0(x) = concat_k ((D_theta(x, sigma_k) - x) / sigma_k^2)  in R^{(K*18) x 8 x 8}.
```

The supervised classifier reads

```text
g(x) = h(BoardStem(x), B(S_0(x)))
```

where `B` is a small Conv1x1 + depthwise-Conv3x3 bottleneck. The denoiser
parameters can be frozen after pretraining via `freeze_score_prior()` so
that the supervised loss never updates the score prior, matching the
cleanest falsification recipe in the packet.

## Falsification Layout

- All-class prior — train `D_theta` without the binary-zero filter.
- Frozen random denoiser — skip pretraining and rely on initialization.
- No-score branch — set `score_bottleneck_channels=0`.
- Score-only — drop the board stem output.

## What is and is not proven

The denoising-score identity is proven under infinite data, sufficient
capacity, Gaussian corruption, and squared-error denoising. The thesis that
puzzle-likeness is detectable from a class-0 score field is empirical and
must be tested against the packet's all-class-prior, random-field, and
material-broadcast controls.
