# Math Thesis

Wavelet Scattering Board Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`.

Batch candidate rank: `2`.

Working thesis: Puzzle-like structure may live in multiscale arrangements of piece planes. A fixed wavelet scattering front end can test whether stable multiscale modulus features help beyond learned CNN filters while avoiding engine-specific priors.

## Operator description

Let `x in R^{C x 8 x 8}` be the simple_18 board tensor (`C = 18`). Define the 2x2 separable Haar bank `{psi_LL, psi_H, psi_V, psi_D}` with

```
psi_LL = (1/2) [[ 1,  1], [ 1,  1]]
psi_H  = (1/2) [[ 1,  1], [-1, -1]]   (horizontal-edge / vertical-difference)
psi_V  = (1/2) [[ 1, -1], [ 1, -1]]   (vertical-edge / horizontal-difference)
psi_D  = (1/2) [[ 1, -1], [-1,  1]]   (diagonal)
```

Three orientations `O = {H, V, D}` are kept; `LL` is the local average (the lowpass).

Scales are realised with dilated 2x2 convolutions: `psi_{s, o}(p, q) = psi_o(p, q)` applied at dilation `s in S = {1, 2, 4}`. We pad with circular boundary conditions so the operator stays the same spatial size as the input.

The first-order scattering field for channel `c` is

```
U^{(1)}_{c, s, o} = | (psi_{s, o} * x_c) |     for o in O
W^{(0)}_{c, s}     =   psi_{s, LL}    * x_c
```

(modulus on the high-pass bands, signed lowpass kept for the LL-energy summary). The second-order scattering field, when the second layer is enabled, is

```
U^{(2)}_{c, s1, o1, s2, o2} = | psi_{s2, o2} * U^{(1)}_{c, s1, o1} |    with s2 > s1
```

We restrict to `s2 > s1` to follow the standard scattering tree (no information at `s2 <= s1` after a modulus contraction).

## Pooled features

Each board is summarised by a fixed feature vector phi(x) with three groups:

- First-order pooled stats: per `(c, s, o)` we report `mean(U^{(1)})`, `std(U^{(1)})`, and `max(U^{(1)})`.
- Lowpass signed energy: per `(c, s)` we report `mean(W^{(0)})`.
- Second-order means: per `(c, s1, o1, s2, o2)` with `s2 > s1` we report `mean(U^{(2)})`.

For `C = 18`, `|S| = 3`, `|O| = 3` the per-channel feature count is `3*9 + 3 + 3*9 = 57`, so phi(x) lives in `R^{1026}`.

## Decision rule

phi(x) is L2-normalised by a `LayerNorm` and fed through a small MLP `R^{1026} -> R^{hidden} -> ... -> R` to produce one logit per board. The puzzle decision flows only through phi(x); the head never reads raw board planes, so all multiscale information is mediated by the fixed scattering operator.

Falsification path: if random fixed filters (the `random_fixed_filters` ablation) or lowpass-only features (`lowpass_only`) match the full Haar scattering, the multiscale-edges-and-modulus structure is not what is helping. If a learned matched-parameter CNN dominates, fixed scattering is not contributing useful structural bias. The `channel_shuffle` ablation tests whether semantic piece channels matter at all to the scattering features.
