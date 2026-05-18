# Math Thesis

Bispectral Phase-Coupling Board Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2110_friday_shanghai_bispectral_phase.md`.

Working thesis: Puzzle-like positions may have distinctive spatial phase-coupling patterns between piece planes, and a bispectral bottleneck can test board arrangement geometry that is invisible to magnitude-only Fourier summaries and different from CNN texture learning.

## Operator description

Let `x in R^{C x 8 x 8}` be the simple_18 board tensor (`C = 18`). A learned `1x1` mixer (optionally fed four deterministic coordinate planes — rank, file, center distance, side-relative forward) maps `x` to `Cmix = 16` real mixed board fields

```
z_c = sum_j W_{c, j} [x; coord]_j     for c in {0, ..., Cmix - 1}
```

For each mixed channel we compute the 2D discrete Fourier transform on the periodic `8 x 8` grid

```
F_c(k) = FFT2(z_c)(k)     for k in {0, ..., 7}^2
```

The power spectrum `|F_c(k)|^2` captures frequency energy but discards spatial phase. The bispectrum keeps third-order phase coupling

```
Bis_c(k, l) = F_c(k) * F_c(l) * conj(F_c(k + l mod (8, 8)))
```

whose phase satisfies

```
angle(Bis_c(k, l)) = angle(F_c(k)) + angle(F_c(l)) - angle(F_c(k + l))
```

## Proposition

The bispectral phase term is invariant to a global periodic translation of the board field. If `z'(s) = z(s - a)` then `F'(k) = exp(-i k . a) F(k)`, and

```
F'(k) F'(l) conj(F'(k+l))
= exp(-i (k + l - (k + l)) . a) F(k) F(l) conj(F(k + l))
= F(k) F(l) conj(F(k + l)).
```

On the finite `8 x 8` torus this holds for circular translations. Chess is not truly translation-invariant, so this is not a desired full symmetry; the useful property is that bispectrum separates arrangement phase coupling from absolute placement, while the optional coordinate planes restore absolute-board context.

## Pooled features

Each board is summarised by a fixed feature vector phi(x) with four groups:

- Bispectral phase + magnitude: for each `(c, (k, l))` we emit `cos(angle(Bis))`, `sin(angle(Bis))`, and `log(1 + |Bis|)`. The pair list contains `T = 48` deterministic structured `(k, l)` pairs (eight directional pairs followed by low x low pairs), or a deterministic random fixed list under the `random_frequency_pairs` ablation.
- Power spectrum: `log(1 + |F_c(k)|^2)` for 16 low-frequency `k` per mixed channel.
- Cross-channel phase coupling: for eight adjacent channel pairs and 12 low-frequency `k` we emit `cos`, `sin`, and `log(1 + |...|)` of `F_a(k) * conj(F_b(k))`.
- Material summary: side-relative piece counts, count delta, total count, and material balance — 20 deterministic scalars derived from the input planes.

For the default configuration (`Cmix = 16`, `T = 48`, 16 power frequencies, 8 channel pairs, 12 cross frequencies) phi(x) lives in `R^{16 * 48 * 3 + 16 * 16 + 8 * 12 * 3 + 20} = R^{2868}`.

## Decision rule

phi(x) is normalised by a `LayerNorm` and fed through a small MLP `R^{2868} -> R^{hidden} -> R^{hidden / 2} -> R` to produce one logit per board. The puzzle decision flows only through phi(x); the head never reads raw board planes, so all information is mediated by the fixed FFT + bispectrum operator on top of the learned channel mixer.

## Falsification path

The central falsifier is `magnitude_only`, which zeros the bispectral phase features but keeps `log(1 + |Bis|)` and the power spectrum. If `magnitude_only` matches the full model, third-order phase coupling is not what is helping. The `power_only` ablation removes the bispectrum entirely and tests whether higher-order coupling matters at all. The `phase_batch_shuffle` ablation rolls the phase features across the batch so they no longer line up with their own labels; if it matches, the phase features are not tied to per-position evidence. The `random_frequency_pairs` ablation replaces the structured `(k, l)` list with a deterministic random list with the same count; if it matches, any spectral sampling is enough. The `channel_pair_shuffle` ablation rolls the cross-channel phase pairs across channels; if it matches, the channel semantics are weak. The `no_coordinate_planes` ablation drops the deterministic coordinate planes; if it improves or matches, absolute-board context is unnecessary.
