# Codex Research Packet: Hadamard Walsh-Spectrum Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1700_tuesday_local_hadamard_walsh_spectrum.md`
- Generated at: 2026-05-05 17:00
- Author: Claude (Opus 4.7, 1M context)
- Status: bespoke implementation already in `src/chess_nn_playground/models/hadamard_spectrum.py`

## Thesis

Apply the **Walsh-Hadamard transform** `WHT_64` (Sylvester construction
`H_{2n} = H_n ⊗ H_2`) to per-channel pooled square signals; classify
puzzle-likeness from top-k Walsh coefficients. The Walsh basis is the boolean
Fourier transform on the 6-dim hypercube of square indices in `{0,1}^6` —
orthogonal in `{-1, +1}`, structurally distinct from DCT, wavelets (i093), and
spectral-Laplacian features.

## Distinct From

- Wavelet scattering (i093): continuous, Daubechies-style, scale-translation invariance.
- Bispectral (i066): three-point correlations of complex Fourier coefficients.
- Character-sum (i067): finite-field characters, not Walsh.

```text
WHT is the unique unitary (orthogonal up to scale) transform whose basis is
{-1,+1}^64; it diagonalizes XOR-convolution on Z_2^6, which has no analog in
the imported transform packets.
```

## Architecture

`HadamardSpectrumNetwork` in `src/chess_nn_playground/models/hadamard_spectrum.py`:

```text
input (B, 18, 8, 8)
  -> BoardConvStem -> (B, C, 8, 8)
  -> 1x1 bank_proj -> (B, C, 8, 8)
  -> flatten -> (B, C, 64)
  -> WHT_64 (fixed buffer) -> (B, C, 64)  # einsum
  -> top-k by |coeff|, signed -> (B, C * k)
  -> concat with global mean -> (B, C * k + C)
  -> MLP -> (B, num_classes)
```

64x64 Walsh matrix is precomputed once via 6 Kronecker products of `[[1,1],[1,-1]]`.

## Ablations

| Ablation | Target |
|---|---|
| `random_walsh_basis` | replace WHT with random orthogonal | tests Walsh structure |
| `top_k_eq_4` | shrink to top-4 coefficients | tests bottleneck |
| `magnitude_only` | drop signs | tests boolean Fourier sign info |
| `dft_swap` | replace WHT with real DFT | tests Z_2^6 vs Z/64 |
| `cnn_same_params` | matched baseline | |

## Falsifier

`random_walsh_basis` should drop PR AUC ≥ 0.01. If not, the `{-1, +1}` boolean Fourier structure is unnecessary and a generic orthogonal projection works equally.

## Targets

PR AUC ≥ 0.82, F1 ≥ 0.76, near-puzzle FPR ≤ 0.20.
