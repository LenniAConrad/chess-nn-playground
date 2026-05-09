# Math Thesis

Tactical Hessian Spectrum Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md`.

Batch candidate rank: `2`.

Working thesis: A real puzzle is a sharp local maximum of tactical
evidence under legal perturbations. Near-puzzles may have high raw
evidence but flatter or less stable local geometry.

## Formal sketch

Let `x` denote the board tensor and let `E(x) ∈ R` be a
differentiable scalar tactical-evidence functional learned by a
compact convolutional encoder. Choose `K` board-shaped perturbation
directions `D_1, ..., D_K` (chess-meaningful, unit Frobenius norm);
the local sharpness of `E` at `x` is captured by the reduced Hessian

```
H_red[i, j] = ∂² E(x + s_i D_i + s_j D_j) / (∂ s_i ∂ s_j)  evaluated at s = 0
```

approximated by symmetric finite differences with step `eps > 0`:

```
H_red[k, k] ≈ (E(x + eps D_k) + E(x - eps D_k) - 2 E(x)) / eps^2
H_red[i, j] ≈ (E(x + eps (D_i + D_j)) - E(x + eps D_i)
               - E(x + eps D_j) + E(x)) / eps^2,   i ≠ j
```

After symmetrising `H_red ← (H_red + H_red^T) / 2`, the eigenvalues
`λ_1 ≤ ... ≤ λ_K` of `H_red` are real. Under the thesis:

- A real puzzle should display strongly negative eigenvalues — the
  evidence surface curves *down* in every probed direction — so
  `concavity := - Σ_{λ_k < 0} λ_k` is large and `λ_K` (top
  eigenvalue) is small or negative.
- A near-puzzle has high raw `E(x)` but a flatter or indefinite
  spectrum — eigenvalues bunched near zero or mixed signs — so
  `concavity` is small and `spectral_radius := max_k |λ_k|` is
  dominated by noise.

The puzzle classifier reads the eigenvalues plus sharpness scalars
(`spectral_gap`, `trace`, `positive_curvature`, `spectral_radius`,
`concavity`), the directional gradient norm `||g||`, the evidence
`E(x)`, and pooled trunk features. The model is therefore a real
implementation of "puzzle = sharp local max of tactical evidence"
rather than a generic mechanism probe over packet-profile features.
