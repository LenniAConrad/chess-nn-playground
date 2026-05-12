# Math Thesis

Harmonic Board Potential Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2045_friday_shanghai_harmonic_potential.md`.

## Working Thesis

Puzzle-like positions may be identifiable by long-range board tension
patterns that appear after solving fixed discrete Poisson equations over
learned current-board charge maps.

## Setup

Let `x in R^{C x 8 x 8}` be a current-board tensor (simple_18). A safe
charge encoder produces signed charge maps `rho_k(x) in R^{64}` for
`k = 1..K` from current channels only via a 1x1 convolution. Let `L` be
the fixed 8x8 grid Laplacian with Neumann boundary, and `lambda_l > 0`
a small set of screening constants. Define for each `(k, l)`:

```
u_{k,l}(x) = (L + lambda_l * I)^{-1} rho_k(x)
E_{k,l}(x) = rho_k(x)^T u_{k,l}(x)
D_{k,l}(x) = sum_{(a,b) in edges} (u_{k,l}(a) - u_{k,l}(b))^2 = u^T L u
```

The hypothesis is that some tactical opportunities create long-range
charge configurations whose smooth potential fields, fluxes, and energies
separate puzzle-like positions from ordinary positions better than local
texture alone.

## Variational Principle

For `lambda > 0`, `u = (L + lambda I)^{-1} rho` is the unique minimizer of

```
J(u) = 0.5 * u^T (L + lambda I) u - rho^T u.
```

Therefore the solver computes the lowest-energy global field matching the
learned charges under the board geometry. This is a fixed, deterministic,
stable, global linear function of the safe current-board charges; the
operator itself is not learned.

## What Is Actually Proven

- `u_{k,l}` is a fixed deterministic global linear function of `rho_k`.
- The set of screening constants `{lambda_l}` defines multiple spatial
  ranges of harmonic coupling.
- Because charges are produced from current-tensor channels only and the
  Green matrices are pure board geometry, the model cannot use engine,
  verification, source, or CRTK metadata as input.

## What Remains Hypothesised

- That learned charges discover chess-relevant tension rather than
  material shortcuts.
- That inverse-Laplacian range coupling is better than CNN receptive
  fields at the current data scale.

## Central Falsifiers (Section 9 of the packet)

The implementation exposes the three falsifier ablations directly via
`ablation` so they can be wired into the same trainer entrypoint:

- `random_orthogonal_solver`: replace each `G_l` with a fixed
  deterministic orthogonal matrix of matched Frobenius norm. If this
  matches the main model, the solver is acting as an ordinary global
  random projection rather than a harmonic field solver.
- `local_gaussian_solver`: replace each `G_l` with a fixed isotropic
  Gaussian blur of comparable spatial scale. If this matches, local
  multiscale smoothing is sufficient and the inverse-Laplacian distance
  law is not contributing useful structure.
- `charge_only_stats`: bypass the solver and feed only charge moments to
  the head. If this matches, the learned charges alone explain the
  signal and the potential field is not adding evidence.

## Self-Critique

Harmonic smoothing may be too generic and may blur away chess-specific
detail. The random-transform and local-blur controls isolate whether the
Green-function distance law matters or whether any global projection is
enough.
