# Math Thesis

Spline Board Surface Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.

Batch candidate rank: `4`.

Working thesis: Chess boards may benefit from a smooth geometric baseline that
is not convolutional. Fit learned tensor-product spline surfaces to piece
planes and classify from low-degree surface coefficients plus residual maps.

## Smooth-surface decomposition

Let `P : {0,...,7}^2 -> R` be a single piece plane and let ``B(y, x)`` be the
tensor-product Bernstein basis of degree ``d = spline_basis_size - 1`` on the
8x8 grid,

``B[y, x, i, j] = b_i^{d}(t_y) * b_j^{d}(t_x)``,

where ``b_i^{d}(t) = C(d, i) * t^i * (1 - t)^{d - i}`` is the i-th Bernstein
polynomial and ``t_y, t_x in {0, 1/7, ..., 1}``.  Flatten ``B`` to a matrix
``Phi : (64, K)`` with ``K = spline_basis_size**2`` columns.

For each plane the smooth surface coefficients are the least-squares fit

``c* = argmin_c || Phi c - vec(P) ||_2^2 = Phi^+ vec(P)``

where ``Phi^+`` is the Moore-Penrose pseudoinverse.  The smooth surface is
``S = Phi c*`` and the residual map is ``R = vec(P) - S``.  This is exactly
the "fit a low-degree tensor-product surface" recipe from the source packet.

## Classification signal

The classifier reads three deterministic features per board:

* the smooth coefficients ``c* : (18, K)`` per piece plane,
* the residual energies ``e = || R ||_2^2 : (18,)`` -- the Frobenius mass
  that the smooth surface failed to capture,
* a compact residual-map summary obtained by a 1x1 channel mixer on the 18
  residual planes followed by mean/max pooling (this is the "optionally
  feed residual maps to a small head" branch from the source packet).

Together these expose the full smooth/sharp split of each board: the smooth
coefficients describe the low-degree geometry, the residual energies score
how non-smooth each piece plane is, and the residual summary preserves
spatial information about *where* the residuals concentrate without
introducing a CNN trunk over the raw board planes.

## Distinctness

This is *not* a wavelet scattering network: the basis is a smooth
low-degree control geometry rather than a multiresolution tight frame.
It is *not* a masked codec: there is no reconstruction pretraining; the
projection is closed-form via the pseudoinverse.  And it is *not* a CNN
over board planes: the only convolution in the model is a 1x1 channel mixer
applied to the residual maps, never to the original planes.
