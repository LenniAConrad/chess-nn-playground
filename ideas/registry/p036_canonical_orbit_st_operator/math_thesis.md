# Math Thesis

Source: `ideas/research/primitives/external_31_canonical_orbit_bdd_wmc_primitives.md`,
rank-1 proposal `primitive_canonical_orbit_st`.

## Working thesis

Given a per-square latent map `X in R^{B, 64, d}` and a finite group `G`
acting on the 64 square indices by permutation actions `T_g`, define a
non-learned hash key `kappa(X)` and the canonical orbit representative

    g^*(X) = argmin_{g in G}  kappa(T_g X),
    Y      = T_{g^*(X)} X.

The argmin is computed under a deterministic lexicographic order over a
fixed quantised random projection of `X`, so it is not a learned function
and does not consume parameters.

For permutation actions, `T_g^{-1} = T_g^T` (in fact `T_g . T_g = I` for
each element of the C2 x C2 group we use), so the Jacobian of `Y` with
respect to `X` along the chosen branch is `T_{g^*}`. PyTorch's
`gather` operator already implements this gradient path, so the
"straight-through" property is automatic for permutation groups.

## Group choice

For chess the natural exact symmetries that preserve channel layout are
the four-element board-geometry group

    e            (identity)
    F            (file mirror, swap files f <-> 7-f)
    R            (rank mirror, swap ranks r <-> 7-r)
    F . R        (180-degree rotation)

These actions touch only the square axis. Colour swap and side-to-move
flip would additionally require swapping the white/black piece planes and
the logit sign, so they are kept as a deferred extension and listed in
`ablations.md`.

## Hash key

The key is

    kappa(X)[k] = sum_{s in 0..63}  w_s * round(<X_s, P_k> / q),

with `P in R^{d, key_dim}` a fixed (`torch.Generator` seed
`0xC0DEC0DE`) Gaussian projection saved as a buffer, `q = hash_quantum`
the quantisation step, and `w_s = linspace(1, 2, 64)` a fixed positional
weight. Quantisation makes the key piecewise-constant in `X`, so ties
are robust to small perturbations within one quantum but distinct
positions still produce distinct keys.

The argmin is lexicographic across the `key_dim` columns. Ties are
broken by favouring the lowest group-element index, which always
prefers identity (`e`). This eliminates the non-deterministic branch
mentioned in the failure-mode catalogue of the source packet.

## Architecture-level claim

    final_logit(x) = i193_trunk(x) + sigmoid(g(joint)) * delta(canonical_pool, residual_pool)

where `canonical_pool` is the mean over squares of the canonical
representative, `residual_pool` is the per-channel RMS of
`canonical - latent`, and `delta` is a small LayerNorm/GELU/Dropout MLP.
The gate is initialised closed (`gate_init = -2.0`) so the operator
starts as a no-op and must learn to fire on positions with strong
symmetry.

## Falsifiers

- Primitive-level: `shuffle_canonical` (in-batch permutation of the
  canonical representative) must lose any aggregate / slice lift versus
  the unablated run.
- `identity_only` (force `g^* = e`) must lose the slice lift if the
  orbit search is load-bearing.
- `fixed_choice` (always choose the file mirror) must also lose the
  slice lift if the orbit decision needs to be input-dependent.
- Architecture-level: p036 must beat i193 on its declared slice (orbit
  gap above the median, i.e. positions with a strong preferred
  orientation) without regressing aggregate PR AUC; `shuffle_canonical`
  must lose >=70% of that lift.

## Why this is not group equivariance

A G-equivariant convolution shares or averages weights across the orbit
copies; this operator picks a single representative by argmin. The
computation graph is therefore different: one branch is selected per
batch element rather than `|G|` branches averaged. The orbit gap
diagnostic exposes the magnitude of that selection. The honesty caveat
already noted in the source packet -- canonicalisation networks exist
in the equivariance literature -- is preserved: the only operator-level
claim is the fixed-key, straight-through selection of a chess-group
representative inside an i193 primitive head.

## Why this is not data augmentation

Augmentation runs the network on each transformed copy and either
averages predictions or trains across the orbit; this operator does the
canonical-quotient *inside* the model in a single forward pass and
exposes orbit-gap diagnostics. Inference cost is `|G| = 4` cheap gather
operations and a `key_dim`-wide reduction, not `|G|` full forwards.
