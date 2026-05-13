# Math Thesis

Source: `ideas/research/primitives/external_21_incremental_delta_linear_color_involution_adjacency.md`
(Incremental Delta-Linear Operator, IDL — first-ranked proposal).

## Operator

Let `x in {0, 1}^{12 x 64}` be the simple_18 piece-plane indicator (12
piece types times 64 squares). The IDL operator is parameterised by a
learned embedding table `E in R^{12 x 64 x d}` and outputs the sparse sum

```
S(x) = sum_{(t, s) : x_{t, s} = 1} E_{t, s}
```

The forward in PyTorch is `S = einsum('bts,tsd->bd', x, E)`. The
"incremental" interpretation comes from the linear-additive structure:
given a chess move that changes ``k`` squares (typically 2-4), updating
``S`` requires reading ``k`` rows of ``E`` and adding their signed delta,
giving the ``O(k)`` per-move update that motivates NNUE's HalfKA.

## What is proven

- ``S(x)`` is linear in ``x``, so changing the set of occupied squares
  produces an additive update equal to the sum of the per-cell rows.
- Backward is `dS/dE_{t, s} = x_{t, s}` and `dE accumulated across the
  batch is `sum_b x_b`. Both are standard, no custom autograd needed.

## What is hypothesised

- The pure sparse linear accumulator carries enough signal *on top of the
  i193 trunk* to improve the puzzle logit on slices where a stable
  per-(piece-type, square) statistic discriminates positives from
  negatives. Examples include simple material-count puzzles and rook /
  back-rank squares.
- The fact that the trunk runs in parallel means the head only needs to
  contribute marginal information. The gate's job is to suppress the
  head's delta on positions where it has no marginal signal.

## Architecture-level claim

```
final_logit(x) = i193_trunk(x) + primitive_gate(x) * primitive_delta(x)
```

where ``primitive_gate(x), primitive_delta(x)`` are MLPs over the fusion
vector ``[norm(S(x)), trunk_diagnostics(x), ||S(x)||]``.

## Failure cases

- The trunk already encodes material/king-square structure densely enough
  that the IDL accumulator adds no marginal signal. In that regime the
  gate collapses to ~0 and the head is silently inert (no regression but
  no lift).
- The embedding table over-fits 12*64*d positions of free parameters on a
  small scout split. We start at d=48 (~36k params) and gate-bias the
  primitive shut at init (`gate_init=-1.5`).
- A pure linear operator on piece presence is **not** rotation-invariant
  or color-symmetric; the IEL primitive in the same research file would
  address this but is out of scope for this head. See the deferred-
  proposals section in `architecture.md`.

## Falsifiers

- `shuffle_squares`: random column permutation of ``x`` before the
  einsum. Decouples the per-square embedding from the actual square.
  If the unablated and shuffled runs match on the target slice, the
  per-square structure of ``E`` is not load-bearing and the primitive is
  dropped.
- `permute_piece_types`: row permutation of ``x``. Same logic but on the
  piece-type axis.
- `zero_accumulator`: hold ``S(x) = 0``. Identifies whether the trunk
  diagnostics in the fusion vector are doing all the work (they should
  not, since the trunk already emits them upstream).
