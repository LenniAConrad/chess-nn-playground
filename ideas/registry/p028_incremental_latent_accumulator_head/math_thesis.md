# Math Thesis

Source: `ideas/research/primitives/external_24_incremental_latent_accumulator_directional_scan.md`
(Incremental Latent Accumulator, ILA — first-ranked proposal).

## Operator

Let `x in {0, 1}^{12 x 64}` be the simple_18 piece-plane indicator
(12 piece types times 64 squares) and `k in {0, ..., 64}` the own-king
square (with 64 meaning "no own king"). The ILA operator factorises
into a global accumulator and a king-anchored accumulator:

```
h_global = sum_{(t, s) : x_{t, s} = 1} G_{t, s}
h_king   = sum_{(t, s) : x_{t, s} = 1} K_{k, t, s}
h_concat = [h_global, h_king]
z        = LayerNorm(phi(h_concat))    (non-linear lift)
```

where `G in R^{12 x 64 x global_dim}` and
`K in R^{65 x 12 x 64 x king_dim}` are learned embedding tables.
`phi` is a small MLP (LayerNorm -> Linear -> GELU -> Linear) that
intentionally distinguishes ILA from `p025` / IDL.

## What is proven

- `h_global` is identical in shape to the IDL accumulator (`p025`),
  so the linear half of the operator is the same primitive.
- `h_king` is a king-anchored linear sum, which is exactly the HalfKA
  refinement that Stockfish NNUE uses. Different king squares route
  to different rows of `K`, so the network can express king-context-
  dependent piece valuations.
- `phi` adds a single non-linearity to the linear accumulator; with
  `phi(x) = x` the operator collapses back to a linear accumulator
  (we expose this via the `linear_only` ablation).

## What is hypothesised

- King-anchored embeddings carry signal that the IDL accumulator
  alone cannot — specifically about king-safety, king-zone activity,
  and endgame king position.
- The non-linear `phi` lift extracts richer features than the pure
  sparse linear sum, particularly on positions with non-trivial
  interactions between distant pieces.

## Architecture-level claim

```
final_logit(x) = i193_trunk(x) + primitive_gate(x) * primitive_delta(x)
```

The primitive head's input is the latent `z` plus the four standard
trunk diagnostics.

## Failure cases

- The trunk already encodes king-zone features (the king stream).
  ILA's king-anchored accumulator may add no marginal information.
  In that regime the gate collapses to ~0 and the head is silent.
- The king-anchored embedding table is large
  (65 * 12 * 64 * king_dim = ~50k floats @ king_dim=16). We start
  with a small `king_dim` and rely on weight decay to keep it
  generalisable.
- Memory cost grows linearly in `king_dim`; we keep it at 16 by
  default.

## Falsifiers

- `zero_global_accumulator`: hold `h_global = 0`. Tests whether the
  global accumulator is load-bearing.
- `zero_king_accumulator`: hold `h_king = 0`. Tests whether the
  king-anchored accumulator is load-bearing. **Primary falsifier.**
- `linear_only`: skip the `phi` non-linearity. Tests whether the
  non-linear lift is load-bearing.
- `shuffle_square_order`: random column permutation of the indicator
  before each sparse sum. Decouples per-square structure from real
  squares; matches IDL's `shuffle_squares` ablation.
