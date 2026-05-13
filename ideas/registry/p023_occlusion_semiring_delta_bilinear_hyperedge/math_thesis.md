# Math Thesis

Source: `ideas/research/primitives/external_19_occlusion_semiring_delta_bilinear_hyperedge.md`,
rank-1 proposal `primitive_1_occlusion_semiring_scan`.

## Working thesis

For each board square `s`, direction `r`, ordered ray cells
`c_{s, r, 1..L}`, occupancy `O`, and value projection `V`:

```
h_{b, r, L} = 0
h_{b, r, t} = (1 - O_{b, c_{r, t+1}}) * h_{b, r, t+1} + V * x_{b, c_{r, t+1}}
```

This is a **backward recurrence** along the ray: the hidden state at
depth `t` aggregates contributions from positions deeper into the
ray, gated by `(1 - O)`. The source-side hidden state `h_{r, 0}` is
the directional ray feature read out at the source square.

After the recurrence, we compute a **bilinear hyperedge** over each
of the 4 opposing-direction pairs `(N, S), (NE, SW), (E, W), (SE,
NW)`:

```
edge_{b, s, p} = (W_L h_{b, left_p, s}) (.) (W_R h_{b, right_p, s})
```

The hyperedge embedding `edge_{b, s, p}` encodes the "attacker -- own
piece -- defender along one line" motif: a non-trivial product
requires both halves of the line to carry information.

## Architecture-level claim

The hyperedge embeddings are concatenated along the pair axis,
mean-pooled across squares, and fed through a LayerNorm + GELU MLP
to produce the primitive delta. A sigmoid gate on the trunk joint
feature controls how much of the delta is added to the i193 base
logit.

## Falsifier

- Primitive-level: `zero_occupancy` (no blocker gate) and
  `uniform_occupancy` (occupancy = 1 everywhere) must each hurt the
  declared slice relative to `none`. If neither does, the
  transmittance is not load-bearing.
- `disable_bilinear` replaces the hyperedge product with a sum
  (`left + right` instead of `left (.) right`). If `disable_bilinear`
  matches `none`, the bilinear hyperedge claim fails.
- Architecture-level: p023 must beat i193 on through-the-square
  motifs without regressing aggregate PR AUC.

## Why this is distinct from p020 and p021

- p020 uses a **forward** recurrence with a hard reset gate
  `(1 - O_t) (.) lambda_d (.) h_{t-1}`; the hidden state at step `t`
  depends on the *prior* steps of the ray. p023's recurrence runs in
  reverse: `h_t` depends on the *later* steps.
- p021 uses a non-recurrent **forward exclusive prefix product**
  `T_{l} = prod_{q<l} (1 - O_q)` to weight each cell's contribution
  independently. p023's backward recurrence carries hidden state.

These three formulations expose different gradient flow per ray
segment and different incremental-update semantics under bounded-
change inputs. They are not mathematical duplicates.

## Why this is not standard hypergraph message passing

Hypergraph NNs typically consume a *supplied* hypergraph structure;
here the hyperedges are generated *inside* the operator by the chess
geometry (opposing-direction pairs) and the per-square content
(through the V projection). The hyperedge embedding is a bilinear
product of opposite-direction ray hidden states, not a generic
permutation-invariant set aggregation.
