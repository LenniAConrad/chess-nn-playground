# Math Thesis

Forcing-Response Front-Door Bottleneck

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-28_0733_tuesday_new_york_forcing_response_bottleneck.md`.

## Working thesis

Let `x` be the current `simple_18` board tensor and `A(x)` the set of
visible-board legal candidate moves. Let `T(x, a)` be the deterministic
afterstate of applying legal move `a`, and `B(T(x, a))` the set of
opponent legal replies. A deterministic rule extractor `rho` produces
square-level planes and per-move and per-response feature vectors; let

```text
M_a = psi_rule(rho(x), a, rho(T(x, a)),
               summarize({rho(T(T(x, a), b)) : b in A(T(x, a))}))
```

be the move-response mediator node, and `M = {M_a : a in A(x)}` the
front-door surrogate mediator set. The model computes

```text
u_a       = phi_theta(M_a)                                    (move-response encoder)
u_a       = MoveGraphTransformer(u_a, ..., move_mask)         (permutation-invariant set)
g_a       = SparseWitnessGate_theta(u_a, move_mask)           (hard-concrete / top-K)
v_a       = Linear_theta(u_a)
Z_c(x)    = LayerNorm( sum_a g_a * v_a / (epsilon + sum_a g_a) )
y_hat(x)  = sigmoid(MLP_binary(Z_c(x)))
```

The *front-door surrogate* claim is the inductive bias

```text
Y ⟂ X_surface | Z_c(M).
```

`Z_c` is a low-capacity witness bottleneck over deterministic legal
intervention responses. The model is structurally forbidden from
routing pooled raw board features to the binary head, so the only path
from `X` to `Y` is through `M`. Curation/style nuisances `U` cannot
write metadata into `M` because `M` is a deterministic consequence of
the legal rules acting on `X`.

## Why this should reduce near-puzzle false positives

A near-puzzle frequently shares the same surface motifs as a true
puzzle (checks, captures, exposed kings, hanging material) but lacks
the response envelope that makes the tactic forcing. The mediator
features expose, per candidate `a`:

- candidate type, capture/check/promotion flags;
- after-move king-ring pressure;
- opponent reply availability and recapture/countercheck channels;
- attack-map and pin/ray deltas;
- escape-square and line-blocking structure.

The sparse witness gate forces the model to commit to a small set of
forcing candidates, so positions whose tactical-looking surface is not
backed by a few clean witnesses receive lower puzzle probability.

## Falsification

The thesis is rejected or weakened if any of the following hold:

1. Removing `response_features` does not change near-puzzle FPR.
2. The selected witnesses are dominated by quiet non-interacting
   moves.
3. Performance disappears when the legal-reply count is clipped to
   `0/1/2+` (mate-like leakage).
4. Mirror consistency under file reflection remains broken on
   validation.
5. Performance depends on the optional fine-label heads being
   enabled.

## Implementation Binding

- Registered model name: `forcing_response_front_door_bottleneck`.
- Source implementation file: `src/chess_nn_playground/models/forcing_response_front_door_bottleneck.py`.
- Idea-local wrapper: `ideas/registry/i081_forcing_response_front_door_bottleneck/model.py`.
