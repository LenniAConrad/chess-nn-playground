# Math Thesis

Tactical Threat-Sheaf Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0255_tuesday_local_threat_sheaf.md`.

The thesis is that puzzle-like positions often contain localized incompatibilities among attack, defense, pin, king-contact, and overloaded target-square relations. Rather than treating those signals as square-grid texture, this architecture builds a pseudo-legal attack-defense complex from the current board and learns a cellular sheaf over its typed edges.

For each board tensor `x`, the model constructs a directed complex `K(x)` whose vertices are board squares and whose edges are deterministic pseudo-legal attacks or defenses. Pawns use diagonal attacks only, leapers use their ordinary jumps, and sliders stop at the first blocker. If a slider attacks an enemy blocker that shields its own king, that first-blocker edge is marked as a pin-line relation.

Every square has a learned stalk vector `z_v in R^d`. For relation type `r`, the model learns source and target restrictions:

```text
A_src[r], A_dst[r]: R^d -> R^d
```

with `diag(a) + U V^T` as the default parameterization. The edge coboundary is:

```text
delta_e = A_src[r(e)] z_src(e) - A_dst[r(e)] z_dst(e)
```

and the fixed-graph, fixed-gate energy is:

```text
E(z; x) = sum_e w_e g_e ||delta_e||_2^2 >= 0
```

where `w_e` is deterministic degree normalization and `g_e` is a learned gate from endpoint states and edge semantics. The sheaf layers perform learned gradient-style diffusion on this energy while injecting contest-cell messages summarizing incoming target-square tension by side.

The readout pools node states, edge tension groups, pin-line tension, contest pressure, overload pressure, and board counts into one puzzle logit. The source packet sketches a two-class head, but this repo idea follows the configured `puzzle_binary` BCE contract: fine labels 0 and 1 are non-puzzle, and fine label 2 is puzzle.
