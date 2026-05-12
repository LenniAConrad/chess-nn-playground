# Math Thesis

Attack-Hodge Sheaf Tension Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0256_tuesday_local_attack_hodge_sheaf.md`.

The central claim is that puzzle-like positions often contain inconsistent local systems of attack, defense, pin, fork, and overload constraints. A square-grid CNN sees the board as local texture, while this model constructs a current-board tactical cell complex and measures learned sheaf-Hodge tension over nodes, attack edges, and tactical faces.

For a board tensor `x`, the rule path constructs:

- square 0-cells `V = {0, ..., 63}`;
- directed attack/control 1-cells `E(x)` from pseudo-legal pawn, knight, king, bishop, rook, and queen geometry;
- tactical 2-cells `F(x)` for fork fans, overload sinks, and ray-pin/x-ray incidence.

Each square has a node cochain `h_v in R^d`, each attack edge has an edge cochain `a_e in R^d`, and each tactical face has a face cochain `c_f in R^d`.

The node-edge coboundary is:

```text
(D0 h)_e = rho_dst[r(e)] h_dst(e) - rho_src[r(e)] h_src(e)
```

where the learned restrictions are diagonal plus low-rank transports tied by side-aware edge type. The edge-face coboundary is:

```text
(D1 a)_f = sum_{e in boundary(f)} sigma(f,e) rho_face[t(f)] a_e
```

The nonnegative Hodge/sheaf tension used by the model is:

```text
E(x) = mean_e ||D0 h||^2 + mean_f ||D1 a||^2
```

with masks for padded edges and faces. The first term measures whether endpoint square stalks can agree on an attack relation. The second term measures curl-like inconsistency around higher-order fork, overload, and ray-pin faces.

The classifier does not use this scalar alone. It pools final node, edge, and face cochains together with per-layer tension summaries. The repo task contract uses one BCE logit where fine labels 0 and 1 are non-puzzle and fine label 2 is puzzle; this overrides the packet's earlier two-logit CE sketch.
