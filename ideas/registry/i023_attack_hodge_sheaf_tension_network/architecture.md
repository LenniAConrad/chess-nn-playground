# Architecture

`Attack-Hodge Sheaf Tension Network` implements the packet's board-only 0/1/2-cell tactical complex for the repo's `puzzle_binary` benchmark. It classifies fine labels 0 and 1 as non-puzzle and fine label 2 as puzzle through the configured one-logit BCE head.

## Implementation Binding

- Registered model name: `attack_hodge_sheaf_tension_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/attack_hodge_sheaf.py`
- Idea-local wrapper: `ideas/registry/i023_attack_hodge_sheaf_tension_network/model.py`

## Encoding Adapter

`EncodingAdapter` validates `(B, C, 8, 8)` board tensors and decodes current piece planes, piece color, piece type, side-to-move, and side-relative roles. The square stem may read all configured input planes, but the tactical complex is built only from decoded current-board pieces and side-to-move. Engine output, source tags, verifier metadata, fine labels, and unresolved-candidate fields are not model inputs.

`simple_18` is the primary binding. LC0-style 112-plane tensors are accepted only under the local assumption that the current piece slice is in the first twelve planes; `lc0_bt4_112` is treated as side-relative for graph construction because absolute original orientation is not recoverable from the tensor alone.

## Attack-Hodge Complex

`AttackComplexBuilder` constructs a padded, per-position cell complex:

- 0-cells are the 64 board squares;
- 1-cells are directed pseudo-legal attack, defense, control, king-contact, and optional one-blocker x-ray edges;
- 2-cells are deterministic tactical incidence faces.

Pawn edges are diagonal attacks only. Knights and kings use their fixed offsets. Bishops, rooks, and queens trace slider rays through empty controlled squares, include the first blocker, and stop. When enabled, x-ray edges continue through exactly one blocker to the next occupied square.

Face families are:

- `fork_fan`: pairs of outgoing attack or king-contact edges from the same source;
- `overload_sink`: pairs of incoming attack/defense/control edges sharing a target square;
- `ray_pin`: a first-blocker slider edge paired with its one-blocker x-ray edge.

Edges and faces are truncated by deterministic source/target/type ordering only, not by labels or engine information.

## Cochains And Restrictions

`SquareStem` creates node cochains `H0` from raw square planes, decoded piece/color/role features, side-to-move, and square coordinates.

`EdgeInitializer` creates edge cochains `H1` from source/target node states plus edge type, tactical group, and geometry embeddings. `FaceInitializer` creates face cochains `H2` from signed boundary edge states plus face type.

`HodgeTensionBlock` maintains learned diagonal-plus-low-rank sheaf restrictions:

```text
D0 h = rho_dst(h_dst) - rho_src(h_src)
D1 a = sum_{e in boundary(f)} sign(f,e) rho_face_type(a_e)
```

Node updates use `D0^T D0` tension, edge updates combine node-edge disagreement with `D1^T D1` face curl, and face updates relax the face curl. All updates are masked and residual-normalized.

## Readout

`MaskedCochainPool` pools node, edge, and face cochains with masked mean, max, and standard deviation. If `use_energy_pool` is enabled, it also appends per-layer node-edge and face-curl energy summaries by tactical group.

The classifier returns `output["logits"]` with shape `(B,)` for `num_classes: 1`. Diagnostics include `sheaf_tension`, `hodge_edge_tension`, `node_edge_energy`, `face_curl_energy`, `attack_energy`, `defense_energy`, `xray_energy`, `fork_fan_energy`, `overload_sink_energy`, `ray_pin_energy`, `edge_density`, and `face_density`.
