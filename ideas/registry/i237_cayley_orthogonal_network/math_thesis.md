# Math Thesis

Cayley Orthogonal Map Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-05-05_1705_tuesday_local_cayley_orthogonal_map.md`.

Working thesis: Builds skew-symmetric A in R^{r x r} from board features and forms the Cayley map Q = (I-A)(I+A)^{-1} in SO(r); rotates a learned reference basis and uses identity-deviation features. Distinct from polar-Procrustes (i063) and QR -- Cayley is an algebraic identity, not a decomposition.

This idea is **implemented as a bespoke torch module** at
`src/chess_nn_playground/models/trunk/cayley_orthogonal.py`
(class `CayleyOrthogonalNetwork`, builder `build_cayley_orthogonal_from_config`); not routed
through the generic ResearchPacketProbe.
