# Implementation Notes

- Bespoke implementation:
  `src/chess_nn_playground/models/clifford_rotor_threat_network.py`
  (`CliffordRotorThreatNetwork` and
  `build_clifford_rotor_threat_network_from_config`).
- Registry key: `clifford_rotor_threat_network`
  (`src/chess_nn_playground/models/registry.py`).
- Idea wrapper: `ideas/all_ideas/registry/i232_clifford_rotor_threat_network/model.py` calls
  the bespoke builder via
  `build_clifford_rotor_threat_network_from_config`.
- Input is the board tensor only; CRTK / source / engine metadata
  remains reporting-only and never enters the model.
- The Cl(3, 0) geometric-product structure tensor (8x8x8) and reverse-sign
  table are precomputed once at module load and stored as buffers; the
  six chess relation adjacencies (king, knight, rank, file, two
  diagonals) are likewise precomputed and registered as a buffer.
- Rotors are computed by a 4-term Taylor series of `exp(B / 2)` followed
  by a `|R| = 1` renormalisation so the sandwich uses
  `R^{-1} = reverse(R)`.
