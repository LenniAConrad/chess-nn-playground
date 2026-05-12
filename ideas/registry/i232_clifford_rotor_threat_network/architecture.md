# Architecture

`Clifford Rotor Threat Network` embeds each square as an 8-dim multivector in
the geometric algebra `Cl(3, 0)` (1 scalar, 3 vectors, 3 bivectors, 1
trivector). Pressure between squares is composed via the non-commutative
geometric product `ab = a . b + a ^ b`; learned rotors `R = exp(B / 2)` act
on neighbour multivectors via the sandwich `x -> R x R^{-1}`; per-grade
pooled features over a fixed family of chess relations capture the
rotational threat structure no real-arithmetic packet can express.

## Pipeline

- Input: board tensor `(B, 18, 8, 8)`. CRTK / source metadata is
  reporting-only and never used as model input.
- Convolutional trunk lifts each square to `channels` features.
- A 1x1 convolution projects to 8 multivector blades per square,
  producing a multivector field `phi: (B, 64, 8)` with blade indexing
  `(scalar, e1, e2, e1 e2, e3, e1 e3, e2 e3, e1 e2 e3)`.
- The bivector grade of `phi` is clipped to `||B|| <= bivector_clip` and a
  rotor field `R = exp(B/2)` is computed via a 4-term Taylor expansion of
  the geometric product, then renormalised so `|R| = 1` (giving
  `R^{-1} = reverse(R)` exactly).
- Per square, the sandwich `phi' = R phi R^{-1}` rotates the local
  multivector frame.
- Six fixed chess relations - king ring, knight, same rank, same file,
  a1-h8 diagonal, a8-h1 anti-diagonal - aggregate `phi'` over neighbours.
- A per-relation geometric-product message
  `m_r[s] = phi'[s] * sum_t W_r[s, t] phi'[t]` is computed via the
  Cl(3, 0) structure tensor and gated by a learned scalar per relation.
- Each message is split by grade (0/1/2/3) and pooled (mean and max
  over squares) into 4 grades x 2 stats per relation.
- Scalar diagnostics: bivector norm, rotor norm, sandwich residual,
  trivector chirality, and per-grade `phi` energies.
- A LayerNorm + GELU MLP head consumes the pooled trunk features,
  per-grade message statistics, and scalar diagnostics, returning one
  puzzle logit plus diagnostic outputs.

## Implementation Binding

- Registered model name: `clifford_rotor_threat_network` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Source implementation file:
  `src/chess_nn_playground/models/clifford_rotor_threat_network.py`
  (`CliffordRotorThreatNetwork` and
  `build_clifford_rotor_threat_network_from_config`).
- Idea-local wrapper:
  `ideas/registry/i232_clifford_rotor_threat_network/model.py` calls
  `build_clifford_rotor_threat_network_from_config`.
- The shared `ResearchPacketProbe` scaffold is no longer used by this
  idea.
