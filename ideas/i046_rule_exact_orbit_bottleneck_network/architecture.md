# Architecture

`Rule-Exact Orbit Bottleneck Network` uses the shared proposal-conditioned research-packet probe.

- Mechanism family: `symmetry`.
- Active proposal profiles: `symmetry`.
- Input: board tensor only; CRTK/source metadata remains reporting-only.
- Board trunk: compact convolutional square encoder over the configured board planes.
- Proposal diagnostics: deterministic board-mechanism features selected from the active profiles, including sheaf/pressure tension, transport imbalance, symmetry residuals, topology and king-path pressure, logic/ray evidence, linear-algebra moments, information and calibration scores, sparse certificate energy, graph/reply pressure, spatial CNN cues, and phase/cost proxies when relevant.
- Head: the classifier receives pooled board features, the mechanism family embedding, profile hash features, active profile flags, and the selected proposal diagnostics. It returns one puzzle logit plus diagnostic outputs such as `mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count`, `sheaf_tension`, `transport_imbalance`, `symmetry_residual`, `topology_pressure`, `ray_language_energy`, `information_surprisal`, `sparse_certificate_energy`, `rank_file_imbalance`, `king_ring_pressure`, `reply_pressure`, and `defense_gap`.
