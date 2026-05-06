# Architecture

## Scaffold-Only Implementation Notice

This folder is not a completed bespoke implementation of the architecture described below. `model.py` is a thin `ResearchPacketProbe` wrapper built with `build_research_packet_probe_from_config`, so this idea remains `implementation_kind: shared_probe_variant` and `implementation_status: probe_scaffold_only` until bespoke model code matching this markdown is added.


`Legal-Reaction Bottleneck Network` uses the shared proposal-conditioned research-packet probe.

- Mechanism family: `graph`.
- Active proposal profiles: `graph`, `defender_reply`.
- Input: board tensor only; CRTK/source metadata remains reporting-only.
- Board trunk: compact convolutional square encoder over the configured board planes.
- Proposal diagnostics: deterministic board-mechanism features selected from the active profiles, including sheaf/pressure tension, transport imbalance, symmetry residuals, topology and king-path pressure, logic/ray evidence, linear-algebra moments, information and calibration scores, sparse certificate energy, graph/reply pressure, spatial CNN cues, and phase/cost proxies when relevant.
- Head: the classifier receives pooled board features, the mechanism family embedding, profile hash features, active profile flags, and the selected proposal diagnostics. It returns one puzzle logit plus diagnostic outputs such as `mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count`, `sheaf_tension`, `transport_imbalance`, `symmetry_residual`, `topology_pressure`, `ray_language_energy`, `information_surprisal`, `sparse_certificate_energy`, `rank_file_imbalance`, `king_ring_pressure`, `reply_pressure`, and `defense_gap`.
