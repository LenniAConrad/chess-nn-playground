# Architecture

## Scaffold-Only Implementation Notice

This folder is not a completed bespoke implementation of the architecture described below. `model.py` is a thin `ResearchPacketProbe` wrapper built with `build_research_packet_probe_from_config`, so this idea remains `implementation_kind: shared_probe_variant` and `implementation_status: probe_scaffold_only` until bespoke model code matching this markdown is added.


`Tracy-Widom Level-Spacing Network` uses the shared proposal-conditioned research-packet probe.

- Mechanism family: `linear_algebra`.
- Active proposal profile: `tracy_widom_level_spacing_network`.
- Input: board tensor only; CRTK/source metadata remains reporting-only.
- Board trunk: compact convolutional square encoder over the configured board planes.
- Proposal diagnostics: deterministic board-mechanism features selected by the
  linear-algebra profile (rank/spectral/moment/displacement-style summaries).
- Head: pooled board features + mechanism family embedding + profile hash features
  + active profile flags + linear-algebra diagnostics, returning one puzzle logit
  plus diagnostic outputs (`mechanism_energy`, `rank_file_imbalance`, etc.).

The bespoke operator described in the source packet (Sylvester / Schur complement /
Bures-Wasserstein / numerical range / Lyapunov / Pfaffian / p-adic / free-probability
/ Williamson / Magnus, depending on the idea) is not yet a hand-written torch
module. Promote this folder to a custom `model.py` when the mechanism-profile
smoke test motivates the cost.
