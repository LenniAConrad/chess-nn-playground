# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/absorbing_threat_markov_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i200_absorbing_threat_markov_network/model.py`
  (a thin adapter over `build_absorbing_threat_markov_network_from_config`).
- Registry key: `absorbing_threat_markov_network`. The shared
  `ResearchPacketProbe` wrapper has been removed.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md`.
- Batch candidate: `Absorbing Threat Markov Network`.
- Strictly board-only: simple_18 tensor in, no engine/source/CRTK
  metadata enters the model. CRTK metadata remains reporting-only.
- Two absorbing states (`proof_absorb`, `disproof_absorb`) live in the
  last two rows of the state-embedding table; the transition matrix
  forces these rows to identity so probability mass cannot leak out.
- Power iteration depth is set by `transition_steps` (default `4`);
  with `transition_steps = 1` the chain becomes a one-step kernel
  (`one_step_only` ablation in the markdown).
