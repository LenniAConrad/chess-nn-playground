# Implementation Notes

- Central code: `src/chess_nn_playground/models/replicator_payoff_piece_dynamics.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i131_replicator_payoff_piece_dynamics/model.py`.
- Registry key: `replicator_payoff_piece_dynamics`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.
- Batch candidate: `Replicator Payoff Piece Dynamics`.
- Inputs: simple_18 board tensor only. CRTK/engine/source metadata is reporting-only.
- Tunables surfaced via config: `max_pieces` (default `32`), `token_dim` (`64`),
  `pair_hidden_dim` (`64`), `num_heads` (`4`), `num_steps` (`5`), `eta` (`0.5`),
  plus standard `channels`/`hidden_dim`/`depth`/`dropout`/`use_batchnorm` knobs.
- Forward returns a dict: `logits` of shape `(batch,)` plus per-head and aggregated
  diagnostics (`entropy`, `top_mass`, `kl_from_initial`, `avg_payoff`,
  `fitness_variance`, mass-on-{own,opp,king,pawn,minor,major}, payoff asymmetry,
  total piece count, backbone feature norm).
