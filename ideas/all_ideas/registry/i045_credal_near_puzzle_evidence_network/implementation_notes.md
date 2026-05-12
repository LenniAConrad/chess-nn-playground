# Implementation Notes

- Bespoke model: `src/chess_nn_playground/models/credal_near_puzzle_evidence.py`.
- Registry key: `credal_near_puzzle_evidence_network`.
- Idea-local wrapper: `ideas/all_ideas/registry/i045_credal_near_puzzle_evidence_network/model.py`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0750_tuesday_los_angeles_credal_evidence.md`.
- Input is intentionally board-only (`simple_18` in the default config); the
  fail-closed adapter rejects any unknown channel count or encoding label
  unless `model.allow_unknown_channels=true` is explicitly set.
- The trainer reads a single binary logit from the model output dict's
  `logits` key. The Dirichlet `alpha`, evidence mass `S`, predictive mean
  `mu_pos` and `uncertainty = 2/S` are exported as auxiliary tensors so
  the report and ablations can read them.
- The credal/evidence loss-shaping constants from `math_thesis.md`
  (`near_tau`, `near_s_max`, `lambda_near_evidence_cap`,
  `lambda_dirichlet_kl`, `kl_anneal_epochs`) are stored as model
  attributes; they are consumed by an idea-specific `CredalEvidenceLoss`
  outside the `forward` pass, never as model input features.
- Engine evaluations, source labels, CRTK metadata, move generation, and
  candidate sets are NOT consumed by the network at any point.
