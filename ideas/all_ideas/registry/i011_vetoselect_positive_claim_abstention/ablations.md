# Ablations

Primary implemented run:

- A2: VetoSelect with self-mined decoys and no rule texture.

Recommended follow-ups:

- A0: LC0 BT4 binary baseline on the same split.
- A1: set `training.veto_select.warmup_epochs` greater than total epochs to disable decoys.
- A3: add deterministic current-board rule texture when the data loader exposes safe rule features. Implemented in `config_v2.yaml`.
- A4: set `lambda_anchor: 0.0`.
- A5: compare raw `puzzle_logit`, `z + a`, accepted-puzzle probability, and exact `selective_puzzle_logit`.

Falsify the idea if near-puzzle false positives do not improve at matched recall or if PR AUC/F1 collapse without a compensating near-puzzle gain.
