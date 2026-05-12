# Implementation Notes

The reusable implementation lives in `src/chess_nn_playground/models/trunk/vetoselect.py`.

Training support is implemented through `training.losses.VetoSelectLoss` and small trainer changes that understand model output dictionaries. Metrics and predictions use `selective_puzzle_logit` as the selected score while saving raw VetoSelect diagnostics in prediction artifacts.

The first pass used board-only decoys with `texture = 1` for negative examples. That corresponds to ablation A2 in the source packet.

The v2/A3 path adds deterministic current-board tactical texture in `src/chess_nn_playground/data/tactical_texture.py`. The model still receives only the board tensor; the texture scalar is emitted by `ChessPositionDataset(include_rule_texture=True)` and is used only by `VetoSelectLoss` to weight self-mined decoy targets after warmup. Prediction artifacts include `rule_texture` for diagnostics.
