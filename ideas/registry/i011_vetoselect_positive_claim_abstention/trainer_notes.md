# Trainer Notes

Use the guarded idea entrypoint:

```bash
python ideas/registry/i011_vetoselect_positive_claim_abstention/train.py
```

The default config requires `device: nvidia`, uses the canonical CRTK tagged split, and trains with `training.loss: veto_select`.

The v2/A3 texture run uses:

```bash
python ideas/registry/i011_vetoselect_positive_claim_abstention/train.py --config ideas/registry/i011_vetoselect_positive_claim_abstention/config_v2.yaml
```

The self-mined decoy target is disabled for the configured warmup epochs. After warmup, negative examples with high detached raw evidence are partly assigned to the rejected-evidence action instead of being forced into ordinary non-puzzle.

When `training.veto_select.use_rule_texture: true`, the decoy target is additionally weighted by deterministic current-board tactical texture. This scalar is not a model input and does not use source labels, engine data, solution moves, or verification metadata.
