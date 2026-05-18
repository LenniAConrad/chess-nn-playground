# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = near_puzzle_hard_negative`
- `model.num_candidates`, `model.num_replies`, `model.token_dim`,
  `model.head_hidden_dim`, `model.head_dropout`,
  `model.reply_temperature`, `model.candidate_temperature`,
  `model.safe_threshold`, `model.king_zone_radius`,
  `model.gate_init`, `model.ablation` for the NPHN head.
- Trunk hyperparameters retain their i193 names with a `trunk_` prefix.
- `training.batch_size` stays at the i193 default 256; the
  candidate/reply pools and bilinear head do not allocate large
  intermediate tensors.

## Loss

`bce_with_logits` on the puzzle logit.

The source primitive describes an *optional* sampler-level pairwise
correction term

    L = L_BCE + lambda_pair [gamma - (logit(p) - logit(n))]_+
              + lambda_veto [gamma_v - (veto(n) - veto(p))]_+

mined from near-puzzle hard negatives. That belongs in the trainer,
not in the primitive; the scaffold here uses the canonical
`bce_with_logits` loss so it slots into the shared trainer unchanged.

## Cost expectation

At defaults (num_candidates=24, num_replies=24, token_dim=32,
head_hidden_dim=64, B=256) the per-step overhead of the NPHN head is
small compared to the trunk: two attention pools over 64 squares,
one bilinear neutralization of size `(num_candidates *
num_replies, token_dim)`, and a few small MLPs over the 11-d
diagnostic vector. Throughput should be within +10% of the i193
baseline.

## Ablation runs

Primary falsifiers:

```yaml
model:
  ablation: no_replies            # zero ReplyMass / Avail / RCI
```

```yaml
model:
  ablation: no_legality_discount  # collapse Disc(m*) to zero
```

Additional ablations:

- `model.ablation: concentration_only`  -- keep Conc/Gap12 only
- `model.ablation: shuffle_replies`     -- in-batch permutation of replies
- `model.ablation: no_overload`         -- drop DOA from z
- `model.ablation: no_king_escape`      -- drop KEP from z
- `model.ablation: zero_delta`          -- i193 baseline
- `model.ablation: trunk_only`          -- strongest control
- `model.ablation: disable_gate`        -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns `nphn_veto_pressure`, `nphn_forcedness_gap`,
and `nphn_reply_availability` should differentiate true puzzles from
near-puzzle false positives on the validation slice; otherwise the
rejection signal is not load-bearing.
