# Trainer Notes

## Shared Trainer Usage

Use:

```text
mode: puzzle_binary
num_classes: 1
loss: BCEWithLogitsLoss
```

## Special Losses

First version:

```text
loss = puzzle_bce
     + legal_weight * legal_bce
     + next_latent_weight * latent_mse
     + reconstruct_weight * next_board_ce
```

Start with:

```text
legal_weight: 0.1
next_latent_weight: 0.05
reconstruct_weight: 0.02
```

## Logging Behavior

Log:

- legal auxiliary accuracy
- next latent MSE
- transition variance
- legal entropy
- near-puzzle false-positive rate

## Metrics

Primary:

- test PR AUC
- test F1
- near-puzzle false-positive rate

Secondary:

- auxiliary accuracy
- runtime per epoch
- puzzle recall

