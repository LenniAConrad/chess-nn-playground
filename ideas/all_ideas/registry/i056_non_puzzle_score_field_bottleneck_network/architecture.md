# Architecture

`Non-Puzzle Score-Field Bottleneck Network` is a bespoke PyTorch architecture
that approximates the smoothed input score of the verified non-puzzle board
distribution at multiple Gaussian noise scales, then funnels the resulting
denoising-residual score stack through a low-dimensional convolutional
bottleneck and fuses it with a compact board encoder before binary puzzle
classification.

## Implementation Binding

- Registered model name: `non_puzzle_score_field_bottleneck_network`
- Source implementation: `src/chess_nn_playground/models/non_puzzle_score_field_bottleneck.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i056_non_puzzle_score_field_bottleneck_network/model.py`

## Modules

- `NoiseLevelEmbedding` lifts `log(sigma)` to a 16-dim feature broadcast to
  `(B,16,8,8)` so the denoiser is conditioned on the active noise scale.
- `OrdinaryScoreDenoiser` is a noise-conditional residual CNN that maps a
  `(B,18,8,8)` board concatenated with the broadcast sigma map to a
  reconstruction `(B,18,8,8)` through `score_prior_blocks` GroupNorm/SiLU
  residual blocks of width `score_prior_hidden`. The denoising-residual
  identity `s_sigma(x) = (D_theta(x, sigma) - x) / sigma^2` from the markdown
  thesis is exposed by `_compute_score_stack`.
- `ScoreFieldBottleneck` projects the `K*18` channel score stack to
  `score_bottleneck_channels` via a Conv1x1 + GroupNorm + SiLU pre-projection
  followed by a depthwise/pointwise Conv3x3+Conv1x1 mixer.
- `_BoardStem` is a compact convolutional encoder over the 18 simple_18
  planes whose output is concatenated with the score bottleneck.
- `_FusionResidualBlock` blocks act on the concatenated `(board ⊕ score)`
  feature map.
- `NonPuzzleScoreFieldBottleneckNetwork` orchestrates the forward pass and
  pools concatenated avg + max features into a small MLP head that returns
  one BCE puzzle logit (`num_classes=1`).

## Forward Contract

Input:

```text
x: (B, 18, 8, 8)  # simple_18 board tensor
```

Per-noise score evaluation (in eval mode, or training when frozen, the
denoiser pass is wrapped in `torch.no_grad()`):

```text
recon_k     = OrdinaryScoreDenoiser(x, sigma_k)            # (B,18,8,8)
score_k     = (recon_k - x) / sigma_k**2                   # (B,18,8,8)
score_stack = concat_k score_k                             # (B,K*18,8,8)
z_score     = ScoreFieldBottleneck(score_stack)            # (B,bottleneck,8,8)
z_board     = BoardStem(x)                                 # (B,channels,8,8)
z           = FusionBlocks(concat[z_board, z_score])
pooled      = concat[avgpool(z), maxpool(z)]
logits      = Head(pooled)                                 # (B,) for num_classes=1
```

The repo's puzzle-binary trainer uses `num_classes: 1` with BCE-with-logits,
so `output["logits"]` has shape `(B,)`. A diagnostic `(B, 2)`
`two_class_logits` tensor is also returned for reporting compatibility.

## Diagnostics

`output` exposes:

- `score_field_norm`, `score_residual_energy`, `score_field_mean_abs`,
  `score_field_max_abs` — global magnitude of the non-puzzle repair field
- `score_per_sigma_norm`, `recon_residual_l2` — per-sigma diagnostics
  splitting evidence across noise scales
- `score_bottleneck_energy` — energy of the bottlenecked feature map
- `mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count`
  — compatibility aliases for the shared reporting pipeline

## Two-Stage Training

The model exposes `denoising_score_matching_loss(clean_board, binary_label)`
that implements the markdown's

```text
L_DSM = E_{x:y=0, sigma, eps} ||D_theta(x + sigma eps, sigma) - x||^2 / (2 sigma^2)
```

with `score_prior_train_on_binary_zero_only=True` filtering rows whose
binary label is `1`. `freeze_score_prior` / `unfreeze_score_prior` toggle the
denoiser parameters for the two-stage pretraining/freezing recipe. Outside
those helpers the model exposes the standard `forward(x) -> dict`
contract that the shared trainer uses.

## Ablations

The architecture supports the central falsifiers from the markdown by
config-only modifications:

- All-class score prior — drop the `binary_label` filter when calling
  `denoising_score_matching_loss`, keeping every other hyperparameter fixed.
- Frozen random denoiser — leave the denoiser at initialization (skip
  pretraining) so the score field becomes a random-feature control.
- Score-only / no-score branches can be reproduced by setting
  `score_bottleneck_channels=0` (no-score) or by zeroing the board stem
  output via a config-only patch in a future ablation runner.
