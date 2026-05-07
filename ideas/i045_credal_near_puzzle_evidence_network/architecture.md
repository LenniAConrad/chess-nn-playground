# Architecture

`Credal Near-Puzzle Evidence Network` (idea i045) is a board-only
`puzzle_binary` classifier whose central operator is a binary Dirichlet
evidential head on top of a compact residual board encoder, exactly as
specified by `math_thesis.md`. The shared `ResearchPacketProbe`
mechanism profile is no longer used here; the implementation is
materially distinct.

## Forward Pipeline

1. **`FailClosedBoardAdapter`.** A `1x1` `Conv2d` projects the board
   tensor from `input_channels` to `hidden_channels = 64`. It refuses
   any unknown channel count unless the config sets
   `allow_unknown_channels=true`, and it cross-checks
   `model.encoding` against the known table
   (`simple_18 -> 18`, `lc0_static_112 -> 112`, `lc0_bt4_112 -> 112`).
2. **`TinyResidualBoardEncoder`.** One `3x3` Conv stem followed by
   `num_res_blocks = 4` residual blocks (each two `3x3` Conv layers at
   width `64`, BatchNorm + ReLU). Output shape `(B, 64, 8, 8)`.
3. **Pool.** Global average pool over the `8x8` board to `(B, 64)`.
4. **MLP.** `Linear(64, hidden_dim=128)` -> ReLU.
5. **`DirichletEvidenceHead`.** Optional dropout, then `Linear(128, 2)`
   gives `raw_evidence`. The Dirichlet parameters are
   `evidence = softplus(raw_evidence)` and
   `alpha = evidence_floor + evidence` with `evidence_floor = 1.0`
   (matching the math thesis `alpha = 1 + e_theta(x)`).
6. **Logit head.** With `num_classes = 1` (the puzzle-binary BCE
   contract used by this idea) the model emits the binary logit
   `puzzle_logit = log(alpha_1 + eps) - log(alpha_0 + eps)` whose
   sigmoid equals the Dirichlet predictive mean
   `mu_pos = alpha_1 / S`. With `num_classes = 2` it emits
   `logits = log(alpha + eps)` so that `softmax(logits)` equals the
   Dirichlet mean.

## Output Contract

`forward(x)` returns a `dict` keyed by:

- `logits`: shape `(B,)` for `num_classes = 1` (single-logit BCE) or
  `(B, 2)` for `num_classes = 2`.
- `alpha`, `alpha_pos`, `alpha_neg`: Dirichlet parameters
  `(B, 2)` and the two scalar columns.
- `evidence`, `evidence_pos`, `evidence_neg`: nonnegative evidence and
  its scalar columns.
- `evidence_mass` (`S = alpha_0 + alpha_1`).
- `mu_pos = alpha_1 / S`.
- `uncertainty = 2 / S`.

These auxiliary tensors are exactly the columns the markdown
`Section 7` lists for the trainer's prediction artifact (`mu_pos`,
`S`, `uncertainty`) and the columns the `CredalEvidenceLoss`
described in `math_thesis.md` reads when it computes
`L_set(z, m)` and `L_ev(z, S)` from fine labels.

## Loss-shaping hyperparameters

The model surface exposes the loss-shaping constants from the
math thesis (`near_tau`, `near_s_max`, `lambda_near_evidence_cap`,
`lambda_dirichlet_kl`, `kl_anneal_epochs`) as attributes so that an
idea-specific `CredalEvidenceLoss` outside `forward` can read them
without re-parsing the YAML. The forward pass itself does not consume
fine labels, which keeps the model trainable from the shared
puzzle-binary BCE-with-logits loss while the credal/evidence-cap
ablations live alongside the trainer.

## Implementation Binding

- Registered model name: `credal_near_puzzle_evidence_network`
- Source implementation file: `src/chess_nn_playground/models/credal_near_puzzle_evidence.py`
- Idea-local wrapper: `ideas/i045_credal_near_puzzle_evidence_network/model.py`
