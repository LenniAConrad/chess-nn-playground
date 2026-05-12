# Math Thesis

Auxiliary Reconstruction BoardNet.

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `3`.

The puzzle decision is modeled by a shared convolutional encoder
`f_theta : R^{C x 8 x 8} -> R^{D x 8 x 8}` with two heads on top of the
same latent map `h = f_theta(x)`:

\[
z(x) = w^\top \mathrm{pool}(h(x)) + b,
\quad
\hat r(x) = D_phi(h(x)) \in R^{R x 8 x 8},
\]

where `z` is the puzzle logit and `\hat r` are per-square reconstruction
logits over `R` selected current-board input planes
`x_S = x[:, S, :, :]` with `S \subset \{0, ..., C-1\}`. The default
choice of `S` is the full set of simple_18 input planes.

The combined ablation objective is

\[
L(\theta, \phi) = \mathrm{BCE}(z(x), y)
   + \lambda_{\text{recon}} \, \mathrm{BCE}(\hat r(x), x_S),
\]

with `lambda_recon = 0.05` by default. The first term is the standard
puzzle_binary BCE-with-logits loss; the second is a per-square BCE
between the decoder logits and the corresponding input planes. Because
`x_S` comes only from the current board, the auxiliary term does not
introduce any future or engine information into training.

The thesis is that the puzzle target is better predicted when the
encoder is regularised against discarding board detail. Reconstruction
gradients flow back through `D_phi` into `f_theta`, so the latent must
preserve enough geometry to reconstruct piece occupancy, side-to-move,
castling rights, and en-passant. Falsification proceeds by collapsing
either the auxiliary or the encoder:

- If the `classifier_only` ablation matches or beats the full model,
  the regulariser is unnecessary.
- If `decoder_no_loss` (decoder parameters present but `lambda_recon
  = 0`) matches the full model, the gain came from added capacity, not
  from reconstruction supervision.
- If reconstruction BCE does not decrease meaningfully under the full
  loss while puzzle accuracy improves, the model has found a shortcut
  that bypasses the auxiliary signal.

The model exposes `reconstruction_logits`, `reconstruction_probs`,
`reconstruction_error`, and per-plane reconstruction BCE so each of these
collapse modes is directly observable in metrics.
