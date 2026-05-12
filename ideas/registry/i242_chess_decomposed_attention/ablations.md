# Ablations

- Remove the global attention stream and keep exchange/king streams only.
- Remove exchange/king attention biases and keep vanilla attention in all streams.
- Replace the learned phase router with uniform stream averaging.
- Drop the residual joint head and use only routed stream logits.
- Compare against i193 under the same split, seed, epoch budget, and slice report.
