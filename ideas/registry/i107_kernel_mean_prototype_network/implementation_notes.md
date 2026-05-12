# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/kernel_mean_prototype_network.py`.
- Idea-local wrapper: `ideas/registry/i107_kernel_mean_prototype_network/model.py`.
- Registry key: `kernel_mean_prototype_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.
- Batch candidate: `Kernel Mean Prototype Network`.
- The model is intentionally board-only and does not consume engine, verification,
  source, or CRTK metadata as input.
- The kernel mean is the only set-level summary the classifier sees: pieces
  interact only through their contribution to ``mu(x)``, and any two boards
  with the same empirical kernel mean produce the same forward pass.
- Prototypes share the kernel feature dimension and have one trainable
  bandwidth each (parameterised as ``log_gamma`` to stay positive).
