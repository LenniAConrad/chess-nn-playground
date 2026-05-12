# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/stripe_selective_mixer_cnn.py` (`StripeSelectiveMixerCNN`, `StripeSelectiveMixerBlock`, `DirectionalStripeConv`).
- Idea-local wrapper: `ideas/all_ideas/registry/i173_stripe_selective_mixer_cnn/model.py` exposes `build_model_from_config(config)` that calls `build_stripe_selective_mixer_cnn_from_config`.
- Registry key: `stripe_selective_mixer_cnn`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.
- Batch candidate: `Stripe-Selective Mixer CNN` (rank 4).
- Stripe scans are implemented as `Conv2d` layers whose kernels are masked to a single chess line direction. The mask is a non-trainable buffer; only the four directional kernels and the local `Conv3x3` plus the gate MLP and `Conv1x1` fuse carry trainable weights.
- Default `stripe_kernel = 5`. With `depth = 2` the network reaches across the full 8x8 board along every chess line.
- Board-only by construction: engine, verification, source, and CRTK metadata are never used as model input.
