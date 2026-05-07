# Architecture

`Forced-Target Funnel Network` is a board-only `puzzle_binary` classifier. The implementation
replaces the shared research-packet probe with a materially distinct
bespoke model so the markdown thesis is exercised by trainable code, not
by a generic scaffold.

## Mechanism

The model accepts the repository `simple_18` board tensor (`B x 18 x 8 x 8`).
A compact convolutional square-encoder produces shared trunk features, and
an idea-specific head implements the architecture's distinguishing
mechanism described in the math thesis. Diagnostic tensors are exposed
alongside the puzzle logit so the trainer can record per-batch evidence
signals defined in `math_thesis.md`.

## Output Contract

Forward returns a `dict` whose `"logits"` entry has shape `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. Idea-specific
diagnostic tensors are always finite and align with the registered
builder.

## Implementation Binding

- Registered model name: `forced_target_funnel_network`
- Source implementation file: `src/chess_nn_playground/models/forced_target_funnel.py`
- Idea-local wrapper: `ideas/i213_forced_target_funnel_network/model.py`
