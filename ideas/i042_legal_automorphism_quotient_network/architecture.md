# Architecture

`Legal Automorphism Quotient Network` is a board-only `puzzle_binary`
classifier whose central operator is the deterministic Reynolds quotient
of the four-element legal chess-rule automorphism group
``G = <m, q> ~ C2 x C2``. The implementation replaces the shared
research-packet probe with a materially distinct bespoke model so the
markdown thesis is exercised by trainable code rather than a generic
mechanism profile.

## Forward Pipeline

1. **Legal automorphism transform.** The simple_18 board tensor
   `(B, 18, 8, 8)` is validated and three deterministic counterfactual
   tensors are constructed: the file mirror `m(x)` (flip files a<->h plus
   swap of king-side and queen-side castling per color and en-passant
   file mirror), the color flip `q(x)` (rank reflection, swap of white
   and black piece planes, side-to-move toggle, swap of White<->Black
   castling preserving king/queen side, en-passant rank flip) and the
   composition `m(q(x))`. The four views are stacked along a new
   dimension into the orbit ``X_orbit`` of shape `(B, 4, 18, 8, 8)`.
   Unsupported encodings fail closed; no move generation, mate flag,
   engine input, CRTK source label, or verification metadata is
   consulted.
2. **Shared residual board encoder.** The orbit is flattened to
   `(4B, 18, 8, 8)` and passed through a single shared CNN tower:
   `Conv(18 -> width) -> norm/GELU` stem, ``num_blocks`` residual blocks
   at ``width`` (each `Conv3x3 -> norm -> GELU -> Conv3x3 -> norm ->
   residual + GELU`), global average pool, and a `Linear -> LayerNorm
   -> GELU` projection to ``latent_dim``. Output shape:
   `(4B, latent_dim)`. Optional `orbit_chunk_size` preserves the
   `[e, m, q, mq]` orbit pairing under low-memory chunking.
3. **Reynolds character projection.** The encoder output is reshaped
   back to `(B, 4, latent_dim)`. The Reynolds invariant latent is
   ``z_inv = (1/|G|) sum_g phi(g . s)`` (the trivial C2 x C2 character)
   and the three nontrivial character components
   ``z_chars[chi] = (1/|G|) sum_g chi(g) phi(g . s)`` are exposed for
   diagnostics and for the optional ``R_char`` regularizer. The fixed
   character table `[[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]] / 4`
   uses orbit order `[e, m, q, mq]`.
4. **Classification head.** Only ``z_inv`` is fed to a two-layer MLP
   (`latent_dim -> head_hidden_dim -> num_classes`) with GELU
   activation and dropout. The head returns the puzzle logit for the
   `puzzle_binary` BCE-with-logits trainer when `num_classes=1`.

## Output Contract

Forward returns a `dict` whose `"logits"` entry has shape `(B,)` for the
puzzle_binary BCE-with-logits trainer when `num_classes=1` (or `(B, 2)`
for cross-entropy when `num_classes=2`). Diagnostic tensors include
`z_invariant`, `invariant_norm`, `character_energy`, `character_norms`
(the `[B, 3]` matrix of nontrivial-character L2 norms in orbit order
`[m, q, mq]`), the per-character convenience scalars
`file_mirror_character_norm`, `color_flip_character_norm` and
`joint_character_norm`, the `orbit_variance` of the per-view latents
around `z_inv`, and the optional `character_penalty` ready to be added
to the supervised loss as the `R_char` energy.

## Implementation Binding

- Registered model name: `legal_automorphism_quotient_network`
- Source implementation file: `src/chess_nn_playground/models/legal_automorphism_quotient_network.py`
- Idea-local wrapper: `ideas/i042_legal_automorphism_quotient_network/model.py`
