# Architecture

`Finite-Field Character-Sum Board Network` is a board-only `puzzle_binary`
classifier that exercises the math thesis: the network reads piece tokens
out of the `simple_18` tensor, evaluates fixed multi-prime polynomial
character sums over those tokens, and builds a deterministic
character/Legendre/residue feature bank that is fed to a small trainable
readout. The idea is to test whether finite-field harmonic features over
piece-coordinate tuples carry signal about puzzle-vs-non-puzzle status
beyond what generic CNN pooling extracts.

## Mechanism

1. `Simple18FiniteFieldEncoder` reads the 12 piece planes plus the
   side-to-move plane and produces up to `max_piece_tokens` occupied
   tokens. Each token carries an integer piece code, color/side codes,
   and rank/file integer coordinates so all downstream arithmetic stays
   in the integer domain expected by character sums.
2. `CharacterProbeTable` registers fixed (non-learned) coefficient
   matrices `coeff_p`, Legendre symbol tables `legendre_p`, and a
   residue remap `remap_p` for each configured prime `p > 8`. The
   coefficients implement the polynomial probes
   `f_q(piece, rank, file, side, color) mod p` used by the additive
   characters; the Legendre table implements the multiplicative
   character. Polynomial degree is configurable (degree 0/1/2 zero out
   the appropriate term blocks) so a degree control is available for
   ablation studies.
3. `FiniteFieldCharacterFeatures` evaluates the polynomial probes per
   prime, derives the residues mod `p`, and emits four families of
   deterministic features: scaled residues, additive-character
   `(cos, sin)` of `2 pi r / (p - 1)`, the Legendre symbol of `r`, and
   a residue histogram over the prime. A small material summary is
   appended so the head sees both character-sum signal and a coarse
   material baseline.
4. `CharacterSumHead` is a `LayerNorm + 2-layer MLP` readout over the
   concatenated feature vector. It returns one puzzle logit plus
   diagnostic statistics — character-sum norm, Legendre mean, residue
   entropy, zero-frequency, and polynomial-value mean — that the
   trainer can record for evidence inspection.

A set of ablations (`residue_only`, `material_polynomial_only`,
`random_residue_remap`, `phase_batch_shuffle`, `single_prime`,
`real_polynomial_mlp`) is supported by the bespoke builder so the
character-sum mechanism can be falsified against simpler baselines.

## Output Contract

Forward returns a `dict` whose `"logits"` entry has shape `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. Diagnostic tensors
(`character_sum_norm`, `legendre_mean`, `zero_frequency`,
`residue_entropy`, `polynomial_value_mean`, `character_feature_norm`,
`material_balance`, `piece_count`) are always finite and are appended to
prediction artifacts.

## Implementation Binding

- Registered model name: `finite_field_character_sum_board_network`
- Source implementation file: `src/chess_nn_playground/models/finite_field_character_sum.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i067_finite_field_character_sum_board_network/model.py`
