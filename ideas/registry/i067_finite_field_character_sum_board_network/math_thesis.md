# Math Thesis

Finite-Field Character-Sum Board Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2115_friday_shanghai_character_sums.md`.

## Working thesis

Finite-field harmonic analysis tells us that any sufficiently rich
function on `F_p` can be decomposed into a sum of additive and
multiplicative characters. We hypothesize that puzzle-relevant board
structure carries arithmetic-residue signal that generic CNN pooling
does not surface. Concretely, every occupied square is encoded as a
tuple `(piece, rank, file, side, color)` of integer codes, and the
network evaluates a small, fixed family of polynomial probes
`f_q(piece, rank, file, side, color) mod p` over multiple primes
`p > 8`. Each probe yields a residue stream over the piece tokens.

Two character families are read off the residue streams:

- the additive characters `chi(x) = exp(2 pi i x / (p - 1))`, projected
  onto `(cos, sin)` so the model sees Gauss-style additive-sum phases;
- the multiplicative Legendre character `(x | p)`, providing
  Jacobi-style quadratic-residue evidence.

A residue histogram and entropy give a non-negative-frequency view, and
a coarse material summary keeps a baseline material signal in the
feature vector. A trainable `LayerNorm + MLP` readout then maps the
deterministic character-sum features to one puzzle logit and a fixed
set of diagnostic statistics. The mechanism is materially distinct from
a generic CNN: the polynomial coefficients, prime moduli, Legendre
tables, and additive-character phases are fixed up front, and the
network's degrees of freedom are confined to the readout over those
fixed harmonic features. Ablations replace pieces of this pipeline
(`residue_only`, `material_polynomial_only`, `random_residue_remap`,
`phase_batch_shuffle`, `single_prime`, `real_polynomial_mlp`) so the
character-sum hypothesis can be falsified against simpler baselines.
