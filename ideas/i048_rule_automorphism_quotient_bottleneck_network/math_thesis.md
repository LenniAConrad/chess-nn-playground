# Math Thesis

Rule-Automorphism Quotient Bottleneck Network (`RAQ-Net`).

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0751_tuesday_pdt_automorphism_quotient.md`.

## Working Thesis

A chess puzzle-like position should remain puzzle-like under the exact
color/turn reversal symmetry of chess, and usually under the file mirror
when castling rights are absent, so training a classifier on the quotient
of these safe rule-automorphism orbits should suppress color, side, and
file-orientation shortcuts without using engines, move trees, attack
graphs, sheaves, or transport.

## Formal Setup

Let `B = (O, s, c, e)` denote the simple_18 board: piece occupancy `O`,
side-to-move `s`, four castling-right bits `c`, and en-passant square `e`.
Define the safe-automorphism transform set

  `G_x \subseteq {I, C, H, HC}`,    `{I, C} \subseteq G_x`

where `C` is the color/turn rank-mirror with white/black piece-plane swap,
side-to-move complement, `KQkq <-> kqKQ` castling swap and en-passant rank
flip; `H` is the file mirror, valid only when all castling-right bits are
zero; and `HC` is the composition, valid whenever `H` is. This is a
groupoid, not a global image-symmetry group, and that asymmetry is exactly
what `RAQ-Net` quotients over.

Write `phi_theta : X -> R^D` for the shared encoder and
`p_theta : R^D -> R^P` for the projection head. The masked Reynolds
quotient latent is

  `z_bar(x) = (1 / |G_x|) sum_{g in G_x} phi_theta(T_g x)`,

and the binary classifier is `f(x) = W z_bar(x) + b`. Auxiliary per-view
logits `f^g(x) = W phi_theta(T_g x) + b` feed the REx-style risk
variance, while `p_theta` feeds the orbit-consistency objective with
VICReg variance and covariance no-collapse terms.

## Core Hypothesis

For every safe `g \in G_x`,
`P(Y = 1 | X = x) = P(Y = 1 | X = T_g x)`.
A representation that factors through the orbit `x / G_x` therefore cannot
encode color, side-to-move, or castling-free file orientation shortcuts.

## Proposition

If `L_orbit_inv = 0` without latent collapse and `p_theta` is injective
on the classifier-relevant subspace, then `p_theta(phi_theta(x))` is
constant on every valid orbit, so the binary logit is exactly invariant
under each `T_g`. Any scalar shortcut `a(x)` with nonzero orbit variance
cannot be a deterministic function of the quotient projection.

## Falsification

The semantics-destroying pseudo-orbit replaces `{C, H, HC}` with the
same-count rank/file/rank-file flips that preserve view count but violate
color/side/castling consistency. This control matches encoder capacity,
parameter count, and loss shape while breaking rule semantics; if the
legal-orbit `RAQ-Net` does not beat it, the bottleneck is not exploiting
chess-rule structure.
