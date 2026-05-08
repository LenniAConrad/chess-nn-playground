# Math Thesis

Tactical Bisimulation Puzzle Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0113_saturday_shanghai_tactical_bisimulation.md`.

Working thesis: a chess representation `z = E(x)` is useful for puzzle
binary classification only if the bisimulation distance
`d(z_i, z_j)` is small precisely when the two positions have similar
tactical legal continuations. Two positions can share piece layout,
material, king pressure, or surface texture and still have very
different one-step legal-consequence behaviour; near-puzzles look
puzzle-like but admit defensive escapes that real puzzles do not. The
network therefore learns three things together:

1. direct puzzle evidence `g(z)`,
2. a learned successor signature
   `mu_x = sum_a pi(a | x) * delta_{T(E(x), a)}`
   built from a board-only policy `pi(a | x)` and a learned latent
   transition `T(z, a)`,
3. a metric `d(z, p)` against a learnable prototype bank that is
   shaped by a Bellman-style bisimulation residual
   `|| z - sum_a pi(a) * T(z, a) ||`.

The puzzle logit is a function of the base evidence, the prototype
distances, the successor-signature stats, and the bisimulation
residual, so the classifier cannot collapse onto raw board texture.
