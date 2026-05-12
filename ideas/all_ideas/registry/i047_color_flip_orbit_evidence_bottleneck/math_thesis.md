# Math Thesis

Color-Flip Orbit Evidence Bottleneck

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0751_tuesday_los_angeles_color_flip_orbit.md`.

## Working Thesis

`CFOEB` (Color-Flip Orbit Evidence Bottleneck) hypothesizes that a
chess position is puzzle-like only if its tactical evidence survives
the exact color-flip/rank-reflection symmetry of chess. Concretely,
let `tau` be the deterministic two-element color-flip orbit on the
`simple_18` board planes (rank mirror + white/black piece-plane swap +
side-to-move complement + `KQkq -> kqKQ` castling swap + en-passant
rank mirror). For a sample `x in X_18`, two views `v_0 = x` and
`v_1 = tau(x)` are passed through a shared compact convolutional
encoder `phi_theta` and a shared evidence head `a_theta`, producing
nonnegative per-view per-class evidence
`E_jc(x) = softplus(a_theta(phi_theta(v_j))_c)` for `j in {0,1}` and
`c in {0,1}`. The classifier intersects the two views with the
harmonic mean
`I_c(x) = 2 E_0c(x) E_1c(x) / (E_0c(x) + E_1c(x) + epsilon)`,
emits two-class scores `s_c(x) = log(1 + I_c(x))`, and reduces them
to a single binary logit `s_1(x) - s_0(x)` for the puzzle-binary
trainer.

## Propositions

**Proposition 1 (exact orbit invariance).** For every `x` with a
valid `simple_18` semantic adapter, `s_theta(tau(x)) = s_theta(x)`.
The two views of `tau(x)` are `{tau(x), tau(tau(x))} = {tau(x), x}`,
the harmonic intersection `I_c` is symmetric in its two arguments,
and `log(1+.)` preserves equality.

**Proposition 2 (risk projection under ideal label invariance).**
If the binary label satisfies `Y(x) = Y(tau(x))` and the evaluation
distribution is symmetrized over the orbit, the orbit-averaged
classifier `p_G = 0.5 (p(.|x) + p(.|tau(x)))` has cross-entropy risk
no larger than the average orbit risk of `p`, by convexity of
`-log`.

## Counterexamples Where the Idea Should Fail

- A dataset source labels white-to-move and black-to-move positions
  differently for non-chess reasons.
- The `simple_18` channel-order metadata is wrong, so `tau` corrupts
  positions.
- Castling/en-passant channels are encoded in a way the adapter does
  not understand.
- Puzzle-likeness is dominated by material/phase priors that survive
  `tau` and are already learned by ordinary CNN baselines.
- Fine label `1` ambiguity is not orbit-related.

The implementation enforces a fail-closed `simple_18` semantic
adapter and ships unit tests for `tau(tau(x)) == x`, side-to-move
toggle, castling swap, and en-passant rank mirror so that an adapter
mistake is caught before benchmarking.
