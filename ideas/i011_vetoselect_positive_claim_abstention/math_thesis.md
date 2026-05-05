# Mathematical Thesis

VetoSelect separates two signals that a standard binary puzzle classifier merges into one logit:

- raw positive puzzle evidence;
- whether that evidence should be trusted as a positive claim.

For a board tensor `x`, the model emits raw evidence `z` and selector/trust logit `a`.

```text
pi_N = sigma(-z)
pi_R = sigma(z) * sigma(-a)
pi_P = sigma(z) * sigma(a)
```

The three masses sum to one. `pi_N` is ordinary non-puzzle, `pi_R` is rejected positive evidence, and `pi_P` is accepted puzzle. The selected puzzle score is:

```text
selective_puzzle_logit = log(pi_P) - log(pi_N + pi_R)
```

The hypothesis is that near-puzzle negatives often deserve high `z` but low `a`: the board looks tactical, yet the model should veto the positive claim. True puzzles should keep both `z` and `a` high.
