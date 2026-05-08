# Math Thesis

Tactical Controllability Gramian Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_2004_saturday_shanghai_tactical_controllability_gramian.md`.

## Working thesis

A chess position is a small linear control system

```text
h_{t+1} = A h_t + B_a u_a + B_d u_d
y_t     = C h_t
```

with attackers as inputs (`B_a`), defenders as inputs (`B_d`), and
critical targets as outputs (`C`). The puzzle signal lives in the
controllability and observability Gramians: a *true* puzzle has
attacker-controllable target modes that the defender's controllability
subspace does not span, while a *near*-puzzle looks locally similar
but is cancelled by global defender propagation.

## Operator construction

The model encodes the position as `X ∈ R^{64 x d}` (compact CNN trunk
over simple_18 planes) and assembles

```text
A(X) = sum_g gate_g(X) * mask_g + U(X) V(X)^T
```

over five chess-geometry masks `g ∈ {ray, knight, pawn, king, defense}`
plus a low-rank board-conditioned update `U V^T`. The operator is
spectrally normalised via a small power-iteration estimate of
`||A||_2`:

```text
A_hat = A / max(1, sigma_hat(A))
```

so the unrolled controllability/observability sums

```text
W_a = sum_{k=0..K} A_hat^k B_a B_a^T (A_hat^T)^k
W_d = sum_{k=0..K} A_hat^k B_d B_d^T (A_hat^T)^k
W_o = sum_{k=0..K} (A_hat^T)^k C^T C A_hat^k
```

remain bounded for the recommended K = `gramian_steps`. The packet's
own solver guidance authorises this finite-unroll form for v1.

## Tactical readouts

For each board the model reads

```text
T_a   = trace(C W_a C^T)        # attacker target reach
T_d   = trace(C W_d C^T)        # defender target reach
T_net = T_a - T_d
H_a   = top singular values of W_o^{1/2} W_a W_o^{1/2}   # attacker Hankel modes
H_d   = top singular values of W_o^{1/2} W_d W_o^{1/2}   # defender cancellation modes
phi   = principal angles(span(W_a) leading, span(W_d) leading)
diag_a = diag(C W_a C^T),  diag_d = diag(C W_d C^T)     # per-target Gramian energies
```

`W_o^{1/2}` is implemented through a symmetric eigendecomposition of
`W_o + ε I`, clamped non-negative. The puzzle logit is a `LayerNorm +
MLP` over the concatenation of these scalars with a pooled board
context, the operator gate weights, the spectral-norm proxy and the
low-rank energy.

## Falsification

The packet's required ablations all live in the bespoke model and run
end-to-end on the same head shape: `attacker_only`, `defender_only`,
`no_observability`, `one_step_gramian`, `random_target_C`,
`random_geometry_A`, `fixed_A_no_gates`, `diag_only_gramian`, and the
trainer-side `cnn_same_params` baseline.
