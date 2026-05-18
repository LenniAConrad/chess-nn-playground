# Math Thesis

HalfKA Dual-Stream LC0 Evaluator (i243).

Parent results: i193 (`exchange_then_king_dual_stream`, 0.8755 test PR AUC at 157k params, simple_18 encoding), i242 (`chess_decomposed_attention`, attention-only baseline).

Working thesis: HalfKA's learnable king-conditional accumulator, i193's exchange/king dual-stream decomposition, and LC0's value+policy heads each independently power a state-of-the-art chess system. The simplest three-way composition keeps the engineering wins (incremental update, tactical decomposition, MCTS-ready heads) in a single trunk.

## Operator description

Let `x in R^{B, 18, 8, 8}` be the simple_18 board. Two learnable embedding tables `E_W, E_B in R^{64 x 64 x 6 x d}` hold side-specific HalfKA embeddings indexed by `(king_sq, piece_sq, piece_type)`. The HalfKA accumulator at square `s` is

```
accum_W,s(x) = sum_{p, t} 1[piece_{W, p, t}(s)] * E_W[k_W(x), s, t]
accum_B,s(x) = sum_{p, t} 1[piece_{B, p, t}(s)] * E_B[k_B(x), s, t]
```

where `k_W(x), k_B(x)` are the white/black king squares (extracted as argmax over the king planes), and `piece_{C, p, t}(s)` is the occupancy of color `C` piece type `t` at square `s`. When one piece moves on the board, only two features in `F_active` change, so the accumulator updates by exactly one subtraction and one addition.

The per-square token grid is

```
T(x)_s = [accum_W,s(x); accum_B,s(x)] in R^{2d}
```

reshaped to `(B, 2d, 8, 8)`.

## Dual-stream backbone (i193 reused)

Two per-square reconstruction MLPs concatenate `T(x)` with stream-specific i193 deterministic planes:

```
ex_token(x) = MLP_E([T(x); x; phi_E(x)])
kg_token(x) = MLP_K([T(x); x; phi_K(x)])
```

where `phi_E, phi_K` are the deterministic exchange / king feature stacks from i193. Each token grid is processed by an i193 `StreamEncoder` (a small 3x3 conv stack) and pooled to a stream embedding `e_E(x), e_K(x) in R^{2C}`.

The phase router is

```
alpha(x) = sigmoid( router( [e_E; e_K] ) ) in (0, 1)
```

and the puzzle logit is

```
y_hat(x) = alpha(x) * h_K(e_K) + (1 - alpha(x)) * h_E(e_E) + h_R([e_E; e_K])
```

with linear heads `h_E, h_K` and a small MLP `h_R`.

## LC0 output heads

On the fused pool `j = [e_E; e_K]` the LC0-style heads produce

```
v_hat(x) = softmax( MLP_v(j) ) in Delta^2     (W, D, L)
pi_hat(x) = softmax( MLP_pi(j) ) in Delta^{P - 1}
```

with `P = 32` in the compact variant; the engine-grade scaled variant would use `P = 1858` and run the policy through legal-move masking. The puzzle_binary trainer ignores the value and policy heads; they are exposed as diagnostics only.

## Decision rule

`y_hat(x)` is the BCE-with-logits puzzle logit. The sigmoid gate `alpha` and the per-stream logits expose what each stream contributes; the WDL value softmax and policy logits enable an engine wrapper to reuse the same trunk for MCTS evaluation.

## Falsification path

The central falsifier is `no_halfka`, which drops the HalfKA accumulator entirely and reduces the model to a deterministic i193-style dual-stream conv. If `no_halfka` matches the full model, the HalfKA learnable king-conditioning is not what is helping at scout scale. `no_dual_stream` replaces the two per-stream encoders with a single shared encoder, testing whether the tactical decomposition buys anything over a single conv tower on top of HalfKA. `no_residual` zeros the residual head. `puzzle_only` zeros the LC0 value+policy logits; the puzzle_binary trainer's deltas should be zero under this ablation (confirming the heads are diagnostics only).
