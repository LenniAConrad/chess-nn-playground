# Math Thesis

Multi-Stream Chess-Decomposed Transformer Evaluator (i241).

Parent results: i193 (`exchange_then_king_dual_stream`, 0.8755 test PR AUC at 157k params, simple_18 encoding).

Working thesis: i193's exchange + king dual-stream decomposition does real work at small scale. Lifting it to attention-based streams over the 64-square token grid and adding a third **positional / structural** stream with relative rank/file attention bias retains the chess prior while letting each stream model long-range dependencies inside its task.

## Operator description

Let `x in R^{B, 18, 8, 8}` be the simple_18 board. The reused i193 feature builder produces

```
phi_E(x) in R^{B, 8, 8, 8}    (exchange planes: piece, value, attacker pressure, defender pressure)
phi_K(x) in R^{B, 8, 8, 8}    (king planes:    own/enemy zone, check, escape, line-to-zone)
A_E(x)   in R^{B, 64, 64}     (attacker/defender attention bias from precomputed attack tables)
A_K(x)   in R^{B, 64, 64}     (king-zone attention bias)
```

Three stream projections produce per-square token embeddings

```
tok_E(x) = Proj_E([x; phi_E(x)])     -> flatten to (B, 64, d)  + pos
tok_K(x) = Proj_K([x; phi_K(x)])     -> flatten to (B, 64, d)  + pos
tok_P(x) = Proj_P(x)                  -> flatten to (B, 64, d)  + pos
```

Each token grid is processed by a stack of pre-LN transformer blocks. Attention scores are

```
scores_E = (Q K^T) / sqrt(d_h) + A_E(x)
scores_K = (Q K^T) / sqrt(d_h) + A_K(x)
scores_P = (Q K^T) / sqrt(d_h) + B_pos(drank, dfile)
```

where `B_pos(drank, dfile) = R_rank(drank) + R_file(dfile)` and `R_rank, R_file` are learnable per-head bias tables indexed by the relative rank and file offsets. This is the standard relative-position attention bias adapted to the 8x8 board.

## Pooled features and fusion

Each stream emits a mean pool `e_*(x) in R^{B, d}`. The phase router is

```
alpha(x) = softmax( MLP_router( [e_E; e_K; e_P] ) ) in Delta^2
```

The puzzle logit is

```
y_hat(x) =   alpha_E h_E(e_E) + alpha_K h_K(e_K) + alpha_P h_P(e_P)
           + h_R( [e_E; e_K; e_P] )
```

with `h_E, h_K, h_P, h_R` linear heads. Auxiliary per-stream diagnostic logits `aux_E(e_E)`, `aux_K(e_K)`, `aux_P(e_P)` live alongside the main logits and can be supervised with stream-specific aux losses at training time (weight `aux_loss_weight`, default 0.05); the puzzle_binary trainer only consumes `y_hat`.

## Decision rule

`y_hat(x)` is the BCE-with-logits puzzle logit. Softmax of the route weights and the per-stream logits expose what each stream is contributing.

## Falsification path

The central falsifier is `no_chess_bias`, which strips all three attention biases and tests whether the chess-aware bias matters beyond the stream-specific input planes. `remove_exchange_stream`, `remove_king_stream`, `remove_positional_stream` zero one stream at a time to isolate per-stream contribution. `no_phase_router` replaces the learned mixture with a uniform 1/3/1/3/1/3 mixture. `no_aux_heads` zeros the auxiliary diagnostic logits and tests whether they carry signal beyond the main heads.
