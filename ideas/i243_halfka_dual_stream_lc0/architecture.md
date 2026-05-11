# i243 — HalfKA Dual-Stream LC0 Evaluator

A three-way composition of architectures that each independently power a
state-of-the-art chess system:

| component | source | role |
|---|---|---|
| HalfKA accumulator | Stockfish NNUE | Rich learnable king-conditional input representation + O(1) incremental update at inference. |
| Exchange/king dual-stream conv | i193 (scout winner) | Replaces NNUE's plain MLP backbone with the dual-stream tactical decomposition that drives the scout's largest within-encoding margin. |
| WDL value + 1858-policy heads | LC0 (BT4 family) | Makes the network usable in an MCTS search loop. |

## Sketch

```
Position (FEN)
   │
   ▼
HalfKA feature extraction (per side):
   active features f_i = (king_sq, piece_color, piece_type, piece_sq)
   accumulator_white = Σ_i  E_white[f_i]      ∈ ℝ^{d_acc}
   accumulator_black = Σ_i  E_black[f_i]      ∈ ℝ^{d_acc}

Reshape to per-square tokens:
   For each square s, gather  accumulator entries whose piece_sq == s
   token_s = MLP_2( concat(accum_white_at_s, accum_black_at_s) ) ∈ ℝ^d
   Output: x_token ∈ ℝ^{[64, d]}

i193 dual-stream backbone (conv on the 8×8 token grid):
   ┌──── exchange stream ────┐   ┌──── king stream ────┐
   │ conv encoder w/ exchange │   │ conv encoder w/ king │
   │ planes input bias        │   │ planes input bias    │
   └────────────┬─────────────┘   └─────────┬────────────┘
                │ pool                       │ pool
                ▼                            ▼
         exchange_pool                 king_pool

Phase-router MLP (i193 design):
   alpha = sigmoid(MLP(exchange_pool ⊕ king_pool))
   fused = alpha · king_pool + (1 - alpha) · exchange_pool + residual_head(joint)

LC0-style heads on fused:
   value  = softmax(W_v fused)       ∈ Δ^2  (W, D, L)
   policy = softmax_legal(W_π fused) ∈ Δ^{1858}
```

## Key equations

**HalfKA accumulator** (incremental update enabled):

$$
a_{\mathrm{side}}(x) \;=\; \sum_{f \in \mathcal{F}_{\mathrm{active}}(x)} E_{\mathrm{side}}[f],
\qquad f = (k_{\mathrm{side}}, \mathrm{color}, \mathrm{type}, s)
$$

When one piece moves: $\mathcal{F}_{\mathrm{active}}$ changes by $\leq 2$ features, so $a_{\mathrm{side}}$ is updated by one subtraction and one addition.

**Per-square token reconstruction** (the bridge from flat accumulator to conv backbone):

$$
\tau_s(x) \;=\; \mathrm{MLP}\!\Bigl(\sum_{f \in \mathcal{F}_{\mathrm{active}}(x),\; \mathrm{piece\_sq}(f)=s} E_{\mathrm{side}(f)}[f] \;\;\Bigm|\Bigm|\;\; \mathrm{geom}_s(x)\Bigr)
$$

where $\mathrm{geom}_s(x)$ is i193's deterministic king-zone / check-ray / attacker-pressure planes at square $s$. The token concatenates HalfKA-learnable + i193-deterministic features.

**i193 fusion (re-used verbatim):**

$$
\hat{e}(x) \;=\; \alpha(x) \cdot h_K(\phi_K(x)) + (1-\alpha(x)) \cdot h_E(\phi_E(x)) + h_R(\phi_K \oplus \phi_E)
$$

**Output heads (LC0):**

$$
\hat{v}(x) = \mathrm{softmax}\!\bigl(W_v\,\hat{e}(x)\bigr), \qquad
\hat{\pi}(x \mid m) = \mathrm{softmax}\!\bigl(\mathrm{mask}_{\mathrm{legal}}(W_\pi\,\hat{e}(x))\bigr)
$$

## Sizing

At BT4-medium scale (~50M params total):

| component | size | params |
|---|---|---:|
| HalfKA white embedding | 49k × 256 | ~12.5M |
| HalfKA black embedding | 49k × 256 | ~12.5M |
| Per-square reconstruction MLP | 256→128→64 | ~50k |
| Dual-stream conv backbone (channels=128, depth=6) | per i193 sizing | ~10M |
| Phase router + residual head | per i193 sizing | ~100k |
| Value head + policy head | LC0 standard | ~3M |
| **Total** | | **~38M** |

This sits at the same scale as LC0 BT4-medium, with most of the parameter budget going to the HalfKA embedding table.

## Why this beats each component alone

| versus | the win |
|---|---|
| Stockfish NNUE (pure MLP backend) | Dual-stream conv decomposition extracts tactical specialisation. Stockfish's MLP cannot distinguish exchange-evaluation features from king-safety features structurally; i243 does. |
| i193 alone (deterministic king features) | Learnable HalfKA accumulator captures fine-grained king-conditional patterns (specific castled-king pawn-shield variants, etc.) that no fixed feature builder reaches. |
| LC0 BT4 alone (generic transformer) | Has neither HalfKA's incremental-update inference advantage nor i193's structural tactical decomposition. |

## Engineering advantages

**Incremental update at inference.** Move-by-move accumulator updates cost ~2 vector additions per move, independent of board complexity. The conv backbone only runs at full cost per evaluation, which fits both alpha-beta engines (millions of evals/s on CPU) and MCTS engines (thousands of evals/s on GPU).

**Composable training.** The HalfKA accumulator can be pre-trained on a different task (e.g. position reconstruction or Stockfish-eval distillation) and the dual-stream backbone fine-tuned later. This separates representation learning from task-specific architecture.

**Interpretable per-stream contributions.** Like i193, the phase-router weights $\alpha(x)$ and per-stream logits expose what each chess concept is contributing to the final eval. Stockfish NNUE's MLP backbone has no comparable inspectability.

## Sizing variants

| variant | embed_dim | total params | when to use |
|---|---:|---:|---|
| `tiny` | 32 | ~2.5M | scout-scale sanity check (puzzle_binary) |
| `small` | 96 | ~10M | research-grade fine-tuning |
| `medium` | 256 | ~38M | engine-grade, matches BT4-medium |
| `large` | 384 | ~75M | engine-grade, matches BT4-large |

## What it would take to train

This proposal cannot be trained on the scout's puzzle_binary corpus alone — the embedding table is wildly overparameterised at 173k samples. Realistic training data:

1. **Stockfish-eval distillation** on a master-game corpus (Stockfish run at fixed depth on tens of millions of positions). Same data source Stockfish NNUE uses.
2. **LC0 self-play data** (publicly available v6/v7 batches, billions of positions).
3. **Hybrid**: pre-train HalfKA accumulator on (1), fine-tune full network on (2).

Realistic compute: 1--2 GPU-weeks for a medium-scale pre-train + fine-tune cycle.

## Hypotheses

- **H1**: at engine training scale, i243 beats both Stockfish NNUE (matched size) and plain i193 (matched size) on Elo against a fixed-depth tournament opponent.
- **H2**: i243's incremental-update inference path gives a $\geq 5\times$ wall-clock speedup over i242 / BT4 / i193 in an engine inner loop, with no quality loss.
- **H3**: the phase-router weights $\alpha(x)$ show interpretable position-type dependence (high $\alpha_K$ on king-attack positions, high $\alpha_E$ on quiet exchange positions), confirming the decomposition is being used.

## Ablations (planned)

- A1 --- replace HalfKA with i193's deterministic king features (asks: does HalfKA buy anything over hand-engineered king conditioning at scale?)
- A2 --- replace dual-stream conv backbone with a plain MLP (asks: does the tactical decomposition buy anything over NNUE's MLP at scale?)
- A3 --- replace LC0 heads with a single puzzle-binary head (sanity-check on scout corpus; expected to underperform engine-trained i243).
- A4 --- disable incremental update in the engine wrapper to measure the wall-clock speedup it buys.

## Status

**Proposed only.** Implementation is non-trivial:
- HalfKA feature computation pipeline (board → active feature indices)
- Embedding table + accumulator code with incremental-update support
- Per-square token reconstruction layer
- Engine-grade training data sourcing (Stockfish or LC0 corpus)

The architecture is the cheap part. The training pipeline + data is the project.
