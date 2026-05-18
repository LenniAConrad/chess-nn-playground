# i243 - HalfKA Dual-Stream LC0 Evaluator

A three-way composition of architectures that each independently power a
state-of-the-art chess system:

| component | source | role |
|---|---|---|
| HalfKA accumulator | Stockfish NNUE | Rich learnable king-conditional input representation + O(1) incremental update at inference. |
| Exchange/king dual-stream conv | i193 (scout winner) | Replaces NNUE's plain MLP backbone with the dual-stream tactical decomposition that drives the scout's largest within-encoding margin. |
| WDL value + 32-policy heads | LC0 (BT4 family) | Makes the network usable in an MCTS search loop. The compact variant uses a 32-dim policy head; the scaled engine variant would use the LC0 1858-move space. |

## Implementation Binding

- Registered model name: `halfka_dual_stream_lc0`
- Source implementation file: `src/chess_nn_playground/models/trunk/halfka_dual_stream_lc0.py`
- Idea-local wrapper: `ideas/registry/i243_halfka_dual_stream_lc0/model.py`
- Training config: `ideas/registry/i243_halfka_dual_stream_lc0/config.yaml`

## Sketch

```
Position (simple_18 board tensor [B, 18, 8, 8])
   |
   v
HalfKA feature extraction (per side):
   active features f_i = (king_sq, piece_color, piece_type, piece_sq)
   white_accumulator = sum_i  white_embedding[white_king_sq, piece_sq_i, piece_type_i]
   black_accumulator = sum_i  black_embedding[black_king_sq, piece_sq_i, piece_type_i]
   per-square reshape -> token_grid [B, 2*embed_dim, 8, 8]

i193 dual-stream backbone (conv on the 8x8 token grid):
   +---------- exchange stream ----------+   +-------- king stream --------+
   | per-square reconstruction MLP        |   | per-square reconstruction MLP|
   | input: token_grid + simple_18 +      |   | input: token_grid + simple_18 +
   |        exchange planes               |   |        king planes           |
   | -> StreamEncoder (3x3 conv stack)    |   | -> StreamEncoder (3x3 conv) |
   +---------------+----------------------+   +-----------+------------------+
                   | mean+max pool                       | mean+max pool
                   v                                      v
            ex_pool [B, 2C]                        kg_pool [B, 2C]

Phase-router MLP (i193 design):
   alpha = sigmoid( router( concat(ex_pool, kg_pool) ) )
   fused = alpha * king_logit + (1 - alpha) * exchange_logit
         + residual_head( concat(ex_pool, kg_pool) )

LC0-style heads on fused pool:
   value_wdl_logits  = MLP(joint) -> 3 logits (W, D, L)
   policy_logits     = MLP(joint) -> 32 compact policy logits
                                     (scaled variant would use 1858)
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
\tau_s(x) \;=\; \mathrm{Proj}\!\Bigl(\bigl[\,\mathrm{accum}_{\mathrm{white}, s}(x) \;\Vert\; \mathrm{accum}_{\mathrm{black}, s}(x)\bigr] \;\Vert\; \mathrm{geom}_s(x)\Bigr)
$$

where $\mathrm{geom}_s(x)$ is i193's deterministic king-zone / check-ray / attacker-pressure planes at square $s$ concatenated with the raw simple_18 tensor.

**i193 fusion (re-used verbatim):**

$$
\hat{y}(x) \;=\; \alpha(x) \cdot h_K(\phi_K(x)) + (1-\alpha(x)) \cdot h_E(\phi_E(x)) + h_R(\phi_K \oplus \phi_E)
$$

**LC0 output heads:**

$$
\hat{v}(x) = \mathrm{softmax}\!\bigl(W_v\,\hat{e}(x)\bigr) \;\in\; \Delta^2 \quad (W, D, L), \qquad
\hat{\pi}(x) = \mathrm{softmax}\!\bigl(W_\pi\,\hat{e}(x)\bigr) \;\in\; \Delta^{P-1}
$$

with $P = 32$ in the compact variant.

## Compact sizing

The compact CPU-testable variant uses `embed_dim=16`, `backbone_channels=48`,
`backbone_depth=2`, `head_hidden=96`, `policy_dim=32`. The HalfKA tables
dominate the parameter count: `2 * (64 * 64 * 6 * 16) = 786432` parameters.
The backbone + heads add ~100k parameters.

The scaled engine variant would use `embed_dim=256, backbone_channels=128,
backbone_depth=6, policy_dim=1858` for ~25--40M parameters at BT4-medium
scale; that variant lives as design notes only.

## Ablation modes

`HalfKADualStreamLC0.ABLATIONS` enumerates the testable variants:

- `none` (default): HalfKA front-end on, dual-stream backbone on, residual
  head on, LC0 heads on.
- `no_halfka`: drop the HalfKA accumulator entirely. The backbone runs only
  on simple_18 + deterministic geometry planes; tests whether HalfKA buys
  anything over i193's hand-engineered king conditioning at scout scale.
- `no_dual_stream`: replace the two per-stream encoders with a single shared
  encoder applied to the half-sum of the two stream inputs. Tests whether
  the tactical decomposition buys anything over a single conv tower.
- `no_residual`: zero the residual head. Tests whether the residual pathway
  carries signal beyond the per-stream mixture.
- `puzzle_only`: zero the LC0 value and policy logits. Used as a sanity-check
  control that puzzle_binary signal does not need the engine heads.

## Diagnostics

`forward(x)` returns:

- `logits`: shape `(B,)`, BCE-compatible for the one-logit puzzle_binary head.
- `prob`: sigmoid of the puzzle logit.
- Per-stream main logits: `exchange_logit`, `king_logit`.
- Mixture weights: `alpha_exchange`, `alpha_king` (sigmoid gate).
- `residual_logit`.
- Per-stream pool norms: `exchange_pool_norm`, `king_pool_norm`.
- LC0 heads: `value_wdl_logits` of shape `(B, 3)`, `policy_logits` of shape
  `(B, policy_dim)`.
- HalfKA accumulator diagnostics: `white_accumulator_norm`,
  `black_accumulator_norm`, `accumulator_norm`, `white_king_sq`,
  `black_king_sq`.
- `mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count`.
- `halfka_dual_stream_ablation`: integer code identifying the active ablation.
- `halfka_embedding_dim`, `policy_logit_count`: scalar size diagnostics.

## Contract

- Input: `(B, 18, 8, 8)` board tensor only. CRTK / verification / source /
  engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit puzzle_binary
  BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine
  label `2` maps to binary target `1`.
- The HalfKA tables and the per-square reconstruction MLP are learnable.
  The deterministic geometry tables in the reused i193 feature builder are
  buffers and never optimised.
- Engine WDL/policy heads are exposed as diagnostics; the puzzle_binary
  trainer does not consume them. Their training requires an engine eval
  pipeline that is out of scope for this implementation.

## Sizing variants

| variant | embed_dim | total params | when to use |
|---|---:|---:|---|
| `tiny` | 8 | ~0.4M | scout-scale CPU sanity check |
| `compact` (default) | 16 | ~0.9M | puzzle_binary scout-scale benchmarking |
| `small` | 96 | ~10M | research-grade fine-tuning (out of scope here) |
| `medium` | 256 | ~38M | engine-grade, matches BT4-medium (out of scope here) |

## What it would take to train at engine grade

This proposal cannot be trained on the scout's puzzle_binary corpus alone
in the engine-grade variant --- the embedding table is wildly
overparameterised at 173k samples. Engine-grade training data and
pipeline are out of scope for this implementation:

1. Stockfish-eval distillation on a master-game corpus.
2. LC0 self-play data (publicly available v6/v7 batches).
3. Hybrid: pre-train HalfKA accumulator on (1), fine-tune full network on (2).
