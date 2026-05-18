# Ablations

The model exposes its central falsifiers through `model.ablation`. Run the main config plus the five ablation configs below.

- `none` (default): HalfKA accumulator on, dual-stream backbone on, residual head on, LC0 value+policy heads exposed.
- `no_halfka`: drop the HalfKA accumulator entirely. The backbone runs on simple_18 + deterministic geometry planes only. Central falsifier for the HalfKA-learnable-king-conditioning claim.
- `no_dual_stream`: replace the two per-stream encoders with a single shared encoder applied to the half-sum of the two stream inputs. Tests whether the i193 tactical decomposition buys anything over a single conv tower on top of HalfKA.
- `no_residual`: zero the residual head. Tests whether the residual pathway carries signal beyond the per-stream mixture.
- `puzzle_only`: zero the LC0 value and policy logits. Sanity-check that puzzle_binary deltas are zero under this ablation (the heads are diagnostics only).

Compare against i193 (`exchange_then_king_dual_stream`), Stockfish NNUE (`stockfish_nnue`), LC0 BT4 (`lc0_bt4`), and i242 (`chess_decomposed_attention`) on the same split and seeds to isolate the gain from each of (a) HalfKA learnable king-conditioning, (b) tactical decomposition on top of HalfKA, and (c) LC0-style heads.
