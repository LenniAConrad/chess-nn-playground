# Architecture

`Puzzle-Binary Benchmark Challengers` promotes the source packet's
**Negative-Class Disentangled Puzzle Head** (Idea 1 in the packet's
priority ranking). It is a board-only `puzzle_binary` classifier: a
compact CNN trunk over the `simple_18` board tensor, three explicit
scalar evidence heads — `e_random`, `e_near`, `e_puzzle` — and a single
final logit obtained from a logsumexp negative competition.

## Mechanism

1. **Board trunk.** `BoardConvStem(input_channels=18, channels, depth,
   use_batchnorm)` produces an `(B, channels, 8, 8)` feature map. The
   trunk consumes `simple_18` only; CRTK / source / engine metadata is
   reporting-only and is never consumed at inference.
2. **Pooled descriptor.** `mean(h)` and `max(h)` are concatenated to a
   `(B, 2 * channels)` vector and projected through a `LayerNorm +
   Linear -> GELU` mixer into the shared evidence space `(B,
   evidence_dim)`.
3. **Three evidence heads.** Three small two-layer MLPs map the shared
   evidence into scalar logits

   ```text
   e_random = head_random(z)
   e_near   = head_near(z)
   e_puzzle = head_puzzle(z)
   ```

   Each head is `Linear -> GELU -> Dropout -> Linear -> 1`.
4. **Single inference logit.** The puzzle logit is

   ```text
   puzzle_logit = e_puzzle - logsumexp([e_random, e_near])
   ```

   so `sigmoid(puzzle_logit) = exp(e_puzzle) / sum(exp([random, near, puzzle]))`.
   This is the form prescribed by the packet: the puzzle channel must
   *win* a competition against both negative channels at once.
5. **Auxiliary 3-way logits.** The raw `[e_random, e_near, e_puzzle]`
   stack is exposed as `aux_3way_logits` in the output dict so a
   trainer can attach the packet's auxiliary 3-way cross-entropy on
   the fine source label (`fine 0 -> random`, `fine 1 -> near`,
   `fine 2 -> puzzle`) at training time. The current in-tree
   `puzzle_binary` trainer runs only the BCE-on-`puzzle_logit` term;
   see `implementation_notes.md`.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. All diagnostic
tensors are finite per batch and are appended to prediction artifacts:

- `evidence_random`, `evidence_near`, `evidence_puzzle`: scalar
  evidence channels per board.
- `aux_3way_logits`: `(B, 3)` stack ordered `[random, near, puzzle]`.
- `negative_margin`: `e_puzzle - max(e_random, e_near)`.
- `random_vs_near_gap`: `|e_random - e_near|` for monitoring negative
  collapse.
- `trunk_energy`: mean square of the trunk feature map.
- `ablation_random_near_merged`, `ablation_aux_only_no_logsumexp`:
  per-batch flags.

## Ablations

The bespoke builder accepts `model.ablation in {"none",
"no_aux_3way", "random_near_merged", "aux_only_no_logsumexp",
"shuffle_fine_negative_labels"}` matching the packet's required
ablation table. `random_near_merged` ties the two negative heads and
should lose near-puzzle discrimination; `aux_only_no_logsumexp`
removes the negative competition; `no_aux_3way` zeroes the aux logits
so the trainer cannot attach a 3-way CE; `shuffle_fine_negative_labels`
is a label-only ablation honored by the trainer (the model itself does
not look at fine labels).

## Implementation Binding

- Registered model name: `puzzle_binary_benchmark_challengers`.
- Source implementation file: `src/chess_nn_playground/models/puzzle_binary_benchmark_challengers.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i074_puzzle_binary_benchmark_challengers/model.py`.
