# Architecture

`Counterfactual Defender Dropout Network` is a board-only puzzle
binary architecture that asks, for each candidate intervention on
the current position, *how much would the network's belief change
if that role were removed?* The thesis (see `math_thesis.md`) is
that real puzzles hinge on a small number of *causally critical*
participants — overloaded defenders, pinning sliders, the one
escape square — so structured deletions of those participants
should produce sharp logit changes, while the same deletions on
near-puzzles should produce small or symmetric changes.

The model is bespoke board-only: it consumes the repository
`simple_18` current-board tensor `(B, 18, 8, 8)` and returns one
puzzle logit for the BCE-with-logits `puzzle_binary` trainer.

## Mechanism

1. **Board encoder.** A compact convolutional square encoder
   (`DefenderDropoutTrunk`) ingests the 18-plane board with two
   coordinate planes and produces:

   - `h`: `(B, channels, 8, 8)` per-square trunk features,
   - `pooled`: `(B, 2 * channels)` mean+max global pool.

2. **Closed-form intervention mask builder.** From the
   `simple_18` planes alone, `DefenderDropoutMaskBuilder` computes
   four structured intervention scores per square using a fixed
   precomputed geometry (per-piece geometric attack tables, slider
   between-square line tables, a king-zone table, occupancy-based
   slider clearance):

   - **defender** — enemy pieces that defend an enemy target the
     side-to-move attacks, plus enemy pieces with own slider rays
     touching the enemy king zone (overloaded / blocking
     defenders);
   - **attacker** — own pieces that hit the enemy king zone or
     hanging enemy material, weighted by own piece value;
   - **king_escape** — empty squares in the enemy king zone that
     are not yet covered by the side-to-move (the candidate one
     escape square);
   - **ray_blocker** — squares on a slider-to-king line whose
     occupancy contributes the *single* blocker on that ray
     (pinning / discovered-attack blockers).

   The top-`k` squares per kind are packed into deterministic
   one-hot intervention masks `masks ∈ {0, 1}^{B × M × 64}` with a
   per-mask kind id `mask_types ∈ {0, 1, 2, 3}^{B × M}` and a
   `valid` flag. A `summary` vector of 14 deterministic scalars
   (per-kind max/mean intervention scores, own/enemy piece counts,
   material balance, attack/ray densities) is also produced.

3. **Base puzzle head.** Pooled trunk features and the
   intervention summary feed a `LayerNorm + Linear + GELU` context
   `context ∈ R^{hidden_dim}`, and a small MLP `base_head`
   produces a baseline puzzle logit `base_logit`.

4. **Counterfactual intervention head.** For every candidate mask
   the `InterventionHead` computes per-mask features

   ```text
   local_mean   = sum_{s in mask} h_s / |mask|
   retained_mean = sum_{s not in mask} h_s / (64 - |mask|)
   abs_gap, product, type_embedding(mask_kind), score, valid
   ```

   and predicts a counterfactual delta `delta ∈ R^{B × M}`. This
   is the differentiable analogue of "remove this defender /
   attacker / escape square / ray-blocker and read off how much
   the puzzle belief should move." Invalid mask slots are zeroed.

5. **Mechanism summaries.** The deltas are aggregated into 13
   counterfactual evidence scalars per batch row:

   - per-kind `top-k` mean sensitivity
     (`defender_sensitivity`, `attacker_sensitivity`,
     `king_escape_sensitivity`, `blocker_sensitivity`),
   - **asymmetry** = `defender_sensitivity − attacker_sensitivity`
     (the central thesis signal: defender deletions should bite
     harder than attacker deletions on a true puzzle),
   - `defender − king_escape`, `defender − blocker`,
   - normalised entropy of `|delta|` over valid masks,
   - max / mean `|delta|`,
   - normalised valid-mask, defender-mask, attacker-mask counts.

6. **Correction head.** A small MLP `correction` reads those 13
   scalars and outputs an additive `correction` to the baseline
   logit. The puzzle output is

   ```text
   logits = base_logit + correction.
   ```

   This factorisation is how the markdown thesis enters the head:
   even if the trunk produces the same baseline for a near-puzzle
   and a true puzzle, only the true puzzle's dropout asymmetry can
   feed the correction head a strongly positive signal.

## Ablations

The constructor accepts one of:

- `none` — the full network described above.
- `random_masks` — replace the four typed score fields with a
  fixed random permutation over squares (gated by piece /
  king-zone / occupancy where appropriate). The defender /
  attacker / escape / blocker masks become structurally
  meaningless, so the asymmetry signal must collapse if the
  thesis is real.
- `defenders_only` — zero out attacker, king-escape and
  ray-blocker scores so the intervention head only sees defender
  deletions. Tests whether defender deletions alone carry the
  signal.
- `no_intervention_head` — force `delta = 0`, so the correction
  head can only see degenerate counterfactual statistics. Tests
  whether the model collapses to the base head when intervention
  evidence is removed.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for
the `puzzle_binary` BCE-with-logits trainer (`num_classes == 1`),
plus diagnostics:

- `logits`, `base_logit`, `intervention_correction`: `(B,)`.
- `intervention_delta`: `(B, M)` per-mask counterfactual delta.
- `intervention_mask_scores`, `intervention_valid`: `(B, M)`.
- `defender_sensitivity`, `attacker_sensitivity`,
  `king_escape_sensitivity`, `blocker_sensitivity`: `(B,)`.
- `sensitivity_asymmetry`, `defender_minus_escape`,
  `defender_minus_blocker`, `sensitivity_entropy`,
  `max_sensitivity`, `mean_sensitivity`: `(B,)`.
- `defender_mask_count`, `attacker_mask_count`,
  `king_escape_mask_count`, `blocker_mask_count`: `(B,)`.
- `mechanism_energy`, `proposal_profile_strength`,
  `proposal_keyword_count`: `(B,)` proposal-profile reporting
  scalars (matching the broader idea-folder reporting contract).

When called with `return_aux=True`, the dict additionally exposes
`intervention_masks` and `intervention_mask_types`.

## Implementation Binding

- Registered model name: `counterfactual_defender_dropout_network`
- Source implementation file: `src/chess_nn_playground/models/counterfactual_defender_dropout.py`
- Idea-local wrapper: `ideas/registry/i189_counterfactual_defender_dropout_network/model.py`
