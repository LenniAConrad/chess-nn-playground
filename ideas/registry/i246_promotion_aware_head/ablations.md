# Ablations — Promotion-Aware Head (i246, PFCT)

## Ablation switches

The model exposes a single `ablation` argument (head-level) and a separate
`trunk_ablation` argument (forwarded to the underlying i193 trunk).

| Head `ablation`             | What it does                                                                                  |
|-----------------------------|-----------------------------------------------------------------------------------------------|
| `none`                      | Full PFCT primitive.                                                                          |
| `copy_baseline_fanout`      | A1 falsifier: replace `F_theta(p, x)` with four copies of the baseline `phi(x)` feature.      |
| `uniform_attention`         | Force the attention distribution to a uniform `1/4` over Q/R/B/N.                             |
| `zero_delta`                | Force `primitive_delta` to zero — collapses the architecture to i193.                          |
| `force_open_gate`           | Bypass the learned gate (`gate == 1` whenever any near-promotion pawn exists).               |
| `trunk_only`                | Alias for `zero_delta` (explicit "primitive disabled" run).                                   |

| Trunk `trunk_ablation`      | What it does                                                                                  |
|-----------------------------|-----------------------------------------------------------------------------------------------|
| `none`                      | Full i193 trunk.                                                                              |
| `shared_stream_only`        | Use a single shared encoder for both streams.                                                 |
| `fixed_half_gate`           | Force the trunk gate to 0.5.                                                                  |
| `king_only`                 | Force the trunk gate to 1 (king stream).                                                      |
| `exchange_only`             | Force the trunk gate to 0 (exchange stream).                                                  |

## What each ablation tests

### `copy_baseline_fanout` — the primary falsifier

This is the matched ablation called out in `claude_03_promotion_fanout_counterfactual.md`.
It keeps the cross-attention head, gate, value projection, and per-square /
per-piece-type embeddings, but it replaces the four counterfactual feature
rows with four copies of the baseline `phi(x)` feature. If the full PFCT
architecture cannot beat this ablation on the promotion slice, then the
four-fold substitution is not adding signal beyond what the architecture
itself encodes, and the primitive must be dropped.

### `uniform_attention`

Disables the learned attention. Tests whether the *weighting* over Q/R/B/N
matters, or whether averaging the fanout is enough. A 1/4-uniform attention
collapses the per-pawn pooled feature to `mean_T value(F_theta(p, x)[T])`.
If this matches the full PFCT, the head is acting like a mean aggregator
and the attention machinery should be removed (cheaper architecture).

### `zero_delta` / `trunk_only`

Forces the primitive delta to zero. Final logit equals `base_logit`. Used
as a sanity check: the model should match the standalone i193 trunk's
puzzle_binary metrics under this ablation, modulo BN/dropout noise on
shared parameters. (The trunk in this idea has its own BN running stats
that are also updated on the counterfactual passes; with `zero_delta` the
counterfactual passes still run, so the BN stats are not strictly identical
to a standalone i193 — but the puzzle logit only depends on the *baseline*
pass, so the metrics should match closely.)

### `force_open_gate`

Bypasses the learned gate. Final logit becomes `base_logit + primitive_delta`
on any position with at least one near-promotion pawn. Tests whether the
gate is load-bearing or whether the primitive can be safely added everywhere
near-promotion pawns exist. If the open-gate run matches `none`, the gate
is decorative and can be removed.

### Trunk ablations

The trunk ablations re-export the i193 baseline ablations. They are
included so that PFCT can be run on top of every flavour of the i193
trunk — useful when interpreting whether PFCT's lift is conditional on
the trunk's exchange/king split.

## Falsification criteria

Drop the primitive (do not promote into the hybrid stack) when *any* of
the following holds on the canonical scout protocol (matched split, seed,
training budget, threshold):

1. `copy_baseline_fanout` reaches within 0.005 PR AUC of the `none`
   ablation on the `promotion` slice → substitution adds nothing.
2. `none` PR AUC on the `promotion` slice is below 0.685 → narrow miss
   on the declared target slice.
3. `none` aggregate test PR AUC regresses by more than 0.005 versus i193
   → the primitive hurts overall performance.
4. The per-slice PR AUC report shows a regression > 0.01 on any
   non-promotion slice (including `crtk_difficulty`, `crtk_phase`,
   `crtk_eval_bucket`, `crtk_tag_families`) → the primitive distorts
   the trunk in ways that hurt non-target slices.

The keep / drop decision is recorded in `runs/decision.md` after the
first scout run.

## Sanity-check tests baked into pytest

The focused test file `tests/test_promotion_aware_head.py` covers the
deterministic shape of the model, the zero-overhead guarantee on positions
without near-promotion pawns, the counterfactual board construction, the
matched-fanout ablation collapse path, and a backward-pass gradient flow
check. These are the smoke checks that gate every CI run.
