# Primitive Research

The primitive corpus is consolidated under [../primitives/](../primitives/) as a single flat folder. File-name prefixes encode the source (`claude_`, `codex_`, `external_`) and full provenance lives in [../primitives/MANIFEST.md](../primitives/MANIFEST.md).

## File Map

| Prefix | Contents |
|---|---|
| `external_*.md` | Fifty imported primitive-focused markdown reports from ChatGPT Deep Research, Claude, and Google/Gemini. |
| `codex_*.md` | Five Codex GPT-5 primitive proposals around reply sets, witnesses, regret, copulas, and antichains. |
| `claude_*.md` | Five Claude Opus 4.7 primitive proposals plus prototype scripts under [../primitives/prototypes/](../primitives/prototypes/). |

Architecture bridge notes that combine primitives live one level up at [../architecture_bridges/](../architecture_bridges/), kept out of the primitive folder to avoid mixing operators with compositions.

## Current Primitive Validation Order

This follows the stacking strategy in [../architecture_bridges/codex_primitive_stacking_strategy.md](../architecture_bridges/codex_primitive_stacking_strategy.md).

1. TSDP, Terminal-State Detection Primitive: cheapest first test, precompute in the data loader, target `mate_in_1`.
2. PFCT, Promotion-Fanout Counterfactual Tensor: gated and sparse, target promotion and underpromotion slices.
3. TDCD, Tempo-Defender Cross-Derivative Operator: test the universal `equal` bucket failure.
4. DHPE, Signed Piece-Existence Hessian Operator: expensive pairwise second-order test for sub-additivity vs super-additivity.
5. CAIO, Complex-Amplitude Interference Operator: speculative phase/interference primitive, run after cheaper targeted tests.

Do not merge these into a hybrid architecture before each primitive clears its own ablation. The current bridge rule is: promote only if the primitive beats matched ablations on its declared target slices.

The full Claude Code implementation handoff is preserved at [../primitives/HANDOFF.md](../primitives/HANDOFF.md).

The operational code/training plan is [../primitives/PRIMITIVE_TRAINING_TODO.md](../primitives/PRIMITIVE_TRAINING_TODO.md).

The pre-scout validation gate for new primitives is [../primitives/PRIMITIVE_VALIDATION_PROTOCOL.md](../primitives/PRIMITIVE_VALIDATION_PROTOCOL.md).

## Authorship Notes

| Prefix | Author/source label | Model label |
|---|---|---|
| `codex_*.md` | Codex-created local proposals | Codex GPT-5 |
| `claude_*.md` | Claude-generated local proposals | Claude Opus 4.7 |
| `external_*.md` | Imported reports | See per-file rows in [../primitives/MANIFEST.md](../primitives/MANIFEST.md); model identity was not always recoverable from the download. |

## Promotion Notes

If a primitive becomes an implementable registered idea, scaffold it from `ideas/registry/template/` into the next available `ideas/registry/i###_<slug>/` folder. The initial recommended trunk is the current exchange/king dual-stream winner noted in the primitive bridge packet, with primitives added as auxiliary heads unless the primitive document explicitly calls for a trunk replacement.

Nearest-neighbor duplicate checks before promotion:

```text
i189, i041, i025-i027, i058, i066, i127, i141
```
