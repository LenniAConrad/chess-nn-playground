# Primitive Research

The primitive corpus is consolidated here:

- [2026-05-12](../primitives/2026-05-12/)

That folder is the single place for primitive imports, Codex-created primitive proposals, Claude Opus 4.7 primitive proposals, prototypes, manifests, and primitive-to-architecture bridge notes from the May 12 session.

## Folder Map

| Folder | Contents |
|---|---|
| [external_imports](../primitives/2026-05-12/external_imports/) | Thirty imported primitive-focused Google/Deep Research markdown files, normalized and named by primitive concept. |
| [codex_candidate_reply_primitives](../primitives/2026-05-12/codex_candidate_reply_primitives/) | Five Codex GPT-5 primitive proposals around reply sets, witnesses, regret, copulas, and antichains. |
| [claude_opus_4_7_primitives](../primitives/2026-05-12/claude_opus_4_7_primitives/) | Five Claude Opus 4.7 primitive proposals plus prototype scripts. |
| [architecture_bridges](../primitives/2026-05-12/architecture_bridges/) | Primitive stacking plan and a bridge architecture sketch. |

## Current Primitive Validation Order

This follows the stacking strategy in [codex_primitive_stacking_strategy.md](../primitives/2026-05-12/architecture_bridges/codex_primitive_stacking_strategy.md).

1. TSDP, Terminal-State Detection Primitive: cheapest first test, precompute in the data loader, target `mate_in_1`.
2. PFCT, Promotion-Fanout Counterfactual Tensor: gated and sparse, target promotion and underpromotion slices.
3. TDCD, Tempo-Defender Cross-Derivative Operator: test the universal `equal` bucket failure.
4. DHPE, Signed Piece-Existence Hessian Operator: expensive pairwise second-order test for sub-additivity vs super-additivity.
5. CAIO, Complex-Amplitude Interference Operator: speculative phase/interference primitive, run after cheaper targeted tests.

Do not merge these into a hybrid architecture before each primitive clears its own ablation. The current bridge rule is: promote only if the primitive beats matched ablations on its declared target slices.

The full Claude Code implementation handoff is preserved at [HANDOFF.md](../primitives/2026-05-12/claude_opus_4_7_primitives/HANDOFF.md).

The operational code/training plan is [PRIMITIVE_TRAINING_TODO.md](../primitives/2026-05-12/PRIMITIVE_TRAINING_TODO.md).

## Authorship Notes

| Set | Author/source label | Model label |
|---|---|---|
| `codex_candidate_reply_primitives/` | Codex-created local proposals | GPT-5 coding agent |
| `claude_opus_4_7_primitives/` | Claude-generated local proposals | Claude Opus 4.7 |
| `external_imports/` | Imported Google/Deep Research files | See the file-level manifest when available; model identity was not always recoverable from the download. |

## Promotion Notes

If a primitive becomes an implementable registered idea, scaffold it from `ideas/all_ideas/registry/template/` into the next available `ideas/all_ideas/registry/i###_<slug>/` folder. The initial recommended trunk is the current exchange/king dual-stream winner noted in the primitive bridge packet, with primitives added as auxiliary heads unless the primitive document explicitly calls for a trunk replacement.

Nearest-neighbor duplicate checks before promotion:

```text
i189, i041, i025-i027, i058, i066, i127, i141
```
