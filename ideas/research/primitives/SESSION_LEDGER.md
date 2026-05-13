# Primitive Research Session Ledger

Author: Codex
Model: GPT-5 Codex coding agent
Last updated: 2026-05-13
Status: active research ledger

This ledger is the human routing table for primitive research files, implementation batches, and follow-up training. The authoritative provenance table is [MANIFEST.md](MANIFEST.md).

## Research Files

| Range | Count | Source | Model | Status |
|---|---:|---|---|---|
| `claude_01`-`claude_05` | 5 | Claude | Claude Opus 4.7 | Implemented separately as legacy `i244`-`i248` |
| `codex_01`-`codex_05` | 5 | Codex | Codex GPT-5 | Implemented in worktree batch `primitive-codex-reply` as `p001`-`p005` |
| `external_01`-`external_20` | 20 | ChatGPT + Claude downloads | GPT-5.5 Pro / Claude Opus 4.7 | Implementation batches launched as `p006`-`p024` |
| `external_21`-`external_30` | 10 | Google / Gemini downloads | Unspecified Gemini model | Implementation batches launched as `p025`-`p035` |
| `external_31`-`external_41` | 11 | ChatGPT downloads | GPT-5.5 Pro | Imported research backlog; reserve `p036`-`p046` |

## Implementation Batches

| Batch | Research files | Reserved IDs | Current disposition |
|---|---|---|---|
| `primitive-codex-reply` | `codex_01`-`codex_05` | `p001`-`p005` | Claude run completed; pending local review and merge |
| `primitive-ray-legal` | `external_02`, `external_03`, `external_04`, `external_05`, `external_12`, `external_14` | `p006`-`p011` | Claude run active / pending final validation |
| `primitive-delta-accumulator` | `external_01`, `external_07`, `external_08`, `external_09`, `external_10`, `external_11`, `external_17` | `p012`-`p018` | Claude run completed; pending local review and merge |
| `primitive-occlusion-blocker` | `external_13`, `external_15`, `external_16`, `external_18`, `external_19`, `external_20` | `p019`-`p024` | Claude run active / pending final validation |
| `primitive-gemini-graph-state` | `external_21`, `external_22`, `external_23`, `external_24`, `external_26`, `external_27` | `p025`-`p030` | Claude run active / pending final validation |
| `primitive-gemini-misc` | `external_06`, `external_25`, `external_28`, `external_29`, `external_30` | `p031`-`p035` | Claude run active; `p035` was still pending at last check |
| `primitive-gpt-polynomial-next` | `external_31`-`external_41` | `p036`-`p046` | Not launched; triage first because several reports duplicate elementary-symmetric / polynomial-ledger themes |

## Integration Rules

1. Merge completed batches only after local checks pass in their worktree.
2. Prefer code/docs/tests/configs over generated audit artifacts from worktrees; regenerate audits in the main tree after integration.
3. Keep raw research files in this folder and implementation folders in `ideas/registry/p###_*`.
4. Do not train from a primitive until its model registry key, static config, focused tests, and dry-run primitive pipeline pass.
5. Treat repeated elementary-symmetric, exterior-product, and polynomial-ledger proposals as a single family during triage; implement one clean representative before adding variants.

## Next Research Triage

The 2026-05-13 GPT batch is heavy on polynomial/set/cohomology primitives. Initial grouping:

| Theme | Files | Suggested first representative |
|---|---|---|
| Elementary-symmetric / polynomial ledger pooling | `external_32`, `external_33`, `external_34`, `external_35`, `external_37`, `external_38`, `external_40` | `external_38_polynomial_ledger_grassmann_rook_primitives.md` |
| Orbit/canonicalization/irrep normalization | `external_31`, `external_33`, `external_39`, `external_41` | `external_39_orbit_irrep_hodge_projection_primitives.md` |
| Matching / rook / matroid constraints | `external_33`, `external_34`, `external_35`, `external_37`, `external_38`, `external_41` | `external_37_truncated_multiset_polynomial_rook_matching_primitives.md` |
| Hodge / graph projection / Green solve | `external_32`, `external_35`, `external_39` | `external_39_orbit_irrep_hodge_projection_primitives.md` |
| BDD / subset log-partition / Boolean solver layers | `external_31`, `external_41` | `external_31_canonical_orbit_bdd_wmc_primitives.md` |

Launch `p036`-`p046` only after deduping this batch into a smaller implementation queue, unless the goal is broad coverage over implementation quality.
