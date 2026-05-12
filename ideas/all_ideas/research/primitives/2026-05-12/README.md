# Primitive Research 2026-05-12

This is the single canonical folder for primitive-level research from the 2026-05-12 session.

The folder is intentionally split so raw primitive reports, Codex-created primitives, prototype code, and architecture bridge notes do not get mixed.

## Folder Map

| Path | Contents | Notes |
|---|---|---|
| [external_imports/](external_imports/) | 30 imported primitive research reports from GPT/ChatGPT Deep Research, Claude, and Google/Gemini | Original downloads were renamed for readability. Provenance and model attribution are in [external_imports/MANIFEST.md](external_imports/MANIFEST.md). |
| [codex_candidate_reply_primitives/](codex_candidate_reply_primitives/) | Five Codex GPT-5 candidate/reply primitives: PAFR, RSP, RCC, TCC, WCQ | These are pure primitive proposals intended to share a future candidate/reply tensor infrastructure; provenance is in the folder manifest. |
| [claude_opus_4_7_primitives/](claude_opus_4_7_primitives/) | Five Claude Opus 4.7 primitive proposals plus prototype scripts | DHPE, TDCD, PFCT, CAIO, and TSDP live here with their README, manifest, and prototype code. |
| [architecture_bridges/](architecture_bridges/) | Architecture notes that use or combine the primitives | Kept separate from primitive definitions so model architecture ideas are not confused with operator proposals. |
| [SESSION_LEDGER.md](SESSION_LEDGER.md) | Human-readable session ledger | Links every Codex-created primitive and related architecture note in one place. |
| [MANIFEST.md](MANIFEST.md) | Folder-level manifest | Counts and ownership map for the whole primitive research tree. |
| [PRIMITIVE_TRAINING_TODO.md](PRIMITIVE_TRAINING_TODO.md) | Training and implementation TODO | Concrete plan for turning primitives into code, tests, scout runs, and hybrid architectures. |

The Claude Code implementation handoff for the Opus 4.7 primitive batch is preserved at [claude_opus_4_7_primitives/HANDOFF.md](claude_opus_4_7_primitives/HANDOFF.md).

## Boundary

Primitive definitions belong in `codex_candidate_reply_primitives/`, `claude_opus_4_7_primitives/`, or `external_imports/`.

Architecture proposals, fusion strategies, and model scaffolds belong in `architecture_bridges/`.

General chess architecture research packets that are not primitive-focused should stay in `ideas/all_ideas/research/packets/classic/` or promoted `ideas/all_ideas/registry/i###_*` folders, not here.
