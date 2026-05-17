# Primitive Research

Single flat folder for every primitive-level research report from the 2026-05-12, 2026-05-13, and 2026-05-16 sessions. File-name prefixes encode the source (`claude_`, `codex_`, `external_`) and the per-source ordinal; full provenance — primitive name, slug, status, prototype, model — lives in [MANIFEST.md](MANIFEST.md).

## Layout

| Path | Contents |
|---|---|
| `claude_*.md` | 5 Claude Opus 4.7 primitive proposals |
| `codex_*.md` | 5 Codex GPT-5 candidate/reply primitives |
| `external_*.md` | 50 imported reports from ChatGPT Deep Research, Claude, and Google/Gemini |
| [prototypes/](prototypes/) | Python prototype scripts for Claude's primitives (DHPE, PFCT, CAIO, TSDP) |
| [HANDOFF.md](HANDOFF.md) | Claude Code handoff for the Opus 4.7 primitive batch |
| [SESSION_LEDGER.md](SESSION_LEDGER.md) | Cross-reference for every primitive and related architecture note |
| [PRIMITIVE_TRAINING_TODO.md](PRIMITIVE_TRAINING_TODO.md) | Concrete plan for promotion, training, ablations, and stacking |
| [PRIMITIVE_VALIDATION_PROTOCOL.md](PRIMITIVE_VALIDATION_PROTOCOL.md) | Pre-scout validation and falsifier protocol for future primitive implementations |
| [MANIFEST.md](MANIFEST.md) | Authoritative provenance table for all 60 primitive research files |

Architecture proposals and fusion strategies live one level up at [../architecture_bridges/](../architecture_bridges/), kept out of this folder so primitive operators are not confused with composed architectures.

Promote any primitive that survives a falsifier to `ideas/registry/p###_<slug>/` with the standard idea scaffold. The legacy Claude handoff implementations currently occupy `i244` through `i248`; new primitive-native implementations should use the `p###` registry.
