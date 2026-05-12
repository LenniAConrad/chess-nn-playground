# Idea Library

This folder is the human navigation layer for the idea corpus. It keeps the research easy to browse while registered implementation folders stay under `ideas/registry/i###_*`.

## Canonical Locations

| Material | Canonical location | Notes |
|---|---|---|
| Registered implementation candidates | `ideas/registry/i###_*` | Source of truth for ideas that can be implemented or benchmarked. Do not move these folders. |
| Raw architecture research packets | `ideas/research/packets/classic/` | Imported and generated markdown packets. Paths are kept stable for citations from registered ideas. |
| Primitive research session | `ideas/research/primitives/` | Dedicated folder for primitive imports, Codex/Claude primitive proposals, prototypes, and bridge notes. |
| Human collections | `ideas/research/collections/` | Curated indexes and grouping notes. These are navigation aids, not benchmark evidence. |

## Collections

- [Primitive Research](primitive_research.md): all primitive-session material and the current validation order.
- [Classic Architecture Packets](classic_architecture_packets.md): pre-primitive packet families grouped by operator style.
- [Registered Ideas](registered_ideas.md): how to use the `i###_*` registry without mixing it with raw packets.

## Organization Policy

Use this folder when deciding where something belongs:

- New primitive research goes under `ideas/research/primitives/` unless it becomes a promoted `i###_*` idea.
- New non-primitive raw research packets go under `ideas/research/packets/classic/`, or a clearly named packet subfolder when a batch is large enough to need separation.
- Implementable ideas are promoted into `ideas/registry/i###_*` and registered in `ideas/registry/registry.jsonl`.
- Architecture bridge notes belong with the research family they connect, unless they are promoted into a registered architecture.

The library can point across all of those places. It should not duplicate full packets or implementation files.
