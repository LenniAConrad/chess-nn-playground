# Ideas

All idea material now lives under one canonical folder:

```text
ideas/all_ideas/
```

Use [all_ideas/ALL_IDEAS.md](all_ideas/ALL_IDEAS.md) as the single inventory. It lists each title, ID, source family, model attribution, and path.

## Folder Map

| Folder | Purpose |
|---|---|
| [all_ideas/registry/](all_ideas/registry/) | Registered `i###_*` implementation ideas, registry metadata, generated index/TODO files, audit outputs, and the idea template. |
| [all_ideas/research/](all_ideas/research/) | Raw research packets, primitive research sessions, prompts, and human collections. |
| [all_ideas/docs/](all_ideas/docs/) | Workflow and benchmark-reporting rules. |

Do not store raw datasets, benchmark outputs, checkpoints, or experiment artifacts here.

## Start Here

- [all_ideas/ALL_IDEAS.md](all_ideas/ALL_IDEAS.md): unified source/model inventory for all idea-like material.
- [all_ideas/registry/INDEX.md](all_ideas/registry/INDEX.md): generated map of registered ideas and research-packet navigation.
- [all_ideas/registry/TODO.md](all_ideas/registry/TODO.md): generated implementation, benchmark, and packet backlog.
- [all_ideas/research/primitives/2026-05-12/PRIMITIVE_TRAINING_TODO.md](all_ideas/research/primitives/2026-05-12/PRIMITIVE_TRAINING_TODO.md): primitive implementation and training plan.
- [all_ideas/docs/WORKFLOW.md](all_ideas/docs/WORKFLOW.md): how to import, promote, implement, benchmark, and reject ideas.
- [all_ideas/docs/BENCHMARK_REPORTING.md](all_ideas/docs/BENCHMARK_REPORTING.md): required slice reporting for every idea.

## Common Commands

Regenerate idea navigation and the unified inventory:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_all_ideas_index.py
```

Regenerate the reusable idea prompt:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_prompt.py
```

Validate implementation metadata:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/audit_implementation_kinds.py --check
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/audit_architecture_conformance.py --check
```
