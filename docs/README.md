# Documentation Index

Use this directory for stable project protocols and reference material. Generated run reports, leaderboards, prompts, plots, and suite logs belong under `reports/`, not `docs/`.

## Core Docs

- [puzzle_binary_benchmark_goal.md](puzzle_binary_benchmark_goal.md): benchmark definition, label mapping, and what counts as a meaningful comparison.
- [experimental_training_pipeline.md](experimental_training_pipeline.md): data flow, benchmark commands, required artifacts, model registration, and encoding notes.
- [reliable_training_protocol.md](reliable_training_protocol.md): smoke, triage, reliable, promotion-grade, and paper-grade training standards.
- [repo_layout.md](repo_layout.md): current folder structure and path stability rules.
- [crtk_export_contract.md](crtk_export_contract.md): required CRTK export fields and import expectations.
- [export_training_data_from_stacks.md](export_training_data_from_stacks.md): short CRTK stack export, import, split, tagging, and audit guide.

## Related Entry Points

- [../README.md](../README.md): top-level operating guide.
- [../configs/README.md](../configs/README.md): benchmark config and suite layout.
- [../scripts/README.md](../scripts/README.md): command entrypoints.
- [../ideas/README.md](../ideas/README.md): research idea workspace.
- [../ideas/all_ideas/registry/TODO.md](../ideas/all_ideas/registry/TODO.md): generated idea and benchmark backlog.

## Maintenance Rules

- Keep this directory concise and stable; do not store one-off run logs here.
- Update docs when moving source paths, changing benchmark contracts, or changing required artifacts.
- Regenerate idea catalogs with `PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py` instead of hand-editing generated idea indexes.
- Prefer repo-relative paths in docs so commands remain copyable from the repo root.
