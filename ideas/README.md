# Future Idea Registry

This folder is the research control room for future architecture ideas. Do not store baseline models, raw datasets, benchmark outputs, or experiment artifacts here.

Start with:

- `INDEX.md`: current map of registered ideas and research packets.
- `TODO.md`: implementation status, performance status, and next action for every registered idea and packet.
- `implementation_audit.md` / `implementation_audit.json`: generated architectural honesty audit from `model.py` wiring.
- `architecture_conformance_audit.md` / `architecture_conformance_audit.json`: generated audit for rows currently marked `implemented` or `tested`.
- `WORKFLOW.md`: how to import, promote, implement, benchmark, and reject ideas.
- `BENCHMARK_REPORTING.md`: required per-difficulty, per-phase, per-motif, and per-tag reporting for every idea.
- `research_packets/CATALOG.md`: generated catalog of raw research packets.

Registered ideas are prompt-compliant research candidates with code/config scaffolds. They are not benchmark evidence unless their `idea.yaml` links a completed result and the run artifacts validate. They are also not automatically bespoke architectures: check `implementation_kind` before treating a folder as architecturally distinct.

Current operating state: only the `bespoke_model` rows are fully implemented architectures. The `shared_probe_variant` rows are retained as scaffolded research packets and provenance, but they are not trainable architecture claims until their markdown thesis is implemented as bespoke model code.

## Registered Ideas

The generated source of truth is:

- `INDEX.md`: all registered ideas and research-packet navigation.
- `TODO.md`: implementation status, linked results, and next benchmark actions.
- `registry.jsonl`: machine-readable registry rows.
- `implementation_audit.json`: machine-readable implementation-kind audit.
- `architecture_conformance_audit.json`: machine-readable non-shell conformance audit for implemented architectures.

There are many registered ideas, so this README intentionally does not duplicate the generated table. Regenerate the table after adding, promoting, rejecting, or benchmarking ideas.

Every future idea should live in its own folder:

```text
ideas/{idea_id}_{idea_slug}/
  idea.yaml
  math_thesis.md
  architecture.md
  implementation_notes.md
  trainer_notes.md
  ablations.md
  model.py
  train.py
  config.yaml
  report_template.md
  runs/
```

Use the registry only for ideas with explicit documentation, implementation status, and links to results.

Validate implementation-kind metadata with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/audit_implementation_kinds.py --check
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/audit_architecture_conformance.py --check
```

Every implemented idea must report more than an aggregate matrix. Its run report must include the standard slice reports over `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`, `crtk_tactic_motifs`, and `crtk_tag_families`, then state what the model appears able and unable to learn.

Regenerate the navigation files with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py
```

## ChatGPT Idea Prompt

Generate the reusable idea-discipline prompt with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_prompt.py
```

This writes `ideas/idea_generation_prompt.md`. The prompt tells future ChatGPT sessions how to search for ideas, reject duplicates, write math theses, separate proof from hypothesis, and avoid treating unresolved candidates as verified labels.

For a more ambitious ChatGPT Pro research pass, use:

```text
ideas/chatgpt_pro_deep_math_research_prompt.md
```

That prompt asks ChatGPT Pro to research one non-obvious, high-math idea, reject common baseline variants, and return a strict Codex handoff packet for implementation, training, and benchmarking.

Imported ChatGPT Pro outputs are kept in:

```text
ideas/research_packets/
```

These files are research handoff packets or download stubs. They are not implemented ideas until one is copied into a proper `ideas/{idea_id}_{idea_slug}/` folder, implemented, trained, and benchmarked.

Do not organize research packets by moving them into nested folders. Keep raw packet paths stable and use `research_packets/CATALOG.md` plus `research_packets/CATALOG.jsonl` for navigation.
