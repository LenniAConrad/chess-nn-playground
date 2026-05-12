# Idea Workflow

This folder has two kinds of research material:

- registered ideas: `ideas/all_ideas/registry/i###_*`
- raw research packets: `ideas/all_ideas/research/packets/classic/` and curated packet subfolders

It also has a human navigation layer:

- idea collections: `ideas/all_ideas/research/collections/`

Registered ideas are the only ideas Codex should implement directly. Research packets are evidence and design input; promote one before coding it.

## Quick Start

To see the current map:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py
```

Then read:

```text
ideas/all_ideas/registry/INDEX.md
ideas/all_ideas/registry/TODO.md
ideas/all_ideas/research/packets/CATALOG.md
```

## Source Of Truth

| Need | Source |
|---|---|
| What is implemented or ready to implement | `ideas/all_ideas/registry/registry.jsonl` and `ideas/all_ideas/registry/i###_*/idea.yaml` |
| Raw imported ChatGPT/Deep Research output | `ideas/all_ideas/research/packets/classic/` |
| Human grouping of all idea material | `ideas/all_ideas/research/collections/` |
| Implementation/performance TODO | `ideas/all_ideas/registry/TODO.md` |
| Full packet catalog | `ideas/all_ideas/research/packets/CATALOG.md` |
| Machine-readable packet catalog | `ideas/all_ideas/research/packets/CATALOG.jsonl` |
| Deep Research duplicate-memory prompt | `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` |
| Generated compact idea prompt | `ideas/all_ideas/research/prompts/idea_generation_prompt.md` |
| Benchmark reporting standard | `ideas/all_ideas/docs/BENCHMARK_REPORTING.md` |

Do not treat a raw research packet as implemented. Do not add a packet to `registry.jsonl` until it has a complete `ideas/all_ideas/registry/i###_*` folder.

## Lifecycle

| Stage | Meaning | Required action |
|---|---|---|
| `packet` | Raw research note in `ideas/all_ideas/research/packets/`. | Catalog it and update prompt duplicate memory if meaningful. |
| `draft` | Registered idea with complete documentation scaffold. | Implement only after reading `math_thesis.md`, `architecture.md`, and `ablations.md`. |
| `implemented` | Model/config/test code exists but benchmark is not complete. | Run smoke tests and a small benchmark. |
| `tested` | Benchmark results exist and are linked from the idea. | Compare against baselines and decide whether to keep, refine, or reject. |
| `rejected` | Falsification criteria failed or the idea duplicated a better approach. | Keep documentation and result links; do not delete. |
| `archived` | Superseded or no longer active. | Keep for duplicate prevention. |

## Promoting A Packet

1. Pick one packet from `ideas/all_ideas/research/packets/CATALOG.md`.
2. Check it against `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` duplicate-memory sections.
3. Choose the next available ID:

```bash
find ideas/all_ideas/registry -maxdepth 1 -type d -name 'i[0-9][0-9][0-9]_*' | sort
```

4. Copy `ideas/all_ideas/registry/template/` to `ideas/all_ideas/registry/i###_<slug>/`.
5. Fill in every file in the scaffold:

```text
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

6. Append one JSON line to `ideas/all_ideas/registry/registry.jsonl`.
7. Regenerate the idea index, TODO, packet catalog, and prompt.
8. Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_idea_registry.py tests/test_idea_prompting.py
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_prompt.py
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py
```

## Implementing A Registered Idea

1. Read only the selected idea folder first.
2. Implement reusable model code in `src/chess_nn_playground/models/`.
3. Register the model in `src/chess_nn_playground/models/registry.py`.
4. Add a benchmark config under `configs/benchmarks/<task>/`, or keep the idea-local `config.yaml` if it is not a shared benchmark yet.
5. Keep the idea folder as documentation and an idea-local wrapper; the reusable model implementation belongs in `src/chess_nn_playground/models/`.
6. Add a narrow forward/smoke test before training.
7. Run the smallest relevant benchmark before scaling up.

## Benchmark Update Checklist

After a run:

1. Link result directory in `idea.yaml.latest_result_path`.
2. Update `status` and `implementation_status`.
3. Add a short run note under `ideas/all_ideas/registry/i###_*/runs/`.
4. Generate `slice_report_val.md` and `slice_report_test.md` with `scripts/reports/report_prediction_slices.py`.
5. Update the idea report with difficulty, phase, eval-bucket, motif, and tag-family performance, including strongest and weakest slices.
6. Record whether the central ablation passed or failed overall and on the declared target slices.
7. Update duplicate-memory text if the idea is rejected or becomes a family to avoid.
8. Regenerate:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_prompt.py
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py
```

Do not accept a run as benchmarked if it only reports an aggregate confusion matrix. Every idea must explain what the model can and cannot learn by difficulty level and CRTK tags.

## Importing New Downloads

1. Copy only relevant Markdown research packets into `ideas/all_ideas/research/packets/classic/`, or into a dedicated subfolder when a batch should stay together.
2. Normalize filenames:

```text
chess_nn_research_<YYYY-MM-DD>_<HHMM>_<weekday>_<timezone>_<slug>.md
```

3. Keep duplicate browser downloads only if they are useful for provenance; mark them as duplicates in `README.md`.
4. Update `ideas/all_ideas/research/packets/README.md` with a compact import summary.
5. Update `ideas/all_ideas/research/collections/` when the batch changes the human grouping of the corpus.
6. Update `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` so future research avoids near-duplicates.
7. Regenerate catalogs and prompts:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_prompt.py
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py
```

## Rules That Prevent Mess

- One implemented idea per registered folder.
- One central falsifier per idea.
- Research packets are not edited into implementations.
- Generated prompt/catalog files are regenerated, not hand-maintained.
- Raw data, benchmark results, and model code stay outside `ideas/`.
- If an idea is bad, mark it `rejected`; do not delete it, because rejected ideas prevent duplicate research later.
