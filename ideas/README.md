# All Ideas

This is the canonical folder for the complete idea corpus.

| Area | Contents |
|---|---|
| [ALL_IDEAS.md](ALL_IDEAS.md) | Single generated inventory with title, ID, source, model attribution, and path. |
| [registry/](registry/) | Registered ideas that can be implemented, trained, audited, or benchmarked. ID prefix encodes the target models/ bucket: `i###` = trunk (whole-architecture), `p###` = primitive operator, `a###` = compositional architecture. Numbering is independent per prefix. |
| [research/](research/) | Raw architecture packets, primitive research, prompts, and human collections. |
| [docs/](docs/) | Workflow and benchmark-reporting standards. |

The folder is intentionally one home with internal sections. Registered ideas should remain in `registry/`; raw research should remain in `research/`; both are indexed together in `ALL_IDEAS.md`.

Regenerate the inventory with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_all_ideas_index.py
```
