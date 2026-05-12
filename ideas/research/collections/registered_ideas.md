# Registered Ideas

Registered ideas are the implementation-ready side of the corpus. They stay at:

```text
ideas/registry/i###_<slug>/
```

Do not move these folders into research packet folders. The registry, audit scripts, training handoffs, and generated prompts expect the `ideas/registry/` layout.

## Source Of Truth

| Need | File |
|---|---|
| Machine-readable registry | `ideas/registry/registry.jsonl` |
| Human index | `ideas/registry/INDEX.md` |
| Status and benchmark next actions | `ideas/registry/TODO.md` |
| Implementation-kind audit | `ideas/registry/audits/implementation_audit.md` |
| Architecture conformance audit | `ideas/registry/audits/architecture_conformance_audit.md` |

## Promotion Rule

Raw packets and primitive proposals are not implemented ideas. Promote only when the candidate has:

1. A distinct mathematical thesis.
2. A central falsifier.
3. A scaffolded `ideas/registry/i###_<slug>/` folder.
4. An `idea.yaml` row with explicit implementation status.
5. A model/config path or a clear reason it remains draft.

The primitive folder can produce future registered ideas, but until that happens it is still research input rather than benchmark evidence.
