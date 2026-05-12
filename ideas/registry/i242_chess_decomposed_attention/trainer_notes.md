# Trainer Notes

Train through the guarded idea wrapper:

```bash
PYTHONDONTWRITEBYTECODE=1 python ideas/registry/i242_chess_decomposed_attention/train.py
```

Use the canonical CRTK split and the same slice reporting contract as other `puzzle_binary` ideas. Compare directly against i193, LC0 BT4, and the strongest current bespoke registry models before treating any gain as meaningful.
