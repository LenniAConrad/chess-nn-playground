# Ablations

The implementation exposes the packet's required ablations via the
`model.ablation` field in `config.yaml`:

- `none` — main model (default).
- `local_only` — drop every stripe branch and keep only the local
  `Conv3x3`. Ordinary CNN control.
- `rank_file_only` — keep ranks and files (rook lines), drop
  diagonals and anti-diagonals.
- `diag_only` — keep diagonals only.
- `random_stripes` — replace each stripe mask with a fixed random
  `K`-position mask so the line geometry is destroyed at matched
  parameter count.
- `no_global_gate` — drop the sigmoid global-context gate so stripe
  branches are summed without selection.

Compare runs against LC0 BT4, NNUE, and the strongest registered
ideas on the same split and seeds.
