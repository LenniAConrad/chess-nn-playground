# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/rule_aware_tactical_head.py`.
- Idea-local wrapper: `ideas/registry/i248_rule_aware_tactical_head/model.py`.
- Registry key: `rule_aware_tactical_head`.
- Source primitive: `ideas/research/primitives/claude_05_terminal_state_detection.md`.
- Source prototype: `ideas/research/primitives/prototypes/tsdp_prototype.py`.
- TSDP feature extraction: `src/chess_nn_playground/data/terminal_state.py`.

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. The 11-dim TSDP feature vector is derived in `_compute_tsdp` by:

1. Reconstructing a `chess.Board` from each sample (piece planes,
   side-to-move plane, four castling-right planes, en-passant plane) via
   `simple_18_to_board`. Halfmove and fullmove counters are not preserved
   by `simple_18` and default to 0 / 1; this does not affect legal-move
   enumeration or check / mate / stalemate detection for puzzle positions.
2. Enumerating `board.legal_moves`, pushing / popping each candidate, and
   accumulating the 11 exact rule counts.

This per-sample python-chess call is the documented temporary fallback noted
in `ideas/research/primitives/PRIMITIVE_TRAINING_TODO.md`. The fallback is
honest about cost: at batch size 256, the data loader does ~256 * ~50
python-chess evaluations per batch on CPU, materially slower than the GPU
forward pass. The production upgrade is:

```text
scripts/data/precompute_primitive_features.py  (TODO)
   reads data/splits/crtk_sample_3class_unique_crtk_tags/{train,val,test}.parquet,
   computes the 11-d TSDP vector per row from normalized_fen,
   writes a sibling split directory
     data/splits/crtk_sample_3class_unique_crtk_tags_primitives/
   that adds `tsdp_features: list[float, 11]` to each row.
```

When that script lands, the dataset can expose `tsdp_features` as a
batch tensor (similar to the existing `include_rule_texture` path), and
`_compute_tsdp` becomes a tensor index instead of a python-chess call. The
shape of `_compute_tsdp(board) -> Tensor[B, 11]` is stable across both
paths, so the swap is local.

## Stop-gradient contract

Rule indicators are integer-valued and not differentiable. They are
computed inside `torch.no_grad()` and converted to a fp tensor. The
gradient flow is entirely through:

- the i193 trunk (unchanged from i193's bespoke implementation)
- the two head MLPs (`gate_mlp`, `delta_mlp`)

Trunk diagnostics fed into the head are also detached (`.detach()`) so the
head cannot leak gradient back into the trunk's gate / pooling logic.

## Output dict contract

The model output is a `dict[str, Tensor]` following the i193 contract,
extended with:

- `logits` (rebound to `base_logit + primitive_delta`)
- `base_logit`           (i193 logit, kept for diagnostics)
- `primitive_delta`      (`primitive_gate * primitive_delta_raw`)
- `primitive_delta_raw`  (head MLP output)
- `primitive_gate`       (sigmoid scalar gate)
- `primitive_gate_logit`
- `tsdp_<name>` for each of the 11 raw TSDP features
  (`mate_in_1`, `mate_count`, `stalemate_threat`, `stalemate_count`,
  `check_count`, `promotion_count`, `capture_count`, `castling_count`,
  `total_legal_moves`, `forcing_density`, `mating_special_count`)

All per-sample scalar tensors are emitted in the standard one-column-
per-key shape so the shared trainer copies them into
`predictions_<split>.parquet`.

## Ablation modes

See `model.ALLOWED_ABLATIONS`. The TSDP-specific falsifier is `shuffle_tsdp`:
in-batch permutation of the 11-d vector. The `zero_delta` and `trunk_only`
ablations recover the i193 behavior.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` that wraps the bespoke i193
`ExchangeThenKingDualStreamNetwork` and adds two new MLP heads. It does
not call `build_research_packet_probe_from_config`, does not delegate to a
shared CNN / MLP / NNUE / LC0 baseline builder, and has its own forward
pass. The `implementation_kind: bespoke_model` declaration is consistent
with the `audit_implementation_kinds.py` heuristics.
