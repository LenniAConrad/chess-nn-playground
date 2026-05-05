# crtk Export Contract for Chess NN Playground

This project wants crtk to export JSONL, one chess position per line. JSONL is preferred over JSON arrays because it streams safely for very large USB/engine datasets.

## Recommended command

Add or expose a command like:

```bash
crtk record-to-training-jsonl \
  --input <record-file-or-dir> \
  --output <out.jsonl> \
  --recursive \
  --label-mode explicit|verified-puzzle|verified-near-puzzle|known-non-puzzle|candidate|propose \
  --include-engine-metadata \
  --include-raw-record-id \
  --manifest <out.manifest.json>
```

`record-to-training-jsonl` should not export tensors. It should export position rows and metadata only. The Python side will build tensors from FEN.

## Label rules

Do not guess verified labels silently.

Use these exact label statuses:

- `known_non_puzzle`
- `candidate_1_or_2_unresolved`
- `verified_near_puzzle`
- `verified_puzzle`

Use these numeric labels:

- known non-puzzle: `coarse_label = 0`, `fine_label = 0`
- unresolved candidate: `coarse_label = 1`, `fine_label = null`
- verified near-puzzle: `coarse_label = 1`, `fine_label = 1`
- verified puzzle: `coarse_label = 1`, `fine_label = 2`

If crtk uses engine/rule logic to classify a position, write that as a proposal:

```json
"proposed_label_status": "engine_proposed_puzzle",
"proposed_fine_label": 2,
"proposal_confidence": 0.87,
"proposal_rule_version": "crtk_puzzle_rules_v1",
"proposal_reasons": ["unique_best_move", "large_pv_gap"]
```

Only write `label_status = verified_puzzle` or `label_status = verified_near_puzzle` when the source or crtk verification procedure truly proves it.

## Required fields

Each JSONL row should include:

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "label_status": "candidate_1_or_2_unresolved",
  "coarse_label": 1,
  "fine_label": null,
  "source_kind": "crtk_record",
  "source_file": "input.record",
  "source_record_index": 123,
  "source_group_id": "optional-source-group",
  "sister_group_id": "optional-puzzle-family",
  "game_id": "optional-game-id",
  "position_index": 42,
  "verification_status": "unresolved"
}
```

## Recommended metadata fields

Engine/search fields are useful metadata and labels, but they must never be used as neural network input features.

```json
{
  "best_move": "e2e4",
  "pv1_cp": 320,
  "pv2_cp": 40,
  "pv_gap_cp": 280,
  "pv1_mate": null,
  "pv2_mate": null,
  "stockfish_nodes": 2500000,
  "stockfish_depth": 18,
  "stockfish_version": "Stockfish 16",
  "multipv": [
    {"rank": 1, "move": "e2e4", "cp": 320, "mate": null, "pv": ["e2e4", "e7e5"]},
    {"rank": 2, "move": "d2d4", "cp": 40, "mate": null, "pv": ["d2d4", "d7d5"]}
  ]
}
```

## Proposal labels

If crtk can mine or verify puzzle-like positions, export both the unresolved canonical label and the proposal:

```json
{
  "fen": "8/8/8/8/8/8/5PPP/6K1 w - - 0 1",
  "label_status": "candidate_1_or_2_unresolved",
  "coarse_label": 1,
  "fine_label": null,
  "proposed_label_status": "engine_proposed_puzzle",
  "proposed_fine_label": 2,
  "proposal_confidence": 0.92,
  "proposal_rule_version": "crtk_puzzle_rules_v1",
  "proposal_reasons": ["verified_unique_solution", "stable_best_move", "large_pv_gap"]
}
```

The Python side can later build a weak-label dataset from proposal fields without contaminating verified labels.

## Manifest

For each export, also write a manifest:

```json
{
  "created_at": "2026-04-20T00:00:00Z",
  "crtk_version": "local-build",
  "command": "crtk record-to-training-jsonl ...",
  "input_paths": ["..."],
  "output_path": "...",
  "rows_seen": 1000000,
  "rows_written": 987654,
  "label_status_counts": {
    "candidate_1_or_2_unresolved": 987654
  },
  "proposal_counts": {
    "engine_proposed_puzzle": 12345,
    "engine_proposed_near_puzzle": 54321
  },
  "rule_version": "crtk_puzzle_rules_v1",
  "notes": "Engine metadata is metadata only, not model input."
}
```

## Command modes

Recommended `--label-mode` behavior:

- `explicit`: use only explicit `kind`, `label_status`, or verification fields from the record.
- `candidate`: force all exported rows to unresolved candidates.
- `known-non-puzzle`: force all exported rows to known non-puzzles.
- `verified-puzzle`: force all exported rows to verified puzzles only when the input source is trusted.
- `verified-near-puzzle`: force all exported rows to verified near-puzzles only when the input source is trusted.
- `propose`: keep canonical labels unresolved unless explicit, and fill proposal fields from crtk rules.

## Minimal first implementation

The first useful crtk change can be small:

1. Add/expose `record-to-training-jsonl`.
2. Accept `--input`, `--output`, `--recursive`, `--label-mode`, `--filter`.
3. Emit `fen`, `label_status`, `coarse_label`, `fine_label`, `source_file`, `source_record_index`, `best_move`, `pv1_cp`, `pv2_cp`, `pv_gap_cp`, `stockfish_nodes`, `verification_status`.
4. Write a manifest JSON.
5. Do not require LC0 weights.

