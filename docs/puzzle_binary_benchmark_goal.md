# Puzzle-Binary Benchmark Goal

## Goal

Build a reproducible benchmark that measures whether a chess neural network can recognize real chess-puzzle signal from a position alone.

The benchmark is not meant to ask "which model can memorize this dataset?" It is meant to ask a sharper question: can a network distinguish actual puzzle positions from positions that look similar, especially hard negatives that are close to puzzle positions but are not puzzles?

## Exact Problem Statement

Input: a chess position, encoded from FEN into the model-specific tensor representation.

Output: one scalar puzzle logit.

Training target:

- source class `0`, known non-puzzle / usually random position -> binary target `0`
- source class `1`, verified near-puzzle / hard negative -> binary target `0`
- source class `2`, verified puzzle -> binary target `1`

Loss: binary cross entropy with logits, normally `BCEWithLogitsLoss`.

Default prediction rule:

```text
sigmoid(logit) >= 0.5 -> predict puzzle
sigmoid(logit) < 0.5  -> predict non-puzzle
```

This is intentionally not a 3-class classifier. The model should only answer "puzzle or not puzzle." The three source classes remain important for diagnostics.

## Diagnostic Matrix

The key confusion matrix is rectangular:

```text
rows:    source classes [random/non-puzzle, near-puzzle, puzzle]
columns: model predictions [predicted non-puzzle, predicted puzzle]
shape:   3x2
```

This matters because a normal binary confusion matrix can hide the most important failure mode. A model can look accurate while still calling too many near-puzzles puzzles.

The most important diagnostic rates are:

- random non-puzzle false-positive rate
- near-puzzle false-positive rate
- verified puzzle recall
- binary F1
- PR AUC
- accuracy, but only as supporting context

Accuracy alone is not enough because the binary split has two non-puzzle rows for every puzzle row. On the current balanced-by-source split, validation and test are 30,000 non-puzzle targets versus 15,000 puzzle targets.

## Training Data

The data came from local CRTK exports made from the available chess stack files.

Canonical exported JSONL:

```text
data/exported/crtk_training_20260419_180229.jsonl
```

Canonical imported Parquet:

```text
data/processed/crtk_training_20260419_180229_fast.parquet
```

Full imported usable rows:

```text
45,002,737
```

Full label distribution:

| Source class | Meaning | Rows |
| ---: | --- | ---: |
| `0` | known non-puzzle | 33,887,586 |
| `1` | verified near-puzzle | 9,287,848 |
| `2` | verified puzzle | 1,827,303 |

The full import report is:

```text
data/reports/crtk_fast_import_20260420_041947.md
```

Current canonical benchmark split:

```text
data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet
```

Split size:

| Split | Random/non-puzzle | Near-puzzle | Puzzle | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 120,000 | 120,000 | 120,000 | 360,000 |
| val | 15,000 | 15,000 | 15,000 | 45,000 |
| test | 15,000 | 15,000 | 15,000 | 45,000 |

After binary mapping, each split has two non-puzzle source groups and one puzzle source group.

## Baseline Architectures

The benchmark currently uses four baseline families:

- Stockfish-style NNUE, used as a practical chess-specific baseline
- plain MLP, used as a weak but simple dense baseline
- plain CNN, used as a spatial baseline
- LC0 BT4-style residual tower, used as the strongest current benchmark baseline

Historical corrected benchmark report:

```text
reports/archive/benchmark_reports_20260424/network_signal_puzzle_binary_benchmark_20260424.md
```

Those numbers are useful orientation, but final claims should rerun the LC0 BT4, NNUE, and candidate configs on the canonical tagged split with the current paper-ready protocol.

Historical test ranking:

| Model | Test F1 | Test PR AUC | Test Acc | Near-puzzle -> puzzle FP | Puzzle recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| LC0 BT4 tower | 0.7445 | 0.8068 | 0.8183 | 0.2477 | 0.7943 |
| Stockfish-style NNUE | 0.7340 | 0.7982 | 0.7988 | 0.3158 | 0.8327 |
| CNN | 0.6823 | 0.7026 | 0.7606 | 0.3361 | 0.7714 |
| MLP | 0.6503 | 0.7089 | 0.6930 | 0.4907 | 0.8562 |

The LC0 BT4-style tower is the reference baseline to beat. A new architecture should aim to beat its F1 and PR AUC while reducing the near-puzzle false-positive rate under a matched current rerun.

## What We Are Trying To Learn

The experiment is meant to answer these questions:

- Can a network detect puzzle-ness as a position-level signal?
- Which architecture family extracts that signal most reliably?
- Which models overfit to superficial puzzle-like patterns?
- Can a new architecture beat a strong chess-shaped baseline without relying on search?
- Does a model that handles near-puzzles well also look like it understands tactical structure better?

The near-puzzle row is the central pressure test. If a model sees every sharp-looking position as a puzzle, it has not learned the right concept.

## User Intuition

The core intuition is:

> Being able to distinguish real chess puzzles from very similar near-puzzles should correlate with a chess neural network's ability to understand tactical and positional patterns. That may relate strongly to playing strength.

I think this intuition is mostly right, with one important boundary.

It is right as a benchmark for tactical position understanding. A real puzzle usually depends on concrete tension: threats, forcing moves, king safety, overloaded defenders, pins, mating nets, tactic geometry, or a decisive material swing. A near-puzzle can share many surface features while missing the decisive tactical fact. So separating puzzles from near-puzzles should reward networks that model chess structure instead of shallow board texture.

It is not a complete benchmark for playing strength. Playing strength also depends on move choice, search, long-term evaluation, quiet positional advantages, endgames, policy calibration, value calibration, time management, and tactical defense. A model can be good at puzzle detection without being a strong engine, and a strong engine component may need more than puzzle detection.

The best framing is:

```text
puzzle-vs-near-puzzle classification is a strong tactical-understanding benchmark,
not a complete Elo benchmark.
```

## How Realistic The Current Benchmark Is

The current implementation is realistic enough to be useful as a first architecture benchmark:

- it uses real local CRTK puzzle-derived data
- it trains the correct single-logit binary target
- it keeps random positions, near-puzzles, and true puzzles visible in diagnostics
- it saves metrics, predictions, plots, checkpoints, and reports
- it compares all models through the same trainer and split

It is not yet the full final benchmark:

- current training uses a 450,000-position sample, not all 45,002,737 rows
- current runs are short and small
- the BT4 tensor is FEN-only; history planes are zero until move history is available
- the NNUE baseline is Stockfish-style, not a production Stockfish network and feature pipeline
- there are not yet repeated seeds, larger sweeps, threshold sweeps, or full-dataset streaming runs
- the local labels and counts were checked, but the upstream CRTK generation process has not been independently audited here

## Success Criteria For New Architectures

A promising new architecture should beat the LC0 BT4-style tower on the same split.

Minimum target:

```text
test F1    > 0.7445
test PR AUC > 0.8068
```

Preferred target:

```text
near-puzzle -> puzzle false-positive rate < 0.20
puzzle recall >= 0.78
```

The best models should improve the hard-negative row without simply becoming too conservative and missing real puzzles.
