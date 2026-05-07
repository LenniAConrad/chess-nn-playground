# Research Reassessment - 2026-05-06

This note records the research-direction discussion and local repo inspection from the 2026-05-06 session. It is meant as a short memory file for the XueTui chess project.

## Project Links

- GitHub: https://github.com/LenniAConrad/chess-nn-playground
- Cloud data share: https://cloud.tsinghua.edu.cn/d/2496398c66334d41a033/

## Context Inspected

The assessment was based on the active `chess-nn-playground` repo, the local CRTK repo, local data reports, sampled Parquet split statistics, current result summaries, and older related folders such as `chess-models`, `lc0-puzzle-finetune`, and thesis/project notes.

Important local docs and code inspected:

- `docs/puzzle_binary_benchmark_goal.md`
- `docs/reliable_training_protocol.md`
- `docs/experimental_training_pipeline.md`
- `ideas/README.md`
- `ideas/INDEX.md`
- `ideas/TODO.md`
- `reports/idea_conformance_audit.md`
- `results/leaderboard.md`
- `reports/paper_ready_all/status.md`
- `reports/paper_ready_all/timeline.md`
- CRTK mining/export docs and training-label export code

## Main Correction

The strongest part of the project is not "fine-tuning a chess engine on puzzles" and not "trying 200 architectures".

The strongest part is the benchmark:

> Can neural chess models distinguish verified tactical puzzles from same-parent near-puzzles that look very similar but destroy the forcing solution?

This is more interesting than ordinary puzzle classification because random non-puzzles are too easy. The real signal is whether the model rejects hard negatives that share the surface structure of a tactic.

## CRTK Label Meaning

The CRTK export defines the useful three-way structure:

- `fine_label=0`: known/random non-puzzle.
- `fine_label=1`: verified near-puzzle. This is a sibling child position from the same parent FEN as a verified puzzle, but it does not pass the puzzle filter.
- `fine_label=2`: verified puzzle. This passes the CRTK puzzle filter.

The current `puzzle_binary` task maps:

- fine labels `0` and `1` to binary `0`.
- fine label `2` to binary `1`.

So the important failure mode is:

> near-puzzle false positives while preserving puzzle recall.

Good language:

- same-parent near-puzzles
- sibling hard negatives
- tactical counterfactual hard negatives
- near-puzzle false positives at matched puzzle recall

Use "minimal edits" carefully. The current data gives sibling legal continuations from the same parent, not a formal proof of minimal edit distance. In sampled paired groups, puzzle/near-puzzle siblings were very close, with about 3.45 mean piece-square differences and median 4, so the similarity claim is plausible.

## Dataset And Split Strength

The data side is stronger than the initial verbal description suggested.

Current imported CRTK export:

- total rows: about 45M
- known non-puzzles: about 33.9M
- verified near-puzzles: about 9.3M
- verified puzzles: about 1.8M

Canonical sampled tagged split:

- train: 360k rows, balanced 120k per fine class
- validation: 45k rows, balanced 15k per fine class
- test: 45k rows, balanced 15k per fine class

The local readiness reports say there are no duplicate FEN leaks, no split-group leaks, and no label-conflicting FENs in the sampled split. Tags are reporting metadata only, not model input.

## Research Level Assessment

Current level:

- Strong undergraduate research if written cleanly.
- Master's-level potential with repeated seeds, ablations, pair-complete evaluation, and statistical reporting.
- Not PhD-level yet.

To become PhD-level, it would need one or more of:

- a formal method for generating/evaluating tactical counterfactual hard negatives;
- a clearly motivated architecture principle that survives ablations and repeated seeds;
- transfer from the benchmark to move prediction, tactical solving, search efficiency, or playing strength;
- evidence that the method generalizes beyond chess or captures a broader principle for symbolic decision domains.

Session rating:

- overall current project: about 7/10
- undergraduate project ceiling: about 9/10
- master's thesis potential: about 7.5-8.5/10 if cleaned up rigorously
- PhD-paper readiness right now: about 4-6/10

## Harsh Assessment

The "200 architectures" framing is risky. It sounds like fishing if presented as "I generated many architectures and will pick the best." In research terms, fishing means trying many hypotheses/configurations and only reporting the lucky winner.

Better framing:

> I used a larger architecture idea bank for exploration, then selected a small number of architecture families for controlled benchmarking.

The main story should be:

> A hard same-parent near-puzzle benchmark plus an architecture study of which inductive biases help neural models distinguish true forcing tactics from similar non-forcing positions.

This lets architecture work stay central without sounding random.

## Baseline Versus Architecture Story

Good research structure:

1. Standard puzzle classification is too easy because random non-puzzles are obvious.
2. Same-parent near-puzzles create a hard counterfactual benchmark.
3. Standard chess-shaped baselines still confuse many near-puzzles with true puzzles.
4. Candidate architectures test which inductive biases reduce that confusion.
5. Repeated seeds, ablations, fixed thresholds, and transfer tests decide whether the architecture claim is real.

Do not lead with the number 200. Lead with the benchmark and the small controlled architecture families.

## Professor Message Drafts

Suggested two-message WeChat version:

```text
Professor, quick update on the XueTui chess project before I spend much more compute. I am reframing it from "fine-tune a chess engine on puzzles" to a harder benchmark for tactical representation: can neural chess models distinguish verified tactical puzzles from same-parent near-puzzles?
```

```text
In my CRTK data, near-puzzles are sibling positions from the same parent FEN as real puzzles: they look very similar but fail the forcing-solution filter. The metric is near-puzzle false positives at high puzzle recall. I am comparing MLP/CNN/NNUE/LC0-style baselines with a small selected set of architectures; the larger idea bank is only for brainstorming, and final claims need repeated seeds/ablations. If a model works well here, the next step is transfer to move prediction, tactical solving, and possibly playing strength.

Code/data:
https://github.com/LenniAConrad/chess-nn-playground
https://cloud.tsinghua.edu.cn/d/2496398c66334d41a033/

Do you think this is a valid research direction?
```

## Queue Clarification

The full paper-ready dry-run state had 2331 tasks.

The full ordering found in `reports/paper_ready_all/timeline.md`:

- experiments `1-36`: current main puzzle-binary baselines: NNUE, MLP, CNN, LC0, with 3 seeds and 3 scales
- experiments `37-72`: optional fine-3class diagnostic baselines
- experiments `73-162`: old coarse-binary CNN/ResNet/LC0-static benchmark variants
- experiments `163-171`: SRPA benchmark/challenger runs
- experiments `172+`: registered idea architectures from `ideas/i001...`

The `73-162` block is mostly legacy queue pollution for the current research story. Those configs use `mode: coarse_binary`, which trains on `coarse_label`. That means puzzles and near-puzzles are both positive, so they do not directly test the current key question of verified puzzle versus near-puzzle hard negative.

The `163-171` SRPA block is not a neutral baseline. SRPA is `idea_i013` promoted into `configs/benchmarks/puzzle_binary/bench_srpa_lc0bt4.yaml` because it already looked promising. It should be treated as a candidate/challenger architecture.

Important caveat:

The local `reports/paper_ready_all/status.md` currently shows a `--limit 1 --no-analysis` dry run with only 9 tasks. If the remote machine is running that exact command, it will only run NNUE across 3 seeds and 3 scales and will never reach the actual idea architectures.

## Recommended Compute Plan

Do not spend a month blindly running the entire queue on one RTX 3070.

Recommended first pass:

1. Run only the main `puzzle_binary` baselines: NNUE, MLP, CNN, LC0 BT4-style.
2. Add SRPA/i013 as the first candidate architecture.
3. Use base scale first; add 3 seeds if runtime allows.
4. Skip old `coarse_binary` configs for the main research claim.
5. After the baseline/challenger comparison, choose a small number of architecture families for deeper runs.

Paper-grade next steps:

1. Build a pair-complete evaluation subset where each parent has at least one puzzle sibling and one near-puzzle sibling.
2. Report near-puzzle false positives at matched puzzle recall, for example recall 80%, 85%, and 90%.
3. Run repeated seeds for the strongest baseline and selected candidates.
4. Add ablations for the best candidate.
5. Test transfer to move prediction, tactical solving, and later possibly playing strength.

## Bottom Line

The better direction is not "train a new chess engine on puzzles." The better direction is:

> build a hard near-puzzle benchmark for tactical representation, then study which neural chess architectures can solve its central failure mode.

Tell the professor now rather than letting the job run for a month without alignment. The current run can be treated as preliminary triage, but it should not define the research story by itself.
