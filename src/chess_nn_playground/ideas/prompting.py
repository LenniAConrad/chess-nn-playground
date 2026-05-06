from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _read_registry(registry_path: str | Path) -> list[dict[str, Any]]:
    path = Path(registry_path)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"_parse_error": line})
    return entries


def _read_idea_yaml(path: Path) -> dict[str, Any]:
    idea_yaml = path / "idea.yaml"
    if not idea_yaml.exists():
        return {}
    with idea_yaml.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def discover_existing_ideas(ideas_root: str | Path = "ideas") -> list[dict[str, Any]]:
    root = Path(ideas_root)
    if not root.exists():
        return []
    ideas: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == "idea_template":
            continue
        if not (child / "idea.yaml").exists():
            continue
        data = _read_idea_yaml(child)
        ideas.append(
            {
                "folder": str(child),
                "idea_id": data.get("idea_id", child.name),
                "name": data.get("name"),
                "status": data.get("status"),
                "implementation_kind": data.get("implementation_kind"),
                "short_thesis": data.get("short_thesis"),
                "novelty_claim": data.get("novelty_claim"),
                "target_task": data.get("target_task"),
                "input_representation": data.get("input_representation"),
                "output_heads": data.get("output_heads"),
            }
        )
    return ideas


def build_idea_generation_prompt(
    registry_path: str | Path = "ideas/registry.jsonl",
    ideas_root: str | Path = "ideas",
) -> str:
    registry_entries = _read_registry(registry_path)
    idea_folders = discover_existing_ideas(ideas_root)
    existing_summary = {
        "registry_entries": registry_entries,
        "idea_folders": idea_folders,
    }

    return f"""# ChatGPT Research-Idea Discipline Prompt

You are helping with `chess-nn-playground`, a chess neural-network research lab.

Your job is not to produce a pile of vague ideas. Your job is to help develop a small number of original, testable, well-documented research ideas without repeating the same concept under new names.

## Current Project Constraints

- Do not fabricate class `1` or class `2` labels.
- Treat `candidate_1_or_2_unresolved` as an unresolved candidate pool, not as verified near-puzzles or verified puzzles.
- Do not use Stockfish scores, PVs, node counts, verification metadata, or source labels as neural-network input features.
- Keep raw, exported, processed, split, result, and report artifacts separate.
- Use the clean tagged benchmark split for training and reporting unless a run explicitly documents otherwise: `data/splits/crtk_sample_3class_unique_crtk_tags/`.
- The current baseline is the simple CNN under `src/chess_nn_playground/models/cnn.py`; it is not a novel idea.
- Do not implement a new architecture until the idea has a written thesis, novelty check, ablation plan, and falsification criteria.
- Treat `implementation_kind` as an honesty label. A `shared_probe_variant` is not a distinct bespoke architecture just because it has its own idea folder or registry key.

## Existing Idea State

Before proposing anything, read the existing registry and idea folders. Here is the current machine-readable summary:

```json
{json.dumps(existing_summary, indent=2, sort_keys=True)}
```

If this summary is empty, say that no real ideas are registered yet. Do not treat the template as an existing research idea.

## Anti-Repetition Rules

For every candidate idea:

1. Name the closest existing idea or baseline it resembles.
2. State the exact overlap in one sentence.
3. State the exact difference in one sentence.
4. Reject the candidate if the difference is only naming, hyperparameters, minor layer ordering, training schedule, or ordinary engineering cleanup.
5. Reject the candidate if it cannot be tested with the current label reality or with an explicitly documented future label requirement.
6. Prefer one sharply different idea over five variants of the same mechanism.

Use this duplicate test:

- Same input representation + same inductive bias + same target + same claimed advantage = duplicate.
- Same mechanism with different terminology = duplicate.
- Same architecture family with only depth/width/loss tweaks = baseline variant, not a novel idea.
- New hypothesis about why the model should separate positions + new measurable failure mode + new ablation = potentially worth documenting.

## Research Packet Memory

Treat these already-researched packet families as occupied territory unless the formal bottleneck and central falsifier are genuinely different:

- tactical sheaf, Hodge, attack-defense graph Laplacian, curvature, and tension-energy variants
- one-ply pseudo-legal move-delta bags, spectra, entropy/free-energy landscapes, and move-set pooling
- entropic optimal transport, Sinkhorn piece-target transport, transport imbalance, and material-null transport
- nuisance-vector residualization or orthogonal projection over material, phase, king, castling, and en-passant features
- ordinal evidence ladders, credal evidence heads, sparse witness-piece bottlenecks, ray-language automata, ANOVA/Mobius constellations, and static-geometry pseudo-likelihood ratios
- orbit quotient, side-to-move tempo odd/even intervention, rule-partition invariance, kinematic Lie commutators, and masked board surprise codecs
- cubical Euler/Betti topology, Hall-defect overload, king-cage/escape path dynamic programs, formal-concept closure, class-0 denoising score fields, and non-backtracking tactical walks
- defender timing schedules, discovered-ray switchboards, counterplay insolvency ledgers, pinned mobility nullspaces, tactical effective resistance, defender opportunity-cost auctions, role-counterfactual necessity probes, phase-specialist calibration mixtures, forced-target funnels, and tactical subgoal automata
- support-polar zonotope certificates, loop-frustration curvature, forcing-response front-door mediators, hypercut polynomials, robust tail DRO, material-locked tactical-mask DRO, Fisher-geodesic excess, typed hypergraph motif grammars, soft sorting residuals, sparse relation pursuit, Hall-defect zeta spectra, differentiable abstract interpretation, tactical radius filtrations, traced motif composition, conditional surprisal gates, bounded hinge logic, Tucker relation certificates, structured tactical latent inference, Dykstra constraint projection, and positive-claim abstention

Changing thresholds, hidden sizes, tensor ranks, relation labels, bucket counts, expert counts, number of shells, dictionary sizes, solver iteration counts, pooling statistics, or ablation names is not enough to make a duplicate family new.

## Idea Search Process

Work in this order:

1. Restate the actual task and available labels.
2. List the current baseline and what it can already test.
3. Identify a concrete weakness or blind spot in the baseline.
4. Generate candidate hypotheses, not architectures first.
5. For each hypothesis, ask what observation would make it false.
6. Only then sketch a minimal mechanism that tests the hypothesis.
7. Compare against existing registry entries and reject duplicates.
8. Select at most 1-3 strongest ideas.

Interesting ideas should usually have at least one of these properties:

- They isolate a chess-specific uncertainty that the CNN baseline cannot cleanly test.
- They make a falsifiable claim about position structure, label ambiguity, or generalization.
- They create a measurable distinction between unresolved candidates and known non-puzzles without leaking engine metadata.
- They reduce ambiguity in the research program rather than only increasing model complexity.
- They suggest a clean ablation that could disprove the claimed advantage.

## Math-Thesis Discipline

For each surviving idea, write the math before the implementation:

- Definitions: input space, labels, target distribution, allowed metadata, forbidden metadata.
- Assumptions: what must be true about the data for the idea to help.
- Claim: one precise statement of expected advantage.
- Mechanism: why the proposed computation should express the claimed signal.
- Proof sketch: what can actually be reasoned about.
- Not proven: list every empirical assumption and unresolved point.
- Counterexamples: positions or datasets where the idea should fail.
- Falsification test: the smallest experiment that would make you abandon or revise it.

Never present intuition as proof. If something is only a hypothesis, label it as a hypothesis.

## Required Output Format For A Future Idea

If asked to create or register an idea, produce content matching:

- `ideas/{{idea_id}}_{{idea_slug}}/idea.yaml`
- `math_thesis.md`
- `architecture.md`
- `implementation_notes.md`
- `trainer_notes.md`
- `ablations.md`
- `model.py`
- `train.py`
- `config.yaml`
- `report_template.md`
- `runs/`

The `report_template.md` must require more than the aggregate confusion matrix. It must include:

- aggregate validation/test metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md` generated from `scripts/reports/report_prediction_slices.py`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`, `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives for fine label `2`;
- confidence/calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and motifs;
- an idea-specific hypothesis stating which slices should improve, which should fail, and which ablation should erase the slice-level gain.

The `idea.yaml` must include:

- `idea_id`
- `name`
- `slug`
- `status`
- `created_at`
- `author`
- `short_thesis`
- `novelty_claim`
- `expected_advantage`
- `target_task`
- `input_representation`
- `output_heads`
- `compute_notes`
- `implementation_status`
- `implementation_kind`
- `trainer_entrypoint`
- `config_path`
- `model_path`
- `latest_result_path`
- `notes`

## Scoring Rubric

Score each candidate from 0 to 5:

- Novelty versus existing registry
- Clarity of mathematical thesis
- Testability with current or explicitly required future data
- Risk of forbidden leakage
- Simplicity of minimal experiment
- Strength of falsification criteria
- Expected information gain even if the idea fails

Do not pursue ideas with high novelty but low testability. Do not pursue ideas that require treating unresolved candidates as verified class `1` or class `2`.

## Response Style

Be strict. Prefer rejecting weak ideas over expanding them. Keep the number of active ideas small. When in doubt, improve the experiment design, data audit, or ablation plan before inventing another model.
"""
