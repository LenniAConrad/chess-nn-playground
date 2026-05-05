#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()


IDEA_DIR_RE = re.compile(r"^i\d{3}_.+")
DATE_RE = re.compile(r"chess_nn_research_(\d{4}-\d{2}-\d{2})_")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _first_paragraph_after_heading(text: str, heading: str) -> str:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == heading:
            start = idx + 1
            break
    if start is None:
        return ""
    paragraph: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("#"):
            break
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("```"):
            break
        paragraph.append(stripped)
        if len(" ".join(paragraph)) > 260:
            break
    return " ".join(paragraph)


def _one_line(text: str, max_len: int = 170) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _packet_status(path: Path, title: str, text: str) -> str:
    name = path.name
    lowered_title = title.lower()
    lowered_head = text[:2000].lower()
    if name == "Pasted markdown.md":
        return "prompt snapshot"
    if name.startswith("deep-research-report"):
        return "link stub"
    if "(1)" in name or "(2)" in name:
        return "duplicate import"
    if title.startswith("Codex Handoff Packet:"):
        return "handoff packet"
    if (
        "architecture_batch" in name
        or "batch" in name
        or "research batch" in lowered_title
        or "architecture batch" in lowered_title
        or "targeted architecture batch" in lowered_head
        or "new candidate ranking" in lowered_head
    ):
        return "batch packet"
    if "synthesis" in text[:2000].lower() or "top3" in name or "best_expansions" in name:
        return "synthesis packet"
    return "research packet"


def _packet_name(title: str, path: Path) -> str:
    if path.name == "Pasted markdown.md":
        return "Prompt snapshot"
    if path.name.startswith("deep-research-report"):
        return path.stem
    prefixes = [
        "Codex Handoff Packet:",
        "Codex Research Packet:",
        "Research Packet:",
    ]
    for prefix in prefixes:
        if title.startswith(prefix):
            return title[len(prefix) :].strip()
    if title:
        return title
    return path.stem.replace("_", " ")


def _tags(name: str, path: Path, summary: str) -> list[str]:
    haystack = f"{name} {path.name} {summary}".lower()
    rules = [
        ("sheaf", ["sheaf", "hodge"]),
        ("move-delta", ["move-delta", "move delta", "move_landscape"]),
        ("transport", ["transport", "sinkhorn", "optimal-transport"]),
        ("symmetry", ["orbit", "automorphism", "color-flip", "reynolds"]),
        ("tempo", ["tempo", "null-move", "tempo-odd"]),
        ("topology", ["topology", "betti", "euler", "percolation"]),
        ("king-path", ["king-cage", "king cage", "king escape", "escape path"]),
        ("logic", ["logic", "clause", "datalog", "psl", "hinge"]),
        ("grammar", ["grammar", "automaton", "language"]),
        ("linear-algebra", ["eigen", "spectrum", "gramian", "tucker", "tensor", "procrustes", "polar", "krylov", "svd"]),
        ("convex", ["convex", "zonotope", "dykstra", "projection", "polar"]),
        ("robustness", ["dro", "robust", "huber", "contamination"]),
        ("information", ["information", "surprisal", "fisher", "geodesic", "codec", "entropy"]),
        ("sparse", ["sparse", "witness", "pursuit", "dictionary"]),
        ("graph", ["graph", "hypergraph", "non-backtracking"]),
        ("calibration", ["calibration", "abstention", "veto", "selective"]),
        ("puzzle-binary", ["puzzle_binary", "near-puzzle", "verified puzzle"]),
    ]
    return [tag for tag, needles in rules if any(needle in haystack for needle in needles)]


def discover_registered_ideas(ideas_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for folder in sorted(ideas_root.iterdir()):
        if not folder.is_dir() or not IDEA_DIR_RE.match(folder.name):
            continue
        idea_yaml = folder / "idea.yaml"
        if not idea_yaml.exists():
            continue
        data = yaml.safe_load(idea_yaml.read_text(encoding="utf-8")) or {}
        rows.append(
            {
                "idea_id": data.get("idea_id", folder.name[:4]),
                "name": data.get("name", folder.name),
                "slug": data.get("slug", folder.name[5:]),
                "status": data.get("status", ""),
                "implementation_status": data.get("implementation_status", ""),
                "target_task": data.get("target_task", ""),
                "short_thesis": data.get("short_thesis", ""),
                "folder": folder.as_posix(),
                "latest_result_path": data.get("latest_result_path"),
            }
        )
    return rows


def merge_registry_metadata(rows: list[dict[str, Any]], registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.exists():
        return rows
    registry_rows: list[dict[str, Any]] = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            registry_rows.append(json.loads(line))
    by_id = {str(row.get("idea_id") or ""): row for row in registry_rows}
    merged: list[dict[str, Any]] = []
    for row in rows:
        metadata = by_id.get(str(row.get("idea_id") or ""), {})
        merged.append({**metadata, **row})
    return merged


def discover_research_packets(packets_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(packets_root.glob("*.md")):
        if path.name in {"README.md", "CATALOG.md"}:
            continue
        text = _read_text(path)
        title = _first_heading(text)
        name = _packet_name(title, path)
        date_match = DATE_RE.search(path.name)
        summary = (
            _first_paragraph_after_heading(text, "## 2. Executive Selection")
            or _first_paragraph_after_heading(text, "## Executive Selection")
            or _first_paragraph_after_heading(text, "## Purpose")
            or _first_paragraph_after_heading(text, "### Thesis")
        )
        summary = _one_line(summary)
        status = _packet_status(path, title, text)
        rows.append(
            {
                "file": path.name,
                "path": path.as_posix(),
                "date": date_match.group(1) if date_match else "",
                "name": name,
                "status": status,
                "tags": _tags(name, path, summary),
                "summary": summary,
            }
        )
    return rows


def _md_link(path: str, label: str) -> str:
    if any(char in path for char in " ()"):
        return f"[{label}](<{path}>)"
    return f"[{label}]({path})"


def _format_idea_id_batches(rows: list[dict[str, Any]], batch_size: int = 24) -> list[str]:
    if not rows:
        return ["- none"]
    lines: list[str] = []
    for batch_index, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[start : start + batch_size]
        ids = ", ".join(f"`{row['idea_id']}`" for row in batch)
        lines.append(f"- Batch {batch_index}: {ids}")
    return lines


def _slugify_name(text: str) -> str:
    text = text.lower().replace("ö", "o")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def build_index_md(registered: list[dict[str, Any]], packets: list[dict[str, Any]]) -> str:
    status_counts = Counter(row["status"] for row in packets)
    tag_counts = Counter(tag for row in packets for tag in row["tags"])
    registered_impl = Counter(row["implementation_status"] for row in registered)
    lines: list[str] = [
        "# Ideas Index",
        "",
        "This is the navigation file for the `ideas/` workspace. It separates implementable registered ideas from raw research packets so future Codex sessions can move directly from research to code.",
        "",
        "## What Goes Where",
        "",
        "| Path | Role | Edit policy |",
        "|---|---|---|",
        "| `ideas/i###_*` | Registered ideas with a fixed documentation scaffold and implementation placeholders. | Update when promoting, implementing, benchmarking, or rejecting an idea. |",
        "| `ideas/registry.jsonl` | Machine-readable list of registered ideas. | Append/update only for registered ideas, not raw packets. |",
        "| `ideas/TODO.md` | Execution checklist with implementation state, performance state, and next action. | Regenerate after changing idea status or packet imports. |",
        "| `ideas/research_packets/` | Raw imported or generated research handoff packets. | Keep packet files immutable except filename/metadata normalization; use catalogs for organization. |",
        "| `ideas/idea_template/` | Scaffold for a future registered idea folder. | Keep as template only. |",
        "| `ideas/BENCHMARK_REPORTING.md` | Required aggregate and slice-level reporting standard. | Update when benchmark metadata or report artifacts change. |",
        "| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Deep Research prompt with duplicate-memory rules. | Update after importing meaningful new packets. |",
        "| `ideas/idea_generation_prompt.md` | Generated prompt from `scripts/ideas/build_idea_prompt.py`. | Do not hand-edit; regenerate. |",
        "",
        "## Current Counts",
        "",
        f"- Registered idea folders: `{len(registered)}`",
        f"- Research packet files cataloged: `{len(packets)}`",
        f"- Registered implementation states: `{dict(sorted(registered_impl.items()))}`",
        f"- Research packet statuses: `{dict(sorted(status_counts.items()))}`",
        "",
        "## Registered Ideas",
        "",
        "| ID | Idea | Status | Implementation | Target |",
        "|---|---|---|---|---|",
    ]
    for row in registered:
        target = _one_line(str(row["target_task"]), 90)
        folder_link = row["folder"].removeprefix("ideas/")
        lines.append(
            f"| `{row['idea_id']}` | {_md_link(folder_link, row['name'])} | `{row['status']}` | `{row['implementation_status']}` | {target} |"
        )
    lines.extend(
        [
            "",
        "## Research Packet Map",
        "",
        f"- Execution TODO: {_md_link('TODO.md', 'TODO.md')}",
        f"- Human catalog: {_md_link('research_packets/CATALOG.md', 'research_packets/CATALOG.md')}",
            f"- Machine catalog: {_md_link('research_packets/CATALOG.jsonl', 'research_packets/CATALOG.jsonl')}",
            f"- Import memory and family warnings: {_md_link('research_packets/README.md', 'research_packets/README.md')}",
            "",
            "Most frequent packet tags:",
            "",
            "| Tag | Count |",
            "|---|---:|",
        ]
    )
    for tag, count in tag_counts.most_common(14):
        lines.append(f"| `{tag}` | {count} |")
    lines.extend(
        [
            "",
            "## Recommended Work Loop",
            "",
            "1. Pick one research packet or registered idea and read only its packet plus this index.",
            "2. If the source is a packet, promote it into the next `ideas/i###_*` folder using `ideas/idea_template/`.",
            "3. Update `ideas/registry.jsonl` only after the promoted folder has the complete scaffold.",
            "4. Implement the model in `src/chess_nn_playground/models/`, not inside the idea folder, unless it is still a stub.",
            "5. Add a config under `configs/benchmarks/<task>/` or keep an idea-local `config.yaml`, then add a focused smoke test before training.",
            "6. Run the benchmark suite, then update the idea folder with result links and status.",
            "",
            "For detailed steps, see `ideas/WORKFLOW.md`.",
            "",
            "Generated by `PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py`.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_catalog_md(packets: list[dict[str, Any]]) -> str:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in packets:
        by_date[row["date"] or "undated"].append(row)
    lines: list[str] = [
        "# Research Packet Catalog",
        "",
        "This generated catalog is the fast index for `ideas/research_packets/`. Raw packet files remain in place; this file gives future Codex sessions enough metadata to choose what to read next.",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted(Counter(row["status"] for row in packets).items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## Packets By Date", ""])
    for date in sorted(by_date.keys(), reverse=True):
        rows = sorted(by_date[date], key=lambda row: row["file"])
        lines.extend(
            [
                f"### {date}",
                "",
                "| Packet | Status | Tags | Summary |",
                "|---|---|---|---|",
            ]
        )
        for row in rows:
            tags = ", ".join(f"`{tag}`" for tag in row["tags"]) or "-"
            lines.append(
                f"| {_md_link(row['file'], row['name'])} | `{row['status']}` | {tags} | {row['summary'] or '-'} |"
            )
        lines.append("")
    lines.append("Generated by `PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py`.\n")
    return "\n".join(lines)


def _next_action_for_registered(row: dict[str, Any]) -> str:
    impl = str(row.get("implementation_status") or "")
    status = str(row.get("status") or "")
    result = row.get("latest_result_path")
    if status == "rejected":
        return "Keep for duplicate prevention; do not implement unless the thesis changes."
    if status == "tested" and row.get("slug") == "vetoselect_positive_claim_abstention":
        return "A3 texture run is the current VetoSelect result; repeat seeds or move to the next architecture candidate."
    if status == "tested":
        return "Compare ablations and decide whether to refine, scale, or archive."
    if impl == "documented_not_implemented":
        return "Implement only if selected as the next coding target; start with model, config, and smoke test."
    if impl and impl != "documented_not_implemented" and not result:
        return "Run paper-grade benchmark, generate slice reports, add a run note, and link the result."
    return "Review status and update idea.yaml."


def _priority_for_packet(
    row: dict[str, Any],
    promoted_slugs: set[str] | None = None,
    source_packet_counts: dict[str, int] | None = None,
) -> tuple[str, str]:
    promoted_slugs = promoted_slugs or set()
    source_packet_counts = source_packet_counts or {}
    name = row["name"].lower()
    status = row["status"]
    if status in {"duplicate import", "link stub", "prompt snapshot"}:
        return ("skip", "No implementation action; keep only for provenance/duplicate prevention.")
    promoted_from_source = source_packet_counts.get(str(row.get("path") or ""), 0)
    if promoted_from_source:
        if status == "batch packet":
            return (
                "skip",
                f"Already mined into `{promoted_from_source}` registered candidate ideas; keep batch packet for provenance.",
            )
        return ("skip", "Already promoted as a registered idea; keep packet for provenance.")
    slug = _slugify_name(row["name"])
    if slug in promoted_slugs:
        return ("skip", f"Already promoted as registered idea `{slug}`; keep packet for provenance.")
    if "vetoselect" in name and "vetoselect_positive_claim_abstention" in promoted_slugs:
        return ("skip", "Already promoted and tested as `i011_vetoselect_positive_claim_abstention`; keep packet for provenance.")
    if "vetoselect" in name:
        return ("next", "Promote to `i011_*`; strongest fit for near-puzzle false-positive rejection.")
    if "dykstra" in name and "dykstra_lcp" in promoted_slugs:
        return ("skip", "Already promoted as `i012_dykstra_lcp`; keep packet for provenance.")
    if "dykstra" in name:
        return ("next", "Promote after VetoSelect or as the high-novelty constraint-solver candidate.")
    if "sparse relation pursuit" in name and "sparse_relation_pursuit_asymmetry" in promoted_slugs:
        return ("skip", "Already promoted as `i013_sparse_relation_pursuit_asymmetry`; keep packet for provenance.")
    if "sparse relation pursuit" in name:
        return ("next", "Promote as an interpretable relation-token sparse-coding candidate.")
    if "contamination-dro" in name and "contamination_dro_huber_tail_rejection" in promoted_slugs:
        return ("skip", "Already promoted as `i014_contamination_dro_huber_tail_rejection`; keep packet for provenance.")
    if "material-locked" in name and "material_locked_tactical_dro" in promoted_slugs:
        return ("skip", "Already promoted as `i015_material_locked_tactical_dro`; keep packet for provenance.")
    if "soft sorting" in name and "soft_sorting_order_residual_ranker" in promoted_slugs:
        return ("skip", "Already promoted as `i016_soft_sorting_order_residual_ranker`; keep packet for provenance.")
    if "conditional surprisal" in name and "conditional_surprisal_gate" in promoted_slugs:
        return ("skip", "Already promoted as `i017_conditional_surprisal_gate`; keep packet for provenance.")
    if "soft sorting" in name:
        return ("near-term", "Consider as an objective-only baseline before heavier architectures.")
    if "conditional surprisal" in name:
        return ("near-term", "Consider if prioritizing information bottlenecks.")
    if "material-locked" in name or "dro" in name:
        return ("near-term", "Consider if prioritizing hard-negative robustness objectives.")
    if status == "handoff packet":
        return ("backlog", "Review for promotion only after current `next` candidates are tested.")
    if status == "batch packet":
        return ("mine", "Mine for a single candidate; do not implement the whole batch.")
    if status == "synthesis packet":
        return ("reference", "Use for prioritization context, not direct implementation.")
    return ("backlog", "Keep cataloged; review if it matches a future coding objective.")


def build_todo_md(registered: list[dict[str, Any]], packets: list[dict[str, Any]]) -> str:
    promoted_slugs = {str(row.get("slug") or "") for row in registered}
    source_packet_counts = Counter(
        str(row.get("source_packet_path") or "")
        for row in registered
        if row.get("source_packet_path") and row.get("source_packet_candidate")
    )
    vetoselect = next((row for row in registered if row.get("slug") == "vetoselect_positive_claim_abstention"), None)
    dykstra = next((row for row in registered if row.get("slug") == "dykstra_lcp"), None)
    srpa = next((row for row in registered if row.get("slug") == "sparse_relation_pursuit_asymmetry"), None)
    trainable_unrun = [
        row
        for row in registered
        if row.get("implementation_status") not in {"", None, "documented_not_implemented"} and not row.get("latest_result_path")
    ]
    priority_order = [
        "tactical_equilibrium_network",
        "neural_proof_number_search",
        "sparse_relation_pursuit_asymmetry",
        "null_move_contrast_puzzle_network",
        "boundary_edit_lagrangian_network",
        "response_minimax_classifier",
        "puzzle_obligation_flow_network",
        "proof_core_set_verifier",
        "factor_agreement_classifier",
        "chess_operator_basis_classifier",
        "rule_consistent_latent_dynamics",
    ]
    priority_index = {slug: idx for idx, slug in enumerate(priority_order)}
    benchmark_queue = sorted(
        trainable_unrun,
        key=lambda row: (priority_index.get(str(row.get("slug") or ""), len(priority_index)), str(row.get("idea_id") or "")),
    )
    benchmark_queue_lines = _format_idea_id_batches(benchmark_queue)
    tested_results = [row for row in registered if row.get("latest_result_path")]
    dykstra_action = (
        "Compare/refine `Soft-Dykstra Latent Constraint Projector` against the latest linked run."
        if dykstra
        else "Promote `Soft-Dykstra Latent Constraint Projector` as the next high-novelty architecture candidate."
    )
    if srpa and not srpa.get("latest_result_path"):
        srpa_action = "Run the first benchmark for `Sparse Relation Pursuit Asymmetry` and add its result link."
    elif srpa:
        srpa_action = "Compare/refine `Sparse Relation Pursuit Asymmetry` against the latest linked run."
    else:
        srpa_action = "Promote `Sparse Relation Pursuit Asymmetry` as the interpretable relation-token candidate."
    if benchmark_queue:
        result = vetoselect.get("latest_result_path") if vetoselect else None
        recommendation_lines = [
            "Stop implementing new architectures for now. All registered idea folders are trainable; the next useful work is benchmark execution and falsification.",
            "",
            "Current execution state:",
            "",
            f"- Registered idea folders: `{len(registered)}`",
            f"- Ideas with linked results: `{len(tested_results)}`",
            f"- Implemented ideas still needing a linked benchmark run: `{len(benchmark_queue)}`",
            "",
            "Recommended immediate sequence:",
            "",
            "1. Materialize the resumable paper-ready plan with `PYTHONDONTWRITEBYTECODE=1 python scripts/run_paper_ready_all.py --dry-run`.",
            "2. Run paper-grade benchmarks for unrun implemented ideas in the batches below.",
            "",
            "Benchmark queue by ID:",
            "",
            *benchmark_queue_lines,
            "",
            "3. For each completed run, add `idea.yaml.latest_result_path`, write a run note under `runs/`, and generate `slice_report_val.md` plus `slice_report_test.md`.",
            "4. Run a matched promotion suite for LC0 BT4, NNUE, `i013`, `i005`, and `i009` under the same convergence budget and seeds `42`, `43`, `44`; use mean/std plus slice reports before calling any result the new best.",
            "5. Compare every result against the LC0 BT4, VetoSelect, and Dykstra baselines before changing architectures again.",
            "",
            "Use `reports/paper_ready_all/state.json` as the resume ledger if the all-runner is interrupted.",
            "Open `reports/paper_ready_all/status.md` for task state, then `reports/paper_ready_all/paper_report.pdf` for the paper-style summary once analysis finishes.",
        ]
        if result:
            recommendation_lines.extend(["", f"Latest VetoSelect run: `{result}`."])
    elif vetoselect:
        result = vetoselect.get("latest_result_path")
        if result and "v2_texture" in str(result):
            recommendation_lines = [
                "Stop generating new ideas for now. The backlog is large enough; the next useful work is implementation and falsification.",
                "",
                "Recommended immediate sequence:",
                "",
                "1. Treat `VetoSelect Positive-Claim Abstention` A3 as the current best tested VetoSelect variant; repeat seeds only if deciding whether to scale it.",
                f"2. {dykstra_action}",
                f"3. {srpa_action}",
            ]
        else:
            recommendation_lines = [
                "Stop generating new ideas for now. The backlog is large enough; the next useful work is implementation and falsification.",
                "",
                "Recommended immediate sequence:",
                "",
                "1. Review/refine `VetoSelect Positive-Claim Abstention` only via a focused A3 rule-texture decoy run; the first board-only A2 run is linked below and is not a baseline win.",
                f"2. {dykstra_action}",
                f"3. {srpa_action}",
            ]
        if result:
            recommendation_lines.extend(["", f"Latest VetoSelect run: `{result}`."])
    else:
        recommendation_lines = [
            "Stop generating new ideas for now. The backlog is large enough; the next useful work is implementation and falsification.",
            "",
            "Recommended immediate sequence:",
            "",
            "1. Promote `VetoSelect Positive-Claim Abstention` into `ideas/i011_vetoselect_positive_claim_abstention/` and implement a small model/head.",
            "2. If VetoSelect is too objective/head-specific, promote `Soft-Dykstra Latent Constraint Projector` as the high-novelty architecture candidate.",
            "3. Promote `Sparse Relation Pursuit Asymmetry` as the interpretable relation-token candidate.",
        ]
    if tested_results:
        performance_status = "Registered idea performance is now tracked per row via result paths; treat informal notes as secondary to linked artifacts."
    else:
        performance_status = "Current idea performance status: no registered ideas or raw research-packet ideas have been implemented and benchmarked yet. Every idea-specific performance cell is therefore `not run` until a result path is linked."
    lines: list[str] = [
        "# Idea TODO",
        "",
        "This is the execution checklist for the idea backlog. It covers registered idea folders and raw research packets. Update it by running:",
        "",
        "```bash",
        "PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py",
        "```",
        "",
        "## Current Recommendation",
        "",
        *recommendation_lines,
        "",
        "Canonical benchmark split: `data/splits/crtk_sample_3class_unique_crtk_tags/`. Every new run must include the slice reporting required by `ideas/BENCHMARK_REPORTING.md`.",
        "",
        "Baseline to beat on `puzzle_binary`: rerun the LC0 BT4-style classifier on the canonical clean tagged split before making final comparisons. Older leaderboard numbers came from earlier split artifacts and are useful only as rough orientation.",
        "",
        performance_status,
        "",
        "## Registered Ideas",
        "",
        "These are scaffolded ideas. They are the only ideas that should be implemented directly.",
        "",
        "| Done | ID | Idea | Implemented? | Performance | Next action |",
        "|---|---|---|---|---|---|",
    ]
    for row in registered:
        implemented = "no" if row["implementation_status"] == "documented_not_implemented" else "yes"
        performance = "not run" if not row.get("latest_result_path") else f"see `{row['latest_result_path']}`"
        done = "[ ]" if implemented == "no" else "[x]"
        folder_link = row["folder"].removeprefix("ideas/")
        lines.append(
            f"| {done} | `{row['idea_id']}` | {_md_link(folder_link, row['name'])} | {implemented} | {performance} | {_next_action_for_registered(row)} |"
        )

    priority_groups = ["next", "near-term", "backlog", "mine", "reference", "skip"]
    packet_rows = []
    for row in packets:
        priority, action = _priority_for_packet(row, promoted_slugs, dict(source_packet_counts))
        packet_rows.append({**row, "priority": priority, "next_action": action})

    lines.extend(
        [
            "",
            "## Research Packet Backlog",
            "",
            "Raw packets below are kept for provenance, synthesis context, or duplicate prevention. Source packets that have been promoted are marked accordingly; code lives under registered `ideas/i###_*` folders.",
            "",
        ]
    )
    for priority in priority_groups:
        rows = [row for row in packet_rows if row["priority"] == priority]
        if not rows:
            continue
        title = {
            "next": "Next Candidates",
            "near-term": "Near-Term Candidates",
            "backlog": "Backlog Handoff Packets",
            "mine": "Batch Packets To Mine",
            "reference": "Synthesis/Reference Packets",
            "skip": "Skip/Provenance Only",
        }[priority]
        lines.extend(
            [
                f"### {title}",
                "",
                "| Done | Packet | Implemented? | Performance | Next action |",
                "|---|---|---|---|---|",
            ]
        )
        for row in rows:
            lines.append(
                f"| [ ] | {_md_link('research_packets/' + row['file'], row['name'])} | no | not run | {row['next_action']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Completion Rules",
            "",
            "- Mark a registered idea done only after its model/config/tests exist and a benchmark or explicit rejection is linked.",
            "- Mark a research packet done only after it is promoted, rejected as duplicate, or deliberately archived.",
            "- Always record performance as a result path plus core metrics, not as an informal note.",
            "- If an idea fails, keep it in the TODO as rejected so future research does not regenerate it.",
            "",
            "Generated by `PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build idea and research packet navigation catalogs.")
    parser.add_argument("--ideas-root", default="ideas")
    parser.add_argument("--packets-root", default="ideas/research_packets")
    parser.add_argument("--index-output", default="ideas/INDEX.md")
    parser.add_argument("--catalog-output", default="ideas/research_packets/CATALOG.md")
    parser.add_argument("--jsonl-output", default="ideas/research_packets/CATALOG.jsonl")
    parser.add_argument("--todo-output", default="ideas/TODO.md")
    args = parser.parse_args()

    ideas_root = Path(args.ideas_root)
    packets_root = Path(args.packets_root)
    registered = merge_registry_metadata(discover_registered_ideas(ideas_root), ideas_root / "registry.jsonl")
    packets = discover_research_packets(packets_root)

    Path(args.index_output).write_text(build_index_md(registered, packets), encoding="utf-8")
    Path(args.catalog_output).write_text(build_catalog_md(packets), encoding="utf-8")
    write_jsonl(packets, Path(args.jsonl_output))
    Path(args.todo_output).write_text(build_todo_md(registered, packets), encoding="utf-8")

    print(f"Wrote {args.index_output}")
    print(f"Wrote {args.catalog_output}")
    print(f"Wrote {args.jsonl_output}")
    print(f"Wrote {args.todo_output}")
    print(f"Registered ideas: {len(registered)}")
    print(f"Research packets: {len(packets)}")


if __name__ == "__main__":
    main()
