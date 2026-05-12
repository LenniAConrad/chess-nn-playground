#!/usr/bin/env python
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()


IMPLEMENTABLE_STATUSES = {"handoff packet", "research packet"}
CANONICAL_SPLIT = "data/splits/crtk_sample_3class_unique_crtk_tags"
EXISTING_PROMOTIONS = {
    "contamination_dro_huber_tail_rejection",
    "material_locked_tactical_mask_dro",
    "soft_sorting_order_residual_ranker",
    "sparse_relation_pursuit_asymmetry",
    "conditional_surprisal_gate",
    "soft_dykstra_latent_constraint_projector",
    "vetoselect_positive_claim_abstention",
}
SECTION_RE = re.compile(r"^##\s+(?:Candidate|Idea|Variant)\s+(\d+):\s+(.+?)\s*$", re.MULTILINE)


def _slugify(text: str) -> str:
    text = text.lower().replace("ö", "o")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")


def _next_id(existing: list[dict[str, Any]]) -> int:
    values = []
    for row in existing:
        match = re.fullmatch(r"i(\d{3})", str(row.get("idea_id", "")))
        if match:
            values.append(int(match.group(1)))
    return max(values, default=17) + 1


def _is_batch_like(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "")
    name = str(row.get("name") or "").lower()
    file_name = str(row.get("file") or "").lower()
    if status == "batch packet":
        return True
    return "batch" in name or "batch" in file_name or "challengers" in name


def _candidate_sections(row: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(row.get("path") or ""))
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(SECTION_RE.finditer(text))
    sections: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        rank = int(match.group(1))
        name = match.group(2).strip()
        section_start = match.end()
        section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[section_start:section_end]
        summary = _section_summary(section)
        sections.append({"rank": rank, "name": name, "summary": summary})
    return sections


def _section_summary(section: str) -> str:
    lines = section.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower() == "### thesis":
            parts: list[str] = []
            for candidate in lines[idx + 1 :]:
                stripped = candidate.strip()
                if stripped.startswith("#"):
                    break
                if not stripped:
                    if parts:
                        break
                    continue
                if stripped.startswith("```"):
                    break
                parts.append(stripped)
                if len(" ".join(parts)) > 260:
                    break
            return _short(" ".join(parts))
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("|") and not stripped.startswith("```"):
            return _short(stripped)
    return ""


def _promotable_rows(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet in packets:
        candidates = _candidate_sections(packet) if _is_batch_like(packet) else []
        if candidates:
            for candidate in candidates:
                rows.append(
                    {
                        **packet,
                        "name": candidate["name"],
                        "summary": candidate["summary"] or f"{candidate['name']} candidate from {packet['name']}.",
                        "source_packet_candidate": candidate["name"],
                        "source_packet_rank": candidate["rank"],
                        "source_packet_status": packet.get("status"),
                    }
                )
            continue
        if packet.get("status") in IMPLEMENTABLE_STATUSES:
            rows.append(packet)
    return rows


def _family_from_packet(name: str, tags: list[str]) -> str:
    haystack = f"{name} {' '.join(tags)}".lower()
    if "line" in haystack or "ray" in haystack or "stripe" in haystack:
        return "grammar"
    if "prototype" in haystack or "dictionary" in haystack or "codebook" in haystack:
        return "sparse"
    if "calibration" in haystack or "evidence" in haystack or "surprisal" in haystack:
        return "information"
    if "margin" in haystack or "rate" in haystack or "dro" in haystack or "robust" in haystack:
        return "robustness"
    if "mixer" in haystack or "scan" in haystack or "memory" in haystack:
        return "generic"
    if "forest" in haystack or "logic" in haystack or "clause" in haystack:
        return "logic"
    if "attention" in haystack or "slot" in haystack or "transformer" in haystack:
        return "graph"
    if "sheaf" in tags or "hodge" in haystack:
        return "sheaf"
    if "move-delta" in tags or "counterfactual" in haystack or "move" in haystack:
        return "move_delta"
    if "transport" in tags:
        return "transport"
    if "symmetry" in tags or "orbit" in haystack or "automorphism" in haystack:
        return "symmetry"
    if "king-path" in tags:
        return "king_path"
    if "topology" in tags or "euler" in haystack or "betti" in haystack or "percolation" in haystack:
        return "topology"
    if "logic" in tags or "lattice" in haystack or "hinge" in haystack:
        return "logic"
    if "grammar" in tags or "automaton" in haystack:
        return "grammar"
    if "linear-algebra" in tags or "spectrum" in haystack or "gramian" in haystack or "tucker" in haystack:
        return "linear_algebra"
    if "information" in tags or "surprisal" in haystack or "fisher" in haystack or "codec" in haystack:
        return "information"
    if "sparse" in tags or "witness" in haystack:
        return "sparse"
    if "graph" in tags or "hypergraph" in haystack:
        return "graph"
    if "convex" in tags or "zonotope" in haystack:
        return "convex"
    if "tempo" in tags:
        return "tempo"
    if "robustness" in tags or "dro" in haystack:
        return "robustness"
    return "generic"


def _short(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _model_py() -> str:
    return '''from __future__ import annotations

from typing import Any

from chess_nn_playground.models.research_packet_probe import ResearchPacketProbe
from chess_nn_playground.models.research_packet_probe import build_research_packet_probe_from_config


def build_model_from_config(config: dict[str, Any]) -> ResearchPacketProbe:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_research_packet_probe_from_config(model_cfg)
'''


def _train_py() -> str:
    return '''#!/usr/bin/env python
from __future__ import annotations

from chess_nn_playground.ideas.implementation import idea_train_cli


if __name__ == "__main__":
    idea_train_cli(__file__)
'''


def _config_yaml(idea_id: str, slug: str, family: str) -> dict[str, Any]:
    return {
        "idea_id": idea_id,
        "run": {"name": f"{idea_id}_{slug}_simple18", "output_dir": "results"},
        "seed": 42,
        "deterministic": True,
        "mode": "puzzle_binary",
        "device": "nvidia",
        "data": {
            "train_path": f"{CANONICAL_SPLIT}/split_train.parquet",
            "val_path": f"{CANONICAL_SPLIT}/split_val.parquet",
            "test_path": f"{CANONICAL_SPLIT}/split_test.parquet",
            "encoding": "simple_18",
            "cache_features": False,
        },
        "model": {
            "name": slug,
            "input_channels": 18,
            "num_classes": 1,
            "channels": 64,
            "hidden_dim": 96,
            "depth": 2,
            "dropout": 0.1,
            "use_batchnorm": True,
            "mechanism_family": family,
            "packet_profile": slug,
        },
        "training": {
            "epochs": 20,
            "batch_size": 256,
            "num_workers": "auto",
            "persistent_workers": True,
            "prefetch_factor": 2,
            "learning_rate": 0.0007,
            "weight_decay": 0.0001,
            "class_weighting": "balanced",
            "loss": "bce_with_logits",
            "early_stopping_patience": 5,
            "mixed_precision": True,
            "allow_tf32": True,
            "matmul_precision": "high",
            "reliability_tier": "paper_grade",
            "min_epochs": 10,
            "min_active_epochs": 10,
            "gradient_clip_norm": 1.0,
            "lr_scheduler": {
                "name": "reduce_on_plateau",
                "factor": 0.5,
                "patience": 2,
                "min_lr": 1.0e-5,
            },
        },
        "notes": "Promoted from a research packet as a board-only mechanism-profile implementation. CRTK metadata is reporting-only.",
    }


def _report_template(name: str, family: str) -> str:
    return f"""# Report Template

## Run

- Result path:
- Config:
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## Packet Diagnostics

- Mechanism family: `{family}`
- Packet auxiliary logit:
- Mechanism energy:
- Sheaf tension / transport imbalance / symmetry residual / topology pressure as applicable:
- Near-puzzle false positives:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `{name}` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior.
"""


def _write_idea_folder(idea_id: str, row: dict[str, Any], slug: str, family: str) -> None:
    folder = Path("ideas/registry") / f"{idea_id}_{slug}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "runs").mkdir(exist_ok=True)
    source_path = row["path"]
    source_candidate = row.get("source_packet_candidate")
    source_rank = row.get("source_packet_rank")
    name = row["name"]
    thesis = _short(row.get("summary") or f"{name} tests whether a {family} board mechanism improves puzzle-binary classification.")
    idea = {
        "idea_id": idea_id,
        "name": name,
        "slug": slug,
        "status": "scaffolded",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "author": "Codex",
        "short_thesis": thesis,
        "novelty_claim": f"Promoted from `{source_path}`; uses a {family} mechanism profile over board-only features rather than generic CNN-only pooling.",
        "expected_advantage": "Test the packet's proposed board-structure signal under the same puzzle_binary benchmark contract as the other ideas.",
        "target_task": "puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps to puzzle.",
        "input_representation": "Current-board simple_18 tensor only; CRTK/source metadata is reporting-only and never used as model input.",
        "output_heads": "One puzzle logit plus packet-profile diagnostics saved to prediction artifacts.",
        "compute_notes": "Compact convolutional trunk plus deterministic board-mechanism diagnostics selected by packet family.",
        "implementation_status": "probe_scaffold_only",
        "implementation_kind": "shared_probe_variant",
        "trainer_entrypoint": f"ideas/registry/{idea_id}_{slug}/train.py",
        "config_path": f"ideas/registry/{idea_id}_{slug}/config.yaml",
        "model_path": f"ideas/registry/{idea_id}_{slug}/model.py",
        "latest_result_path": None,
        "notes": f"Research-packet promotion from `{source_path}`. Scaffold-only ResearchPacketProbe wrapper; not a completed bespoke implementation of the markdown architecture. Do not benchmark or describe this folder as an implemented architecture until bespoke model code replaces the shared probe.",
        "source_packet_path": source_path,
        "source_packet_candidate": source_candidate,
        "source_packet_rank": source_rank,
        "source_packet_status": row.get("source_packet_status", row.get("status")),
        "mechanism_family": family,
    }
    (folder / "idea.yaml").write_text(yaml.safe_dump(idea, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (folder / "config.yaml").write_text(yaml.safe_dump(_config_yaml(idea_id, slug, family), sort_keys=False), encoding="utf-8")
    (folder / "model.py").write_text(_model_py(), encoding="utf-8")
    (folder / "train.py").write_text(_train_py(), encoding="utf-8")
    candidate_rank_note = f"Batch candidate rank: `{source_rank}`.\n\n" if source_candidate else ""
    candidate_impl_note = f"- Batch candidate: `{source_candidate}`.\n" if source_candidate else ""
    docs = {
        "math_thesis.md": f"# Math Thesis\n\n{name}\n\nSource packet: `{source_path}`.\n\n{candidate_rank_note}Working thesis: {thesis}\n\nThis folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not evidence that the bespoke architecture has been implemented.\n",
        "architecture.md": f"# Architecture\n\n## Scaffold-Only Implementation Notice\n\nThis folder is not a completed bespoke implementation of the architecture described below. `model.py` is a thin `ResearchPacketProbe` wrapper built with `build_research_packet_probe_from_config`, so this idea remains `implementation_kind: shared_probe_variant` and `implementation_status: probe_scaffold_only` until bespoke model code matching this markdown is added.\n\nThe current scaffold uses a compact board encoder, deterministic board statistics, and the `{family}` mechanism branch. It returns one puzzle logit plus diagnostics such as `packet_aux_logit`, `mechanism_energy`, `sheaf_tension`, `transport_imbalance`, `symmetry_residual`, `topology_pressure`, and `information_surprisal` when relevant.\n",
        "implementation_notes.md": f"# Implementation Notes\n\n- Central scaffold code: `src/chess_nn_playground/models/research_packet_probe.py`.\n- Registry key: `{slug}`.\n- Source packet: `{source_path}`.\n{candidate_impl_note}- This is not a completed bespoke architecture. Replace the shared probe with model code matching `architecture.md` before marking it implemented or benchmarking it as an architecture.\n",
        "trainer_notes.md": "# Trainer Notes\n\nThe guarded idea `train.py` will reject this folder while it remains `implementation_status: probe_scaffold_only`. Mark it trainable only after replacing the shared probe with bespoke registered model code.\n",
        "ablations.md": f"# Ablations\n\n- No architecture ablations are valid while this folder is scaffold-only.\n- After bespoke code exists, compare `{name}` against LC0 BT4, NNUE, and the strongest registered idea runs on the same split and seeds.\n",
        "report_template.md": _report_template(name, family),
    }
    for filename, text in docs.items():
        (folder / filename).write_text(text, encoding="utf-8")


def main() -> None:
    catalog_path = Path("ideas/research/packets/CATALOG.jsonl")
    registry_path = Path("ideas/registry/registry.jsonl")
    packets = _load_jsonl(catalog_path)
    registry = _load_jsonl(registry_path)
    existing_slugs = {str(row.get("slug", "")) for row in registry}
    existing_names = {str(row.get("name", "")).lower() for row in registry}
    next_id = _next_id(registry)
    added: list[dict[str, Any]] = []
    packet_model_names: list[str] = []
    for row in _promotable_rows(packets):
        name = str(row["name"])
        slug = _slugify(name)
        if slug in EXISTING_PROMOTIONS or slug in existing_slugs or name.lower() in existing_names:
            continue
        idea_id = f"i{next_id:03d}"
        next_id += 1
        family = _family_from_packet(name, list(row.get("tags") or []))
        _write_idea_folder(idea_id, row, slug, family)
        entry = {
            "idea_id": idea_id,
            "name": name,
            "slug": slug,
            "status": "implemented",
            "folder": f"ideas/registry/{idea_id}_{slug}",
            "target_task": "puzzle_binary",
            "short_thesis": _short(row.get("summary") or f"{name} research-packet promotion."),
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "implementation_kind": "shared_probe_variant",
            "source_packet_path": row["path"],
            "source_packet_candidate": row.get("source_packet_candidate"),
            "source_packet_rank": row.get("source_packet_rank"),
            "source_packet_status": row.get("source_packet_status", row.get("status")),
            "mechanism_family": family,
        }
        registry.append(entry)
        added.append(entry)
        packet_model_names.append(slug)
        existing_slugs.add(slug)
        existing_names.add(name.lower())
    _write_jsonl(registry_path, registry)
    all_packet_slugs = [
        str(row.get("slug"))
        for row in registry
        if str(row.get("folder", "")).startswith("ideas/registry/i") and Path(str(row.get("folder", ""))).exists()
        and str(row.get("source_packet_path", "")).startswith("ideas/research/packets/classic/")
    ]
    Path("src/chess_nn_playground/models/research_packet_registry.py").write_text(
        "from __future__ import annotations\n\n\n"
        f"RESEARCH_PACKET_MODEL_NAMES: list[str] = {all_packet_slugs!r}\n",
        encoding="utf-8",
    )
    print(f"Added {len(added)} research packet idea implementations")
    for entry in added:
        print(f"{entry['idea_id']} {entry['slug']} ({entry['mechanism_family']})")


if __name__ == "__main__":
    main()
