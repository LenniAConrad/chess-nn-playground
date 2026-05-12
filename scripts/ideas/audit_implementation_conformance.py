#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.ideas.schema import discover_idea_folders
from chess_nn_playground.models.research_packet_probe import PROFILE_NAMES
from chess_nn_playground.models.research_packet_probe import _profile_flags
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES


FAMILY_RULES: list[tuple[str, list[str]]] = [
    ("sheaf", ["sheaf", "hodge", "tension"]),
    ("move_delta", ["move_delta", "counterfactual", "one_ply", "null_move", "minimal_edit", "edit_distance"]),
    ("transport", ["transport", "sinkhorn", "assignment"]),
    ("symmetry", ["symmetry", "symmetric", "orbit", "automorphism", "mirror", "color_flip", "invariant", "commutative"]),
    (
        "topology",
        [
            "topology",
            "euler",
            "betti",
            "percolation",
            "curvature",
            "frustration",
            "filtration",
            "radius",
            "geometry",
            "neighborhood",
            "harmonic",
            "potential",
        ],
    ),
    ("king_path", ["king", "cage", "escape", "shelter"]),
    (
        "logic",
        [
            "logic",
            "clause",
            "resolution",
            "lattice",
            "hinge",
            "boolean",
            "matroid",
            "hall",
            "zeta",
            "concept",
            "formal_concept",
            "bisimulation",
            "fixed_point",
            "verifier",
            "soundness",
            "forest",
            "decision_forest",
            "conjunction",
            "multiplicative",
            "tropical",
            "circuit",
            "scratchpad",
            "disproof",
        ],
    ),
    ("grammar", ["grammar", "automaton", "ray", "line", "stripe", "walk", "program", "scan", "run_length"]),
    (
        "linear_algebra",
        [
            "spectrum",
            "spectral",
            "matrix",
            "pencil",
            "rank",
            "tucker",
            "tensor",
            "gramian",
            "hessian",
            "nullspace",
            "orthogonal",
            "displacement",
            "moment",
            "schur",
            "bispectral",
            "bitboard",
            "finite_field",
            "commutator",
            "determinantal",
            "grassmannian",
            "procrustes",
            "krylov",
            "resolvent",
            "parity",
            "syndrome",
            "wavelet",
            "tensorsketch",
            "spline",
            "invertible",
            "bilinear",
            "derivative",
            "curl",
            "divergence",
            "morphological",
            "replicator",
            "mobius",
            "constellation",
            "pivot",
            "trace",
            "elimination",
            "row_file",
            "factor",
            "maxout",
            "signature",
            "sylvester",
            "lyapunov",
            "complement",
            "bures",
            "wasserstein",
            "numerical",
            "range",
            "pfaffian",
            "skew",
            "padic",
            "ultrametric",
            "newton",
            "free_probability",
            "r_transform",
            "williamson",
            "symplectic",
            "magnus",
            "bch",
            "coupling",
            "series",
            "riccati",
            "hamiltonian",
            "clifford",
            "rotor",
            "multivector",
            "bivector",
            "tracy",
            "widom",
            "rmt",
            "spacing",
            "lindstrom",
            "gessel",
            "viennot",
            "determinant",
            "toda",
            "isospectral",
            "lax",
            "manakov",
        ],
    ),
    (
        "information",
        [
            "information",
            "surprisal",
            "surprise",
            "codec",
            "entropy",
            "fisher",
            "zobrist",
            "credal",
            "bayesian",
            "temperature",
            "evidence",
            "likelihood",
            "score_field",
            "sieve",
            "variational",
            "inference",
            "variance",
            "agreement",
        ],
    ),
    ("sparse", ["sparse", "witness", "prototype", "dictionary", "codebook", "expert", "capsule", "motif"]),
    (
        "graph",
        [
            "graph",
            "hypergraph",
            "relation",
            "attention",
            "slot",
            "transformer",
            "query",
            "effective_resistance",
            "markov",
            "token",
            "cross_stitch",
            "defense",
            "defender",
            "reply",
            "reaction",
            "counterplay",
            "safe_reply",
            "option",
            "front_door",
            "causal",
            "interaction",
            "tree",
            "hypernetwork",
        ],
    ),
    (
        "convex",
        [
            "convex",
            "zonotope",
            "projection",
            "submodular",
            "support_function",
            "barrier",
            "cut",
            "budget",
            "boundary",
            "distance",
            "liability",
            "opportunity",
            "funnel",
            "empty_square",
            "hypercut",
        ],
    ),
    ("tempo", ["tempo", "phase", "timing", "recurrent", "cellular", "iterative", "cascade", "early_exit"]),
    (
        "robustness",
        [
            "robust",
            "dro",
            "margin",
            "calibration",
            "source_rate",
            "disentangled",
            "negative_class",
            "stability",
            "dropout",
            "consensus",
        ],
    ),
]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _normalized_terms(text: str) -> tuple[str, set[str]]:
    normalized = str(text).strip().lower().replace("-", "_").replace(" ", "_")
    return f"_{normalized}_", {token for token in normalized.split("_") if token}


def _contains_term(haystack: str, tokens: set[str], term: str) -> bool:
    term = term.lower().replace("-", "_").replace(" ", "_")
    if "_" in term:
        return f"_{term}_" in haystack
    return term in tokens


def infer_family(*, slug: str, name: str, current: str | None = None) -> str:
    haystack, term_set = _normalized_terms(f"{slug}_{name}")
    for family, family_terms in FAMILY_RULES:
        if any(_contains_term(haystack, term_set, term) for term in family_terms):
            return family
    return "generic"


def active_profiles(slug: str, family: str) -> list[str]:
    flags = _profile_flags(slug, family).tolist()
    return [name for name, enabled in zip(PROFILE_NAMES, flags) if float(enabled) > 0.0]


def sync_registry(rows: list[dict[str, Any]], path: Path = Path("ideas/registry/registry.jsonl")) -> int:
    if not path.exists():
        return 0
    by_id = {str(row["idea_id"]): row for row in rows if row["packet_probe"]}
    changed = 0
    updated_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        row = by_id.get(str(entry.get("idea_id")))
        if row is not None and entry.get("mechanism_family") != row["family"]:
            entry["mechanism_family"] = row["family"]
            changed += 1
        updated_lines.append(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
    if changed:
        path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return changed


def _architecture_text(name: str, family: str, profiles: list[str]) -> str:
    profiles_text = ", ".join(f"`{item}`" for item in profiles)
    return (
        "# Architecture\n\n"
        f"`{name}` uses the shared proposal-conditioned research-packet probe.\n\n"
        f"- Mechanism family: `{family}`.\n"
        f"- Active proposal profiles: {profiles_text}.\n"
        "- Input: board tensor only; CRTK/source metadata remains reporting-only.\n"
        "- Board trunk: compact convolutional square encoder over the configured board planes.\n"
        "- Proposal diagnostics: deterministic board-mechanism features selected from the active profiles, including "
        "sheaf/pressure tension, transport imbalance, symmetry residuals, topology and king-path pressure, logic/ray "
        "evidence, linear-algebra moments, information and calibration scores, sparse certificate energy, graph/reply "
        "pressure, spatial CNN cues, and phase/cost proxies when relevant.\n"
        "- Head: the classifier receives pooled board features, the mechanism family embedding, profile hash features, "
        "active profile flags, and the selected proposal diagnostics. It returns one puzzle logit plus diagnostic "
        "outputs such as `mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count`, `sheaf_tension`, "
        "`transport_imbalance`, `symmetry_residual`, `topology_pressure`, `ray_language_energy`, `information_surprisal`, "
        "`sparse_certificate_energy`, `rank_file_imbalance`, `king_ring_pressure`, `reply_pressure`, and `defense_gap`.\n"
    )


def audit_one(folder: Path, *, fix: bool) -> dict[str, Any]:
    idea_path = folder / "idea.yaml"
    config_path = folder / "config.yaml"
    architecture_path = folder / "architecture.md"
    idea = _load_yaml(idea_path)
    config = _load_yaml(config_path)
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    slug = str(idea.get("slug") or model_cfg.get("name") or folder.name.split("_", 1)[-1])
    name = str(idea.get("name") or slug)
    model_name = str(model_cfg.get("name") or "")
    is_packet_probe = model_name in RESEARCH_PACKET_MODEL_NAMES
    current_family = model_cfg.get("mechanism_family")
    expected_family = infer_family(slug=slug, name=name, current=str(current_family or "generic"))
    profiles = active_profiles(slug, expected_family)

    issues: list[str] = []
    actions: list[str] = []
    if model_name != slug:
        issues.append(f"model.name {model_name!r} does not match slug {slug!r}")
    if is_packet_probe:
        if current_family != expected_family:
            issues.append(f"mechanism_family {current_family!r} should be {expected_family!r}")
            if fix:
                model_cfg["mechanism_family"] = expected_family
                idea["mechanism_family"] = expected_family
                actions.append("updated mechanism_family")
        if model_cfg.get("packet_profile") != slug:
            issues.append(f"packet_profile {model_cfg.get('packet_profile')!r} should be {slug!r}")
            if fix:
                model_cfg["packet_profile"] = slug
                actions.append("updated packet_profile")
        if fix:
            architecture = _architecture_text(name, expected_family, profiles)
            existing_architecture = architecture_path.read_text(encoding="utf-8") if architecture_path.exists() else ""
            if existing_architecture != architecture:
                architecture_path.write_text(architecture, encoding="utf-8")
                actions.append("rewrote architecture.md")
    config["model"] = model_cfg
    if fix and actions:
        _write_yaml(config_path, config)
        _write_yaml(idea_path, idea)

    return {
        "idea_id": idea.get("idea_id"),
        "slug": slug,
        "model_name": model_name,
        "packet_probe": is_packet_probe,
        "family": model_cfg.get("mechanism_family"),
        "expected_family": expected_family if is_packet_probe else None,
        "active_profiles": profiles if is_packet_probe else [],
        "issues": issues,
        "actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit registered ideas for proposal/implementation conformance.")
    parser.add_argument("--fix", action="store_true", help="Update packet-probe configs and architecture notes.")
    parser.add_argument("--output", default="reports/idea_conformance_audit.json")
    parser.add_argument("--markdown", default="reports/idea_conformance_audit.md")
    args = parser.parse_args()

    rows = [audit_one(folder, fix=args.fix) for folder in discover_idea_folders(Path("ideas/registry"))]
    registry_rows_changed = sync_registry(rows) if args.fix else 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Idea Implementation Conformance Audit",
        "",
        f"- Ideas audited: `{len(rows)}`",
        f"- Packet-probe ideas: `{sum(1 for row in rows if row['packet_probe'])}`",
        f"- Rows with issues before fix: `{sum(1 for row in rows if row['issues'])}`",
        f"- Rows changed: `{sum(1 for row in rows if row['actions'])}`",
        f"- Registry rows synchronized: `{registry_rows_changed}`",
        "",
        "| ID | Slug | Packet Probe | Family | Active Profiles | Issues | Actions |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['idea_id']}` | `{row['slug']}` | `{row['packet_probe']}` | "
            f"`{row['family'] or '-'}` | {', '.join(f'`{item}`' for item in row['active_profiles']) or '-'} | "
            f"{'; '.join(row['issues']) or '-'} | {', '.join(row['actions']) or '-'} |"
        )
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audited {len(rows)} ideas")
    print(f"Packet-probe ideas: {sum(1 for row in rows if row['packet_probe'])}")
    print(f"Rows with issues before fix: {sum(1 for row in rows if row['issues'])}")
    print(f"Rows changed: {sum(1 for row in rows if row['actions'])}")
    print(f"Registry rows synchronized: {registry_rows_changed}")
    print(f"Saved {output}")
    print(f"Saved {args.markdown}")
    if any(row["issues"] for row in rows) and not args.fix:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
