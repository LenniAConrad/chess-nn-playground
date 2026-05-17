#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Any

import yaml



from chess_nn_playground.ideas.implementation_kind import audit_implementation_kinds


TARGET_KIND = "shared_probe_variant"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _one_line(value: object, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _matches(row: object, selector: str) -> bool:
    folder = str(getattr(row, "folder"))
    folder_name = Path(folder).name
    return selector in {
        str(getattr(row, "idea_id")),
        folder_name,
        folder,
        Path(folder).as_posix(),
    }


def _prompt_for(row: object) -> str:
    folder = str(getattr(row, "folder"))
    idea = _load_yaml(Path(folder) / "idea.yaml")
    config = _load_yaml(Path(folder) / "config.yaml")
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    idea_id = str(getattr(row, "idea_id"))
    slug = str(getattr(row, "slug"))
    name = str(idea.get("name") or slug)
    target_task = _one_line(idea.get("target_task"))
    thesis = _one_line(idea.get("short_thesis"))
    current_model_name = str(model_cfg.get("name") or "")

    return dedent(
        f"""
        You are Codex working in the local repository `chess-nn-playground`.

        Goal: implement `{folder}` as a real bespoke architecture that faithfully follows its markdown idea. Do not leave it as a `ResearchPacketProbe` shell.

        Idea metadata:
        - Idea ID: `{idea_id}`
        - Folder: `{folder}`
        - Name: `{name}`
        - Slug / current model name: `{slug}` / `{current_model_name}`
        - Current implementation kind: `shared_probe_variant`
        - Current implementation status: `{idea.get("implementation_status")}`
        - Target task: {target_task or "<read from idea.yaml>"}
        - Short thesis: {thesis or "<read from math_thesis.md>"}

        Read these files first:
        - `{folder}/math_thesis.md`
        - `{folder}/architecture.md`
        - `{folder}/implementation_notes.md`
        - `{folder}/trainer_notes.md`
        - `{folder}/idea.yaml`
        - `{folder}/config.yaml`
        - `{folder}/model.py`

        Hard rules:
        - Do not fake implementation status.
        - Do not keep `ResearchPacketProbe` or `build_research_packet_probe_from_config` in `{folder}/model.py`.
        - Do not mark this idea as `bespoke_model`, `implemented`, or `tested` unless the bespoke model is actually implemented, registered, smoke-tested, and validated.
        - Preserve the existing idea ID, folder, task contract, and config identity unless the codebase requires a narrowly scoped correction.
        - Keep unrelated benchmark results and unrelated files untouched.
        - Do NOT modify any file under `ideas/<other_idea>/`. The only `ideas/` folder you may write to is `{folder}/`. Repo-wide `ideas/` files (INDEX.md, TODO.md, audit reports) may be touched only if the conformance audit requires it. Touching another idea's folder is grounds for rejection of this run.
        - If the markdown idea is underspecified or impossible to implement faithfully, stop and document the exact gap instead of pretending.

        Implementation requirements:
        1. Derive the architecture from the markdown, especially `math_thesis.md` and `architecture.md`.
        2. Implement a materially distinct PyTorch model, preferably in `src/chess_nn_playground/models/`.
        3. Register the model in the project model registry.
        4. Keep `{folder}/model.py` as a thin idea-local `build_model_from_config(config)` wrapper around the bespoke implementation.
        5. Make the model accept the repo board tensor contract and return logits in the expected shape for the puzzle-binary trainer.
        6. Remove scaffold-only notices only after the shared probe wrapper is gone.
        7. Update `{folder}/architecture.md` with an `Implementation Binding` section that names:
           - the registered model name
           - the source implementation file
           - the idea-local wrapper
        8. Update `{folder}/idea.yaml` only after the implementation is real:
           - `implementation_kind: bespoke_model`
           - `implementation_status: implemented`
           - `status: implemented`
        9. Add or update focused tests proving:
           - the model builds from `{folder}/config.yaml`
           - a forward pass works
           - output shape is correct
           - the idea no longer imports or calls `ResearchPacketProbe`
           - implementation-kind and architecture-conformance validation pass

        Verification commands to run before finishing:
        ```bash
        chess-nn-audit-ideas --check
        chess-nn-audit-architectures --check
        PYTHONDONTWRITEBYTECODE=1 pytest tests/test_idea_registry.py tests/test_research_architectures.py
        ```

        If registry/index/report generation changes, also run:
        ```bash
        chess-nn-build-idea-catalog
        PYTHONDONTWRITEBYTECODE=1 python -m scripts.ideas.build_idea_prompt
        ```

        Final response requirements:
        - List changed files.
        - Confirm `{folder}/model.py` no longer imports or calls `ResearchPacketProbe` / `build_research_packet_probe_from_config`.
        - Confirm the idea is detected as `bespoke_model`.
        - Confirm validation commands and tests pass.
        - If any markdown requirement was approximated rather than implemented directly, say exactly what differs and why.
        """
    ).strip()


def _clipboard_command() -> list[str] | None:
    candidates = [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["pbcopy"],
        ["clip.exe"],
    ]
    for command in candidates:
        if shutil.which(command[0]):
            return command
    return None


def _copy_to_clipboard(text: str) -> None:
    command = _clipboard_command()
    if command is None:
        raise RuntimeError(
            "No clipboard command found. Install one of: wl-copy, xclip, xsel, pbcopy, or clip.exe."
        )
    subprocess.run(command, input=text, text=True, check=True)


def _copy_one(row: object, *, delay: float, index: int = 1, total: int = 1) -> None:
    folder = str(getattr(row, "folder"))
    _copy_to_clipboard(_prompt_for(row))
    print(f"Copied prompt {index}/{total} to clipboard: {folder}", flush=True)
    if delay > 0:
        time.sleep(delay)


def _copy_loop(rows: list[object], *, delay: float, auto_advance: bool) -> None:
    total = len(rows)
    if total == 0:
        print("No matching scaffold-only ideas found.")
        return
    for index, row in enumerate(rows, start=1):
        _copy_one(row, delay=delay, index=index, total=total)
        if index == total:
            print("Done. No more matching prompts.")
            return
        next_folder = str(getattr(rows[index], "folder"))
        if auto_advance:
            print(f"Next prompt will copy after {delay:g}s: {next_folder}", flush=True)
            if delay > 0:
                time.sleep(delay)
            continue
        response = input(f"Paste/send that prompt now. Press Enter for next ({next_folder}), or q to quit: ")
        if response.strip().lower() in {"q", "quit", "exit"}:
            print(f"Stopped after {index}/{total}. Resume with --offset {index}.")
            return


def _matching_rows(args: argparse.Namespace) -> list[object]:
    rows = [row for row in audit_implementation_kinds(args.ideas_root) if row.detected_kind == TARGET_KIND]
    if args.idea:
        selectors = set(args.idea)
        rows = [row for row in rows if any(_matches(row, selector) for selector in selectors)]
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Print copy-paste Codex prompts for scaffold-only ideas.")
    parser.add_argument("--ideas-root", default="ideas/registry")
    parser.add_argument(
        "--idea",
        action="append",
        help="Idea id, folder name, or folder path. May be repeated. If omitted, prints prompts for matching scaffold ideas.",
    )
    parser.add_argument("--list", action="store_true", help="List matching scaffold idea folders instead of printing prompts.")
    parser.add_argument("--copy-one", action="store_true", help="Copy only the first matching prompt to the clipboard and exit.")
    parser.add_argument("--copy-loop", action="store_true", help="Copy prompts to the clipboard one at a time without printing them.")
    parser.add_argument("--auto-advance", action="store_true", help="With --copy-loop, do not wait for Enter before the next prompt.")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait after copying a prompt. Default: 2.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of prompts/folders printed.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many matching scaffold ideas.")
    args = parser.parse_args()

    rows = _matching_rows(args)
    if args.list:
        for row in rows:
            print(getattr(row, "folder"))
        return
    if args.copy_one:
        if not rows:
            raise SystemExit("No matching scaffold-only ideas found.")
        _copy_one(rows[0], delay=args.delay, index=1, total=len(rows))
        return
    if args.copy_loop:
        _copy_loop(rows, delay=args.delay, auto_advance=args.auto_advance)
        return

    for index, row in enumerate(rows, start=1):
        if index > 1:
            print("\n" + "=" * 88 + "\n")
        print(_prompt_for(row))


if __name__ == "__main__":
    main()
