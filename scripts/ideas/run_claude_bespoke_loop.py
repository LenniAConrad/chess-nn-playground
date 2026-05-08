#!/usr/bin/env python
"""Run Claude Code headlessly across every scaffold-only idea.

One fresh `claude -p` invocation per idea. Fresh session = clean context, no
cross-idea token spillover, cheapest per-idea cost.

After each idea finishes, the runner re-runs the implementation-kind and
architecture-conformance audits plus the focused pytest suite. On green it
auto-commits the touched files for that idea; on red it leaves the working
tree dirty (or restores it, with --restore-on-fail) and moves on.

Defaults to `--dangerously-skip-permissions` so the verification commands
embedded in the prompt actually run unattended. Switch to `--safe` to use
`--permission-mode acceptEdits` with a Bash allowlist instead.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from print_codex_bespoke_prompt import _matching_rows, _prompt_for  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "claude_bespoke"

VERIFY_COMMANDS: list[list[str]] = [
    ["python", "scripts/ideas/audit_implementation_kinds.py", "--check"],
    ["python", "scripts/ideas/audit_architecture_conformance.py", "--check"],
]


def _verification_commands(idea_id: str) -> list[list[str]]:
    """Per-idea verify: the audits cover the global honesty gate; the focused
    pytest only runs the current idea's test so a backlog of pre-existing
    broken tests in tests/test_research_architectures.py does not block
    progress on unrelated ideas.
    """
    # test_idea_registry_validation is a "whole repo is valid" gate; it
    # iterates every idea and fails if any one is invalid. Deselect it here and
    # let the per-idea -k filter exercise the current idea's bespoke test.
    return [
        *VERIFY_COMMANDS,
        [
            "pytest",
            "tests/test_idea_registry.py",
            "tests/test_research_architectures.py",
            "-q",
            "--deselect",
            "tests/test_idea_registry.py::test_idea_registry_validation",
            "-k",
            idea_id,
        ],
    ]


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _format_dry_run_verify_cmd(cmd: list[str]) -> str:
    parts: list[str] = []
    quote_next = False
    for arg in cmd:
        if quote_next:
            escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'"{escaped}"')
            quote_next = False
        else:
            parts.append(shlex.quote(arg))
            quote_next = arg == "-k"
    return " ".join(parts)


def _selector_idea_id(selector: str) -> str:
    return Path(selector).name.split("_", 1)[0]


def _print_dry_run_verify_commands(idea_id: str, *, label: str = "verify commands") -> None:
    print(f"   [dry-run] {label}:", flush=True)
    for cmd in _verification_commands(idea_id):
        print(f"   [dry-run]   {_format_dry_run_verify_cmd(cmd)}", flush=True)


SAFE_ALLOWED_TOOLS = [
    "Read",
    "Edit",
    "Write",
    "Glob",
    "Grep",
    "Bash(python:*)",
    "Bash(pytest:*)",
    "Bash(git status:*)",
    "Bash(git diff:*)",
    "Bash(ls:*)",
]


def _ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_claude_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("claude")
    if not found:
        raise SystemExit(
            "claude CLI not found on PATH. Install with: npm install -g @anthropic-ai/claude-code"
        )
    return found


def _build_claude_argv(
    *,
    claude_bin: str,
    safe: bool,
    max_turns: int,
    model: str | None,
    effort: str | None,
) -> list[str]:
    argv = [claude_bin, "-p", "--verbose", "--output-format", "stream-json"]
    if max_turns > 0:
        argv += ["--max-turns", str(max_turns)]
    if model:
        argv += ["--model", model]
    if effort:
        argv += ["--effort", effort]
    if safe:
        argv += ["--permission-mode", "acceptEdits"]
        argv += ["--allowedTools", ",".join(SAFE_ALLOWED_TOOLS)]
    else:
        argv += ["--dangerously-skip-permissions"]
    return argv


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def _working_tree_dirty() -> bool:
    return bool(_git(["status", "--porcelain"]).stdout.strip())


def _changed_paths() -> list[str]:
    raw = _git(["status", "--porcelain"]).stdout
    paths: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        # porcelain v1: "XY path" or "XY path -> path" for renames; we want the post-rename path
        rest = line[3:]
        if " -> " in rest:
            rest = rest.split(" -> ", 1)[1]
        paths.append(rest.strip().strip('"'))
    return paths


def _scope_violations(target_folder: str) -> list[str]:
    """Return files claude touched in OTHER ideas/i*/ folders.

    Top-level files in ideas/ (INDEX, TODO, audits) are repo-wide and allowed.
    Files in src/, tests/, scripts/ are legitimately cross-cutting and allowed.
    """
    target = target_folder.rstrip("/") + "/"
    violations: list[str] = []
    for path in _changed_paths():
        if not path.startswith("ideas/"):
            continue
        rel = path[len("ideas/") :]
        if "/" not in rel:
            # top-level ideas/<file>, e.g. INDEX.md or audit json — allowed
            continue
        if path.startswith(target):
            continue
        violations.append(path)
    return violations


def _run_verifications(log_handle, idea_id: str) -> tuple[bool, str]:
    for cmd in _verification_commands(idea_id):
        log_handle.write(f"\n$ {_format_cmd(cmd)}\n")
        log_handle.flush()
        env = os.environ.copy()
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        log_handle.write(result.stdout)
        if result.stderr:
            log_handle.write("\n[stderr]\n" + result.stderr)
        log_handle.flush()
        if result.returncode != 0:
            return False, " ".join(cmd)
    return True, ""


def _commit(idea_folder: str, idea_id: str) -> None:
    _git(["add", "-A"])
    if not _git(["diff", "--cached", "--quiet"], check=False).returncode:
        # nothing staged — claude made no changes
        return
    message = f"{idea_id}: implement bespoke model via headless claude\n\nFolder: {idea_folder}"
    _git(["commit", "-m", message])


def _restore() -> None:
    _git(["reset", "--hard"], check=False)
    _git(["clean", "-fd"], check=False)


def _run_one_idea(
    *,
    row: object,
    index: int,
    total: int,
    claude_argv: list[str],
    log_dir: Path,
    timeout_s: int,
    restore_on_fail: bool,
    dry_run: bool,
) -> bool:
    folder = str(getattr(row, "folder"))
    idea_id = str(getattr(row, "idea_id"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{_ts()}_{idea_id}.log"

    prompt = _prompt_for(row)

    print(f"\n[{index}/{total}] {idea_id} -> {folder}", flush=True)
    print(f"   log: {log_path}", flush=True)

    if dry_run:
        print("   [dry-run] would run:", " ".join(shlex.quote(a) for a in claude_argv), flush=True)
        print(f"   [dry-run] prompt length: {len(prompt)} chars", flush=True)
        _print_dry_run_verify_commands(idea_id)
        return True

    if _working_tree_dirty():
        if restore_on_fail:
            print("   pre-idea cleanup: tree dirty from previous run, restoring.", flush=True)
            _restore()
        else:
            print("   refusing to start: working tree is dirty. commit or stash first.", flush=True)
            return False

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# idea: {idea_id}\n# folder: {folder}\n# argv: {claude_argv}\n\n")
        log.write("=== PROMPT ===\n")
        log.write(prompt)
        log.write("\n\n=== CLAUDE OUTPUT ===\n")
        log.flush()
        try:
            result = subprocess.run(
                claude_argv,
                cwd=REPO_ROOT,
                input=prompt,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            log.write(f"\n[runner] claude timed out after {timeout_s}s\n")
            print(f"   timeout after {timeout_s}s", flush=True)
            if restore_on_fail:
                _restore()
            return False
        if result.returncode != 0:
            log.write(f"\n[runner] claude exited {result.returncode} (typically max_turns or rate limit). Falling through to scope check + verification — the audits are the source of truth, not the exit code.\n")
            print(f"   claude exited {result.returncode}; running audits anyway", flush=True)

        log.write("\n=== SCOPE CHECK ===\n")
        out_of_scope = _scope_violations(folder)
        if out_of_scope:
            log.write("[runner] claude modified files outside target idea folder:\n")
            for path in out_of_scope:
                log.write(f"  - {path}\n")
            log.flush()
            print(f"   scope violation: {len(out_of_scope)} file(s) outside {folder}", flush=True)
            for path in out_of_scope[:5]:
                print(f"     - {path}", flush=True)
            if restore_on_fail:
                _restore()
                print("   working tree restored", flush=True)
            return False
        log.write("ok\n")

        log.write("\n=== VERIFICATION ===\n")
        ok, failed_cmd = _run_verifications(log, idea_id)

    if not ok:
        print(f"   verification failed: {failed_cmd}", flush=True)
        if restore_on_fail:
            _restore()
            print("   working tree restored", flush=True)
        else:
            print("   leaving dirty tree for manual inspection", flush=True)
        return False

    _commit(folder, idea_id)
    print("   committed.", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ideas-root", default="ideas")
    parser.add_argument("--idea", action="append", help="Idea id, folder name, or folder path. May be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=80, help="Per-idea --max-turns. 0 disables.")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-idea wallclock timeout in seconds.")
    parser.add_argument("--model", default=None, help="Override Claude model alias or full id (e.g. 'opus', 'sonnet', 'claude-opus-4-7'). Default: claude CLI's own default (Opus on a Max subscription).")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max", "none"], help="Extended-thinking effort level. Default: high. Use 'none' to omit the flag entirely.")
    parser.add_argument("--claude-bin", default=None, help="Path to claude CLI (default: PATH lookup).")
    parser.add_argument("--safe", action="store_true", help="Use acceptEdits + tool allowlist instead of dangerously-skip-permissions.")
    parser.add_argument(
        "--restore-on-fail",
        dest="restore_on_fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On idea failure, git reset --hard + clean -fd so the next idea starts clean. Default: on. Use --no-restore-on-fail to keep the dirty tree for inspection (will halt the loop).",
    )
    parser.add_argument("--stop-on-fail", action="store_true", help="Halt the loop on the first failure.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without invoking claude.")
    args = parser.parse_args()

    rows = _matching_rows(args)
    if not rows:
        print("No matching scaffold-only ideas found.")
        if args.dry_run and args.idea:
            for selector in args.idea:
                idea_id = _selector_idea_id(selector)
                _print_dry_run_verify_commands(idea_id, label=f"verify commands for {idea_id}")
        return

    claude_bin = _resolve_claude_bin(args.claude_bin) if not args.dry_run else (args.claude_bin or "claude")
    effort = None if args.effort == "none" else args.effort
    claude_argv = _build_claude_argv(
        claude_bin=claude_bin,
        safe=args.safe,
        max_turns=args.max_turns,
        model=args.model,
        effort=effort,
    )

    log_dir = Path(args.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir

    print(f"queued: {len(rows)} ideas")
    print(f"argv:   {' '.join(shlex.quote(a) for a in claude_argv)}")
    print(f"logs:   {log_dir}")

    successes = 0
    failures: list[str] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        ok = _run_one_idea(
            row=row,
            index=index,
            total=total,
            claude_argv=claude_argv,
            log_dir=log_dir,
            timeout_s=args.timeout,
            restore_on_fail=args.restore_on_fail,
            dry_run=args.dry_run,
        )
        if ok:
            successes += 1
        else:
            failures.append(str(getattr(row, "folder")))
            if args.stop_on_fail:
                break

    print(f"\nfinished. ok={successes} failed={len(failures)} skipped={total - successes - len(failures)}")
    if failures:
        print("failed:")
        for folder in failures:
            print(f"  - {folder}")


if __name__ == "__main__":
    main()
