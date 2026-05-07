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
    ["pytest", "tests/test_idea_registry.py", "tests/test_research_architectures.py", "-q"],
]

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


def _build_claude_argv(*, claude_bin: str, safe: bool, max_turns: int, model: str | None) -> list[str]:
    argv = [claude_bin, "-p", "--output-format", "text"]
    if max_turns > 0:
        argv += ["--max-turns", str(max_turns)]
    if model:
        argv += ["--model", model]
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


def _run_verifications(log_handle) -> tuple[bool, str]:
    for cmd in VERIFY_COMMANDS:
        log_handle.write(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}\n")
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
        return True

    if _working_tree_dirty():
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
            log.write(f"\n[runner] claude exited {result.returncode}\n")
            print(f"   claude exited {result.returncode}", flush=True)
            if restore_on_fail:
                _restore()
            return False

        log.write("\n=== VERIFICATION ===\n")
        ok, failed_cmd = _run_verifications(log)

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
    parser.add_argument("--max-turns", type=int, default=40, help="Per-idea --max-turns. 0 disables.")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-idea wallclock timeout in seconds.")
    parser.add_argument("--model", default=None, help="Override Claude model (e.g. claude-sonnet-4-6).")
    parser.add_argument("--claude-bin", default=None, help="Path to claude CLI (default: PATH lookup).")
    parser.add_argument("--safe", action="store_true", help="Use acceptEdits + tool allowlist instead of dangerously-skip-permissions.")
    parser.add_argument("--restore-on-fail", action="store_true", help="git reset --hard + clean -fd if an idea fails.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Halt the loop on the first failure.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without invoking claude.")
    args = parser.parse_args()

    rows = _matching_rows(args)
    if not rows:
        print("No matching scaffold-only ideas found.")
        return

    claude_bin = _resolve_claude_bin(args.claude_bin) if not args.dry_run else (args.claude_bin or "claude")
    claude_argv = _build_claude_argv(
        claude_bin=claude_bin,
        safe=args.safe,
        max_turns=args.max_turns,
        model=args.model,
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
