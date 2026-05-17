# Agent Automation

This folder contains experimental local automation for Claude/Codex-style implementation handoffs. These scripts are intentionally isolated from the normal training, benchmark, reporting, and CI path.

Default behavior is conservative:

- Launchers use `CLAUDE_PERMISSION_MODE=acceptEdits` by default.
- `bypassPermissions` requires both `CLAUDE_PERMISSION_MODE=bypassPermissions` and `CLAUDE_ALLOW_BYPASS_PERMISSIONS=1`.
- The Python loop uses `--permission-mode acceptEdits` plus a small tool allowlist by default.
- `--unsafe-skip-permissions` is explicit and should be used only in disposable worktrees.
- Expensive training remains gated by `CLAUDE_ALLOW_TRAINING=1`.
- `CLAUDE_DRY_RUN=1` / `--dry-run` prints commands and prompts without invoking Claude.

## Commands

Primitive handoff prompt:

```bash
CLAUDE_DRY_RUN=1 scripts/agents/run_primitive_implementation_with_claude.sh
```

Explicit research primitive batch:

```bash
CLAUDE_DRY_RUN=1 \
CLAUDE_RESEARCH_TARGET_FILES="ideas/research/primitives/external_31_canonical_orbit_bdd_wmc_primitives.md" \
scripts/agents/run_research_primitive_implementation_with_claude.sh
```

Prepared worktree/tmux launcher:

```bash
LAUNCH_DRY_RUN=1 scripts/agents/launch_remaining_primitive_implementation_batches.sh
```

Scaffold-only idea loop:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m scripts.agents.run_claude_bespoke_loop --dry-run --limit 1
```

Repair old idea breakage, then implement queued proposals/research primitives in one tmux-managed Claude run:

```bash
CLAUDE_DRY_RUN=1 CLAUDE_TOTAL_LIMIT=2 \
scripts/agents/run_claude_idea_repair_and_implementation.sh --inside-tmux
```

For a real run, omit `CLAUDE_DRY_RUN=1`. The launcher defaults to `claude-opus-4-7`,
max reasoning, `acceptEdits`, tmux supervision, and an ETA/status file under
`reports/claude_idea_repair_and_implementation/`.

These scripts may create logs under `logs/` or `reports/*claude*/` when actually run. Keep generated logs out of review unless they are intentional evidence.
