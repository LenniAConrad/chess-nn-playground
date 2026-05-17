from __future__ import annotations

from pathlib import Path

from scripts.agents.run_claude_bespoke_loop import _build_claude_argv


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_LAUNCHERS = [
    "run_primitive_implementation_with_claude.sh",
    "run_research_primitive_implementation_with_claude.sh",
    "launch_remaining_primitive_implementation_batches.sh",
]


def test_agent_launchers_are_not_in_repo_root_or_idea_scripts() -> None:
    for path in ROOT_LAUNCHERS:
        assert not (REPO_ROOT / path).exists(), path
    assert not (REPO_ROOT / "scripts/ideas/run_claude_bespoke_loop.py").exists()

    agents_dir = REPO_ROOT / "scripts/agents"
    assert (agents_dir / "run_primitive_implementation_with_claude.sh").exists()
    assert (agents_dir / "run_research_primitive_implementation_with_claude.sh").exists()
    assert (agents_dir / "launch_remaining_primitive_implementation_batches.sh").exists()
    assert (agents_dir / "run_claude_idea_repair_and_implementation.sh").exists()
    assert (agents_dir / "run_claude_bespoke_loop.py").exists()


def test_shell_agent_launchers_default_to_safe_permission_mode() -> None:
    for path in [
        REPO_ROOT / "scripts/agents/run_primitive_implementation_with_claude.sh",
        REPO_ROOT / "scripts/agents/run_research_primitive_implementation_with_claude.sh",
        REPO_ROOT / "scripts/agents/launch_remaining_primitive_implementation_batches.sh",
        REPO_ROOT / "scripts/agents/run_claude_idea_repair_and_implementation.sh",
    ]:
        text = path.read_text(encoding="utf-8")
        assert 'CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-acceptEdits}"' in text
        assert "CLAUDE_ALLOW_BYPASS_PERMISSIONS" in text
        assert 'CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-bypassPermissions}"' not in text


def test_idea_repair_agent_launcher_has_tmux_eta_and_safe_opus_defaults() -> None:
    text = (REPO_ROOT / "scripts/agents/run_claude_idea_repair_and_implementation.sh").read_text(encoding="utf-8")

    assert 'CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-7}"' in text
    assert 'CLAUDE_EFFORT="${CLAUDE_EFFORT:-max}"' in text
    assert 'CLAUDE_DRY_RUN="${CLAUDE_DRY_RUN:-0}"' in text
    assert "tmux new-session" in text
    assert "--inside-tmux" in text
    assert "progress_bar()" in text
    assert "ETA:" in text
    assert "discover_queue()" in text
    assert "audit_architecture_conformance" in text
    assert "validate_idea_for_training" in text
    assert "research_primitive" in text


def test_python_agent_loop_defaults_to_accept_edits() -> None:
    argv = _build_claude_argv(
        claude_bin="claude",
        unsafe_skip_permissions=False,
        max_turns=1,
        model=None,
        effort="high",
    )

    assert "--permission-mode" in argv
    assert "acceptEdits" in argv
    assert "--allowedTools" in argv
    assert "--dangerously-skip-permissions" not in argv


def test_python_agent_loop_unsafe_mode_is_explicit() -> None:
    argv = _build_claude_argv(
        claude_bin="claude",
        unsafe_skip_permissions=True,
        max_turns=1,
        model=None,
        effort="high",
    )

    assert "--dangerously-skip-permissions" in argv
    assert "--permission-mode" not in argv
