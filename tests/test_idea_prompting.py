from __future__ import annotations

from chess_nn_playground.ideas.prompting import build_idea_generation_prompt


def test_idea_generation_prompt_contains_core_rules():
    prompt = build_idea_generation_prompt()
    assert "Anti-Repetition Rules" in prompt
    assert "Math-Thesis Discipline" in prompt
    assert "Do not fabricate class `1` or class `2` labels" in prompt
    assert "Stockfish scores" in prompt
    assert "Chess Operator Basis Classifier" in prompt
    assert "Puzzle Obligation Flow Network" in prompt
    assert "Null-Move Contrast Puzzle Network" in prompt
    assert "Proof-Core Set Verifier" in prompt
    assert "Neural Proof-Number Search Network" in prompt
    assert "Boundary-Edit Lagrangian Network" in prompt
    assert "Tactical Equilibrium Network" in prompt
    assert "Rule-Consistent Latent Dynamics Network" in prompt
    assert '"folder": "ideas/research/packets"' not in prompt
