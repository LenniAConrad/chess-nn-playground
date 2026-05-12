#!/usr/bin/env python
from __future__ import annotations

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.evaluation.chatgpt_prompt import build_chatgpt_run_prompt
from chess_nn_playground.utils.logging import write_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a ChatGPT Pro prompt containing the latest tested run context."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--leaderboard", default="results/leaderboard.md")
    parser.add_argument("--registry", default="ideas/all_ideas/registry/registry.jsonl")
    parser.add_argument("--output", default="reports/prompts/chatgpt_pro_run_prompt.md")
    parser.add_argument("--max-runs", type=int, default=25)
    parser.add_argument("--print", action="store_true", help="Also print the prompt to stdout")
    args = parser.parse_args()

    prompt = build_chatgpt_run_prompt(
        results_dir=args.results_dir,
        leaderboard_path=args.leaderboard,
        registry_path=args.registry,
        max_runs=args.max_runs,
    )
    write_text(prompt, args.output)
    if args.print:
        print(prompt)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
