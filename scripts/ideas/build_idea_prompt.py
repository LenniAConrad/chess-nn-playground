#!/usr/bin/env python
from __future__ import annotations

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.ideas.prompting import build_idea_generation_prompt
from chess_nn_playground.utils.logging import write_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reusable ChatGPT prompt for disciplined research idea generation."
    )
    parser.add_argument("--registry", default="ideas/all_ideas/registry/registry.jsonl")
    parser.add_argument("--ideas-root", default="ideas/all_ideas/registry")
    parser.add_argument("--output", default="ideas/all_ideas/research/prompts/idea_generation_prompt.md")
    parser.add_argument("--print", action="store_true", help="Also print the prompt to stdout")
    args = parser.parse_args()

    prompt = build_idea_generation_prompt(
        registry_path=args.registry,
        ideas_root=args.ideas_root,
    )
    write_text(prompt, args.output)
    if args.print:
        print(prompt)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
