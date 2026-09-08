"""Command-line entry point.

Examples:
    python main.py --ingest
    python main.py "How long do I have to return an electrical item?"
    python main.py            # interactive REPL
"""
from __future__ import annotations

import argparse
import sys


def run_once(question: str, show_trace: bool = True) -> None:
    from src.graph import answer_question

    result = answer_question(question)
    print("\n" + "=" * 70)
    print("Q:", question)
    print("-" * 70)
    print(result.get("generation", "(no answer)"))
    if show_trace:
        print("-" * 70)
        print("Agent trace:")
        for step in result.get("trace", []):
            print("  •", step)
    print("=" * 70 + "\n")


def repl() -> None:
    from src.graph import answer_question

    print("Retail Agentic RAG — interactive mode. Type 'exit' to quit.\n")
    while True:
        try:
            q = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        result = answer_question(q)
        print("\nassistant ›", result.get("generation", "(no answer)"))
        print("  [", " | ".join(result.get("trace", [])), "]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail Agentic RAG assistant")
    parser.add_argument("question", nargs="*", help="A question to ask")
    parser.add_argument("--ingest", action="store_true", help="(Re)build the vector store then exit")
    parser.add_argument("--no-trace", action="store_true", help="Hide the agent decision trace")
    args = parser.parse_args()

    if args.ingest:
        from src.ingest import build_vectorstore
        build_vectorstore(reset=True)
        return

    if args.question:
        run_once(" ".join(args.question), show_trace=not args.no_trace)
    else:
        repl()


if __name__ == "__main__":
    sys.exit(main())
