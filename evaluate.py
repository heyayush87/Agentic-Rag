"""Lightweight evaluation harness — proves the system works with numbers.

Three metrics, computed over data/eval_questions.json:

1. Retrieval hit-rate: did the expected source document appear in the context?
2. Keyword recall:      did the answer contain the required fact(s)?
3. Faithfulness:        LLM-as-judge — is the answer grounded in the context?
4. Answer relevance:    LLM-as-judge — does the answer address the question?

Run:
    python evaluate.py

These are the RAGAS-style metrics, implemented without extra dependencies so
the harness runs against whichever provider you configured.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from src.graph import answer_question
from src.llm import get_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


def _judge(system: str, user: str) -> bool:
    llm = get_chat_model()
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content.strip().split()[0].strip('."\'').lower() == "yes"


FAITHFUL_SYS = (
    "You judge whether every factual claim in ANSWER is supported by CONTEXT. "
    "Reply only 'yes' or 'no'."
)
RELEVANCE_SYS = (
    "You judge whether ANSWER addresses QUESTION. Reply only 'yes' or 'no'."
)


def evaluate() -> None:
    cases = json.loads((config.DATA_DIR / "eval_questions.json").read_text())
    n = len(cases)
    hits = kw = faithful = relevant = 0
    rows = []

    for c in cases:
        result = answer_question(c["question"])
        answer = result.get("generation", "")
        context = result.get("context", "")

        hit = c["expected_source"] in context
        kw_ok = all(tok.lower() in answer.lower() for tok in c.get("must_include", []))
        faith = _judge(FAITHFUL_SYS, f"CONTEXT:\n{context}\n\nANSWER:\n{answer}")
        rel = _judge(RELEVANCE_SYS, f"QUESTION: {c['question']}\n\nANSWER:\n{answer}")

        hits += hit
        kw += kw_ok
        faithful += faith
        relevant += rel
        rows.append((c["question"][:48], hit, kw_ok, faith, rel))

    print("\nPer-question results")
    print(f"{'question':50} {'src?':5}{'kw?':5}{'faith':6}{'rel':5}")
    print("-" * 72)
    for q, h, k, f, r in rows:
        print(f"{q:50} {str(h):5}{str(k):5}{str(f):6}{str(r):5}")

    print("\nAggregate metrics")
    print(f"  Retrieval hit-rate : {hits}/{n}  = {hits/n:.0%}")
    print(f"  Keyword recall     : {kw}/{n}  = {kw/n:.0%}")
    print(f"  Faithfulness       : {faithful}/{n}  = {faithful/n:.0%}")
    print(f"  Answer relevance   : {relevant}/{n}  = {relevant/n:.0%}")


if __name__ == "__main__":
    evaluate()
