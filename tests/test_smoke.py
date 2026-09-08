"""Import/structure smoke tests — run without any API key or network.

    python -m pytest -q         (if pytest installed)
    python tests/test_smoke.py  (plain run)

These verify the code is wired correctly (modules import, the graph compiles,
the knowledge base is present) without making any LLM calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def test_config_paths():
    assert config.DATA_DIR.exists(), "data directory missing"
    md_files = list(config.DATA_DIR.glob("*.md"))
    assert len(md_files) >= 4, "expected the retail knowledge base docs"


def test_graph_compiles():
    # Building the graph compiles the state machine but makes no LLM calls.
    from src.graph import build_graph
    app = build_graph()
    assert app is not None


def test_nodes_importable():
    from src import nodes
    for fn in ["route_question", "retrieve", "grade_documents", "rewrite_query",
               "do_web_search", "generate", "direct_answer", "grade_generation"]:
        assert hasattr(nodes, fn), f"missing node: {fn}"


def test_eval_set_valid():
    import json
    data = json.loads((config.DATA_DIR / "eval_questions.json").read_text())
    assert len(data) >= 5
    for row in data:
        assert "question" in row and "expected_source" in row


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
            passed += 1
    print(f"\n{passed} smoke tests passed.")
