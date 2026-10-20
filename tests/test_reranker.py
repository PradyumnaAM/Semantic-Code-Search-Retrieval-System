
from unittest.mock import patch
import numpy as np

from backend.reranker import rerank_results


def make_results():
    return {
        "documents": [[
            "def greet(): return 'Hello'",
            "def build_context(): return 'LLM context'",
            "def calculate_total(): return 100"
        ]],
        "metadatas": [[
            {"name": "greet", "path": "example.py"},
            {"name": "build_context", "path": "rag.py"},
            {"name": "calculate_total", "path": "math.py"}
        ]]
    }


def test_reranker_orders_by_score():
    results = make_results()

    with patch(
        "backend.reranker.reranker.predict",
        return_value=np.array([0.2, 0.9, 0.4])
    ):
        ranked = rerank_results(
            "Which function builds context?",
            results,
            top_k=3
        )

    names = [item["metadata"]["name"] for item in ranked]

    assert names == [
        "build_context",
        "calculate_total",
        "greet"
    ]


def test_reranker_respects_top_k():
    results = make_results()

    with patch(
        "backend.reranker.reranker.predict",
        return_value=np.array([0.2, 0.9, 0.4])
    ):
        ranked = rerank_results(
            "Which function builds context?",
            results,
            top_k=2
        )

    assert len(ranked) == 2
    assert ranked[0]["metadata"]["name"] == "build_context"


def test_reranker_handles_empty_results():
    results = {
        "documents": [[]],
        "metadatas": [[]]
    }

    ranked = rerank_results(
        "Which function builds context?",
        results,
        top_k=3
    )

    assert ranked == []
