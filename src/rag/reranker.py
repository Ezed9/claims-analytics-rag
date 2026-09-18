import sys
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RERANK_THRESHOLD, RERANK_TOP_N, RERANKER_MODEL_NAME

_CROSS_ENCODER: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder(RERANKER_MODEL_NAME)
    return _CROSS_ENCODER


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def rerank(query: str, candidates: list[dict]) -> tuple[list[dict], bool]:
    if not candidates:
        return [], False

    cross_encoder = _get_cross_encoder()
    pairs = [(query, c["chunk"].contextualized_text) for c in candidates]
    logits = cross_encoder.predict(pairs, convert_to_numpy=True)
    scores = _sigmoid(np.asarray(logits, dtype=float))

    scored = sorted(
        (
            {"chunk": c["chunk"], "retrieval_score": c["score"], "rerank_score": float(s)}
            for c, s in zip(candidates, scores)
        ),
        key=lambda r: r["rerank_score"],
        reverse=True,
    )
    passing = [r for r in scored if r["rerank_score"] >= RERANK_THRESHOLD][:RERANK_TOP_N]
    insufficient_evidence = len(passing) == 0
    return passing, insufficient_evidence


def test_rerank() -> None:
    from src.rag.hybrid_retriever import get_retriever

    query = "iPhone 15 cracked screen"
    candidates = get_retriever().search(query, mode="hybrid_contextual")
    passing, insufficient = rerank(query, candidates)
    print(f"query={query!r} insufficient_evidence={insufficient}")
    for result in passing:
        print(f"  {result['chunk'].chunk_id:35s} rerank_score={result['rerank_score']:.4f}")
    if insufficient:
        raise RuntimeError("Expected at least one chunk to pass the rerank threshold.")


if __name__ == "__main__":
    test_rerank()
