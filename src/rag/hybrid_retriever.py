import re
import sys
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import EMBED_MODEL_NAME, RRF_K, TOP_K_CANDIDATES
from src.rag.contextual_chunker import Chunk, build_chunks

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "for", "with", "and", "or", "after",
    "before", "this", "that", "it", "its", "as", "from", "under", "into",
    "about", "than", "then", "so", "no", "not", "if", "when", "while",
    "document", "scope", "section",
}

_MODEL: SentenceTransformer | None = None
_RETRIEVER: "HybridRetriever | None" = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _MODEL


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


class HybridRetriever:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = build_chunks()
        if not self.chunks:
            raise RuntimeError("No chunks were produced from data/raw_documents.")
        self.chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}

        model = _get_model()
        contextual_texts = [c.contextualized_text for c in self.chunks]
        raw_texts = [c.raw_text for c in self.chunks]

        self.contextual_embeddings = model.encode(contextual_texts, normalize_embeddings=True)
        self.raw_embeddings = model.encode(raw_texts, normalize_embeddings=True)
        self.bm25 = BM25Okapi([_tokenize(t) for t in contextual_texts])

        # In-memory: the index is rebuilt from the documents on every start, and a
        # shared on-disk store let one process delete another process's collections.
        self.chroma_client = chromadb.EphemeralClient()
        self._contextual_collection = self._sync_collection(
            "chunks_contextual", contextual_texts, self.contextual_embeddings
        )
        self._raw_collection = self._sync_collection("chunks_raw", raw_texts, self.raw_embeddings)

    def _sync_collection(self, name: str, texts: list[str], embeddings: np.ndarray):
        try:
            self.chroma_client.delete_collection(name)
        except Exception:
            pass
        collection = self.chroma_client.create_collection(name, metadata={"hnsw:space": "cosine"})
        collection.add(
            ids=[c.chunk_id for c in self.chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
        )
        return collection

    def _dense_rank(self, query: str, which: str) -> list[tuple[str, float]]:
        collection = self._contextual_collection if which == "contextual" else self._raw_collection
        model = _get_model()
        q_embedding = model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)[0]
        result = collection.query(
            query_embeddings=[q_embedding.tolist()], n_results=len(self.chunks)
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [(chunk_id, 1.0 - dist) for chunk_id, dist in zip(ids, distances)]

    def _bm25_rank(self, query: str) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores)
        return [(self.chunks[i].chunk_id, float(scores[i])) for i in order]

    @staticmethod
    def _rrf_fuse(rank_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
        scores: dict[str, float] = defaultdict(float)
        for rank_list in rank_lists:
            for rank, chunk_id in enumerate(rank_list):
                scores[chunk_id] += 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def search(
        self, query: str, mode: str = "hybrid_contextual", top_k: int = TOP_K_CANDIDATES
    ) -> list[dict]:
        if mode == "dense_raw":
            ranked = self._dense_rank(query, "raw")
        elif mode == "hybrid_contextual":
            dense_ranked = self._dense_rank(query, "contextual")
            bm25_ranked = self._bm25_rank(query)
            ranked = self._rrf_fuse(
                [[cid for cid, _ in dense_ranked], [cid for cid, _ in bm25_ranked]]
            )
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        results = []
        for chunk_id, score in ranked[:top_k]:
            results.append({"chunk": self.chunk_by_id[chunk_id], "score": score})
        return results


def get_retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever()
    return _RETRIEVER


def test_search() -> None:
    retriever = get_retriever()
    query = "iPhone 15 cracked screen"
    for mode in ("hybrid_contextual", "dense_raw"):
        results = retriever.search(query, mode=mode, top_k=5)
        print(f"Mode={mode} query={query!r}")
        for result in results:
            chunk = result["chunk"]
            print(f"  {chunk.chunk_id:35s} score={result['score']:.4f}  {chunk.heading_path}")
        if not results:
            raise RuntimeError(f"No results returned for mode={mode}")


if __name__ == "__main__":
    test_search()
