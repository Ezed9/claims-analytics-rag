import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.orchestrator import run_triage
from src.config import EVAL_RESULTS_PATH, GOLDEN_DATASET_PATH, RERANK_TOP_N, has_active_llm
from src.rag.hybrid_retriever import get_retriever
from src.rag.reranker import rerank

RETRIEVAL_MODES = ["hybrid_contextual", "dense_raw"]


def _load_golden_dataset() -> list[dict[str, Any]]:
    return json.loads(GOLDEN_DATASET_PATH.read_text())


def _context_precision_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for chunk_id in retrieved_ids if chunk_id in relevant_set)
    return hits / len(retrieved_ids)


def _retrieve_and_rerank(query: str, mode: str) -> list[str]:
    retriever = get_retriever()
    candidates = retriever.search(query, mode=mode)
    passing, _ = rerank(query, candidates)
    if passing:
        return [r["chunk"].chunk_id for r in passing]
    return [r["chunk"].chunk_id for r in candidates[:RERANK_TOP_N]]


def _run_ragas_metrics(cases: list[dict[str, Any]], case_results: list[dict[str, Any]]) -> None:
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference

    from src.config import EMBED_MODEL_NAME
    from src.llm_client import judge_llm

    judge = judge_llm()
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME))
    faithfulness_metric = Faithfulness(llm=judge)
    context_precision_metric = LLMContextPrecisionWithReference(llm=judge)

    for case, result in zip(cases, case_results):
        query = case["claim_text"]
        retrieved_ids = _retrieve_and_rerank(query, "hybrid_contextual")
        retriever = get_retriever()
        contexts = [retriever.chunk_by_id[cid].raw_text for cid in retrieved_ids]
        response = (
            f"Verdict: {result['verdict_actual']}. "
            f"Incident type: {case['expected_incident_type']}."
        )
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=contexts or [""],
            reference=case["ground_truth_answer"],
        )
        try:
            result["faithfulness"] = faithfulness_metric.single_turn_score(sample)
            result["context_precision_llm"] = context_precision_metric.single_turn_score(sample)
        except Exception as exc:
            result["faithfulness"] = None
            result["context_precision_llm"] = None
            result["ragas_error"] = str(exc)
    _ = embeddings  # embeddings wrapper is wired for parity with the Ragas judge configuration


def evaluate() -> dict[str, Any]:
    cases = _load_golden_dataset()
    case_results: list[dict[str, Any]] = []

    for case in cases:
        triage = run_triage(
            case["claim_text"],
            case["carrier"],
            device_model=case.get("device_model"),
            claim_date=case.get("claim_date"),
            policy_effective_date=case.get("policy_effective_date"),
        )
        verdict_actual = triage["verdict"]
        verdict_correct = verdict_actual == case["expected_verdict"]

        precision_by_mode = {
            mode: _context_precision_at_k(
                _retrieve_and_rerank(case["claim_text"], mode), case["relevant_chunk_ids"]
            )
            for mode in RETRIEVAL_MODES
        }

        case_results.append(
            {
                "case_id": case["case_id"],
                "verdict_actual": verdict_actual,
                "verdict_expected": case["expected_verdict"],
                "verdict_correct": verdict_correct,
                "context_precision_hybrid_contextual": precision_by_mode["hybrid_contextual"],
                "context_precision_dense_raw": precision_by_mode["dense_raw"],
                "faithfulness": None,
                "context_precision_llm": None,
            }
        )

    ragas_skipped_reason = None
    if has_active_llm():
        _run_ragas_metrics(cases, case_results)
    else:
        ragas_skipped_reason = (
            "No GEMINI_API_KEY or ANTHROPIC_API_KEY set; Ragas LLM-judged metrics "
            "(faithfulness, context_precision) were skipped. Reference-based, "
            "non-LLM context precision still ran for both retrieval modes."
        )

    verdict_accuracy = sum(r["verdict_correct"] for r in case_results) / len(case_results)
    hybrid_scores = [r["context_precision_hybrid_contextual"] for r in case_results]
    dense_scores = [r["context_precision_dense_raw"] for r in case_results]
    faithfulness_scores = [r["faithfulness"] for r in case_results if r["faithfulness"] is not None]

    summary = {
        "num_cases": len(case_results),
        "verdict_accuracy": round(verdict_accuracy, 4),
        "context_precision_hybrid_contextual_mean": round(statistics.mean(hybrid_scores), 4),
        "context_precision_hybrid_contextual_std": round(
            statistics.pstdev(hybrid_scores), 4
        ),
        "context_precision_dense_raw_mean": round(statistics.mean(dense_scores), 4),
        "context_precision_dense_raw_std": round(statistics.pstdev(dense_scores), 4),
        "faithfulness_mean": round(statistics.mean(faithfulness_scores), 4)
        if faithfulness_scores
        else None,
        "ragas_skipped_reason": ragas_skipped_reason,
    }

    output = {"cases": case_results, "summary": summary}
    EVAL_RESULTS_PATH.write_text(json.dumps(output, indent=2))
    return output


def _print_summary(output: dict[str, Any]) -> None:
    summary = output["summary"]
    print(f"Cases evaluated: {summary['num_cases']}")
    print(f"Verdict accuracy: {summary['verdict_accuracy'] * 100:.1f}%")
    print(
        "Context precision (non-LLM, reference-based) — hybrid_contextual: "
        f"{summary['context_precision_hybrid_contextual_mean']:.4f} "
        f"± {summary['context_precision_hybrid_contextual_std']:.4f}"
    )
    print(
        "Context precision (non-LLM, reference-based) — dense_raw: "
        f"{summary['context_precision_dense_raw_mean']:.4f} "
        f"± {summary['context_precision_dense_raw_std']:.4f}"
    )
    if summary["ragas_skipped_reason"]:
        print(f"Ragas LLM metrics skipped: {summary['ragas_skipped_reason']}")
    else:
        print(f"Ragas faithfulness mean: {summary['faithfulness_mean']}")


if __name__ == "__main__":
    _print_summary(evaluate())
