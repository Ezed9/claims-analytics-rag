import json
import sys
from pathlib import Path

from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import EVAL_RESULTS_PATH
from eval.evaluate_pipeline import evaluate

ALPHA = 0.05


def _load_or_compute_results() -> dict:
    if EVAL_RESULTS_PATH.exists():
        return json.loads(EVAL_RESULTS_PATH.read_text())
    return evaluate()


def run_significance_test() -> dict:
    results = _load_or_compute_results()
    cases = results["cases"]
    hybrid_scores = [c["context_precision_hybrid_contextual"] for c in cases]
    dense_scores = [c["context_precision_dense_raw"] for c in cases]

    differences = [h - d for h, d in zip(hybrid_scores, dense_scores)]
    non_zero_differences = [d for d in differences if d != 0]

    if not non_zero_differences:
        return {
            "n_cases": len(cases),
            "n_nonzero_differences": 0,
            "statistic": None,
            "p_value": None,
            "alpha": ALPHA,
            "decision": "fail_to_reject",
            "reason": "All paired differences between hybrid_contextual and dense_raw "
            "context precision were zero; the Wilcoxon signed-rank test is undefined.",
        }

    statistic, p_value = stats.wilcoxon(hybrid_scores, dense_scores, alternative="greater")
    decision = "reject" if p_value < ALPHA else "fail_to_reject"
    return {
        "n_cases": len(cases),
        "n_nonzero_differences": len(non_zero_differences),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": ALPHA,
        "decision": decision,
        "reason": (
            f"Wilcoxon signed-rank test (alternative='greater'), H0: hybrid_contextual context "
            f"precision <= dense_raw context precision. At alpha={ALPHA}, we "
            f"{'reject' if decision == 'reject' else 'fail to reject'} H0."
        ),
    }


if __name__ == "__main__":
    outcome = run_significance_test()
    print(f"n_cases={outcome['n_cases']} n_nonzero_differences={outcome['n_nonzero_differences']}")
    print(f"Wilcoxon statistic: {outcome['statistic']}")
    print(f"p-value: {outcome['p_value']}")
    print(f"Decision at alpha={outcome['alpha']}: {outcome['decision'].upper()}")
    print(outcome["reason"])
