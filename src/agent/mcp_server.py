import sys
from datetime import date, timedelta
from pathlib import Path

from mcp.server.mcpserver import MCPServer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.db_engine import get_coverage_rule
from src.rag.hybrid_retriever import get_retriever
from src.rag.reranker import rerank

DEFAULT_TENURE_DAYS = 400


def verify_policy_rules(
    carrier: str,
    incident_type: str,
    claim_date: str,
    policy_effective_date: str | None = None,
) -> dict:
    """Look up the deductible, waiting period, and repair/replacement rules for a
    carrier's plan and incident type, and evaluate whether a claim is covered
    given the policy's effective/enrollment date."""
    rules = get_coverage_rule(incident_type, carrier=carrier)
    if rules.empty:
        return {
            "is_covered": False,
            "rule_covered": False,
            "waiting_period_violation": False,
            "reason": "no_matching_policy",
            "carrier_partner": carrier,
            "incident_type": incident_type,
        }

    rule = rules.iloc[0].to_dict()
    claim_dt = date.fromisoformat(claim_date)
    if policy_effective_date is not None:
        effective_dt = date.fromisoformat(policy_effective_date)
    else:
        effective_dt = claim_dt - timedelta(days=DEFAULT_TENURE_DAYS)
    tenure_days = (claim_dt - effective_dt).days

    rule_covered = bool(rule["is_covered"])
    waiting_period_violation = rule_covered and tenure_days < int(rule["waiting_period_days"])
    is_covered = rule_covered and not waiting_period_violation

    return {
        "is_covered": is_covered,
        "rule_covered": rule_covered,
        "waiting_period_violation": waiting_period_violation,
        "deductible_usd": float(rule["deductible_usd"]),
        "waiting_period_days": int(rule["waiting_period_days"]),
        "repair_allowed": bool(rule["repair_allowed"]),
        "replacement_allowed": bool(rule["replacement_allowed"]),
        "carrier_partner": rule["carrier_partner"],
        "plan_name": rule["plan_name"],
        "policy_key": int(rule["policy_key"]),
        "tenure_days": tenure_days,
    }


def search_repair_sops(device_model: str, damage_symptom: str) -> dict:
    """Retrieve and rerank repair-SOP evidence chunks for a device model and damage
    symptom, returning the passing chunks (rerank score >= 0.60) or flagging
    insufficient_evidence when none pass."""
    query = f"{device_model} {damage_symptom}".strip()
    candidates = get_retriever().search(query, mode="hybrid_contextual")
    passing, insufficient_evidence = rerank(query, candidates)
    evidence = [
        {
            "chunk_id": result["chunk"].chunk_id,
            "heading_path": result["chunk"].heading_path,
            "text": result["chunk"].raw_text,
            "rerank_score": result["rerank_score"],
        }
        for result in passing
    ]
    return {"insufficient_evidence": insufficient_evidence, "evidence": evidence}


mcp = MCPServer("claims-triage")
mcp.tool()(verify_policy_rules)
mcp.tool()(search_repair_sops)


if __name__ == "__main__":
    mcp.run()
