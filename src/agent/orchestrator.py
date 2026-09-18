import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.audit_logger import write_audit_log
from src.agent.economic_engine import compute_economics
from src.agent.extractor import extract_claim_info
from src.agent.mcp_server import search_repair_sops, verify_policy_rules
from src.config import resolve_provider


def run_triage(
    claim_text: str,
    carrier: str,
    device_model: str | None = None,
    claim_date: str | None = None,
    policy_effective_date: str | None = None,
) -> dict[str, Any]:
    claim_date = claim_date or date.today().isoformat()

    extraction = extract_claim_info(claim_text, carrier_hint=carrier)
    device_model = device_model or extraction.device_model

    policy_check = verify_policy_rules(
        carrier=carrier,
        incident_type=extraction.incident_type.value,
        claim_date=claim_date,
        policy_effective_date=policy_effective_date,
    )

    evidence: dict[str, Any] = {"insufficient_evidence": None, "evidence": []}
    economics: dict[str, Any] | None = None

    if not policy_check.get("rule_covered", False):
        verdict = "DENIED_NOT_COVERED"
    elif policy_check.get("waiting_period_violation", False):
        verdict = "DENIED_WAITING_PERIOD"
    elif not policy_check.get("repair_allowed", True):
        # Repair is not physically possible (e.g. a lost/stolen device), so
        # there is no repair SOP to look up: replacement is the only option
        # and no RAG evidence is needed to reach that verdict.
        verdict = "RECOMMEND_REPLACEMENT"
    else:
        # The full claim narrative is a far better query for the dense
        # retriever and cross-encoder reranker than a bag of extracted
        # keywords: keyword lists (e.g. "water submerged rain") read as
        # unnatural text and produce uninformative reranker scores.
        evidence = search_repair_sops(device_model or "unknown device", claim_text)
        if evidence["insufficient_evidence"]:
            verdict = "MANUAL_REVIEW"
        else:
            breakdown = compute_economics(device_model, evidence["evidence"])
            economics = breakdown.__dict__
            if breakdown.recommended_action == "RECOMMEND_REPLACEMENT" and not policy_check.get(
                "replacement_allowed", True
            ):
                verdict = "RECOMMEND_REPAIR"
            else:
                verdict = breakdown.recommended_action

    provider = resolve_provider()
    model_name = {
        "anthropic": "claude-sonnet-5",
        "gemini": "gemini-2.5-flash",
        "none": "rule_based_fallback",
    }[provider if provider in ("anthropic", "gemini") else "none"]

    audit_record = {
        "claim_text": claim_text,
        "carrier": carrier,
        "device_model": device_model,
        "claim_date": claim_date,
        "extraction": extraction.model_dump(),
        "policy_check": policy_check,
        "evidence": evidence,
        "economics": economics,
        "verdict": verdict,
        "llm_provider": provider,
        "model_name": model_name,
    }
    audit_path = write_audit_log(audit_record)

    return {
        "verdict": verdict,
        "extraction": extraction.model_dump(),
        "policy_check": policy_check,
        "evidence": evidence,
        "economics": economics,
        "audit_path": str(audit_path),
        "llm_provider": provider,
        "model_name": model_name,
    }


if __name__ == "__main__":
    result = run_triage("iPhone 15 cracked screen", "Verizon")
    print(result)
