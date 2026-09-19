import json
from pathlib import Path

import pytest

import src.config as config
from src.agent import orchestrator
from src.agent.audit_logger import verify_audit, write_audit_log
from src.agent.economic_engine import compute_economics
from src.agent.extractor import IncidentType, extract_rule_based
from src.agent.mcp_server import verify_policy_rules


@pytest.fixture(autouse=True)
def no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "has_active_llm", lambda: False)


def test_audit_log_detects_tampering(tmp_path: Path) -> None:
    path = write_audit_log({"verdict": "MANUAL_REVIEW"}, audit_dir=tmp_path)
    assert verify_audit(path)
    path.chmod(0o644)
    payload = json.loads(path.read_text())
    payload["verdict"] = "RECOMMEND_REPAIR"
    path.write_text(json.dumps(payload))
    assert not verify_audit(path)


def test_economics_uses_sop_evidence_when_present() -> None:
    evidence = [{"text": "Part cost: $100. Labor: 60 min"}]
    result = compute_economics("iPhone 15", evidence)
    assert result.cost_source == "sop_evidence"
    assert result.part_cost_usd == 100.0
    assert result.labor_minutes == 60


def test_economics_falls_back_without_evidence() -> None:
    assert compute_economics("iPhone 15", []).cost_source == "device_default_fallback"


def test_waiting_period_violation_flagged() -> None:
    result = verify_policy_rules(
        "Verizon", "SCREEN_DAMAGE", "2026-01-01", policy_effective_date="2025-12-31"
    )
    assert result["waiting_period_days"] == 15
    assert result["waiting_period_violation"] is True
    assert result["is_covered"] is False


def test_verizon_screen_rule_values() -> None:
    result = verify_policy_rules("Verizon", "SCREEN_DAMAGE", "2026-01-01")
    assert result["deductible_usd"] == 29.0
    assert result["rule_covered"] is True
    assert result["waiting_period_violation"] is False


def test_unknown_carrier_not_covered() -> None:
    assert verify_policy_rules("NoSuchCarrier", "SCREEN_DAMAGE", "2026-01-01")["rule_covered"] is False


def test_rule_based_extraction() -> None:
    info = extract_rule_based("My iPhone 15 Pro screen is cracked", carrier_hint="Verizon")
    assert info.device_model == "iPhone 15 Pro"
    assert info.incident_type == IncidentType.SCREEN_DAMAGE


def test_lost_device_recommends_replacement(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        orchestrator, "write_audit_log", lambda record: write_audit_log(record, audit_dir=tmp_path)
    )
    result = orchestrator.run_triage(
        "My iPhone 15 was stolen at the airport", "Verizon", policy_effective_date="2020-01-01"
    )
    assert result["policy_check"]["repair_allowed"] is False
    assert result["verdict"] == "RECOMMEND_REPLACEMENT"
    assert verify_audit(Path(result["audit_path"]))
