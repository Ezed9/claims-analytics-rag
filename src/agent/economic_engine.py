import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import REPAIR_THRESHOLD
from src.db_engine import query_df

PART_COST_RE = re.compile(r"Part cost:\s*\$(\d+(?:\.\d+)?)")
LABOR_RE = re.compile(r"Labor:\s*(\d+)\s*min")

DEFAULT_MSRP_USD = 800.0
DEFAULT_LABOR_RATE_USD_PER_HOUR = 30.0
DEFAULT_PART_COST_RATIO = 0.15
DEFAULT_LABOR_MINUTES = 45


@dataclass
class EconomicBreakdown:
    device_model: str
    msrp_usd: float
    part_cost_usd: float
    labor_minutes: int
    labor_rate_usd_per_hour: float
    labor_cost_usd: float
    total_repair_cost_usd: float
    replacement_cost_usd: float
    cost_ratio: float
    recommended_action: str
    cost_source: str


def _lookup_msrp(device_model: str | None) -> float:
    if not device_model:
        return DEFAULT_MSRP_USD
    result = query_df(
        "SELECT msrp_usd FROM dim_device WHERE lower(model_name) = lower(?)", [device_model]
    )
    if result.empty:
        return DEFAULT_MSRP_USD
    return float(result.iloc[0]["msrp_usd"])


def _lookup_labor_rate(depot_key: int | None) -> float:
    if depot_key is not None:
        result = query_df(
            "SELECT avg_labor_rate_usd FROM dim_repair_depot WHERE depot_key = ?", [depot_key]
        )
        if not result.empty:
            return float(result.iloc[0]["avg_labor_rate_usd"])
    result = query_df("SELECT AVG(avg_labor_rate_usd) AS avg_rate FROM dim_repair_depot")
    return float(result.iloc[0]["avg_rate"])


def _parse_evidence(evidence_chunks: list[dict]) -> tuple[float | None, int | None]:
    for item in evidence_chunks:
        text = item["chunk"].raw_text if "chunk" in item else item.get("text", "")
        part_match = PART_COST_RE.search(text)
        labor_match = LABOR_RE.search(text)
        if part_match and labor_match:
            return float(part_match.group(1)), int(labor_match.group(1))
    return None, None


def compute_economics(
    device_model: str | None,
    evidence_chunks: list[dict],
    depot_key: int | None = None,
) -> EconomicBreakdown:
    msrp = _lookup_msrp(device_model)
    labor_rate = _lookup_labor_rate(depot_key)

    part_cost, labor_minutes = _parse_evidence(evidence_chunks)
    if part_cost is not None and labor_minutes is not None:
        cost_source = "sop_evidence"
    else:
        part_cost = msrp * DEFAULT_PART_COST_RATIO
        labor_minutes = DEFAULT_LABOR_MINUTES
        cost_source = "device_default_fallback"

    labor_cost = (labor_minutes / 60.0) * labor_rate
    total_repair_cost = part_cost + labor_cost
    replacement_cost = msrp * 0.85
    cost_ratio = total_repair_cost / msrp if msrp else 1.0
    recommended_action = "RECOMMEND_REPAIR" if cost_ratio < REPAIR_THRESHOLD else "RECOMMEND_REPLACEMENT"

    return EconomicBreakdown(
        device_model=device_model or "unknown",
        msrp_usd=round(msrp, 2),
        part_cost_usd=round(part_cost, 2),
        labor_minutes=labor_minutes,
        labor_rate_usd_per_hour=round(labor_rate, 2),
        labor_cost_usd=round(labor_cost, 2),
        total_repair_cost_usd=round(total_repair_cost, 2),
        replacement_cost_usd=round(replacement_cost, 2),
        cost_ratio=round(cost_ratio, 4),
        recommended_action=recommended_action,
        cost_source=cost_source,
    )


if __name__ == "__main__":
    breakdown = compute_economics("iPhone 15", [])
    print(breakdown)
