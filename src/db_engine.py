import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DB_PATH


@contextmanager
def get_connection(read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def query_df(sql: str, params: list | None = None, read_only: bool = True) -> pd.DataFrame:
    with get_connection(read_only=read_only) as con:
        return con.execute(sql, params or []).df()


def run_sql_file(path: Path, read_only: bool = False) -> None:
    with get_connection(read_only=read_only) as con:
        con.execute(path.read_text())


def get_coverage_rule(
    incident_type: str, policy_key: int | None = None, carrier: str | None = None
) -> pd.DataFrame:
    if policy_key is not None:
        sql = """
            SELECT r.*, p.carrier_partner, p.plan_name
            FROM policy_coverage_rules r
            JOIN dim_policy p ON p.policy_key = r.policy_key
            WHERE r.policy_key = ? AND r.incident_type = ?
        """
        params = [policy_key, incident_type]
    elif carrier is not None:
        sql = """
            SELECT r.*, p.carrier_partner, p.plan_name
            FROM policy_coverage_rules r
            JOIN dim_policy p ON p.policy_key = r.policy_key
            WHERE p.carrier_partner = ? AND r.incident_type = ?
            ORDER BY p.policy_key
        """
        params = [carrier, incident_type]
    else:
        raise ValueError("Either policy_key or carrier must be provided.")
    return query_df(sql, params)


def get_velocity_scores(tier: str | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM vw_claim_velocity_risk"
    if tier is not None:
        sql += " WHERE risk_tier = ?"
        return query_df(sql, [tier])
    return query_df(sql)


def test_queries() -> None:
    with get_connection(read_only=True) as con:
        row_count = con.execute("SELECT COUNT(*) FROM fact_claims").fetchone()[0]
        print(f"fact_claims row count: {row_count}")

    rule_sample = get_coverage_rule("SCREEN_DAMAGE", carrier="Verizon")
    print("Sample coverage rule (Verizon / SCREEN_DAMAGE):")
    print(rule_sample.to_string(index=False))

    velocity = get_velocity_scores()
    print("Velocity risk tier distribution:")
    print(velocity["risk_tier"].value_counts().to_string())


if __name__ == "__main__":
    test_queries()
