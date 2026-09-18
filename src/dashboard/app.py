import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.orchestrator import run_triage
from src.db_engine import query_df


def load_kpis() -> dict:
    row = query_df(
        """
        SELECT
            COUNT(*) AS total_claims,
            ROUND(AVG(CASE WHEN is_first_contact_resolved THEN 1.0 ELSE 0.0 END), 3) AS fcr_rate,
            ROUND(AVG(triage_cycle_time_hours), 2) AS avg_cycle_time_hours,
            ROUND(SUM(net_payout), 2) AS total_net_payout
        FROM fact_claims
        """
    ).iloc[0]
    return row.to_dict()


def load_carrier_volume() -> pd.DataFrame:
    return query_df(
        """
        SELECT p.carrier_partner, COUNT(*) AS claim_count
        FROM fact_claims fc
        JOIN dim_policy p ON p.policy_key = fc.policy_key
        GROUP BY p.carrier_partner
        ORDER BY claim_count DESC
        """
    )


def load_resolution_breakdown() -> pd.DataFrame:
    return query_df(
        """
        SELECT recommended_action, COUNT(*) AS claim_count
        FROM fact_claims
        GROUP BY recommended_action
        ORDER BY claim_count DESC
        """
    )


def load_velocity_table(tiers: list[str] | None, model_filter: str | None) -> pd.DataFrame:
    sql = "SELECT * FROM vw_claim_velocity_risk WHERE 1=1"
    params: list = []
    if tiers:
        placeholders = ", ".join("?" for _ in tiers)
        sql += f" AND risk_tier IN ({placeholders})"
        params.extend(tiers)
    if model_filter:
        sql += " AND model_name = ?"
        params.append(model_filter)
    sql += " ORDER BY velocity_score DESC LIMIT 500"
    return query_df(sql, params)


def load_depot_ranking() -> pd.DataFrame:
    return query_df("SELECT * FROM vw_depot_cycle_time_ranking")


def load_available_models() -> list[str]:
    return query_df("SELECT DISTINCT model_name FROM dim_device ORDER BY model_name")[
        "model_name"
    ].tolist()


def load_carriers() -> list[str]:
    return query_df("SELECT DISTINCT carrier_partner FROM dim_policy ORDER BY carrier_partner")[
        "carrier_partner"
    ].tolist()


def render_overview_tab() -> None:
    kpis = load_kpis()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Claims", f"{int(kpis['total_claims']):,}")
    col2.metric("First-Contact Resolution", f"{kpis['fcr_rate'] * 100:.1f}%")
    col3.metric("Avg Triage Cycle Time", f"{kpis['avg_cycle_time_hours']:.1f} hrs")
    col4.metric("Total Net Payout", f"${kpis['total_net_payout']:,.0f}")

    col_left, col_right = st.columns(2)
    with col_left:
        volume = load_carrier_volume()
        fig = px.bar(volume, x="carrier_partner", y="claim_count", title="Claim Volume by Carrier")
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        breakdown = load_resolution_breakdown()
        fig = px.pie(
            breakdown,
            names="recommended_action",
            values="claim_count",
            hole=0.45,
            title="Recommended Action Mix",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_velocity_tab() -> None:
    tiers = st.multiselect("Risk Tier", ["ELEVATED", "MODERATE", "LOW"], default=["ELEVATED"])
    models = ["All"] + load_available_models()
    model_choice = st.selectbox("Device Model", models)
    model_filter = None if model_choice == "All" else model_choice

    velocity = load_velocity_table(tiers or None, model_filter)
    st.dataframe(velocity, use_container_width=True)

    depot_ranking = load_depot_ranking()
    fig = px.bar(
        depot_ranking,
        x="depot_name",
        y="avg_cycle_time_hours",
        color="region",
        title="Depot Cycle Time Ranking (lower is better)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_triage_tab() -> None:
    claim_text = st.text_area("Claim Narrative", "Customer's iPhone 15 has a cracked screen.")
    carrier = st.selectbox("Carrier", load_carriers())

    if st.button("Run Triage"):
        result = run_triage(claim_text, carrier)
        st.subheader(f"Verdict: {result['verdict']}")

        st.markdown("**Extraction**")
        st.json(result["extraction"])

        st.markdown("**Policy Check**")
        st.json(result["policy_check"])

        st.markdown("**Evidence Chunks**")
        st.json(result["evidence"])

        st.markdown("**Economic Breakdown**")
        st.json(result["economics"])

        st.markdown("**Audit Log**")
        audit_path = Path(result["audit_path"])
        audit_text = audit_path.read_text()
        st.json(audit_text)
        st.download_button("Download Audit JSON", audit_text, file_name=audit_path.name)


def main() -> None:
    st.set_page_config(page_title="Claims Analytics & Policy Triage", layout="wide")
    st.title("Warranty Claims Analytics & Policy Triage")

    tab1, tab2, tab3 = st.tabs(["Overview", "Velocity & Anomalies", "Claim Triage"])
    with tab1:
        render_overview_tab()
    with tab2:
        render_velocity_tab()
    with tab3:
        render_triage_tab()


if __name__ == "__main__":
    main()
