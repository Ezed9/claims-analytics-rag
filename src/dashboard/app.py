import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.orchestrator import run_triage
from src.db_engine import query_df

VERDICT_STYLE = {
    "RECOMMEND_REPAIR": ("Repair", "green", ":material/build:"),
    "RECOMMEND_REPLACEMENT": ("Replace", "violet", ":material/swap_horiz:"),
    "MANUAL_REVIEW": ("Manual review", "orange", ":material/visibility:"),
    "DENIED_NOT_COVERED": ("Denied: not covered", "red", ":material/block:"),
    "DENIED_WAITING_PERIOD": ("Denied: waiting period", "red", ":material/schedule:"),
}
ACTION_COLORS = {
    "REPAIR": "#0F62FE",
    "REPLACEMENT": "#6929C4",
    "DENIED_NOT_COVERED": "#525252",
    "DENIED_WAITING_PERIOD": "#A8A8A8",
}
EXAMPLES = {
    "Cracked screen": ("Customer's iPhone 15 has a cracked screen after a drop.", "Verizon"),
    "Water damage": ("iPhone 15 Pro fell in a pool and won't turn on.", "Verizon"),
    "Stolen phone": ("My iPhone 15 was stolen at the airport.", "Verizon"),
    "Unsupported device": ("Galaxy S23 screen is cracked.", "Verizon"),
}


@st.cache_data(ttl=600)
def load_kpis() -> dict:
    return query_df(
        """
        SELECT
            COUNT(*) AS total_claims,
            AVG(CASE WHEN is_first_contact_resolved THEN 1.0 ELSE 0.0 END) AS fcr_rate,
            AVG(triage_cycle_time_hours) AS avg_cycle_time_hours,
            SUM(net_payout) AS total_net_payout
        FROM fact_claims
        """
    ).iloc[0].to_dict()


@st.cache_data(ttl=600)
def load_carrier_volume() -> pd.DataFrame:
    return query_df(
        """
        SELECT p.carrier_partner AS carrier, COUNT(*) AS claims
        FROM fact_claims fc JOIN dim_policy p ON p.policy_key = fc.policy_key
        GROUP BY 1 ORDER BY claims DESC
        """
    )


@st.cache_data(ttl=600)
def load_resolution_breakdown() -> pd.DataFrame:
    return query_df(
        "SELECT recommended_action AS action, COUNT(*) AS claims FROM fact_claims GROUP BY 1"
    )


@st.cache_data(ttl=600)
def load_tier_counts() -> pd.DataFrame:
    return query_df("SELECT risk_tier AS tier, COUNT(*) AS claims FROM vw_claim_velocity_risk GROUP BY 1")


@st.cache_data(ttl=600)
def load_velocity_table(tiers: tuple[str, ...], model_filter: str | None) -> pd.DataFrame:
    sql = """
        SELECT claim_key, model_name, incident_type, filing_date, claims_in_30d,
               net_payout, velocity_score, risk_tier
        FROM vw_claim_velocity_risk WHERE 1=1
    """
    params: list = []
    if tiers:
        sql += f" AND risk_tier IN ({', '.join('?' for _ in tiers)})"
        params.extend(tiers)
    if model_filter:
        sql += " AND model_name = ?"
        params.append(model_filter)
    sql += " ORDER BY velocity_score DESC LIMIT 500"
    return query_df(sql, params)


@st.cache_data(ttl=600)
def load_depot_ranking() -> pd.DataFrame:
    return query_df("SELECT * FROM vw_depot_cycle_time_ranking ORDER BY avg_cycle_time_hours")


@st.cache_data(ttl=600)
def load_available_models() -> list[str]:
    return query_df("SELECT DISTINCT model_name FROM dim_device ORDER BY model_name")["model_name"].tolist()


@st.cache_data(ttl=600)
def load_carriers() -> list[str]:
    return query_df("SELECT DISTINCT carrier_partner FROM dim_policy ORDER BY 1")["carrier_partner"].tolist()


def render_overview_tab() -> None:
    kpis = load_kpis()
    with st.container(horizontal=True):
        st.metric("Total claims", f"{int(kpis['total_claims']):,}", border=True)
        st.metric("First-contact resolution", f"{kpis['fcr_rate'] * 100:.1f}%", border=True)
        st.metric("Avg triage cycle time", f"{kpis['avg_cycle_time_hours']:.1f} hrs", border=True)
        st.metric("Total net payout", f"${kpis['total_net_payout'] / 1e6:,.1f}M", border=True)

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Claim volume by carrier")
        chart = (
            alt.Chart(load_carrier_volume())
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#0F62FE")
            .encode(x=alt.X("carrier:N", sort="-y", title=None), y=alt.Y("claims:Q", title="Claims"),
                    tooltip=["carrier", "claims"])
        )
        st.altair_chart(chart, height=320)
    with right, st.container(border=True):
        st.subheader("Recommended action mix")
        breakdown = load_resolution_breakdown()
        chart = (
            alt.Chart(breakdown)
            .mark_arc(innerRadius=70)
            .encode(theta="claims:Q", color=alt.Color(
                "action:N",
                title=None,
                scale=alt.Scale(domain=list(ACTION_COLORS), range=list(ACTION_COLORS.values())),
            ), tooltip=["action", "claims"])
        )
        st.altair_chart(chart, height=320)


def render_velocity_tab() -> None:
    st.caption("Claims are scored on 30-day filing velocity and payout z-score per policy and device.")
    tier_counts = load_tier_counts().set_index("tier")["claims"]
    with st.container(horizontal=True):
        for tier in ("ELEVATED", "MODERATE", "LOW"):
            st.metric(f"{tier.title()} risk", f"{int(tier_counts.get(tier, 0)):,}", border=True)

    with st.container(border=True):
        col_a, col_b = st.columns([2, 1])
        tiers = col_a.pills(
            "Risk tier", ["ELEVATED", "MODERATE", "LOW"], selection_mode="multi", default=["ELEVATED"]
        )
        model_choice = col_b.selectbox("Device model", ["All", *load_available_models()])
        velocity = load_velocity_table(tuple(tiers or ()), None if model_choice == "All" else model_choice)
        st.dataframe(
            velocity,
            hide_index=True,
            column_config={
                "velocity_score": st.column_config.ProgressColumn(
                    "Velocity score", format="%.1f", min_value=0, max_value=100
                ),
                "net_payout": st.column_config.NumberColumn("Net payout", format="$%.2f"),
                "filing_date": st.column_config.DateColumn("Filed"),
                "claims_in_30d": "Claims in 30d",
                "model_name": "Device",
                "incident_type": "Incident",
                "claim_key": "Claim",
                "risk_tier": "Tier",
            },
        )
        st.caption(f"Showing top {len(velocity)} claims by velocity score (max 500).")

    with st.container(border=True):
        st.subheader("Depot cycle time (lower is better)")
        chart = (
            alt.Chart(load_depot_ranking())
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                y=alt.Y("depot_name:N", sort="x", title=None),
                x=alt.X("avg_cycle_time_hours:Q", title="Avg cycle time (hrs)"),
                color=alt.Color("region:N", title="Region"),
                tooltip=["depot_name", "region", "avg_cycle_time_hours", "fcr_rate"],
            )
        )
        st.altair_chart(chart, height=280)


def render_result(result: dict) -> None:
    label, color, icon = VERDICT_STYLE.get(result["verdict"], (result["verdict"], "gray", ""))
    with st.container(border=True):
        st.markdown(f"### :{color}[{icon} {label}]")
        st.caption(f"Extraction by `{result['model_name']}` · audit `{Path(result['audit_path']).name}`")

        ext, pol = result["extraction"], result["policy_check"]
        with st.container(horizontal=True):
            st.metric("Device", ext.get("device_model") or "Unknown", border=True)
            st.metric("Incident", ext["incident_type"].replace("_", " ").title(), border=True)
            st.metric("Deductible", f"${pol['deductible_usd']:,.0f}" if "deductible_usd" in pol else "n/a", border=True)
            st.metric("Confidence", f"{ext['confidence']:.0%}", border=True)

        if econ := result["economics"]:
            with st.container(border=True):
                st.markdown("**Repair vs. replace**")
                st.progress(
                    min(econ["cost_ratio"], 1.0),
                    text=f"Repair \\${econ['total_repair_cost_usd']:,.0f} of \\${econ['msrp_usd']:,.0f} MSRP "
                    f"({econ['cost_ratio']:.0%}, replace above 65%)",
                )

    with st.expander(":material/gavel: Policy check"):
        st.json(pol)
    ev = result["evidence"].get("evidence", [])
    with st.expander(f":material/description: Evidence ({len(ev)} chunks)"):
        if not ev:
            st.caption("No repair evidence was needed or none passed the reranker threshold.")
        for item in ev:
            st.markdown(f"**{item['heading_path']}** · score {item['rerank_score']:.2f}")
            st.caption(item["text"])
    audit_text = Path(result["audit_path"]).read_text()
    with st.expander(":material/verified: Audit log"):
        st.code(audit_text, language="json")
        st.download_button("Download audit JSON", audit_text, file_name=Path(result["audit_path"]).name)


def render_triage_tab() -> None:
    st.caption("An LLM only extracts entities. Coverage and the repair/replace verdict come from SQL rules and a cost model.")
    example = st.pills("Try an example", list(EXAMPLES), key="example")
    default_text, default_carrier = EXAMPLES.get(example, ("Customer's iPhone 15 has a cracked screen.", None))
    carriers = load_carriers()
    with st.form("triage"):
        claim_text = st.text_area("Claim narrative", default_text, key=f"text_{example}")
        carrier = st.selectbox(
            "Carrier", carriers, index=carriers.index(default_carrier) if default_carrier in carriers else 0,
            key=f"carrier_{example}",
        )
        submitted = st.form_submit_button("Run triage", type="primary", icon=":material/play_arrow:")
    if submitted:
        with st.spinner("Running triage..."):
            st.session_state["result"] = run_triage(claim_text, carrier)
    if "result" in st.session_state:
        render_result(st.session_state["result"])


def main() -> None:
    st.set_page_config(page_title="Claims triage", page_icon=":material/shield:", layout="wide")
    st.title(":material/shield: Warranty claims analytics & policy triage")
    st.caption("DuckDB warehouse · velocity fraud scoring · hybrid RAG · auditable triage agent")

    overview, velocity, triage = st.tabs(
        [":material/dashboard: Overview", ":material/query_stats: Velocity & anomalies", ":material/gavel: Claim triage"]
    )
    with overview:
        render_overview_tab()
    with velocity:
        render_velocity_tab()
    with triage:
        render_triage_tab()


if __name__ == "__main__":
    main()
