import random
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from faker import Faker

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    DB_PATH,
    FRAUD_RING_RATE,
    NUM_CLAIMS,
    RANDOM_SEED,
    REPAIR_THRESHOLD,
    SCHEMA_DDL_PATH,
    VELOCITY_SQL_PATH,
)

INCIDENT_TYPES = [
    "SCREEN_DAMAGE",
    "LIQUID_DAMAGE",
    "BATTERY",
    "LOST_STOLEN",
    "HARDWARE_MALFUNCTION",
    "OTHER_DAMAGE",
]
INCIDENT_TYPE_WEIGHTS = [0.38, 0.14, 0.16, 0.10, 0.12, 0.10]

REPAIR_COST_RATIO = {
    "SCREEN_DAMAGE": 0.12,
    "LIQUID_DAMAGE": 0.34,
    "BATTERY": 0.08,
    "LOST_STOLEN": 0.0,
    "HARDWARE_MALFUNCTION": 0.15,
    "OTHER_DAMAGE": 0.20,
}

SYMPTOM_TEMPLATES = {
    "SCREEN_DAMAGE": [
        "Customer reports a cracked screen on their {device} after dropping it {context}.",
        "Screen is shattered on the {device}; touch input is unresponsive {context}.",
        "Spiderweb crack across the {device} display with glass shards visible {context}.",
    ],
    "LIQUID_DAMAGE": [
        "{device} was submerged in water {context} and now will not power on.",
        "Customer spilled a drink on their {device} {context}; charging port works intermittently.",
        "Liquid contact indicator triggered on {device} after exposure to rain {context}.",
    ],
    "BATTERY": [
        "Battery on the {device} drains fully within two hours {context}.",
        "{device} shuts down unexpectedly at 30% battery {context}.",
        "Customer says the {device} battery has swelled and the case is bulging {context}.",
    ],
    "LOST_STOLEN": [
        "{device} was stolen from the customer's bag {context}.",
        "Customer lost their {device} {context} and cannot locate it via Find My.",
        "{device} was taken during a break-in {context}.",
    ],
    "HARDWARE_MALFUNCTION": [
        "{device} camera module stopped focusing {context}.",
        "Speaker on the {device} produces no sound {context}.",
        "{device} biometric sensor fails intermittently {context}.",
    ],
    "OTHER_DAMAGE": [
        "{device} back glass is cracked after a fall {context}.",
        "{device} charging port is bent and the cable will not seat {context}.",
        "Customer reports cosmetic frame damage to the {device} {context}.",
    ],
}

CONTEXT_PHRASES = [
    "last week",
    "yesterday",
    "during a hiking trip",
    "at the gym",
    "on public transit",
    "while at work",
    "over the weekend",
    "during a rainstorm",
    "at home",
    "while traveling",
]


def build_dim_device() -> pd.DataFrame:
    rows = [
        ("Apple", "iPhone 13", 2021, 699.00, 6),
        ("Apple", "iPhone 13 mini", 2021, 599.00, 6),
        ("Apple", "iPhone 13 Pro", 2021, 999.00, 5),
        ("Apple", "iPhone 13 Pro Max", 2021, 1099.00, 5),
        ("Apple", "iPhone 14", 2022, 799.00, 6),
        ("Apple", "iPhone 14 Plus", 2022, 899.00, 6),
        ("Apple", "iPhone 14 Pro", 2022, 999.00, 5),
        ("Apple", "iPhone 14 Pro Max", 2022, 1099.00, 5),
        ("Apple", "iPhone 15", 2023, 799.00, 6),
        ("Apple", "iPhone 15 Plus", 2023, 899.00, 6),
        ("Apple", "iPhone 15 Pro", 2023, 999.00, 5),
        ("Apple", "iPhone 15 Pro Max", 2023, 1199.00, 5),
        ("Apple", "iPhone 16", 2024, 799.00, 6),
        ("Apple", "iPhone 16 Plus", 2024, 899.00, 6),
        ("Apple", "iPhone 16 Pro", 2024, 999.00, 5),
        ("Apple", "iPhone 16 Pro Max", 2024, 1199.00, 5),
        ("Samsung", "Galaxy S23", 2023, 799.00, 7),
        ("Samsung", "Galaxy S24", 2024, 799.00, 7),
        ("Google", "Pixel 8", 2023, 699.00, 7),
        ("Google", "Pixel 9", 2024, 799.00, 7),
    ]
    df = pd.DataFrame(
        rows, columns=["brand", "model_name", "release_year", "msrp_usd", "repairability_score"]
    )
    df.insert(0, "device_key", range(1, len(df) + 1))
    return df


def build_dim_policy() -> pd.DataFrame:
    rows = [
        ("Verizon", "Verizon Total Mobile Protection", 17.00, 3),
        ("Apple", "AppleCare+", 9.99, None),
        ("Apple", "AppleCare+ with Theft and Loss", 13.99, None),
        ("T-Mobile", "Asurion T-Mobile Protection<Plus>", 18.00, 3),
        ("AT&T", "AT&T Protect Advantage", 17.00, 3),
    ]
    df = pd.DataFrame(
        rows, columns=["carrier_partner", "plan_name", "monthly_premium_usd", "max_claims_annual"]
    )
    df.insert(0, "policy_key", range(1, len(df) + 1))
    return df


def build_policy_coverage_rules(dim_policy: pd.DataFrame) -> pd.DataFrame:
    plan_rules = {
        "Verizon Total Mobile Protection": {
            "wait": 15,
            "rules": {
                "SCREEN_DAMAGE": (True, 29.00, True, True),
                "LIQUID_DAMAGE": (True, 99.00, True, True),
                "LOST_STOLEN": (True, 199.00, False, True),
                "BATTERY": (True, 49.00, True, False),
                "HARDWARE_MALFUNCTION": (True, 49.00, True, True),
                "OTHER_DAMAGE": (True, 99.00, True, True),
            },
        },
        "AppleCare+": {
            "wait": 0,
            "rules": {
                "SCREEN_DAMAGE": (True, 29.00, True, False),
                "LIQUID_DAMAGE": (True, 99.00, True, True),
                "LOST_STOLEN": (False, 0.00, False, False),
                "BATTERY": (True, 99.00, True, False),
                "HARDWARE_MALFUNCTION": (True, 0.00, True, True),
                "OTHER_DAMAGE": (True, 99.00, True, True),
            },
        },
        "AppleCare+ with Theft and Loss": {
            "wait": 0,
            "rules": {
                "SCREEN_DAMAGE": (True, 29.00, True, False),
                "LIQUID_DAMAGE": (True, 99.00, True, True),
                "LOST_STOLEN": (True, 99.00, False, True),
                "BATTERY": (True, 99.00, True, False),
                "HARDWARE_MALFUNCTION": (True, 0.00, True, True),
                "OTHER_DAMAGE": (True, 99.00, True, True),
            },
        },
        "Asurion T-Mobile Protection<Plus>": {
            "wait": 30,
            "rules": {
                "SCREEN_DAMAGE": (True, 29.00, True, False),
                "LIQUID_DAMAGE": (True, 149.00, True, True),
                "LOST_STOLEN": (True, 229.00, False, True),
                "BATTERY": (True, 49.00, True, False),
                "HARDWARE_MALFUNCTION": (True, 49.00, True, True),
                "OTHER_DAMAGE": (True, 149.00, True, True),
            },
        },
        "AT&T Protect Advantage": {
            "wait": 30,
            "rules": {
                "SCREEN_DAMAGE": (True, 29.00, True, False),
                "LIQUID_DAMAGE": (True, 125.00, True, True),
                "LOST_STOLEN": (True, 225.00, False, True),
                "BATTERY": (True, 49.00, True, False),
                "HARDWARE_MALFUNCTION": (True, 49.00, True, True),
                "OTHER_DAMAGE": (True, 125.00, True, True),
            },
        },
    }
    records = []
    rule_id = 1
    for _, policy_row in dim_policy.iterrows():
        plan = plan_rules[policy_row["plan_name"]]
        wait_days = plan["wait"]
        for incident_type in INCIDENT_TYPES:
            is_covered, deductible, repair_allowed, replacement_allowed = plan["rules"][incident_type]
            records.append(
                (
                    rule_id,
                    policy_row["policy_key"],
                    incident_type,
                    is_covered,
                    deductible,
                    wait_days if is_covered else 0,
                    repair_allowed,
                    replacement_allowed,
                )
            )
            rule_id += 1
    return pd.DataFrame(
        records,
        columns=[
            "rule_id",
            "policy_key",
            "incident_type",
            "is_covered",
            "deductible_usd",
            "waiting_period_days",
            "repair_allowed",
            "replacement_allowed",
        ],
    )


def build_dim_repair_depot() -> pd.DataFrame:
    rows = [
        ("TechFix Northeast Hub", "Northeast", 32.00),
        ("TechFix Southeast Hub", "Southeast", 27.00),
        ("TechFix Midwest Hub", "Midwest", 26.00),
        ("TechFix Southwest Hub", "Southwest", 28.00),
        ("TechFix West Coast Hub", "West", 38.00),
        ("TechFix Pacific Northwest Hub", "Pacific Northwest", 35.00),
    ]
    df = pd.DataFrame(rows, columns=["depot_name", "region", "avg_labor_rate_usd"])
    df.insert(0, "depot_key", range(1, len(df) + 1))
    return df


def build_dim_date() -> pd.DataFrame:
    full_dates = pd.date_range(start="2023-10-01", periods=731, freq="D")
    df = pd.DataFrame({"full_date": full_dates})
    df["date_key"] = df["full_date"].dt.strftime("%Y%m%d").astype(int)
    df["calendar_year"] = df["full_date"].dt.year
    df["calendar_month"] = df["full_date"].dt.month
    return df[["date_key", "full_date", "calendar_year", "calendar_month"]]


def _generate_narratives(
    rng: np.random.Generator, incident_types: np.ndarray, device_names: np.ndarray
) -> np.ndarray:
    narratives = np.empty(len(incident_types), dtype=object)
    for incident_type in INCIDENT_TYPES:
        mask = incident_types == incident_type
        count = int(mask.sum())
        if count == 0:
            continue
        templates = rng.choice(SYMPTOM_TEMPLATES[incident_type], size=count)
        contexts = rng.choice(CONTEXT_PHRASES, size=count)
        devices_subset = device_names[mask]
        narratives[mask] = [
            template.format(device=device, context=context)
            for template, device, context in zip(templates, devices_subset, contexts)
        ]
    return narratives


def _assign_actions(
    is_covered_final: np.ndarray,
    repair_allowed: np.ndarray,
    replacement_allowed: np.ndarray,
    cost_ratio: np.ndarray,
    incident_types: np.ndarray,
    rule_covered: np.ndarray,
    waiting_violation: np.ndarray,
) -> np.ndarray:
    action = np.empty(len(is_covered_final), dtype=object)
    action[~rule_covered] = "DENIED_NOT_COVERED"
    action[rule_covered & waiting_violation] = "DENIED_WAITING_PERIOD"

    approved = is_covered_final
    prefer_repair = approved & repair_allowed & (cost_ratio < REPAIR_THRESHOLD)
    action[prefer_repair] = "REPAIR"

    prefer_replacement = approved & ~prefer_repair & replacement_allowed
    action[prefer_replacement] = "REPLACEMENT"

    fallback = approved & (action == None)  # noqa: E711
    action[fallback & repair_allowed] = "REPAIR"
    action[fallback & ~repair_allowed] = "REPLACEMENT"
    return action


def _build_claims_block(
    rng: np.random.Generator,
    n: int,
    dim_device: pd.DataFrame,
    dim_policy: pd.DataFrame,
    coverage_rules: pd.DataFrame,
    dim_repair_depot: pd.DataFrame,
    dim_date: pd.DataFrame,
    fraud_multiplier: np.ndarray,
    fixed_policy_key: np.ndarray | None = None,
    fixed_device_key: np.ndarray | None = None,
    fixed_date_key: np.ndarray | None = None,
    incident_weights: list[float] | None = None,
) -> pd.DataFrame:
    device_key = (
        fixed_device_key
        if fixed_device_key is not None
        else rng.integers(1, len(dim_device) + 1, n)
    )
    policy_key = (
        fixed_policy_key
        if fixed_policy_key is not None
        else rng.integers(1, len(dim_policy) + 1, n)
    )
    depot_key = rng.integers(1, len(dim_repair_depot) + 1, n)
    date_key = (
        fixed_date_key
        if fixed_date_key is not None
        else rng.choice(dim_date["date_key"].to_numpy(), size=n)
    )
    incident_type = rng.choice(
        INCIDENT_TYPES, size=n, p=incident_weights if incident_weights else INCIDENT_TYPE_WEIGHTS
    )

    base = pd.DataFrame(
        {
            "policy_key": policy_key,
            "device_key": device_key,
            "depot_key": depot_key,
            "filing_date_key": date_key,
            "incident_type": incident_type,
        }
    )
    base = base.merge(
        dim_device[["device_key", "brand", "model_name", "msrp_usd"]], on="device_key", how="left"
    )
    base = base.merge(
        coverage_rules[
            [
                "policy_key",
                "incident_type",
                "is_covered",
                "deductible_usd",
                "waiting_period_days",
                "repair_allowed",
                "replacement_allowed",
            ]
        ],
        on=["policy_key", "incident_type"],
        how="left",
    )

    rule_covered = base["is_covered"].to_numpy(dtype=bool)
    waiting_violation = (rng.random(n) < 0.05) & (base["waiting_period_days"].to_numpy() > 0)
    is_covered_final = rule_covered & ~waiting_violation

    msrp = base["msrp_usd"].to_numpy(dtype=float)
    base_ratio = np.array([REPAIR_COST_RATIO[t] for t in incident_type])
    noise = rng.normal(1.0, 0.15, n).clip(0.6, 1.6)
    repair_estimate = (base_ratio * msrp * noise * fraud_multiplier).clip(min=15.0)
    replacement_estimate = msrp * rng.uniform(0.75, 0.95, n) * fraud_multiplier
    cost_ratio = repair_estimate / msrp

    action = _assign_actions(
        is_covered_final,
        base["repair_allowed"].to_numpy(dtype=bool),
        base["replacement_allowed"].to_numpy(dtype=bool),
        cost_ratio,
        incident_type,
        rule_covered,
        waiting_violation,
    )

    actual_repair_cost = np.where(action == "REPAIR", repair_estimate, 0.0)
    replacement_device_cost = np.where(action == "REPLACEMENT", replacement_estimate, 0.0)
    deductible_collected = np.where(is_covered_final, base["deductible_usd"].to_numpy(), 0.0)
    cost_used = np.where(action == "REPAIR", actual_repair_cost, replacement_device_cost)
    net_payout = np.where(
        is_covered_final, np.clip(cost_used - deductible_collected, 0.0, None), 0.0
    )

    adjudication_status = np.where(is_covered_final, "APPROVED", "DENIED")

    cycle_base = rng.lognormal(mean=np.log(24), sigma=0.6, size=n)
    triage_cycle_time_hours = (cycle_base * np.where(fraud_multiplier > 1.0, 1.8, 1.0)).clip(1, 300)
    fcr_prob = np.where(action == "REPAIR", 0.78, 0.55)
    fcr_prob = np.where(fraud_multiplier > 1.0, fcr_prob * 0.5, fcr_prob)
    is_first_contact_resolved = rng.random(n) < fcr_prob

    device_names = base["model_name"].to_numpy(dtype=object)
    narrative = _generate_narratives(rng, incident_type, device_names)

    return pd.DataFrame(
        {
            "policy_key": policy_key,
            "device_key": device_key,
            "depot_key": depot_key,
            "filing_date_key": date_key,
            "incident_type": incident_type,
            "claim_narrative": narrative,
            "adjudication_status": adjudication_status,
            "recommended_action": action,
            "actual_repair_cost": np.round(actual_repair_cost, 2),
            "replacement_device_cost": np.round(replacement_device_cost, 2),
            "deductible_collected": np.round(deductible_collected, 2),
            "net_payout": np.round(net_payout, 2),
            "triage_cycle_time_hours": np.round(triage_cycle_time_hours, 2),
            "is_first_contact_resolved": is_first_contact_resolved,
        }
    )


def build_fact_claims(
    dim_device: pd.DataFrame,
    dim_policy: pd.DataFrame,
    coverage_rules: pd.DataFrame,
    dim_repair_depot: pd.DataFrame,
    dim_date: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    fraud_count = round(NUM_CLAIMS * FRAUD_RING_RATE)
    normal_count = NUM_CLAIMS - fraud_count

    normal_df = _build_claims_block(
        rng,
        normal_count,
        dim_device,
        dim_policy,
        coverage_rules,
        dim_repair_depot,
        dim_date,
        fraud_multiplier=np.ones(normal_count),
    )

    num_clusters = 90
    cluster_sizes = rng.integers(30, 50, num_clusters)
    cluster_sizes[-1] += fraud_count - cluster_sizes.sum()

    fraud_policy_key = np.repeat(rng.integers(1, len(dim_policy) + 1, num_clusters), cluster_sizes)
    fraud_device_key = np.repeat(rng.integers(1, len(dim_device) + 1, num_clusters), cluster_sizes)

    date_pool = dim_date["date_key"].to_numpy()
    cluster_start_idx = rng.integers(0, len(date_pool) - 10, num_clusters)
    cluster_windows = rng.integers(5, 10, num_clusters)
    fraud_date_key = np.concatenate(
        [
            date_pool[start + rng.integers(0, window, size)]
            for start, window, size in zip(cluster_start_idx, cluster_windows, cluster_sizes)
        ]
    )

    fraud_df = _build_claims_block(
        rng,
        int(fraud_count),
        dim_device,
        dim_policy,
        coverage_rules,
        dim_repair_depot,
        dim_date,
        fraud_multiplier=rng.uniform(1.5, 2.5, int(fraud_count)),
        fixed_policy_key=fraud_policy_key,
        fixed_device_key=fraud_device_key,
        fixed_date_key=fraud_date_key,
        incident_weights=[0.30, 0.10, 0.05, 0.45, 0.05, 0.05],
    )

    claims = pd.concat([normal_df, fraud_df], ignore_index=True)
    claims = claims.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    claims.insert(0, "claim_key", range(1, len(claims) + 1))
    claims.insert(1, "claim_id", [f"CLM-{key:07d}" for key in claims["claim_key"]])
    return claims


def load_warehouse(db_path: Path = DB_PATH) -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    Faker.seed(RANDOM_SEED)

    dim_device = build_dim_device()
    dim_policy = build_dim_policy()
    coverage_rules = build_policy_coverage_rules(dim_policy)
    dim_repair_depot = build_dim_repair_depot()
    dim_date = build_dim_date()
    fact_claims = build_fact_claims(dim_device, dim_policy, coverage_rules, dim_repair_depot, dim_date)

    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        con.execute(SCHEMA_DDL_PATH.read_text())
        con.register("dim_device_df", dim_device)
        con.register("dim_policy_df", dim_policy)
        con.register("policy_coverage_rules_df", coverage_rules)
        con.register("dim_repair_depot_df", dim_repair_depot)
        con.register("dim_date_df", dim_date)
        con.register("fact_claims_df", fact_claims)

        con.execute("INSERT INTO dim_device SELECT * FROM dim_device_df")
        con.execute("INSERT INTO dim_policy SELECT * FROM dim_policy_df")
        con.execute("INSERT INTO policy_coverage_rules SELECT * FROM policy_coverage_rules_df")
        con.execute("INSERT INTO dim_repair_depot SELECT * FROM dim_repair_depot_df")
        con.execute("INSERT INTO dim_date SELECT * FROM dim_date_df")
        con.execute("INSERT INTO fact_claims SELECT * FROM fact_claims_df")
        con.execute(VELOCITY_SQL_PATH.read_text())

        row_count = con.execute("SELECT COUNT(*) FROM fact_claims").fetchone()[0]
    finally:
        con.close()

    print(f"Loaded warehouse at {db_path} with {row_count} fact_claims rows.")


if __name__ == "__main__":
    load_warehouse()
