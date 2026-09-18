-- Warranty Claims Analytics star schema.

DROP TABLE IF EXISTS fact_claims;
DROP TABLE IF EXISTS policy_coverage_rules;
DROP TABLE IF EXISTS dim_repair_depot;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_policy;
DROP TABLE IF EXISTS dim_device;

CREATE TABLE dim_device (
    device_key INTEGER PRIMARY KEY,
    brand VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    release_year INTEGER NOT NULL,
    msrp_usd DECIMAL(10, 2) NOT NULL,
    repairability_score INTEGER NOT NULL
);

CREATE TABLE dim_policy (
    policy_key INTEGER PRIMARY KEY,
    carrier_partner VARCHAR NOT NULL,
    plan_name VARCHAR NOT NULL,
    monthly_premium_usd DECIMAL(10, 2) NOT NULL,
    max_claims_annual INTEGER
);

CREATE TABLE policy_coverage_rules (
    rule_id INTEGER PRIMARY KEY,
    policy_key INTEGER NOT NULL REFERENCES dim_policy(policy_key),
    incident_type VARCHAR NOT NULL,
    is_covered BOOLEAN NOT NULL,
    deductible_usd DECIMAL(10, 2) NOT NULL,
    waiting_period_days INTEGER NOT NULL,
    repair_allowed BOOLEAN NOT NULL,
    replacement_allowed BOOLEAN NOT NULL
);

CREATE TABLE dim_repair_depot (
    depot_key INTEGER PRIMARY KEY,
    depot_name VARCHAR NOT NULL,
    region VARCHAR NOT NULL,
    avg_labor_rate_usd DECIMAL(10, 2) NOT NULL
);

CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE NOT NULL,
    calendar_year INTEGER NOT NULL,
    calendar_month INTEGER NOT NULL
);

CREATE TABLE fact_claims (
    claim_key INTEGER PRIMARY KEY,
    claim_id VARCHAR NOT NULL,
    policy_key INTEGER NOT NULL REFERENCES dim_policy(policy_key),
    device_key INTEGER NOT NULL REFERENCES dim_device(device_key),
    depot_key INTEGER NOT NULL REFERENCES dim_repair_depot(depot_key),
    filing_date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    incident_type VARCHAR NOT NULL,
    claim_narrative VARCHAR NOT NULL,
    adjudication_status VARCHAR NOT NULL,
    recommended_action VARCHAR NOT NULL,
    actual_repair_cost DECIMAL(10, 2) NOT NULL,
    replacement_device_cost DECIMAL(10, 2) NOT NULL,
    deductible_collected DECIMAL(10, 2) NOT NULL,
    net_payout DECIMAL(10, 2) NOT NULL,
    triage_cycle_time_hours DECIMAL(10, 2) NOT NULL,
    is_first_contact_resolved BOOLEAN NOT NULL
);
