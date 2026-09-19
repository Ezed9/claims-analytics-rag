-- Claim velocity / fraud-ring anomaly scoring, exposed as a view so the
-- dashboard and db_engine can query it directly.
--
-- The schema has no per-customer dimension (dim_policy is a plan PRODUCT,
-- shared by every enrollee of that carrier/plan), so a (policy_key,
-- device_key) pair is the finest available proxy for "a claimant filing
-- against a given device". Because that proxy is still shared by many
-- real customers, its ambient claim volume is high (tens of claims per
-- rolling 30-day window even with no fraud at all), so a fixed count
-- threshold like ">= 10 claims in 30 days => risky" would flag almost
-- every claim. Instead every component below is a z-score computed
-- relative to that same (policy_key, device_key) partition's own
-- history, so a claim only scores high when it is unusual FOR THAT
-- PARTITION, not merely because the partition itself is high-volume.

CREATE OR REPLACE VIEW vw_claim_velocity_risk AS
WITH claim_dates AS (
    SELECT
        fc.claim_key,
        fc.policy_key,
        fc.device_key,
        fc.net_payout,
        fc.incident_type,
        dd.model_name,
        dt.full_date AS filing_date
    FROM fact_claims fc
    JOIN dim_date dt ON dt.date_key = fc.filing_date_key
    JOIN dim_device dd ON dd.device_key = fc.device_key
),
inter_claim AS (
    SELECT
        *,
        DATE_DIFF(
            'day',
            LAG(filing_date) OVER (PARTITION BY policy_key, device_key ORDER BY filing_date, claim_key),
            filing_date
        ) AS days_since_last_claim
    FROM claim_dates
),
freq_window AS (
    SELECT
        *,
        COUNT(*) OVER (
            PARTITION BY policy_key, device_key ORDER BY filing_date
            RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW
        ) AS claims_in_30d
    FROM inter_claim
),
partition_stats AS (
    SELECT
        *,
        AVG(claims_in_30d) OVER (PARTITION BY policy_key, device_key) AS mean_claims_30d,
        STDDEV_POP(claims_in_30d) OVER (PARTITION BY policy_key, device_key) AS std_claims_30d,
        AVG(days_since_last_claim) OVER (PARTITION BY policy_key, device_key) AS mean_gap_days,
        STDDEV_POP(days_since_last_claim) OVER (PARTITION BY policy_key, device_key) AS std_gap_days,
        AVG(net_payout) OVER (PARTITION BY model_name) AS mean_payout,
        STDDEV_POP(net_payout) OVER (PARTITION BY model_name) AS std_payout
    FROM freq_window
),
scored AS (
    SELECT
        *,
        CASE
            WHEN std_claims_30d IS NULL OR std_claims_30d = 0 THEN 0
            ELSE (claims_in_30d - mean_claims_30d) / std_claims_30d
        END AS frequency_zscore,
        CASE
            WHEN days_since_last_claim IS NULL OR std_gap_days IS NULL OR std_gap_days = 0 THEN 0
            ELSE (mean_gap_days - days_since_last_claim) / std_gap_days
        END AS recency_zscore,
        CASE
            WHEN std_payout IS NULL OR std_payout = 0 THEN 0
            ELSE (net_payout - mean_payout) / std_payout
        END AS payout_zscore
    FROM partition_stats
),
final AS (
    SELECT
        claim_key,
        policy_key,
        device_key,
        model_name,
        incident_type,
        filing_date,
        days_since_last_claim,
        claims_in_30d,
        net_payout,
        ROUND(payout_zscore, 3) AS payout_zscore,
        LEAST(
            (
                GREATEST(frequency_zscore, 0) * 0.4
                + GREATEST(recency_zscore, 0) * 0.3
                + GREATEST(payout_zscore, 0) * 0.3
            ) * 40,
            100
        ) AS velocity_score
    FROM scored
)
SELECT
    claim_key,
    policy_key,
    device_key,
    model_name,
    incident_type,
    filing_date,
    days_since_last_claim,
    claims_in_30d,
    net_payout,
    payout_zscore,
    ROUND(velocity_score, 2) AS velocity_score,
    CASE
        WHEN velocity_score >= 60 THEN 'ELEVATED'
        WHEN velocity_score >= 35 THEN 'MODERATE'
        ELSE 'LOW'
    END AS risk_tier
FROM final;

CREATE OR REPLACE VIEW vw_depot_cycle_time_ranking AS
SELECT
    d.depot_key,
    d.depot_name,
    d.region,
    d.avg_labor_rate_usd,
    COUNT(*) AS claim_count,
    ROUND(AVG(fc.triage_cycle_time_hours), 2) AS avg_cycle_time_hours,
    ROUND(AVG(CASE WHEN fc.is_first_contact_resolved THEN 1.0 ELSE 0.0 END), 3) AS fcr_rate,
    RANK() OVER (ORDER BY AVG(fc.triage_cycle_time_hours) ASC) AS cycle_time_rank
FROM fact_claims fc
JOIN dim_repair_depot d ON d.depot_key = fc.depot_key
GROUP BY d.depot_key, d.depot_name, d.region, d.avg_labor_rate_usd
ORDER BY avg_cycle_time_hours ASC;
