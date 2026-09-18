-- Claim velocity / fraud-ring anomaly scoring, exposed as a view so the
-- dashboard and db_engine can query it directly.

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
        DATE_DIFF('day', LAG(filing_date) OVER (PARTITION BY policy_key ORDER BY filing_date), filing_date)
            AS days_since_last_claim
    FROM claim_dates
),
freq_window AS (
    SELECT
        *,
        COUNT(*) OVER (
            PARTITION BY policy_key ORDER BY filing_date
            RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW
        ) AS claims_in_30d
    FROM inter_claim
),
payout_stats AS (
    SELECT
        *,
        AVG(net_payout) OVER (PARTITION BY model_name) AS payout_mean,
        STDDEV_POP(net_payout) OVER (PARTITION BY model_name) AS payout_std
    FROM freq_window
),
scored AS (
    SELECT
        *,
        CASE
            WHEN payout_std IS NULL OR payout_std = 0 THEN 0
            ELSE (net_payout - payout_mean) / payout_std
        END AS payout_zscore
    FROM payout_stats
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
        payout_zscore,
        LEAST(claims_in_30d * 4.0, 40) AS frequency_score,
        CASE
            WHEN days_since_last_claim IS NULL THEN 0
            WHEN days_since_last_claim <= 30 THEN GREATEST(30 - days_since_last_claim, 0) / 30.0 * 30
            ELSE 0
        END AS recency_score,
        LEAST(GREATEST(payout_zscore, 0) * 10, 30) AS payout_score
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
    ROUND(payout_zscore, 3) AS payout_zscore,
    ROUND(frequency_score + recency_score + payout_score, 2) AS velocity_score,
    CASE
        WHEN frequency_score + recency_score + payout_score >= 60 THEN 'ELEVATED'
        WHEN frequency_score + recency_score + payout_score >= 35 THEN 'MODERATE'
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
