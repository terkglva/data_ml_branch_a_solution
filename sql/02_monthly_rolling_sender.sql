WITH monthly AS (
    SELECT
        sender_id,
        strftime('%Y-%m', date_iso) AS month,
        SUM(amount_kzt) AS monthly_turnover_kzt
    FROM transactions_clean
    WHERE amount_kzt IS NOT NULL AND date_iso IS NOT NULL
    GROUP BY sender_id, strftime('%Y-%m', date_iso)
)
SELECT
    sender_id,
    month,
    ROUND(monthly_turnover_kzt, 2) AS monthly_turnover_kzt,
    ROUND(
        SUM(monthly_turnover_kzt) OVER (
            PARTITION BY sender_id
            ORDER BY month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS rolling_3m_turnover_kzt
FROM monthly
ORDER BY sender_id, month;
