SELECT
    sender_id,
    receiver_id,
    COUNT(*) AS operations_count,
    ROUND(SUM(amount_kzt), 2) AS turnover_kzt
FROM transactions_clean
WHERE amount_kzt IS NOT NULL
GROUP BY sender_id, receiver_id
ORDER BY turnover_kzt DESC
LIMIT 10;
