WITH incoming_by_source AS (
    SELECT
        receiver_id,
        sender_id,
        SUM(amount_kzt) AS source_incoming_kzt
    FROM transactions_clean
    WHERE amount_kzt > 0
    GROUP BY receiver_id, sender_id
), shares AS (
    SELECT
        receiver_id,
        sender_id,
        source_incoming_kzt,
        SUM(source_incoming_kzt) OVER (PARTITION BY receiver_id) AS total_incoming_kzt
    FROM incoming_by_source
)
SELECT
    receiver_id,
    sender_id AS dominant_source_id,
    ROUND(source_incoming_kzt, 2) AS dominant_source_incoming_kzt,
    ROUND(total_incoming_kzt, 2) AS total_incoming_kzt,
    ROUND(source_incoming_kzt / total_incoming_kzt, 4) AS dominant_source_share
FROM shares
WHERE total_incoming_kzt > 0
  AND source_incoming_kzt / total_incoming_kzt > 0.70
ORDER BY dominant_source_share DESC, total_incoming_kzt DESC;
