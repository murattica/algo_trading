{{
    config(
        materialized='incremental',
        unique_key=['symbol', 'timestamp'],
        incremental_strategy='delete+insert',
        on_schema_change='append_new_columns',
        alias="ichimoku_minutes"
    )
}}


WITH data_with_ichimoku AS (
    SELECT
        symbol,
        timestamp,
        high,
        low,
        close,

        -- Tenkan-sen (9-period high + 9-period low) / 2
        (MAX(high) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 8 PRECEDING AND CURRENT ROW) +
         MIN(low) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 8 PRECEDING AND CURRENT ROW)) / 2
         AS tenkan_sen,

        -- Kijun-sen (26-period high + 26-period low) / 2
        (MAX(high) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) +
         MIN(low) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 25 PRECEDING AND CURRENT ROW)) / 2
         AS kijun_sen,

        -- Senkou Span B (52-period high + 52-period low) / 2
        (MAX(high) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 51 PRECEDING AND CURRENT ROW) +
         MIN(low) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 51 PRECEDING AND CURRENT ROW)) / 2
         AS senkou_span_b,

        -- Chikou Span (close shifted back 26 periods)
        LAG(close, 26) OVER (PARTITION BY symbol ORDER BY timestamp) AS chikou_span

    FROM {{ source('public', 'ticker_data') }}
    WHERE 1=1
        {% if is_incremental() %}
        and timestamp >= (SELECT max(timestamp) - 51 * 60 * 1000  FROM  {{ this }} )
        {% else %}
        and timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '6 weeks')::bigint  
        {% endif %}
),

ichimoku_with_senkou AS (
    SELECT
        *,
        -- Senkou Span A: Average of Tenkan-sen and Kijun-sen
        (tenkan_sen + kijun_sen) / 2 AS senkou_span_a
    FROM
        data_with_ichimoku
    WHERE 1=1
        {% if is_incremental() %}
        and timestamp > (SELECT max(timestamp)  FROM  {{ this }} )
        {% endif %}
)
    
    
SELECT
    symbol,
    timestamp,
    -- Signal Generation Logic
    CASE
        -- Buy Signal: Tenkan-sen crosses above Kijun-sen and price is above the cloud
        WHEN tenkan_sen > kijun_sen
             AND close > senkou_span_a
             AND close > senkou_span_b THEN 0  -- 'BUY'

        -- Sell Signal: Tenkan-sen crosses below Kijun-sen and price is below the cloud
        WHEN tenkan_sen < kijun_sen
             AND close < senkou_span_a
             AND close < senkou_span_b THEN 1  -- 'SELL'

        -- Exit Signal (For Long): Price falls below Kijun-sen or enters the cloud
        WHEN (close < kijun_sen AND close BETWEEN LEAST(senkou_span_a, senkou_span_b) AND GREATEST(senkou_span_a, senkou_span_b))
     		or (close > kijun_sen AND close BETWEEN LEAST(senkou_span_a, senkou_span_b) AND GREATEST(senkou_span_a, senkou_span_b))
        THEN 2 -- 'EXIT'

        -- No Signal
        ELSE 3 -- 'HOLD'
    END AS ichimoku 
FROM
    ichimoku_with_senkou a
ORDER BY
    symbol, timestamp