{{
    config(
        materialized='incremental',
        unique_key=['symbol', 'timestamp'],
        incremental_strategy='delete+insert',
        on_schema_change='append_new_columns',
        alias="rsi_minutes"
    )
}}

{% set windows = ["7", "14", "30"] %}

WITH price_changes AS 
(
    SELECT
        symbol,
        timestamp,
        close,
        close - LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) AS price_change
    FROM {{ source('public', 'ticker_data') }}
    WHERE 1=1
        {% if is_incremental() %}
        and timestamp >= (SELECT max(timestamp) - 31 * 60 * 1000  FROM  {{ this }} )
        {% else %}
        and timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '6 weeks')::bigint  
        {% endif %}
),

gains_and_losses AS 
(
    SELECT
        symbol,
        timestamp,
        close,
        GREATEST(price_change, 0) AS gain,  -- Positive changes
        ABS(LEAST(price_change, 0)) AS loss -- Absolute value of negative changes
    FROM
        price_changes
),

average_gains_losses AS 
(
    SELECT
        symbol,
        timestamp,
        {%- for win in windows %}
        AVG(gain) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN {{ win }} - 1  PRECEDING AND CURRENT ROW) AS avg_gain_{{ win }},
        AVG(loss) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN {{ win }} - 1 PRECEDING AND CURRENT ROW) AS avg_loss_{{ win }},
        {% endfor %}
        close
    FROM gains_and_losses
),

rsi_calculation AS 
(
    SELECT
        symbol,
        timestamp,
        {%- for win in windows %}
        avg_gain_{{ win }},
        avg_loss_{{ win }},
        {% endfor %}

        {%- for win in windows %}
        CASE
            WHEN avg_loss_{{ win }} = 0 THEN 100  -- If there is no loss, RSI is 100
            ELSE 100 - (100 / (1 + (avg_gain_{{ win }} / avg_loss_{{ win }})))  -- Standard RSI formula
        END AS rsi_{{ win }},
        {% endfor %}
        close
    FROM
        average_gains_losses
    WHERE 1=1
        {% if is_incremental() %}
        and timestamp > (SELECT max(timestamp)  FROM  {{ this }} )
        {% endif %}
)

SELECT
    symbol,
    {%- for win in windows %}
    rsi_{{ win }} / 100 as rsi_{{ win }},
    {% endfor %}
    timestamp
FROM
    rsi_calculation
ORDER BY
    symbol, timestamp