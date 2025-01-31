{{
    config(
        materialized='incremental',
        unique_key=['symbol', 'timestamp'],
        incremental_strategy='delete+insert',
        on_schema_change='append_new_columns',
        alias="bolinger_minutes"
    )
}}

WITH bollinger_data AS (
    SELECT 
    	symbol,
        timestamp,
        close,
        AVG(close::NUMERIC) OVER (partition by symbol
            ORDER BY timestamp
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS sma, -- 20-period SMA
        STDDEV(close::NUMERIC) OVER (partition by symbol
            ORDER BY timestamp
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS stddev -- Standard Deviation
    FROM {{ source('public', 'ticker_data') }}
    WHERE 1=1
        {% if is_incremental() %}
        and timestamp >= (SELECT max(timestamp) - 10 * 60 * 1000  FROM  {{ this }} )
        {% else %}
        and timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '6 weeks')::bigint  
        {% endif %}
),

base as
(
	SELECT 
		symbol,
	    timestamp,
	    close,
	    sma AS middle_band,
	    sma + 2 * stddev AS upper_band, -- Upper Band
	    sma - 2 * stddev AS lower_band -- Lower Band
	FROM bollinger_data
	WHERE sma IS NOT NULL -- Exclude rows before 20th observation
)

SELECT 
	symbol,
    timestamp,
    case when close > upper_band then 1
    	 when close < lower_band then -1
    else 0 end as bolinger
FROM base
WHERE 1=1
    {% if is_incremental() %}
    and timestamp > (SELECT max(timestamp)  FROM  {{ this }} )
    {% endif %}