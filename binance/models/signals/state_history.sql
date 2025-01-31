{{
    config(
        materialized='incremental',
        unique_key=['symbol', 'timestamp'],
        incremental_strategy='delete+insert',
        on_schema_change='append_new_columns',
        alias="state_history"
    )
}}

select 
	t1.*, 
	t2.rsi_7,
	t2.rsi_14,
	t2.rsi_30,
	t3.ichimoku,
    t4.bolinger
from  ticker_data t1
left join {{ ref('rsi') }} t2
on t1.symbol = t2.symbol
	and t1.timestamp = t2.timestamp
left join {{ ref('ichimoku') }} t3
on t1.symbol = t3.symbol
	and t1.timestamp = t3.timestamp
left join {{ ref('bolinger') }} t4
on t1.symbol = t4.symbol
	and t1.timestamp = t4.timestamp
where 1=1
    {% if is_incremental() %}
    and t1.timestamp >= (select max(timestamp) FROM  {{ this }} )
    {% else %}
    and t1.timestamp > extract(epoch from now() - interval '6 weeks')::bigint  
    {% endif %}