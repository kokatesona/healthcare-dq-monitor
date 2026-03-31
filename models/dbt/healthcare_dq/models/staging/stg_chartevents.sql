-- Design decision: filter to 6 vitals at staging so mart models
-- receive clean typed data without repeated CASE logic.
with source as (
    select * from raw_chartevents
),
cleaned as (
    select
        hadm_id,
        subject_id,
        cast(charttime as timestamp)  as charttime,
        vital_name,
        cast(valuenum as double)      as valuenum,
        current_timestamp             as _loaded_at
    from source
    where hadm_id    is not null
      and vital_name is not null
      and vital_name in (
          'heart_rate', 'systolic_bp', 'diastolic_bp',
          'spo2', 'temperature_c', 'respiratory_rate'
      )
)
select * from cleaned
