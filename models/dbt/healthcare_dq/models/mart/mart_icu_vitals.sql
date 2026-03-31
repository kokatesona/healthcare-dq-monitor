-- Design decision: pivot at mart layer (not staging) so GE can validate
-- the wide format directly and ML has one clean feature table.
with vitals as (
    select * from {{ ref('stg_chartevents') }}
),
admissions as (
    select * from {{ ref('stg_admissions') }}
),
pivoted as (
    select
        hadm_id,
        subject_id,
        charttime,
        max(case when vital_name = 'heart_rate'       then valuenum end) as heart_rate,
        max(case when vital_name = 'systolic_bp'      then valuenum end) as systolic_bp,
        max(case when vital_name = 'diastolic_bp'     then valuenum end) as diastolic_bp,
        max(case when vital_name = 'spo2'             then valuenum end) as spo2,
        max(case when vital_name = 'temperature_c'    then valuenum end) as temperature_c,
        max(case when vital_name = 'respiratory_rate' then valuenum end) as respiratory_rate
    from vitals
    group by hadm_id, subject_id, charttime
),
final as (
    select
        p.hadm_id,
        p.subject_id,
        p.charttime,
        a.admittime,
        a.los_hours,
        a.admission_type,
        a.hospital_expire_flag,
        date_diff('hour', a.admittime, p.charttime) as hours_since_admit,
        p.heart_rate,
        p.systolic_bp,
        p.diastolic_bp,
        p.spo2,
        p.temperature_c,
        p.respiratory_rate,
        (
            case when p.heart_rate       is null then 1 else 0 end +
            case when p.systolic_bp      is null then 1 else 0 end +
            case when p.diastolic_bp     is null then 1 else 0 end +
            case when p.spo2             is null then 1 else 0 end +
            case when p.temperature_c    is null then 1 else 0 end +
            case when p.respiratory_rate is null then 1 else 0 end
        )                                           as missing_vitals_count,
        current_timestamp                           as _loaded_at
    from pivoted p
    left join admissions a using (hadm_id)
)
select * from final
