with vitals as (
    select * from {{ ref('mart_icu_vitals') }}
),
diagnoses as (
    select * from {{ ref('stg_diagnoses_icd') }}
),
vital_agg as (
    select
        hadm_id,
        subject_id,
        admission_type,
        hospital_expire_flag,
        los_hours,
        count(*)                                                              as total_chart_rows,
        sum(missing_vitals_count)                                             as total_missing_vitals,
        avg(heart_rate)                                                       as avg_heart_rate,
        avg(systolic_bp)                                                      as avg_systolic_bp,
        avg(spo2)                                                             as avg_spo2,
        avg(temperature_c)                                                    as avg_temperature_c,
        avg(respiratory_rate)                                                 as avg_respiratory_rate,
        sum(case when heart_rate < 20  or heart_rate > 300       then 1 else 0 end) as hr_oor_count,
        sum(case when spo2 < 50        or spo2 > 100             then 1 else 0 end) as spo2_oor_count,
        sum(case when respiratory_rate < 4 or respiratory_rate > 60 then 1 else 0 end) as rr_oor_count,
        min(charttime)                                                        as first_chart_time,
        max(charttime)                                                        as last_chart_time,
        current_timestamp                                                     as _loaded_at
    from vitals
    group by hadm_id, subject_id, admission_type, hospital_expire_flag, los_hours
),
diag_agg as (
    select
        hadm_id,
        count(*)                                              as total_diagnoses,
        sum(case when not is_valid_icd then 1 else 0 end)    as invalid_icd_count
    from diagnoses
    group by hadm_id
)
select
    v.*,
    coalesce(d.total_diagnoses,  0)  as total_diagnoses,
    coalesce(d.invalid_icd_count, 0) as invalid_icd_count,
    round(100.0 * (
        1.0
        - (0.3 * least(v.total_missing_vitals::double / nullif(v.total_chart_rows * 6, 0), 1.0))
        - (0.2 * least((v.hr_oor_count + v.spo2_oor_count + v.rr_oor_count)::double / nullif(v.total_chart_rows, 0), 1.0))
        - (0.2 * least(coalesce(d.invalid_icd_count, 0)::double / nullif(d.total_diagnoses, 0), 1.0))
    ), 1)                            as raw_dq_score
from vital_agg v
left join diag_agg d using (hadm_id)
