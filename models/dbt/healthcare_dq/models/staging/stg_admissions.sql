with source as (
    select * from raw_admissions
),
cleaned as (
    select
        hadm_id,
        subject_id,
        cast(admittime as timestamp)                              as admittime,
        cast(dischtime as timestamp)                              as dischtime,
        date_diff('hour',
            cast(admittime as timestamp),
            cast(dischtime as timestamp))                         as los_hours,
        upper(trim(admission_type))                               as admission_type,
        cast(hospital_expire_flag as integer)                     as hospital_expire_flag,
        current_timestamp                                         as _loaded_at
    from source
    where hadm_id    is not null
      and subject_id is not null
      and admittime  is not null
      and dischtime  is not null
      and cast(dischtime as timestamp) >= cast(admittime as timestamp)
)
select * from cleaned
