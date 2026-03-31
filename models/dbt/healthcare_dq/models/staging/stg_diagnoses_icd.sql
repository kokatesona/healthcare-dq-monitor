with source as (
    select * from raw_diagnoses_icd
),
valid_codes as (
    select icd9_code from (
        values ('41001'),('41071'),('4280'),('4019'),('25000'),
               ('496'),('2724'),('5849'),('51881'),('99592')
    ) t(icd9_code)
),
cleaned as (
    select
        d.hadm_id,
        d.seq_num,
        upper(trim(d.icd9_code))                                   as icd9_code,
        case
            when upper(trim(d.icd9_code)) in (select icd9_code from valid_codes)
            then true else false
        end                                                        as is_valid_icd,
        current_timestamp                                          as _loaded_at
    from source d
    where d.hadm_id   is not null
      and d.icd9_code is not null
)
select * from cleaned
