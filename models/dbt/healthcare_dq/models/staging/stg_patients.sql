-- Design decision: subject_id is already a salted SHA-256 hash from ingest
-- (HIPAA-aware) — no PII ever stored in this layer.
with source as (
    select * from raw_patients
),
cleaned as (
    select
        subject_id,
        upper(gender)      as gender,
        anchor_age,
        anchor_year,
        current_timestamp  as _loaded_at
    from source
    where subject_id is not null
      and gender in ('M', 'F')
)
select * from cleaned
