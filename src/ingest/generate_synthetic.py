"""
Synthetic ICU data generator.

Produces ~40K records that mimic MIMIC-III structure and intentionally
injects dirty rows so GE + ML have something to catch:
  - Missing vitals        (~5%)
  - Out-of-range values   (~3%)
  - Duplicate records     (~2%)
  - Timestamp gaps >4h    (~4%)
  - ICD code mismatches   (~3%)
  - Vitals spikes         (~2%)

Run:
    python src/ingest/generate_synthetic.py
"""

import hashlib
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from faker import Faker

load_dotenv()

fake = Faker()
random.seed(42)
np.random.seed(42)

DUCKDB_PATH  = os.getenv("DUCKDB_PATH", "./data/healthcare.duckdb")
N_PATIENTS   = 5_000
N_ADMISSIONS = 8_000
N_CHARTEVENTS = 40_000

VALID_ICD_CODES = [
    "41001", "41071", "4280", "4019", "25000",
    "496",   "2724",  "5849", "51881", "99592",
]

VITAL_RANGES = {
    "heart_rate":       (40,   180),
    "systolic_bp":      (60,   220),
    "diastolic_bp":     (30,   130),
    "spo2":             (70,   100),
    "temperature_c":    (35.0, 40.5),
    "respiratory_rate": (8,    40),
}


def _hash_id(raw_id: int) -> str:
    salt = os.getenv("ANON_SALT", "dev-salt-change-in-prod")
    return hashlib.sha256(f"{salt}{raw_id}".encode()).hexdigest()[:16]


def generate_patients(n: int) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "subject_id":  _hash_id(i),
            "gender":      random.choice(["M", "F"]),
            "anchor_age":  random.randint(18, 95),
            "anchor_year": random.randint(2100, 2180),
        })
    return pd.DataFrame(rows)


def generate_admissions(patients: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    subject_ids = patients["subject_id"].tolist()
    for i in range(1, n + 1):
        admit_dt  = fake.date_time_between(start_date="-5y", end_date="now")
        los_hours = random.randint(2, 720)
        rows.append({
            "hadm_id":              _hash_id(i + 100_000),
            "subject_id":           random.choice(subject_ids),
            "admittime":            admit_dt,
            "dischtime":            admit_dt + timedelta(hours=los_hours),
            "admission_type":       random.choice(["EMERGENCY", "ELECTIVE", "URGENT"]),
            "hospital_expire_flag": random.choices([0, 1], weights=[0.9, 0.1])[0],
        })
    return pd.DataFrame(rows)


def generate_diagnoses(admissions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hadm_id in admissions["hadm_id"].tolist():
        for seq in range(1, random.randint(1, 5) + 1):
            icd = random.choice(VALID_ICD_CODES) if random.random() > 0.03 else "INVALID"
            rows.append({"hadm_id": hadm_id, "seq_num": seq, "icd9_code": icd})
    return pd.DataFrame(rows)


def generate_chartevents(admissions: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    hadm_sample = admissions.sample(n=min(n, len(admissions)), replace=True)

    for _, adm in hadm_sample.iterrows():
        admit_dt = adm["admittime"]
        if pd.isnull(admit_dt):
            continue

        chart_dt = admit_dt + timedelta(hours=random.uniform(0, 24))
        vital    = random.choice(list(VITAL_RANGES.keys()))
        lo, hi   = VITAL_RANGES[vital]
        value    = round(random.uniform(lo, hi), 1)

        r = random.random()
        if r < 0.05:
            value = None
        elif r < 0.08:
            value = value * random.uniform(2, 4)
        elif r < 0.10:
            chart_dt = chart_dt + timedelta(hours=random.uniform(5, 24))

        repeat = 2 if random.random() < 0.02 else 1
        for _ in range(repeat):
            rows.append({
                "hadm_id":    adm["hadm_id"],
                "subject_id": adm["subject_id"],
                "charttime":  chart_dt,
                "vital_name": vital,
                "valuenum":   value,
            })

    return pd.DataFrame(rows[:n])


def save_to_duckdb(patients, admissions, diagnoses, chartevents) -> None:
    Path(DUCKDB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DUCKDB_PATH)

    for table, df in [
        ("raw_patients",     patients),
        ("raw_admissions",   admissions),
        ("raw_diagnoses_icd", diagnoses),
        ("raw_chartevents",  chartevents),
    ]:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        print(f"  {table:25s}: {len(df):,} rows")

    con.close()


def main() -> None:
    print("Generating synthetic ICU data...")
    patients    = generate_patients(N_PATIENTS)
    admissions  = generate_admissions(patients, N_ADMISSIONS)
    diagnoses   = generate_diagnoses(admissions)
    chartevents = generate_chartevents(admissions, N_CHARTEVENTS)

    print(f"Saving to {DUCKDB_PATH}...")
    save_to_duckdb(patients, admissions, diagnoses, chartevents)
    print("Done. Next: make dbt-run")


if __name__ == "__main__":
    main()
