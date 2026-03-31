"""Tests for src/ingest — run with: pytest tests/test_ingest.py -v"""

import duckdb

from src.ingest.generate_synthetic import (
    VITAL_RANGES,
    _hash_id,
    generate_admissions,
    generate_chartevents,
    generate_diagnoses,
    generate_patients,
)


class TestHashId:
    def test_returns_16_char_string(self):
        assert len(_hash_id(1)) == 16

    def test_deterministic(self):
        assert _hash_id(99) == _hash_id(99)

    def test_different_ids_differ(self):
        assert _hash_id(1) != _hash_id(2)

    def test_no_raw_id_in_output(self):
        assert "12345" not in _hash_id(12345)


class TestGeneratePatients:
    def setup_method(self):
        self.df = generate_patients(100)

    def test_row_count(self):
        assert len(self.df) == 100

    def test_required_columns(self):
        assert {"subject_id", "gender", "anchor_age", "anchor_year"}.issubset(self.df.columns)

    def test_no_null_subject_ids(self):
        assert self.df["subject_id"].notnull().all()

    def test_gender_values(self):
        assert set(self.df["gender"].unique()).issubset({"M", "F"})

    def test_age_range(self):
        assert self.df["anchor_age"].between(18, 95).all()


class TestGenerateAdmissions:
    def setup_method(self):
        self.patients = generate_patients(50)
        self.df = generate_admissions(self.patients, 80)

    def test_row_count(self):
        assert len(self.df) == 80

    def test_dischtime_after_admittime(self):
        valid = self.df.dropna(subset=["admittime", "dischtime"])
        assert (valid["dischtime"] >= valid["admittime"]).all()

    def test_admission_types(self):
        assert set(self.df["admission_type"].unique()).issubset(
            {"EMERGENCY", "ELECTIVE", "URGENT"}
        )


class TestGenerateChartevents:
    def setup_method(self):
        patients = generate_patients(100)
        admissions = generate_admissions(patients, 500)
        self.df = generate_chartevents(admissions, 500)

    def test_has_rows(self):
        assert len(self.df) > 400

    def test_vital_names_valid(self):
        valid = set(VITAL_RANGES.keys())
        assert set(self.df["vital_name"].dropna().unique()).issubset(valid)

    def test_contains_nulls(self):
        assert self.df["valuenum"].isnull().sum() > 0


class TestDuckDBRoundtrip:
    def test_roundtrip(self, tmp_path):
        patients   = generate_patients(20)
        admissions = generate_admissions(patients, 30)

        con = duckdb.connect(str(tmp_path / "test.duckdb"))
        con.register("patients",   patients)
        con.register("admissions", admissions)
        con.register("diagnoses",   generate_diagnoses(admissions))
        con.register("chartevents", generate_chartevents(admissions, 100))

        con.execute("CREATE TABLE raw_patients      AS SELECT * FROM patients")
        con.execute("CREATE TABLE raw_admissions    AS SELECT * FROM admissions")
        con.execute("CREATE TABLE raw_diagnoses_icd AS SELECT * FROM diagnoses")
        con.execute("CREATE TABLE raw_chartevents   AS SELECT * FROM chartevents")

        assert con.execute("SELECT COUNT(*) FROM raw_patients").fetchone()[0] == 20
        assert con.execute("SELECT COUNT(*) FROM raw_admissions").fetchone()[0] == 30
        assert con.execute("SELECT COUNT(*) FROM raw_diagnoses_icd").fetchone()[0] > 0
        assert con.execute("SELECT COUNT(*) FROM raw_chartevents").fetchone()[0] > 0
        con.close()
