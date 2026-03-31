"""
Healthcare DQ Monitor — Airflow DAG
Daily pipeline: ingest → dbt → validate → score → agent

Design decision: Airflow over Prefect — larger job market for DE roles,
first-class dbt and Great Expectations operators available.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner":            "sonakshi",
    "depends_on_past":  False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="healthcare_dq_pipeline",
    default_args=DEFAULT_ARGS,
    description="Healthcare DQ monitor — ingest, transform, validate, score, agent",
    schedule="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["healthcare", "dq", "ml"],
) as dag:

    def run_ingest():
        import subprocess, sys
        r = subprocess.run(
            [sys.executable, "src/ingest/generate_synthetic.py"],
            capture_output=True, text=True
        )
        print(r.stdout)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)

    def run_dbt_run():
        import subprocess
        r = subprocess.run(
            ["dbt", "run", "--profiles-dir", "."],
            cwd="models/dbt/healthcare_dq",
            capture_output=True, text=True,
        )
        print(r.stdout)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)

    def run_dbt_test():
        import subprocess
        r = subprocess.run(
            ["dbt", "test", "--profiles-dir", "."],
            cwd="models/dbt/healthcare_dq",
            capture_output=True, text=True,
        )
        print(r.stdout)
        if r.returncode != 0:
            raise RuntimeError(f"dbt tests failed:\n{r.stderr}")

    def run_anomaly_scoring():
        import os, sys
        sys.path.insert(0, ".")
        os.environ.setdefault("DUCKDB_PATH", "./data/healthcare.duckdb")
        os.environ.setdefault("MLFLOW_TRACKING_URI", "./mlruns")
        from src.ml.autoencoder import score_records
        score_records()

    def run_trust_scoring():
        import os, sys
        sys.path.insert(0, ".")
        os.environ.setdefault("DUCKDB_PATH", "./data/healthcare.duckdb")
        os.environ.setdefault("MLFLOW_TRACKING_URI", "./mlruns")
        from src.ml.trust_scorer import score_all
        score_all()

    def run_agent():
        import os, sys
        sys.path.insert(0, ".")
        os.environ.setdefault("DUCKDB_PATH", "./data/healthcare.duckdb")
        try:
            from src.agent.repair_agent import run_agent_on_flagged
            run_agent_on_flagged(top_n=10)
        except Exception as e:
            print(f"Agent step skipped (non-blocking): {e}")

    t1 = PythonOperator(task_id="ingest_data",     python_callable=run_ingest)
    t2 = PythonOperator(task_id="dbt_run",         python_callable=run_dbt_run)
    t3 = PythonOperator(task_id="dbt_test",        python_callable=run_dbt_test)
    t4 = PythonOperator(task_id="anomaly_scoring", python_callable=run_anomaly_scoring)
    t5 = PythonOperator(task_id="trust_scoring",   python_callable=run_trust_scoring)
    t6 = PythonOperator(task_id="agent_repair",    python_callable=run_agent)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
