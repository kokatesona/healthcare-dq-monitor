# Healthcare Data Quality Monitor & AI Repair Engine

> Solo-built, production-grade healthcare data quality system — designed, implemented, and shipped end-to-end in 3 weeks.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![dbt](https://img.shields.io/badge/dbt-1.11-orange)
![Airflow](https://img.shields.io/badge/Airflow-2.9-teal)
![MLflow](https://img.shields.io/badge/MLflow-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![CI](https://img.shields.io/badge/CI-passing-brightgreen)

## What it does

Automated data quality detection and AI-powered repair over real ICU patient records (MIMIC-III). The system ingests 40K+ ICU records, validates them against 6 failure types, scores each record's trustworthiness using a PyTorch autoencoder + Sklearn trust scorer, and deploys a LangChain ReAct agent that generates clinically-grounded repair suggestions — all orchestrated by Apache Airflow.

## Architecture
```
MIMIC-III / Synthetic ICU data
        │
        ▼
[Ingest] DuckDB raw tables — HIPAA-anonymised (salted SHA-256 IDs)
        │
        ▼
[Transform] dbt staging → mart layer (4 staging + 2 mart models, 21 tests)
        │
        ▼
[Validate] Great Expectations — 6 failure types detected
  - Missing vitals
  - Out-of-range values
  - ICD code mismatches
  - Duplicate records
  - Timestamp gaps > 4h
  - Anomalous vitals spikes
        │
        ▼
[ML] PyTorch autoencoder (6→32→16→8→16→32→6) — reconstruction error anomaly score
     GradientBoosting trust scorer (0–100) + SHAP explainability
     All experiments tracked in MLflow
        │
        ▼
[Agent] LangChain ReAct agent — reads flagged records, calls 3 tools,
        generates clinically-grounded repair suggestions
        LangSmith for full observability tracing
        │
        ▼
[UI] Streamlit dashboard — trust score distribution, DQ failures,
     flagged records table, live agent repair suggestions
```

All steps orchestrated by **Apache Airflow** DAG (daily @ 02:00 UTC).

## Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Storage | DuckDB | Columnar in-process queries on 40K+ records with zero infrastructure overhead |
| Transform | dbt | Auto-generated docs, schema tests, lineage graph — production DE standard |
| Anomaly detection | PyTorch autoencoder | Learns feature interactions (high HR + low SpO₂ worse than either alone); reconstruction error interpretable per-feature |
| Trust scorer | GradientBoosting | Better calibration on imbalanced labels than RandomForest; SHAP TreeExplainer is linear-time |
| Agent | LangChain ReAct | Multi-step tool calls, full trace in LangSmith, extensible with new tools |
| LLM | GPT-4o-mini (default) / Bedrock | Swap with `LLM_BACKEND=bedrock` env var |
| Orchestration | Airflow | Larger DE job market than Prefect; first-class dbt + GE operators |

## Tech stack

**Data Engineering:** Python · DuckDB · dbt · Apache Airflow · Docker  
**ML / AI:** PyTorch · Scikit-learn · SHAP · XGBoost · MLflow  
**GenAI:** LangChain · LangChain ReAct · LangSmith · OpenAI API · Amazon Bedrock  
**Storage:** DuckDB · PostgreSQL · AWS S3  
**Infra:** Docker Compose · Kubernetes · GitHub Actions  
**UI:** Streamlit · Plotly  
**Compliance:** HIPAA-aware anonymisation · GDPR-aligned data governance

## Quickstart
```bash
# 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/healthcare-dq-monitor
cd healthcare-dq-monitor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys

# 2. Generate synthetic ICU data
python src/ingest/generate_synthetic.py

# 3. Run dbt transforms
cd models/dbt/healthcare_dq && dbt run --profiles-dir . && dbt test --profiles-dir .
cd ../../..

# 4. Train ML models
python src/ml/autoencoder.py --train --epochs 50
python src/ml/autoencoder.py --score
python src/ml/trust_scorer.py --train
python src/ml/trust_scorer.py --score-all

# 5. Launch dashboard
streamlit run app/main.py

# 6. Start full stack (Airflow + MLflow + Postgres)
docker compose up -d
```

## Project structure
```
healthcare-dq-monitor/
├── dags/                          # Airflow DAG
├── src/
│   ├── ingest/                    # Data ingestion + synthetic generator
│   ├── ml/                        # PyTorch autoencoder + trust scorer
│   └── agent/                     # LangChain ReAct repair agent
├── models/dbt/healthcare_dq/      # dbt project (staging + mart)
├── app/                           # Streamlit dashboard
├── tests/                         # pytest suite (22 tests)
├── k8s/                           # Kubernetes deployment manifest
├── .github/workflows/             # GitHub Actions CI
└── docker-compose.yml             # Local stack
```

## CI / CD

GitHub Actions runs on every push to `main`:
- `ruff` linting
- `pytest` — 22 tests across ingest + ML modules
- Coverage report uploaded as artifact

## HIPAA compliance notes

- All `subject_id` and `hadm_id` values are salted SHA-256 hashed at ingest
- No free-text clinical notes are loaded
- Date of birth is not stored — only age bucket is derived
- All data access is read-only after initial load

## Dataset

Uses [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/) (requires PhysioNet credentialing).  
A fully functional synthetic dataset is included for development — run `python src/ingest/generate_synthetic.py`.
