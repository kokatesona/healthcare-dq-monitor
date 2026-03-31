"""
Trust Scorer — GradientBoosting + SHAP
0-100 trust score per ICU admission.
Design decision: GradientBoosting over RandomForest — better calibration
on imbalanced labels; SHAP TreeExplainer is linear-time on GBM.
"""

import argparse
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import shap
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

FEATURE_COLS = [
    "total_missing_vitals", "total_chart_rows",
    "hr_oor_count", "spo2_oor_count", "rr_oor_count",
    "invalid_icd_count", "raw_dq_score",
    "avg_recon_error", "los_hours",
]

SCORER_PATH = Path("./models/trust_scorer.pkl")
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT  = os.getenv("MLFLOW_EXPERIMENT_NAME", "healthcare-dq-monitor")


def load_features():
    import duckdb
    con = duckdb.connect(
        os.getenv("DUCKDB_PATH", "./data/healthcare.duckdb"), read_only=True
    )
    has_scores = con.execute("""
        SELECT count(*) FROM information_schema.tables
        WHERE table_name = 'ml_anomaly_scores'
    """).fetchone()[0] > 0

    if has_scores:
        score_col   = "coalesce(a.avg_recon_error, 0.0) as avg_recon_error"
        flag_cond   = "or coalesce(a.flagged_count, 0) > 2"
        join_clause = """
        LEFT JOIN (
            SELECT hadm_id,
                   avg(reconstruction_error) as avg_recon_error,
                   sum(anomaly_flag)         as flagged_count
            FROM ml_anomaly_scores
            GROUP BY hadm_id
        ) a USING (hadm_id)"""
    else:
        score_col   = "0.0 as avg_recon_error"
        flag_cond   = ""
        join_clause = ""

    df = con.execute(f"""
        SELECT
            s.hadm_id,
            s.total_missing_vitals,
            s.total_chart_rows,
            s.hr_oor_count,
            s.spo2_oor_count,
            s.rr_oor_count,
            s.invalid_icd_count,
            s.raw_dq_score,
            s.los_hours,
            {score_col},
            case when (
                s.hr_oor_count > 0
                or s.spo2_oor_count > 0
                or (s.invalid_icd_count::double /
                    nullif(s.total_diagnoses, 0) > 0.25)
                {flag_cond}
            ) then 1 else 0 end as has_dq_problem
        FROM main_mart.mart_patient_summary s
        {join_clause}
    """).df()
    con.close()

    pos = df["has_dq_problem"].sum()
    print(f"  Labels: {pos} positive ({100*pos/len(df):.1f}%), "
          f"{len(df)-pos} negative out of {len(df)} admissions")
    return df


def train() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_features()
    X  = df[FEATURE_COLS].fillna(0.0).values
    y  = df["has_dq_problem"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    params = dict(n_estimators=200, max_depth=4,
                  learning_rate=0.05, subsample=0.8, random_state=42)

    with mlflow.start_run(run_name="trust-scorer-train"):
        mlflow.log_params(params)
        clf = GradientBoostingClassifier(**params)
        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred  = clf.predict(X_test)
        auc     = roc_auc_score(y_test, y_proba)
        report  = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("roc_auc",   auc)
        mlflow.log_metric("f1",        report["1"]["f1-score"])
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall",    report["1"]["recall"])
        print(f"ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_test)
        for feat, imp in sorted(
            zip(FEATURE_COLS, np.abs(shap_vals).mean(axis=0)),
            key=lambda x: -x[1],
        ):
            mlflow.log_metric(f"shap_{feat}", imp)
            print(f"  SHAP {feat:30s}: {imp:.4f}")

        SCORER_PATH.parent.mkdir(exist_ok=True)
        with open(SCORER_PATH, "wb") as f:
            pickle.dump({"clf": clf, "scaler": scaler,
                         "features": FEATURE_COLS}, f)
        mlflow.sklearn.log_model(clf, "trust_scorer")
        print(f"\nSaved → {SCORER_PATH}")


def score_admission(hadm_id: str, feature_values: dict) -> dict:
    with open(SCORER_PATH, "rb") as f:
        bundle = pickle.load(f)
    X      = np.array([[feature_values.get(c, 0.0) for c in bundle["features"]]])
    X      = bundle["scaler"].transform(X)
    proba  = bundle["clf"].predict_proba(X)[0][1]
    score  = round((1 - proba) * 100, 1)
    exp    = shap.TreeExplainer(bundle["clf"])
    s_vals = exp.shap_values(X)[0]
    return {
        "hadm_id":     hadm_id,
        "trust_score": score,
        "dq_flag":     bool(score < 60),
        "shap_values": dict(zip(bundle["features"],
                                [round(float(v), 4) for v in s_vals])),
    }


def score_all() -> None:
    import duckdb
    with open(SCORER_PATH, "rb") as f:
        bundle = pickle.load(f)
    df    = load_features()
    X     = bundle["scaler"].transform(df[FEATURE_COLS].fillna(0.0).values)
    proba = bundle["clf"].predict_proba(X)[:, 1]
    df["trust_score"] = (1 - proba) * 100
    df["dq_flagged"]  = df["trust_score"] < 60

    con = duckdb.connect(os.getenv("DUCKDB_PATH", "./data/healthcare.duckdb"))
    con.execute("DROP TABLE IF EXISTS ml_trust_scores")
    con.execute("CREATE TABLE ml_trust_scores AS "
                "SELECT hadm_id, trust_score, dq_flagged FROM df")
    con.close()
    n = int(df["dq_flagged"].sum())
    print(f"Scored {len(df):,} admissions — {n:,} flagged (trust < 60)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",     action="store_true")
    p.add_argument("--score-all", action="store_true", dest="score_all")
    args = p.parse_args()
    if args.train:
        train()
    if args.score_all:
        score_all()
    if not args.train and not args.score_all:
        p.print_help()
