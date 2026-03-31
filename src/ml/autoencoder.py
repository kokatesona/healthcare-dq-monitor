"""
PyTorch Autoencoder — ICU Vitals Anomaly Detection
Architecture: 6 → 32 → 16 → 8 → 16 → 32 → 6
Design decision: autoencoder over Isolation Forest — learns feature
interactions and gives per-feature reconstruction error for explainability.
"""

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset

load_dotenv()

VITAL_COLS = [
    "heart_rate", "systolic_bp", "diastolic_bp",
    "spo2", "temperature_c", "respiratory_rate",
]
VITAL_RANGES = {
    "heart_rate":       (20.0, 300.0),
    "systolic_bp":      (40.0, 280.0),
    "diastolic_bp":     (10.0, 180.0),
    "spo2":             (50.0, 100.0),
    "temperature_c":    (33.0,  42.0),
    "respiratory_rate": ( 4.0,  60.0),
}

MODEL_DIR  = Path("./models")
MODEL_PATH = MODEL_DIR / "autoencoder.pt"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "healthcare-dq-monitor")


class VitalsAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 6, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 16), nn.ReLU(), nn.BatchNorm1d(16),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        with torch.no_grad():
            return ((x - self.forward(x)) ** 2).mean(dim=1)


def minmax_scale(arr: np.ndarray) -> np.ndarray:
    mins = np.array([VITAL_RANGES[c][0] for c in VITAL_COLS])
    maxs = np.array([VITAL_RANGES[c][1] for c in VITAL_COLS])
    return np.clip((arr - mins) / (maxs - mins), 0.0, 1.0)


def load_clean_vitals() -> np.ndarray:
    import duckdb
    con = duckdb.connect(
        os.getenv("DUCKDB_PATH", "./data/healthcare.duckdb"), read_only=True
    )
    # Load per-vital readings from staging (long format) and stack into matrix
    rows = []
    for vital in VITAL_COLS:
        lo, hi = VITAL_RANGES[vital]
        df = con.execute(f"""
            SELECT valuenum FROM main_staging.stg_chartevents
            WHERE vital_name = '{vital}'
              AND valuenum IS NOT NULL
              AND valuenum BETWEEN {lo} AND {hi * 1.5}
            LIMIT 5000
        """).df()
        rows.append(df["valuenum"].values)
    con.close()
    n = min(len(r) for r in rows)
    arr = np.stack([r[:n] for r in rows], axis=1).astype(np.float32)
    print(f"  Loaded {n:,} clean readings per vital")
    return arr


def score_records() -> None:
    import duckdb
    assert MODEL_PATH.exists(), "Run --train first"
    ckpt  = torch.load(MODEL_PATH, weights_only=True)
    model = VitalsAutoencoder()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    con = duckdb.connect(os.getenv("DUCKDB_PATH", "./data/healthcare.duckdb"))

    # Pivot long-format staging into wide format for scoring
    pivot_cases = ", ".join([
        f"max(case when vital_name = '{c}' then valuenum end) as {c}"
        for c in VITAL_COLS
    ])
    df = con.execute(f"""
        SELECT hadm_id, charttime, {pivot_cases}
        FROM main_staging.stg_chartevents
        GROUP BY hadm_id, charttime
    """).df()

    filled = df[VITAL_COLS].copy()
    for c in VITAL_COLS:
        filled[c] = filled[c].fillna(filled[c].median())

    X      = torch.tensor(
        minmax_scale(filled.values.astype(np.float32)), dtype=torch.float32
    )
    errors = model.reconstruction_error(X).numpy()

    df["reconstruction_error"] = errors
    df["anomaly_flag"]         = (errors > ckpt["threshold"]).astype(int)

    con.execute("DROP TABLE IF EXISTS ml_anomaly_scores")
    con.execute("CREATE TABLE ml_anomaly_scores AS SELECT * FROM df")
    con.close()

    n = int(df["anomaly_flag"].sum())
    print(f"Scored {len(df):,} records — {n:,} flagged ({100*n/len(df):.1f}%)")


def train(epochs: int = 50, batch_size: int = 256, lr: float = 1e-3) -> float:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    raw    = load_clean_vitals()
    scaled = minmax_scale(raw)
    X      = torch.tensor(scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model     = VitalsAutoencoder()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    with mlflow.start_run(run_name="autoencoder-train"):
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "architecture": "6-32-16-8-16-32-6", "n_records": len(X),
        })
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                loss = criterion(model(batch), batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(batch)
            epoch_loss /= len(X)
            if epoch % 10 == 0:
                mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                print(f"  epoch {epoch:3d}/{epochs}  loss={epoch_loss:.6f}")

        model.eval()
        errors    = model.reconstruction_error(X).numpy()
        threshold = float(np.percentile(errors, 95))
        mlflow.log_metric("anomaly_threshold_p95", threshold)
        mlflow.log_metric("mean_recon_error", float(errors.mean()))

        MODEL_DIR.mkdir(exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "threshold":   threshold,
            "vital_cols":  VITAL_COLS,
        }, MODEL_PATH)
        mlflow.pytorch.log_model(model, "autoencoder")
        print(f"\nSaved → {MODEL_PATH}  threshold={threshold:.6f}")
        return threshold


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()
    if args.train:
        train(epochs=args.epochs)
    if args.score:
        score_records()
    if not args.train and not args.score:
        p.print_help()
