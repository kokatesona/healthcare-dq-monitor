"""Tests for src/ml — run with: pytest tests/test_ml.py -v"""

import numpy as np
import pickle
import torch


class TestVitalsAutoencoder:
    def setup_method(self):
        from src.ml.autoencoder import VitalsAutoencoder, minmax_scale, VITAL_RANGES
        self.model      = VitalsAutoencoder()
        self.model.eval()
        self.VITAL_RANGES = VITAL_RANGES
        self.minmax_scale = minmax_scale

    def test_forward_output_shape(self):
        x = torch.randn(16, 6)
        assert self.model(x).shape == (16, 6)

    def test_reconstruction_error_shape(self):
        x = torch.randn(32, 6)
        assert self.model.reconstruction_error(x).shape == (32,)

    def test_reconstruction_error_non_negative(self):
        x = torch.randn(32, 6)
        assert (self.model.reconstruction_error(x) >= 0).all()

    def test_minmax_scale_clips_to_01(self):
        raw = np.array([
            [v[0] for v in self.VITAL_RANGES.values()],
            [v[1] for v in self.VITAL_RANGES.values()],
        ])
        scaled = self.minmax_scale(raw)
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0

    def test_anomalous_records_higher_error(self):
        clean = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
        noisy = torch.tensor([[5.0, -3.0, 8.0, -2.0, 4.0, 6.0]], dtype=torch.float32)
        assert (self.model.reconstruction_error(noisy) >
                self.model.reconstruction_error(clean)).all()


class TestTrustScorer:
    def test_score_admission_structure(self, tmp_path):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import MinMaxScaler
        import src.ml.trust_scorer as ts

        FEATURE_COLS = ts.FEATURE_COLS
        clf = GradientBoostingClassifier(n_estimators=5, random_state=42)
        X   = np.random.rand(50, len(FEATURE_COLS))
        y   = np.random.randint(0, 2, 50)
        clf.fit(X, y)
        scaler = MinMaxScaler().fit(X)

        scorer_path = tmp_path / "trust_scorer.pkl"
        with open(scorer_path, "wb") as f:
            pickle.dump({"clf": clf, "scaler": scaler,
                         "features": FEATURE_COLS}, f)

        original       = ts.SCORER_PATH
        ts.SCORER_PATH = scorer_path
        result         = ts.score_admission("test-hadm",
                                            {c: 0.5 for c in FEATURE_COLS})
        ts.SCORER_PATH = original

        assert "trust_score"  in result
        assert "dq_flag"      in result
        assert "shap_values"  in result
        assert 0 <= result["trust_score"] <= 100
        assert isinstance(result["dq_flag"], (bool, np.bool_))
        assert set(result["shap_values"].keys()) == set(FEATURE_COLS)
