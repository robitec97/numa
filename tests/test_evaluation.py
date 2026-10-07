"""Regressioni su probabilita', inferenza e separazione temporale."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import hypergeom

from src.backtest import run_backtest
from src.features import build_features
from src.inference import exact_pvalue, holm_adjust, total_hits_pmf
from src.models import normalize_to_probs
from src.robust import TemporalEnsembleModel


class EvaluationTests(unittest.TestCase):
    def test_probabilities_concentrated_and_extreme(self):
        for scores in (np.r_[1.0, np.zeros(89)], np.full(90, 1e308),
                       np.zeros(90), np.arange(90) - 50,
                       np.geomspace(1e-300, 1e300, 90)):
            p = normalize_to_probs(scores)
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p >= 1e-6) & (p <= 1 - 1e-6)).all())
            self.assertAlmostEqual(p.sum(), 6, places=12)

    def test_normalization_preserves_ordinary_probabilities(self):
        scores = np.linspace(0.03, 0.1, 90)
        np.testing.assert_allclose(normalize_to_probs(scores), 6 * scores / scores.sum(), atol=1e-15)

    def test_invalid_scores_rejected(self):
        for scores in (np.zeros(89), np.full(90, np.nan), np.full(90, np.inf)):
            with self.assertRaises(ValueError):
                normalize_to_probs(scores)

    def test_exact_tail_matches_single_draw(self):
        for k in range(7):
            self.assertAlmostEqual(exact_pvalue(k, 1), hypergeom.sf(k - 1, 90, 6, 6), places=14)
        self.assertEqual(exact_pvalue(0, 0), 1.0)

    def test_exact_distribution_moments(self):
        n = 600
        pmf = total_hits_pmf(n)
        support = np.arange(len(pmf))
        mean = support @ pmf
        self.assertAlmostEqual(mean, 0.4 * n, places=10)
        self.assertAlmostEqual((support - mean) ** 2 @ pmf,
                               n * hypergeom.var(90, 6, 6), places=9)

    def test_holm_controls_reported_family(self):
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(len(holm_adjust([])), 0)
        with self.assertRaises(ValueError):
            holm_adjust([float("nan")])

    def test_features_do_not_use_current_or_future_draws(self):
        rng = np.random.default_rng(31)
        P = np.zeros((80, 90), dtype=bool)
        for row in P:
            row[rng.choice(90, 6, replace=False)] = True
        modified = P.copy()
        modified[40:] = np.roll(modified[40:], 7, axis=1)
        np.testing.assert_array_equal(build_features(P)[:41], build_features(modified)[:41])
        np.testing.assert_array_equal(build_features(P[:41]), build_features(P)[:41])

    def test_ensemble_splits_whole_draws_and_excludes_calibration_from_fit(self):
        captured = []

        class RecordingEstimator:
            n_iter_ = 1

            def __init__(self, **kwargs):
                pass

            def fit(self, X, y, sample_weight, X_val, y_val):
                captured.append((X.copy(), X_val.copy(), sample_weight.copy()))

            def predict_proba(self, X):
                # Uniforme: il pareggio deve scegliere peso zero.
                return np.tile([14 / 15, 1 / 15], (len(X), 1))

        X = np.repeat(np.arange(204), 90).reshape(-1, 1)
        y = np.tile(np.r_[np.ones(6), np.zeros(84)], 204)
        with patch("src.robust.HistGradientBoostingClassifier", RecordingEstimator):
            model = TemporalEnsembleModel(validation_draws=2, calibration_draws=2)
            model.fit(X, y)
        self.assertEqual(len(captured), 3)
        for train, val, weights in captured:
            self.assertEqual(train.max(), 199)
            self.assertEqual(val.min(), 200)
            self.assertEqual(val.max(), 201)
            self.assertTrue((np.ptp(weights.reshape(-1, 90), axis=1) == 0).all())
            self.assertEqual(weights.sum(), 200 * 90)
        self.assertEqual(model.fit_history[0]["validation_end"], 202)
        self.assertEqual(model.weight, 0)

    def test_backtest_fits_only_past_and_prediction_survives_future_changes(self):
        class RecordingModel:
            name = "recording"
            needs_fit = True

            def fit(self, X, y):
                self.last_training_draw = int(X.max())
                self.scores = y.reshape(-1, 90).mean(axis=0)

            def predict_scores(self, X):
                return self.scores

        X = np.broadcast_to(np.arange(45)[:, None, None], (45, 90, 1)).copy()
        P = np.zeros((45, 90), dtype=bool)
        P[:, :6] = True
        model = RecordingModel()
        _, result = run_backtest(X, P, 40, [model])
        self.assertEqual(model.last_training_draw, 39)
        changed = P.copy()
        changed[40:] = np.roll(changed[40:], 6, axis=1)
        _, other = run_backtest(X, changed, 40, [RecordingModel()])
        np.testing.assert_array_equal(result[model.name]["probs"], other[model.name]["probs"])


if __name__ == "__main__":
    unittest.main()
