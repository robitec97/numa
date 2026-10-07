"""Challenger con bagging per concorso, validazione temporale e shrinkage.

Protocollo fisso: tre bootstrap del training, 240 concorsi per early
stopping, poi 120 concorsi distinti per calibrare il peso verso l'uniforme.
Nessuna di queste scelte usa il periodo di test. Il modello non viene
riallenato sui blocchi di validazione/calibrazione, per mantenere la stessa
distribuzione dei punteggi che e' stata calibrata.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import N_BALLS, N_DRAWN
from .models import normalize_to_probs


class TemporalEnsembleModel:
    name = "ensemble temporale calibrato"
    needs_fit = True

    def __init__(self, seeds=(1, 7, 42), validation_draws=240,
                 calibration_draws=120, max_iter=150):
        if not seeds or validation_draws < 1 or calibration_draws < 1 or max_iter < 1:
            raise ValueError("parametri del protocollo non validi")
        self.seeds = tuple(seeds)
        self.validation_draws = validation_draws
        self.calibration_draws = calibration_draws
        self.max_iter = max_iter
        self.fit_history = []

    def fit(self, X, y):
        if len(X) != len(y) or len(X) % N_BALLS:
            raise ValueError("training deve contenere concorsi interi da 90 righe")
        n = len(X) // N_BALLS
        train_end = n - self.validation_draws - self.calibration_draws
        if train_end < 200:
            raise ValueError("servono almeno 200 concorsi prima di validazione e calibrazione")
        val_end = n - self.calibration_draws
        a, b = train_end * N_BALLS, val_end * N_BALLS
        self.estimators = []
        for seed in self.seeds:
            # Bootstrap di intere estrazioni: i 90 esempi di un concorso
            # rimangono insieme, anche nella scelta dei pesi.
            rng = np.random.default_rng(seed)
            counts = np.bincount(rng.integers(train_end, size=train_end), minlength=train_end)
            est = HistGradientBoostingClassifier(
                max_iter=self.max_iter, learning_rate=0.05, max_depth=3,
                min_samples_leaf=400, l2_regularization=10.0,
                early_stopping=True, n_iter_no_change=10, tol=1e-7,
                random_state=seed,
            )
            est.fit(X[:a], y[:a], sample_weight=np.repeat(counts, N_BALLS),
                    X_val=X[a:b], y_val=y[a:b])
            self.estimators.append(est)
        raw = np.mean([e.predict_proba(X[b:])[:, 1] for e in self.estimators], axis=0)
        p = np.array([normalize_to_probs(row) for row in raw.reshape(-1, N_BALLS)])
        labels = y[b:].reshape(-1, N_BALLS)
        baseline = N_DRAWN / N_BALLS
        # Griglia piccola fissata a priori; parita' risolta a favore
        # dell'uniforme. La calibrazione non cambia il ranking se peso > 0.
        weights = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
        losses = []
        for weight in weights:
            mixed = baseline + weight * (p - baseline)
            losses.append(float(-np.mean(labels * np.log(mixed) + (1 - labels) * np.log1p(-mixed))))
        self.weight = float(weights[np.argmin(losses)])
        self.fit_history.append({
            "seen_draws": n, "train_end": train_end, "validation_end": val_end,
            "weight": self.weight, "iterations": [int(e.n_iter_) for e in self.estimators],
            "calibration_logloss": losses,
        })

    def predict_scores(self, X_draw):
        p = normalize_to_probs(np.mean([e.predict_proba(X_draw)[:, 1] for e in self.estimators], axis=0))
        baseline = N_DRAWN / N_BALLS
        return baseline + self.weight * (p - baseline)
