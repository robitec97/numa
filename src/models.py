"""Modelli e strategie a confronto.

Interfaccia comune:
  fit(X_train, y_train)          — X: (n_righe, n_feature), y: 0/1
  predict_scores(X_draw)         — X_draw: (90, n_feature) per un singolo
                                   concorso; ritorna 90 punteggi (piu' alto
                                   = piu' "probabile" secondo il modello)

I punteggi vengono poi normalizzati in probabilita' per-numero che sommano
a 6 (il numero di palline estratte) per il confronto su log-loss/Brier.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import N_BALLS, N_DRAWN
from .features import FEATURE_NAMES


class UniformModel:
    """Baseline teorico: ogni numero ha probabilita' 6/90."""

    name = "caso (uniforme)"
    needs_fit = False

    def fit(self, X, y):
        pass

    def predict_scores(self, X_draw):
        return np.full(N_BALLS, N_DRAWN / N_BALLS)


class FeatureRankModel:
    """Strategia che ordina i numeri secondo una singola feature."""

    needs_fit = False

    def __init__(self, name, feature, sign=1.0):
        self.name = name
        self._idx = FEATURE_NAMES.index(feature)
        self._sign = sign

    def fit(self, X, y):
        pass

    def predict_scores(self, X_draw):
        return self._sign * X_draw[:, self._idx].astype(np.float64)


def hot_model():
    """Numeri 'caldi': i piu' frequenti nelle ultime 50 estrazioni."""
    return FeatureRankModel("numeri caldi (freq. ultime 50)", "freq_50")


def cold_model():
    """Numeri 'ritardatari': quelli assenti da piu' tempo."""
    return FeatureRankModel("ritardatari (gap massimo)", "gap")


def global_freq_model():
    """Numeri storicamente piu' frequenti dall'inizio del gioco."""
    return FeatureRankModel("freq. storica globale", "global_freq")


class SklearnModel:
    needs_fit = True

    def __init__(self, name, estimator):
        self.name = name
        self.est = estimator

    def fit(self, X, y):
        self.est.fit(X, y)

    def predict_scores(self, X_draw):
        return self.est.predict_proba(X_draw)[:, 1]


def logistic_model():
    est = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    return SklearnModel("regressione logistica", est)


def gbm_model(seed=42):
    est = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=seed,
    )
    return SklearnModel("gradient boosting", est)


def all_models():
    return [
        UniformModel(),
        global_freq_model(),
        hot_model(),
        cold_model(),
        logistic_model(),
        gbm_model(),
    ]


def normalize_to_probs(scores: np.ndarray) -> np.ndarray:
    """Trasforma 90 punteggi in probabilita' per-numero che sommano a 6.

    Mantiene sia la somma a 6 sia i limiti stretti (0, 1), anche quando
    quasi tutto il punteggio e' concentrato su un solo numero.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (N_BALLS,) or not np.isfinite(s).all():
        raise ValueError("servono 90 punteggi finiti")
    s = np.clip(s, 0.0, None)
    if s.max() > 0:
        s = s / s.max()  # evita overflow senza cambiare le proporzioni
    if s.sum() <= 0:
        p = np.full(N_BALLS, N_DRAWN / N_BALLS)
    else:
        p = s / s.sum() * N_DRAWN
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    normalized = clipped / clipped.sum() * N_DRAWN
    if normalized.min() >= 1e-6 and normalized.max() <= 1 - 1e-6:
        return normalized
    # Proiezione sul simplesso limitato: clip e successiva rinormalizzazione
    # da soli possono restituire probabilita' maggiori di 1.
    lo, hi = -float(N_DRAWN), float(N_DRAWN)
    for _ in range(80):
        shift = (lo + hi) / 2
        if np.clip(p + shift, 1e-6, 1 - 1e-6).sum() < N_DRAWN:
            lo = shift
        else:
            hi = shift
    return np.clip(p + (lo + hi) / 2, 1e-6, 1 - 1e-6)
