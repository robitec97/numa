"""Feature engineering: per ogni (concorso t, numero n) costruisce feature
calcolate esclusivamente sulla storia precedente a t (nessun leakage).
"""

import numpy as np

from .data import N_BALLS, N_DRAWN

WINDOWS = (10, 50, 200)
EWMA_ALPHAS = (0.02, 0.1)

FEATURE_NAMES = (
    [f"freq_{w}" for w in WINDOWS]
    + [f"ewma_{a}" for a in EWMA_ALPHAS]
    + ["gap", "gap_ratio", "global_freq", "number", "parity", "decade"]
)


def build_features(P: np.ndarray) -> np.ndarray:
    """Ritorna X di forma (T, 90, n_feature).

    X[t, n] usa solo le estrazioni 0..t-1. La riga t=0 non ha storia:
    le feature di frequenza valgono 0 e il gap vale il massimo osservabile.
    """
    T = P.shape[0]
    Pf = P.astype(np.float64)
    n_feat = len(FEATURE_NAMES)
    X = np.zeros((T, N_BALLS, n_feat), dtype=np.float32)
    col = 0

    # cumsum "shiftata": C[t] = somma delle presenze nelle estrazioni < t
    C = np.zeros((T + 1, N_BALLS))
    np.cumsum(Pf, axis=0, out=C[1:])

    for w in WINDOWS:
        lo = np.clip(np.arange(T) - w, 0, None)
        width = np.arange(T) - lo
        width[0] = 1  # evita 0/0 al primo concorso
        X[:, :, col] = (C[np.arange(T)] - C[lo]) / width[:, None]
        col += 1

    for a in EWMA_ALPHAS:
        ew = np.zeros(N_BALLS)
        for t in range(T):
            X[t, :, col] = ew
            ew = (1 - a) * ew + a * Pf[t]
        col += 1

    # gap: concorsi trascorsi dall'ultima uscita (prima della prima uscita: t+1)
    last_seen = np.full(N_BALLS, -1)
    gap = np.empty((T, N_BALLS))
    for t in range(T):
        gap[t] = t - last_seen
        last_seen[P[t]] = t
    expected_gap = N_BALLS / N_DRAWN  # 15
    X[:, :, col] = np.log1p(gap)
    X[:, :, col + 1] = gap / expected_gap
    col += 2

    counts = C[:T]
    denom = np.clip(np.arange(T), 1, None).astype(float)
    X[:, :, col] = counts / denom[:, None]
    col += 1

    n = np.arange(1, N_BALLS + 1)
    X[:, :, col] = (n / N_BALLS)[None, :]
    X[:, :, col + 1] = (n % 2)[None, :]
    X[:, :, col + 2] = ((n - 1) // 10 / 8.0)[None, :]

    return X
