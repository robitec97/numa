"""Le sestine di ogni strategia per il prossimo concorso.

Tutte usano soltanto le estrazioni gia' avvenute. Le parti casuali sono
riproducibili: il seme dipende solo dalla data del concorso.

  numa         sestina poco giocata dagli altri (src/popularity.py): stessa
               probabilita' di vincere, quota attesa piu' alta quando si vince
  ensemble     ensemble temporale calibrato (src/robust.py), il candidato
               migliore del backtest; peso spesso 0 = equivale al caso
  gbm          gradient boosting originale (src/models.py)
  caldi        i 6 numeri piu' frequenti nelle ultime 50 estrazioni
  ritardatari  i 6 numeri assenti da piu' tempo
  caso         6 numeri a caso
"""

from datetime import date

import numpy as np
import pandas as pd

from .data import N_BALLS, N_DRAWN, NUM_COLS, presence_matrix
from .features import FEATURE_NAMES, build_features
from .models import gbm_model
from .popularity import PopularityModel, generate, past_combinations
from .robust import TemporalEnsembleModel

LABELS = {
    "numa": "Numa · sestina poco giocata",
    "ensemble": "Ensemble ML calibrato",
    "gbm": "Gradient boosting",
    "caldi": "Numeri caldi (ultime 50)",
    "ritardatari": "Ritardatari",
    "caso": "Caso puro",
}


def seed_for(target: date, salt: int = 0) -> int:
    return int(target.strftime("%Y%m%d")) * 10 + salt


def top6(scores: np.ndarray, rng: np.random.Generator) -> list[int]:
    jitter = rng.uniform(0, 1e-9, N_BALLS)  # spareggio neutro tra punteggi uguali
    return sorted(int(n) + 1 for n in np.argsort(-(scores + jitter))[:N_DRAWN])


def next_features(P: np.ndarray) -> np.ndarray:
    """Feature del concorso successivo all'ultimo osservato (solo storia passata)."""
    P_ext = np.vstack([P, np.zeros((1, N_BALLS), dtype=bool)])
    return build_features(P_ext)[-1]


def predict(draws: pd.DataFrame, popularity: PopularityModel, target: date,
            with_ml: bool = True) -> dict:
    """Sestine di tutte le strategie per il concorso `target`."""
    P = presence_matrix(draws)
    x_next = next_features(P)
    rng = lambda salt: np.random.default_rng(seed_for(target, salt))
    picks, info = {}, {}

    last = sorted(int(v) for v in draws.iloc[-1][NUM_COLS])
    numa = generate(popularity, rng(1), n=1, past=past_combinations(draws), last=last)[0]
    picks["numa"] = [int(v) for v in numa]
    info["numa_popularity"] = round(float(popularity.score(numa[None])[0]), 4)

    if with_ml:
        X = build_features(P)
        flatX, flaty = X.reshape(-1, X.shape[-1]), P.reshape(-1).astype(int)
        ens = TemporalEnsembleModel()
        ens.fit(flatX, flaty)
        picks["ensemble"] = top6(ens.predict_scores(x_next), rng(2))
        info["ensemble_weight"] = ens.weight
        gbm = gbm_model()
        gbm.fit(flatX, flaty)
        picks["gbm"] = top6(gbm.predict_scores(x_next), rng(3))

    picks["caldi"] = top6(x_next[:, FEATURE_NAMES.index("freq_50")].astype(float), rng(4))
    picks["ritardatari"] = top6(x_next[:, FEATURE_NAMES.index("gap")].astype(float), rng(5))
    picks["caso"] = sorted(int(n) + 1 for n in rng(6).choice(N_BALLS, N_DRAWN, replace=False))
    return {"picks": picks, "info": info}


def superstar_hint(popularity: PopularityModel, exclude, target: date) -> int:
    """Numero SuperStar suggerito: uno a caso tra i 30 meno giocati."""
    order = [int(n) + 1 for n in np.argsort(popularity.theta) if int(n) + 1 not in set(exclude)]
    return int(np.random.default_rng(seed_for(target, 7)).choice(order[:30]))
