"""Backtest walk-forward: per ogni concorso del periodo di valutazione i
modelli vedono solo la storia precedente. I modelli ML vengono riallenati
ogni RETRAIN_EVERY concorsi su finestra espandente.

Metriche per modello:
  - hits: numeri indovinati giocando ogni concorso la sestina top-6
  - log-loss e Brier per-numero (vs previsione probabilistica)
  - p-value Monte Carlo: probabilita' che una strategia puramente casuale
    ottenga almeno altrettanti hit totali
  - bilancio economico simulato di una schedina da 1 colonna a concorso
"""

import numpy as np

from .data import N_BALLS, N_DRAWN
from .models import normalize_to_probs

RETRAIN_EVERY = 30

# Vincite medie indicative per categoria (punti 2..6), in euro.
# Fonti: medie storiche pubblicate da Sisal; il jackpot (punti 6) e' posto
# a 50M come ordine di grandezza. Valori volutamente approssimativi.
PRIZE = {0: 0.0, 1: 0.0, 2: 5.0, 3: 25.0, 4: 300.0, 5: 32_000.0, 6: 50_000_000.0}
TICKET_COST = 1.0


def run_backtest(X, P, eval_start, models, seed=123, eval_end=None,
                 retrain_every=RETRAIN_EVERY):
    """X: (T, 90, F) feature, P: (T, 90) presenze, valuta i concorsi [eval_start, eval_end).

    Modelli con is_sequential=True ricevono direttamente la storia delle
    presenze (fit_seq / predict_scores_seq) invece delle feature tabellari.
    """
    T = X.shape[0]
    eval_idx = np.arange(eval_start, T if eval_end is None else eval_end)
    results = {
        m.name: {
            "hits": np.zeros(len(eval_idx), dtype=int),
            "logloss": np.zeros(len(eval_idx)),
            "brier": np.zeros(len(eval_idx)),
            "balance": np.zeros(len(eval_idx)),
            "probs": np.zeros((len(eval_idx), N_BALLS)),
            "picks": np.zeros((len(eval_idx), N_DRAWN), dtype=int),
        }
        for m in models
    }

    flatX = X.reshape(T * N_BALLS, -1)
    flaty = P.reshape(T * N_BALLS).astype(int)

    for j, t in enumerate(eval_idx):
        if any(m.needs_fit for m in models) and (j % retrain_every == 0):
            n_train_rows = t * N_BALLS
            for m in models:
                if not m.needs_fit:
                    continue
                if getattr(m, "is_sequential", False):
                    m.fit_seq(P[:t])
                else:
                    m.fit(flatX[:n_train_rows], flaty[:n_train_rows])

        actual = np.flatnonzero(P[t])  # indici 0-based dei 6 estratti
        y = P[t].astype(float)
        for m in models:
            if getattr(m, "is_sequential", False):
                scores = m.predict_scores_seq(P[:t])
            else:
                scores = m.predict_scores(X[t])
            p = normalize_to_probs(scores)
            # top-6 con tie-break deterministico ma neutro (hash del concorso)
            rng = np.random.default_rng(seed + t)
            jitter = rng.uniform(0, 1e-9, N_BALLS)
            picks = np.argsort(-(scores + jitter))[:N_DRAWN]
            hits = len(set(picks) & set(actual))

            r = results[m.name]
            r["hits"][j] = hits
            r["probs"][j] = p
            r["picks"][j] = np.sort(picks) + 1
            r["logloss"][j] = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            r["brier"][j] = np.mean((p - y) ** 2)
            r["balance"][j] = PRIZE[hits] - TICKET_COST

    return eval_idx, results


def random_hits_distribution(n_draws, n_sims=200_000, seed=7):
    """Distribuzione Monte Carlo degli hit totali di un giocatore casuale.

    Gli hit di una sestina casuale contro un'estrazione seguono una
    ipergeometrica(N=90, K=6, n=6); qui simuliamo la somma su n_draws concorsi.
    """
    rng = np.random.default_rng(seed)
    per_draw = rng.hypergeometric(N_DRAWN, N_BALLS - N_DRAWN, N_DRAWN, size=(n_sims, n_draws))
    return per_draw.sum(axis=1)


def mc_pvalue(total_hits, totals_mc):
    """P(caso >= osservato) e P(caso <= osservato), a due code."""
    ge = (totals_mc >= total_hits).mean()
    le = (totals_mc <= total_hits).mean()
    return ge, le
