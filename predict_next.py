"""Genera la 'previsione' dei modelli per il prossimo concorso, dopo aver
aggiornato automaticamente il dataset dalla fonte remota.

ATTENZIONE (ed e' il punto di tutto il progetto): il backtest dimostra che
queste sestine NON hanno piu' probabilita' di vincere di sei numeri a caso.
Ogni sestina ha probabilita' 1 su C(90,6) = 622.614.630 di fare 6.

Uso: .venv/bin/python predict_next.py [--no-update]
"""

import os
import sys

import numpy as np

import update_data
from src.data import N_BALLS, load_draws, presence_matrix
from src.features import build_features
from src.models import gbm_model, normalize_to_probs
from src.rnn import LSTMModel


def refresh_dataset():
    try:
        n_new = update_data.update()
    except Exception as e:
        print(f"[!] Aggiornamento non riuscito ({e}): uso i dati locali.\n")
        return
    if n_new:
        print(f"Dataset aggiornato: +{n_new} estrazioni nuove.\n")
    else:
        print("Dataset gia' aggiornato.\n")


def report(name, scores):
    p = normalize_to_probs(scores)
    top = np.sort(np.argsort(-scores)[:6])
    nums = "  ".join(f"{n+1:2d}" for n in top)
    probs = ", ".join(f"{n+1}: {100*p[n]:.2f}%" for n in top)
    print(f"{name}\n  sestina: {nums}\n  P stimate (teorico 6.67%): {probs}\n")


def main():
    if "--no-update" not in sys.argv:
        refresh_dataset()

    df = load_draws()
    P = presence_matrix(df)
    X = build_features(P)

    # feature per il concorso successivo all'ultimo osservato: appendo una
    # riga fittizia e ricostruisco (le feature di t usano solo la storia < t)
    P_ext = np.vstack([P, np.zeros((1, N_BALLS), dtype=bool)])
    x_next = build_features(P_ext)[-1]

    print(f"Ultimo concorso nel dataset: {df.data.iloc[-1].date()}\n")

    gbm = gbm_model()
    gbm.fit(X.reshape(-1, X.shape[-1]), P.reshape(-1).astype(int))
    report("Gradient boosting", gbm.predict_scores(x_next))

    try:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() - 2))
        lstm = LSTMModel(seed=42)
        lstm.fit_seq(P)
        report("LSTM (rete ricorrente)", lstm.predict_scores_seq(P))
    except ImportError:
        print("[!] PyTorch non installato: salto la previsione LSTM.\n")

    print("Nota: il backtest su 1.800 concorsi mostra che queste 'previsioni'")
    print("non battono il caso. Giocare queste sestine = giocarne una qualsiasi.")


if __name__ == "__main__":
    main()
