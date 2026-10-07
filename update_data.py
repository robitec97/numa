"""Aggiornamento automatico del dataset dalla fonte GitHub.

Scarica lo storico aggiornato, lo valida (range, duplicati, date) e — prima
di sovrascrivere — verifica che le estrazioni gia' presenti in locale
coincidano con quelle remote. Se la fonte remota contraddice il passato,
NON aggiorna e segnala il problema (i dati storici non cambiano: se
cambiano, e' la fonte a essere sospetta).

Uso: .venv/bin/python update_data.py
     oppure via predict_next.py, che lo invoca automaticamente.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data import DATA_PATH, N_BALLS, N_DRAWN, NUM_COLS

REMOTE_URL = ("https://raw.githubusercontent.com/Lottopyrhon/"
              "Estrazioni_Superenalotto/main/superenalotto.txt")
COLS = ["concorso", "data", *NUM_COLS, "jolly", "superstar"]


def _parse(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), names=COLS, dayfirst=True, parse_dates=["data"])
    df = df.sort_values("data").reset_index(drop=True)
    nums = df[NUM_COLS].to_numpy()
    if not (nums.min() >= 1 and nums.max() <= N_BALLS):
        raise ValueError("dati remoti: numeri fuori range 1-90")
    if any(len(set(r)) < N_DRAWN for r in nums):
        raise ValueError("dati remoti: numeri duplicati in un'estrazione")
    if df["data"].duplicated().any():
        raise ValueError("dati remoti: date duplicate")
    return df


def update(path: Path = DATA_PATH, timeout: int = 20) -> int:
    """Scarica e aggiorna il file. Ritorna il numero di estrazioni nuove.

    Solleva eccezioni su problemi di rete o validazione: sta al chiamante
    decidere se proseguire con i dati locali.
    """
    local_text = path.read_text()
    local = _parse(local_text)

    resp = requests.get(REMOTE_URL, timeout=timeout)
    resp.raise_for_status()
    remote = _parse(resp.text)

    # le estrazioni gia' note devono coincidere (confronto sui 6 numeri ordinati)
    m = local.merge(remote, on="data", suffixes=("_l", "_r"))
    if len(m) < len(local):
        raise ValueError(f"la fonte remota ha perso {len(local) - len(m)} estrazioni locali")
    sl = np.sort(m[[f"{c}_l" for c in NUM_COLS]].to_numpy(), axis=1)
    sr = np.sort(m[[f"{c}_r" for c in NUM_COLS]].to_numpy(), axis=1)
    bad = (sl != sr).any(axis=1)
    if bad.any():
        dates = m.loc[bad, "data"].dt.date.tolist()[:5]
        raise ValueError(f"la fonte remota contraddice {bad.sum()} estrazioni storiche: {dates}")

    n_new = len(remote) - len(local)
    if n_new > 0:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(resp.text)
        tmp.replace(path)
    return n_new


def main():
    try:
        n_new = update()
    except Exception as e:
        print(f"Aggiornamento fallito ({e}); si continua con i dati locali.")
        sys.exit(1)
    if n_new:
        print(f"Dataset aggiornato: +{n_new} estrazioni nuove.")
    else:
        print("Dataset gia' aggiornato.")


if __name__ == "__main__":
    main()
