"""Caricamento e validazione dello storico estrazioni SuperEnalotto.

Dataset primario: data/superenalotto_full.txt (repo GitHub Lottopyrhon/
Estrazioni_Superenalotto, aggiornato al 30/07/2026), incrociato con
luigimassa/superenalotto-archivio e superenalotto.net per la validazione.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "superenalotto_full.txt"

N_BALLS = 90
N_DRAWN = 6

# Concorso del 27/12/2004: la fonte primaria riporta 41, ma sia
# luigimassa/superenalotto-archivio sia superenalotto.net riportano 4.
KNOWN_FIXES = {("2004-12-27", 41): 4}

NUM_COLS = [f"n{i}" for i in range(1, 7)]


def load_draws(path: Path = DATA_PATH) -> pd.DataFrame:
    cols = ["concorso", "data", *NUM_COLS, "jolly", "superstar"]
    df = pd.read_csv(path, names=cols, dayfirst=True, parse_dates=["data"])
    df = df.sort_values("data").reset_index(drop=True)

    for (date, wrong), right in KNOWN_FIXES.items():
        mask = df["data"] == date
        row = df.loc[mask, NUM_COLS]
        df.loc[mask, NUM_COLS] = row.replace(wrong, right).values

    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    nums = df[NUM_COLS].to_numpy()
    assert nums.min() >= 1 and nums.max() <= N_BALLS, "numeri fuori range 1-90"
    assert not any(len(set(r)) < N_DRAWN for r in nums), "numeri duplicati in un'estrazione"
    assert df["data"].is_monotonic_increasing, "date non ordinate"
    assert not df["data"].duplicated().any(), "date duplicate"


def presence_matrix(df: pd.DataFrame) -> np.ndarray:
    """Matrice booleana (n_estrazioni, 90): P[t, n-1] = numero n uscito al concorso t."""
    nums = df[NUM_COLS].to_numpy()
    P = np.zeros((len(df), N_BALLS), dtype=bool)
    rows = np.repeat(np.arange(len(df)), N_DRAWN)
    P[rows, nums.ravel() - 1] = True
    return P
