"""Valutazione della rete ricorrente (LSTM) con la stessa metodologia
degli altri modelli: backtest walk-forward su 3 finestre disgiunte da 600
concorsi + sensibilita' al seed sulla finestra piu' recente.

Salva results/rnn.json (usato da make_figures.py per integrare le figure).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.backtest import mc_pvalue, random_hits_distribution, run_backtest
from src.data import load_draws, presence_matrix
from src.features import build_features
from src.rnn import LSTMModel

W = 600
OUT = Path("results")
torch.set_num_threads(max(1, os.cpu_count() - 2))


def main():
    df = load_draws()
    P = presence_matrix(df)
    X = build_features(P)  # ignorate dall'LSTM, richieste dall'harness
    T = len(df)
    totals_mc = random_hits_distribution(W)

    windows = {
        "A (piu' vecchia)": (T - 3 * W, T - 2 * W),
        "B (intermedia)": (T - 2 * W, T - W),
        "C (originale)": (T - W, T),
    }

    out = {"windows": {}, "seeds": {}}
    for label, (s, e) in windows.items():
        t0 = time.time()
        _, res = run_backtest(X, P, s, models=[LSTMModel(seed=42)], eval_end=e)
        r = next(iter(res.values()))
        tot = int(r["hits"].sum())
        p_ge, _ = mc_pvalue(tot, totals_mc)
        period = f"{df.data.iloc[s].date()} -> {df.data.iloc[e-1].date()}"
        out["windows"][label] = {
            "period": period,
            "hits": tot,
            "p_beats_chance": p_ge,
            "logloss": float(r["logloss"].mean()),
            "brier": float(r["brier"].mean()),
            "balance_eur": float(r["balance"].sum()),
            "hit_counts": np.bincount(r["hits"], minlength=7).tolist(),
        }
        print(f"Finestra {label} [{period}]  hit={tot} (attesi 240)  "
              f"p(>=caso)={p_ge:.3f}  logloss={r['logloss'].mean():.5f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    print("\nSensibilita' al seed, finestra C:")
    for seed in (1, 7, 42, 99, 2026):
        if seed == 42:
            tot = out["windows"]["C (originale)"]["hits"]
        else:
            _, res = run_backtest(X, P, T - W, models=[LSTMModel(seed=seed)])
            tot = int(next(iter(res.values()))["hits"].sum())
        p_ge, _ = mc_pvalue(tot, totals_mc)
        out["seeds"][seed] = tot
        print(f"  seed={seed:5d}  hit={tot}  p(>=caso)={p_ge:.3f}", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "rnn.json").write_text(json.dumps(out, indent=2))
    print("\nSalvato: results/rnn.json")


if __name__ == "__main__":
    main()
