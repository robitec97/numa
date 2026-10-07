"""Replica del backtest su finestre temporali disgiunte.

Il gradient boosting ha battuto il caso nella finestra piu' recente
(p=0.008). Prima di credere a un segnale predittivo va replicato su
periodi indipendenti: un effetto reale si ripete, un colpo di fortuna no.
Testiamo anche 5 seed diversi del GBM sulla finestra originale per
misurare quanto il risultato dipenda dal caso interno al modello.
"""

import json
from pathlib import Path

import numpy as np

from src.backtest import mc_pvalue, random_hits_distribution, run_backtest
from src.data import load_draws, presence_matrix
from src.features import build_features
from src.models import all_models, gbm_model

W = 600
OUT = Path("results")


def evaluate(X, P, start, end, models):
    _, results = run_backtest(X, P, start, models, eval_end=end)
    return {name: int(r["hits"].sum()) for name, r in results.items()}


def main():
    df = load_draws()
    P = presence_matrix(df)
    X = build_features(P)
    T = len(df)

    totals_mc = random_hits_distribution(W)
    windows = {
        "A (piu' vecchia)": (T - 3 * W, T - 2 * W),
        "B (intermedia)": (T - 2 * W, T - W),
        "C (originale)": (T - W, T),
    }

    out = {"windows": {}, "gbm_seeds": {}}
    for label, (s, e) in windows.items():
        period = f"{df.data.iloc[s].date()} -> {df.data.iloc[e-1].date()}"
        hits = evaluate(X, P, s, e, all_models())
        out["windows"][label] = {"period": period, "hits": hits}
        print(f"\nFinestra {label}  [{period}]  — atteso dal caso: 240")
        for name, h in hits.items():
            p_ge, _ = mc_pvalue(h, totals_mc)
            print(f"  {name:32s} {h:4d}  p(>=caso)={p_ge:.3f}")

    print("\nGBM con 5 seed diversi, finestra C:")
    for seed in (1, 7, 42, 99, 2026):
        hits = evaluate(X, P, T - W, T, [gbm_model(seed=seed)])
        h = list(hits.values())[0]
        p_ge, _ = mc_pvalue(h, totals_mc)
        out["gbm_seeds"][seed] = h
        print(f"  seed={seed:5d}  hit={h:4d}  p(>=caso)={p_ge:.3f}")

    OUT.mkdir(exist_ok=True)
    (OUT / "replication.json").write_text(json.dumps(out, indent=2))
    print("\nSalvato: results/replication.json")


if __name__ == "__main__":
    main()
