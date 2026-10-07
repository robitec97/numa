"""Allena e valuta tutti i modelli in walk-forward sugli ultimi 600 concorsi.

Uso: .venv/bin/python run_backtest.py
Salva risultati in results/backtest.json (metriche) e results/backtest.npz.
"""

import json
import time
from pathlib import Path

import numpy as np

from src.backtest import mc_pvalue, random_hits_distribution, run_backtest
from src.data import load_draws, presence_matrix
from src.features import build_features
from src.models import all_models

EVAL_DRAWS = 600
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def main():
    df = load_draws()
    P = presence_matrix(df)
    print(f"Estrazioni: {len(df)} | costruzione feature...", flush=True)
    X = build_features(P)

    eval_start = len(df) - EVAL_DRAWS
    print(f"Backtest su concorsi {eval_start}..{len(df)-1} "
          f"({df.data.iloc[eval_start].date()} -> {df.data.iloc[-1].date()})", flush=True)

    t0 = time.time()
    models = all_models()
    eval_idx, results = run_backtest(X, P, eval_start, models)
    print(f"Backtest completato in {time.time()-t0:.0f}s", flush=True)

    totals_mc = random_hits_distribution(EVAL_DRAWS)
    expected_hits = EVAL_DRAWS * 6 * 6 / 90

    summary = {}
    print(f"\nAtteso per puro caso: {expected_hits:.0f} hit totali "
          f"(media {expected_hits/EVAL_DRAWS:.3f}/concorso, sd MC {totals_mc.std():.1f})\n")
    header = f"{'modello':32s} {'hit':>5s} {'hit/conc':>9s} {'p(>=caso)':>10s} {'logloss':>9s} {'brier':>9s} {'bilancio EUR':>13s}"
    print(header)
    for name, r in results.items():
        tot = int(r["hits"].sum())
        p_ge, p_le = mc_pvalue(tot, totals_mc)
        summary[name] = {
            "total_hits": tot,
            "hits_per_draw": tot / EVAL_DRAWS,
            "p_beats_chance": p_ge,
            "p_worse_than_chance": p_le,
            "logloss": float(r["logloss"].mean()),
            "brier": float(r["brier"].mean()),
            "balance_eur": float(r["balance"].sum()),
            "hit_counts": np.bincount(r["hits"], minlength=7).tolist(),
        }
        s = summary[name]
        print(f"{name:32s} {tot:5d} {s['hits_per_draw']:9.3f} {p_ge:10.3f} "
              f"{s['logloss']:9.5f} {s['brier']:9.5f} {s['balance_eur']:13.0f}")

    meta = {
        "eval_draws": EVAL_DRAWS,
        "eval_period": [str(df.data.iloc[eval_start].date()), str(df.data.iloc[-1].date())],
        "expected_hits_chance": expected_hits,
        "mc_std": float(totals_mc.std()),
        "models": summary,
    }
    (OUT / "backtest.json").write_text(json.dumps(meta, indent=2))
    np.savez_compressed(
        OUT / "backtest.npz",
        eval_idx=eval_idx,
        totals_mc=totals_mc[:20000],
        **{f"hits__{n}": r["hits"] for n, r in results.items()},
        **{f"probs__{n}": r["probs"].astype(np.float32) for n, r in results.items()},
        **{f"balance__{n}": r["balance"] for n, r in results.items()},
    )
    print("\nSalvato: results/backtest.json, results/backtest.npz")


if __name__ == "__main__":
    main()
