"""Confronto esplorativo riproducibile, senza sovrascrivere i risultati originali.

Uso: .venv/bin/python run_improvement.py
La fine del periodo resta fissa al 30/07/2026 anche se il dataset cresce.
Tutte le finestre sono gia' state esplorate: servira' un test prospettico.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from src.backtest import run_backtest
from src.data import DATA_PATH, NUM_COLS, load_draws, presence_matrix
from src.features import build_features
from src.inference import exact_pvalue, holm_adjust
from src.models import UniformModel, gbm_model
from src.robust import TemporalEnsembleModel

ROOT = Path(__file__).resolve().parent


def summarize(result):
    n = len(result["hits"])
    hits = int(result["hits"].sum())
    return {
        "draws": n, "total_hits": hits, "hits_per_draw": hits / n,
        "expected_hits": n * 0.4, "p_chance_ge_exact": exact_pvalue(hits, n),
        "logloss": float(result["logloss"].mean()),
        "brier": float(result["brier"].mean()),
        "hit_counts": np.bincount(result["hits"], minlength=7).tolist(),
    }


def paired_interval(differences, block=30, seed=20260905):
    """IC bootstrap approssimato, blocchi di retraining, stratificato per finestra.

    differences ha forma (finestre, concorsi). Il campionamento mantiene
    appaiati i due modelli e non tratta le 90 palline come indipendenti.
    """
    d = np.asarray(differences, dtype=float)
    if d.ndim != 2 or d.shape[1] % block:
        raise ValueError("ogni finestra deve contenere blocchi interi")
    blocks = d.reshape(d.shape[0], -1, block).mean(axis=2)
    rng = np.random.default_rng(seed)
    estimates = np.zeros(4000)
    for window in blocks:
        samples = rng.integers(len(window), size=(len(estimates), len(window)))
        estimates += window[samples].mean(axis=1) / len(blocks)
    return {"mean_difference": float(d.mean()),
            "ci95_block_bootstrap": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "block_draws": block, "bootstrap_samples": len(estimates)}


def audit_originals():
    backtest = json.loads((ROOT / "results/backtest.json").read_text())
    replication = json.loads((ROOT / "results/replication.json").read_text())
    rnn = json.loads((ROOT / "results/rnn.json").read_text())
    names = list(backtest["models"])
    n = backtest["eval_draws"]
    raw = [exact_pvalue(backtest["models"][name]["total_hits"], n) for name in names]
    lstm_name = "LSTM (rete ricorrente)"
    lstm = list(rnn["windows"].values())[-1]
    full_names = names + [lstm_name]
    full_p = raw + [exact_pvalue(lstm["hits"], n)]
    pooled_hits = {name: sum(w["hits"][name] for w in replication["windows"].values()) for name in names}
    pooled_hits[lstm_name] = sum(w["hits"] for w in rnn["windows"].values())
    pooled_n = n * len(replication["windows"])
    pooled_p = [exact_pvalue(pooled_hits[name], pooled_n) for name in full_names]
    return {
        "note": "Correzioni descrittive sulle famiglie riportate; non comprendono tutte le scelte esplorative o i seed provati.",
        "window_C": {name: {"p_exact": p, "p_holm_7": float(h)}
                     for name, p, h in zip(full_names, full_p, holm_adjust(full_p))},
        "gbm_window_C_holm_original_6": float(holm_adjust(raw)[names.index("gradient boosting")]),
        "pooled_draws": pooled_n,
        "pooled": {name: {"total_hits": pooled_hits[name], "p_exact": p, "p_holm_7": float(h)}
                   for name, p, h in zip(full_names, pooled_p, holm_adjust(pooled_p))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-end", default="2026-07-30")
    parser.add_argument("--window", type=int, default=600)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "results/improvement.json")
    args = parser.parse_args()
    if args.window < 30 or args.window % 30 or args.threads < 1:
        parser.error("--window deve essere multiplo positivo di 30; --threads positivo")
    df = load_draws()
    df = df[df.data <= pd.Timestamp(args.history_end)].reset_index(drop=True)
    if len(df) < 3 * args.window + 560:
        parser.error("storico insufficiente per tre finestre e i blocchi di training/validazione/calibrazione")
    P = presence_matrix(df)
    X = build_features(P)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory_retrospective",
        "history_end": str(df.data.iloc[-1].date()), "window_draws": args.window,
        "windows": 3, "retrain_every": 30, "tie_seed": 123,
        "challenger": {"seeds": [1, 7, 42], "validation_draws": 240,
                       "calibration_draws": 120, "max_iter": 150,
                       "weights": [0, 0.1, 0.25, 0.5, 0.75, 1]},
        "primary_metric": "pooled logloss difference vs uniform (negative is better)",
        "secondary_metric": "top-6 hits; exact upper tail under independent uniform draws",
        "limitations": "Finestre gia' esaminate nel progetto: nessun risultato costituisce conferma su dati nuovi. IC bootstrap approssimati, nessuna garanzia di profitto.",
        "data_path": str(DATA_PATH),
        "evaluated_data_sha256": hashlib.sha256(df[["data", *NUM_COLS]].to_csv(index=False).encode()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ["src/robust.py", "src/models.py", "src/features.py", "src/backtest.py", "src/data.py", "src/inference.py", "run_improvement.py"]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".protocol.json").write_text(json.dumps(protocol, indent=2))
    out = {"protocol": protocol, "original_audit": audit_originals(), "windows": {}}
    records = []
    started = time.monotonic()
    with threadpool_limits(limits=args.threads):
        for i, label in enumerate("ABC"):
            start = len(df) - (3 - i) * args.window
            end = start + args.window
            challenger = TemporalEnsembleModel()
            print(f"Finestra {label}: {df.data.iloc[start].date()} -> {df.data.iloc[end-1].date()}", flush=True)
            idx, result = run_backtest(X, P, start, [UniformModel(), gbm_model(), challenger], eval_end=end)
            records.append(result)
            out["windows"][label] = {
                "period": [str(df.data.iloc[start].date()), str(df.data.iloc[end-1].date())],
                "models": {name: summarize(r) for name, r in result.items()},
                "challenger_fits": challenger.fit_history,
            }
            for name, r in out["windows"][label]["models"].items():
                print(f"  {name}: {r['total_hits']} hit, logloss={r['logloss']:.8f}, p={r['p_chance_ge_exact']:.4f}", flush=True)
    out["pooled"] = {
        name: summarize({key: np.concatenate([r[name][key] for r in records]) for key in ("hits", "logloss", "brier")})
        for name in records[0]
    }
    out["paired_comparisons"] = {
        baseline: {metric: paired_interval([r[TemporalEnsembleModel.name][metric] - r[baseline][metric] for r in records])
                   for metric in ("hits", "logloss", "brier")}
        for baseline in (UniformModel.name, "gradient boosting")
    }
    out["elapsed_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(out, indent=2))
    np.savez_compressed(args.output.with_suffix(".npz"),
                        eval_idx=np.arange(len(df) - 3 * args.window, len(df)),
                        **{f"{key}__{name}": np.concatenate([r[name][key] for r in records])
                           for name in records[0] for key in ("hits", "logloss", "brier", "probs", "picks")})
    print("Totali:", json.dumps(out["pooled"], indent=2), flush=True)
    print(f"Salvato: {args.output} [{out['elapsed_seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
