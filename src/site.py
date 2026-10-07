"""Dati del sito (site/data/*.json): tutto cio' che la pagina mostra.

latest.json  prossimo concorso, sestine registrate, classifica prospettica,
             ultima estrazione, statistiche, modello di popolarita',
             riassunto dei risultati retrospettivi
draws.json   storico compatto (numeri, Jolly, SuperStar, premi per categoria)
             per i controlli fatti nel browser
ledger.json  registro delle previsioni (src/ledger.py)
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import ledger
from .data import N_BALLS, N_DRAWN, NUM_COLS, presence_matrix
from .features import FEATURE_NAMES, build_features
from .popularity import PopularityModel, random_tickets
from .quote import ATTRIBUTION
from .schedule import ROME, SALES_CLOSE, draw_moment, italian_date
from .strategies import LABELS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "site" / "data"
RESULTS = ROOT / "results"
ALGORITHM_FILES = ("src/strategies.py", "src/popularity.py", "src/robust.py",
                   "src/models.py", "src/features.py", "src/data.py")


def code_version() -> str:
    h = hashlib.sha256()
    for name in ALGORITHM_FILES:
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


def data_digest(draws: pd.DataFrame) -> str:
    return hashlib.sha256(draws[["data", *NUM_COLS]].to_csv(index=False).encode()).hexdigest()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def draw_slot(target) -> dict:
    return {"date": str(target), "label": italian_date(target),
            "draw_utc": _iso(draw_moment(target)),
            "sales_close": _iso(datetime.combine(target, SALES_CLOSE, ROME))}


def _r(x, nd=2):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def last_draw(draws, quote, model: PopularityModel) -> dict:
    row = draws.iloc[-1]
    nums = sorted(int(row[c]) for c in NUM_COLS)
    out = {"date": str(row["data"].date()), "label": italian_date(row["data"].date()),
           "numbers": nums, "jolly": int(row["jolly"]),
           "superstar": None if pd.isna(row["superstar"]) else int(row["superstar"]),
           "popularity": round(float(model.score(np.array([nums]))[0]), 3)}
    if quote is not None and pd.notna(quote.iloc[-1].get("payout_3")):
        q = quote.iloc[-1]
        out["prizes"] = {k: _r(q.get(f"payout_{k}")) for k in ("2", "3", "4", "5", "5_1", "6")}
        out["pop_3"] = _r(q.get("pop_3"), 3)
    return out


def stats(draws) -> dict:
    """Frequenze, ritardi attuali e la verifica della 'legge dei ritardatari'."""
    P = presence_matrix(draws)
    X = build_features(np.vstack([P, np.zeros((1, N_BALLS), dtype=bool)]))
    gap_now = X[-1, :, FEATURE_NAMES.index("gap_ratio")] * (N_BALLS / N_DRAWN)
    hot = X[-1, :, FEATURE_NAMES.index("freq_50")] * 50
    # P(esce | assente da almeno g concorsi): resta al 6,67% per ogni g
    G = X[:-1, :, FEATURE_NAMES.index("gap_ratio")] * (N_BALLS / N_DRAWN)
    G, Y = G[200:], P[200:]   # salta i primi concorsi, quando i ritardi non sono definiti
    curve = []
    for g in (1, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100):
        m = G >= g
        if m.sum() >= 2000:   # sotto, il margine d'errore copre l'intero grafico
            curve.append({"gap": g, "p": round(float(Y[m].mean()), 4), "n": int(m.sum())})
    return {"frequency": P.sum(axis=0).astype(int).tolist(),
            "gap": np.rint(gap_now).astype(int).tolist(),
            "hot50": np.rint(hot).astype(int).tolist(),
            "p_exit_given_gap": curve, "draws": int(len(draws))}


def retrospective() -> dict:
    """Riassunto dei risultati gia' calcolati in results/ (se presenti)."""
    out = {}
    try:
        imp = json.loads((RESULTS / "improvement.json").read_text())
        rep = json.loads((RESULTS / "replication.json").read_text())
        rnn = json.loads((RESULTS / "rnn.json").read_text())
        pooled = {name: sum(w["hits"][name] for w in rep["windows"].values())
                  for name in next(iter(rep["windows"].values()))["hits"]}
        pooled["LSTM (rete ricorrente)"] = sum(w["hits"] for w in rnn["windows"].values())
        pooled["ensemble temporale calibrato"] = imp["pooled"]["ensemble temporale calibrato"]["total_hits"]
        out["backtest_1800"] = {"draws": 1800, "expected": 720, "hits": pooled,
                                "period": ["2015-12-24", "2026-07-30"]}
    except (OSError, KeyError, ValueError):
        pass
    try:
        hold = json.loads((RESULTS / "holdout.json").read_text())
        if hold.get("status") == "evaluated":
            out["holdout"] = {"period": hold["period"], "draws": hold["draws"],
                              "models": {k: {"hits": v["total_hits"], "expected": v["expected_hits"],
                                             "p": v["p_chance_ge_exact"]} for k, v in hold["models"].items()}}
    except (OSError, KeyError, ValueError):
        pass
    try:
        st = json.loads((RESULTS / "stats.json").read_text())
        out["tests"] = {"uniformity_p": st["uniformity"].get("p_mc", st["uniformity"].get("p")),
                        "overlap_p": st["overlap"]["p"], "gaps_p": st["gaps"]["gof_p"],
                        "autocorr_p": st["autocorr"].get("ljungbox_p")}
    except (OSError, KeyError, ValueError):
        pass
    try:
        pop = json.loads((RESULTS / "popularity.json").read_text())
        bt = pop["backtest"]
        out["popularity"] = {
            "validation": {k: round(v["r2_within_year"], 3) for k, v in pop["validation"]["categories"].items()},
            "years": bt["years"], "draws": bt["draws"],
            "uniform_return": bt["uniform_exact_return_per_euro_2_3_4"],
            "strategies": {k: {"return": v["return_per_euro_2_3_4"], "vs_uniform": v["vs_uniform_2_3_4"],
                               "ci95": v["difference_per_euro_ci95"], "by_category": v["vs_uniform_by_category"],
                               "hits_per_ticket": v["hits_per_ticket"]}
                           for k, v in bt["strategies"].items()},
            "relative_popularity": pop.get("numa_relative_popularity"),
        }
    except (OSError, KeyError, ValueError):
        pass
    return out


def baseline_median(model: PopularityModel) -> float:
    """Punteggio mediano di una sestina a caso: il riferimento per "giocata N volte"."""
    rng = np.random.default_rng(90)
    return round(float(np.median(model.score(random_tickets(rng, 200_000)))), 5)


def draws_table(draws, quote) -> dict:
    """Storico compatto: [data, [6 numeri], jolly, superstar, prezzo, [premi 2,3,4,5,5+1,6]]."""
    rows = []
    for i, row in enumerate(draws.itertuples(index=False)):
        r = row._asdict()
        prizes = None
        price = 0.5 if r["data"] < pd.Timestamp("2016-02-02") else 1.0
        if quote is not None and pd.notna(quote.iloc[i].get("payout_3")):
            q = quote.iloc[i]
            prizes = [_r(q.get(f"payout_{k}")) for k in ("2", "3", "4", "5", "5_1", "6")]
        rows.append([str(r["data"].date()), sorted(int(r[c]) for c in NUM_COLS), int(r["jolly"]),
                     None if pd.isna(r["superstar"]) else int(r["superstar"]), price, prizes])
    return {"columns": ["data", "numeri", "jolly", "superstar", "prezzo", "premi_2_3_4_5_5+1_6"],
            "attribution": ["Estrazioni: github.com/Lottopyrhon/Estrazioni_Superenalotto", ATTRIBUTION],
            "rows": rows}


def write(draws, quote, model, entries, next_slot, notes, now=None, preview=None) -> dict:
    now = now or datetime.now(timezone.utc)
    current = next((e for e in reversed(entries) if e["target_date"] == next_slot["date"]), None)
    if current is None and preview is not None:
        current = {**preview, "preview": True}
    latest = {
        "generated_utc": _iso(now),
        "notes": notes,
        "data": {"draws": len(draws), "first": str(draws.data.iloc[0].date()),
                 "last": str(draws.data.iloc[-1].date()),
                 "sources": ["github.com/Lottopyrhon/Estrazioni_Superenalotto", ATTRIBUTION]},
        "next_draw": next_slot,
        "prediction": current,
        "labels": LABELS,
        "leaderboard": ledger.leaderboard(entries),
        "recent": [{"target_date": e["target_date"], "picks": e["picks"], "result": e.get("result")}
                   for e in entries[-12:]][::-1],
        "last_draw": last_draw(draws, quote, model),
        "stats": stats(draws),
        "popularity_model": {**model.to_dict(), "baseline_median": baseline_median(model)},
        "retrospective": retrospective(),
        "code_version": code_version(),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, separators=(",", ":")))
    (DATA_DIR / "draws.json").write_text(json.dumps(draws_table(draws, quote), separators=(",", ":")))
    return latest
