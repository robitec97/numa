"""Sestine impopolari: stima della popolarita' e verifica sui premi reali.

1. Stima il modello di popolarita' (src/popularity.py) dalle quote storiche
   e ne misura la capacita' predittiva su anni non usati per la stima.
2. Backtest walk-forward dei premi: per ogni anno dal 2012 il modello e'
   stimato solo sugli anni precedenti; per ogni concorso si calcola quanto
   avrebbe incassato in media una colonna di ogni strategia, con le quote
   effettivamente pagate quel giorno (2, 3, 4, 5 e 5+1 punti; il jackpot e'
   escluso: P = 1/622.614.630 rende la sua media incalcolabile da un campione).
   Controllo placebo: stessa procedura con theta permutato a caso.

La probabilita' di fare punti NON cambia tra le strategie (lo verifica
anche questo script); cambia quanto si incassa quando si fanno.

Uso: .venv.nosync/bin/python run_popularity.py [--tickets 20000]
Scrive results/popularity.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import N_BALLS, load_draws, presence_matrix
from src.quote import ATTRIBUTION, P_HITS, P_HITS_5_1, P_HITS_5_ONLY, load_quote
from src import popularity as pop

ROOT = Path(__file__).resolve().parent
FIRST_TEST_YEAR = 2012
VALIDATION_SPLIT = "2019-12-31"


def best_of(model, rng, n, candidates, past, batch=200):
    """Versione vettoriale di popularity.generate (senza vincolo di sovrapposizione)."""
    out = []
    while sum(len(o) for o in out) < n:
        pool = pop.random_tickets(rng, batch * candidates).reshape(batch, candidates, 6)
        order = np.argsort(model.score(pool.reshape(-1, 6)).reshape(batch, candidates), axis=1)
        picked = []
        for row, idx in zip(pool, order):
            for i in idx[:20]:
                if pop.pattern_ok(row[i][None], past)[0]:
                    picked.append(row[i])
                    break
        out.append(np.array(picked))
    return np.concatenate(out)[:n]


PRIZE_COLS = {"2": "payout_2", "3": "payout_3", "4": "payout_4", "5": "payout_5", "5+1": "payout_5_1"}
MEASURED = ("2", "3", "4")   # categorie frequenti: stima precisa


def ticket_payouts(tickets, P, jolly, quote, rows):
    """Incasso medio per colonna, per categoria e concorso: {cat: (draws,)}."""
    hits = P[rows][:, tickets - 1].sum(axis=2)            # (draws, tickets)
    has_jolly = (tickets[None, :, :] == jolly[rows][:, None, None]).any(axis=2)
    q = quote.iloc[rows]
    won = {"2": hits == 2, "3": hits == 3, "4": hits == 4,
           "5": (hits == 5) & ~has_jolly, "5+1": (hits == 5) & has_jolly}
    pay = {c: (w * q[PRIZE_COLS[c]].to_numpy()[:, None]).mean(axis=1) for c, w in won.items()}
    wins = {c: int(w.sum()) for c, w in won.items()}
    return pay, wins, np.bincount(hits.ravel(), minlength=7)


def exact_uniform(quote, rows):
    q = quote.iloc[rows]
    prob = {"2": P_HITS[2], "3": P_HITS[3], "4": P_HITS[4], "5": P_HITS_5_ONLY, "5+1": P_HITS_5_1}
    return {c: prob[c] * q[PRIZE_COLS[c]].to_numpy() for c in PRIZE_COLS}


def block_ci(diff, block=50, n_boot=4000, seed=20261007):
    diff = np.asarray(diff)
    nb = len(diff) // block
    blocks = diff[: nb * block].reshape(nb, block).mean(axis=1)
    rng = np.random.default_rng(seed)
    est = blocks[rng.integers(nb, size=(n_boot, nb))].mean(axis=1)
    return np.quantile(est, [0.025, 0.975]).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickets", type=int, default=20000, help="colonne simulate per anno e strategia")
    ap.add_argument("--candidates", type=int, default=500)
    args = ap.parse_args()
    started = time.monotonic()

    df = load_draws()
    P = presence_matrix(df)
    quote = load_quote(df)
    jolly = df["jolly"].to_numpy()
    price = quote["price"].to_numpy()
    years = df["data"].dt.year.to_numpy()

    # 1) stima e validazione fuori campione (2002-2019 -> 2020-oggi)
    early = pop.fit(df, quote, end=VALIDATION_SPLIT)
    validation = pop.within_r2(early, df, quote, start="2020-01-01")
    final = pop.fit(df, quote)
    print("R^2 fuori campione (dentro l'anno):",
          {k: round(v["r2_within_year"], 3) for k, v in validation.items()}, flush=True)

    # 2) backtest walk-forward dei premi
    strategies = {
        "caso": None,
        "numa (meno giocata tra 20)": 20,
        f"numa (meno giocata tra {args.candidates})": args.candidates,
        "placebo (popolarita' permutata)": args.candidates,
    }
    per_draw = {name: {c: [] for c in PRIZE_COLS} for name in strategies}
    wins = {name: {c: 0 for c in PRIZE_COLS} for name in strategies}
    hit_counts = {name: np.zeros(7, dtype=int) for name in strategies}
    test_rows = []
    rng = np.random.default_rng(20261007)
    for year in range(FIRST_TEST_YEAR, years.max() + 1):
        rows = np.flatnonzero((years == year) & quote["payout_3"].notna().to_numpy())
        if not len(rows):
            continue
        model = pop.fit(df, quote, end=f"{year - 1}-12-31")
        past = pop.past_combinations(df.iloc[: rows[0]])
        placebo = pop.PopularityModel(rng.permutation(model.theta), model.phi)
        for name, k in strategies.items():
            if k is None:
                tickets = pop.random_tickets(rng, args.tickets)
            else:
                m = placebo if name.startswith("placebo") else model
                tickets = best_of(m, rng, args.tickets, k, past)
            pay, won, counts = ticket_payouts(tickets, P, jolly, quote, rows)
            for c in PRIZE_COLS:
                per_draw[name][c].append(pay[c])
                wins[name][c] += won[c]
            hit_counts[name] += counts
        test_rows.append(rows)
        print(f"{year}: {len(rows)} concorsi | resa 2-4 punti: " + " | ".join(
            f"{n.split(' (')[0]} {sum(v[c][-1].sum() for c in MEASURED) / price[rows].sum():.3f}"
            for n, v in per_draw.items()), flush=True)

    rows = np.concatenate(test_rows)
    spent = price[rows].sum()
    exact = exact_uniform(quote, rows)
    exact_234 = sum(exact[c] for c in MEASURED)
    summary = {}
    for name in strategies:
        pay = {c: np.concatenate(per_draw[name][c]) for c in PRIZE_COLS}
        pay_234 = sum(pay[c] for c in MEASURED)
        n_tickets = hit_counts[name].sum() / len(rows)
        summary[name] = {
            "return_per_euro_2_3_4": float(pay_234.sum() / spent),
            "vs_uniform_2_3_4": float(pay_234.sum() / exact_234.sum() - 1),
            "difference_per_euro_ci95": block_ci((pay_234 - exact_234) / price[rows]),
            "vs_uniform_by_category": {c: float(pay[c].sum() / exact[c].sum() - 1) for c in PRIZE_COLS},
            "mean_prize_when_won": {c: float(pay[c].sum() * n_tickets / wins[name][c]) if wins[name][c] else None
                                    for c in PRIZE_COLS},
            "wins": wins[name],
            "hits_per_ticket": float((hit_counts[name] * np.arange(7)).sum() / hit_counts[name].sum()),
            "hit_counts": hit_counts[name].tolist(),
        }
        r = summary[name]
        print(f"{name:36s} resa 2-4 punti {r['return_per_euro_2_3_4']:.4f} EUR/EUR  "
              f"vs caso {100 * r['vs_uniform_2_3_4']:+.1f}% IC {np.round(r['difference_per_euro_ci95'], 4)}  "
              f"hit/colonna {r['hits_per_ticket']:.4f}  per categoria "
              + " ".join(f"{c}:{100 * v:+.0f}%" for c, v in r["vs_uniform_by_category"].items()), flush=True)

    # quanto e' giocata la sestina tipica di Numa rispetto a una qualsiasi
    sample = pop.random_tickets(rng, 200000)
    base = np.log(np.mean(np.exp(final.score(sample))))
    numa_scores = final.score(best_of(final, rng, 2000, args.candidates, pop.past_combinations(df)))
    out = {
        "attribution": ATTRIBUTION,
        "method": "ridge su log(vincitori/attesi) per 2, 3, 4 punti; intercetta per categoria e anno",
        "model": final.to_dict(),
        "validation": {"train_end": VALIDATION_SPLIT, "test_start": "2020-01-01", "categories": validation},
        "most_played": (np.argsort(-final.theta)[:15] + 1).tolist(),
        "least_played": (np.argsort(final.theta)[:15] + 1).tolist(),
        "backtest": {
            "years": [FIRST_TEST_YEAR, int(years.max())], "draws": int(len(rows)),
            "tickets_per_year": args.tickets,
            "uniform_exact_return_per_euro_2_3_4": float(exact_234.sum() / spent),
            "uniform_exact_return_per_euro_2_to_5_1": float(sum(e.sum() for e in exact.values()) / spent),
            "measured_categories": list(MEASURED),
            "notes": "Metrica principale: premi da 2, 3 e 4 punti (frequenti, stima precisa). 5 e 5+1 riportati per completezza: troppo rari per una stima affidabile. Jackpot escluso. Le quote reali includono i contributi aggiuntivi su 3 e 4 punti; importi lordi.",
            "strategies": summary,
        },
        "numa_relative_popularity": {
            "median": float(np.exp(np.median(numa_scores) - base)),
            "uniform_median": float(np.exp(np.median(final.score(sample)) - base)),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    path = ROOT / "results" / "popularity.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Salvato {path} [{out['elapsed_seconds']:.0f}s]")


if __name__ == "__main__":
    main()
