"""numa — il SuperEnalotto, onestamente.

Uso (dalla cartella del progetto, con l'ambiente in .venv.nosync):
  python numa.py                    aggiorna dati e sito, poi lo apre su http://localhost:8765
  python numa.py build              aggiorna dati, registro e sito, senza server (automazione)
  python numa.py serve              avvia solo il server locale
  python numa.py genera [N]         stampa N sestine poco giocate
  python numa.py controlla 1 2 3 4 5 6
                                    storico di una sestina: punti fatti e premi dal 1997

Opzioni:
  --no-update    non scarica nulla (usa i dati locali)
  --no-ml        salta i modelli ML (piu' veloce; la classifica li omette)
  --no-register  non scrive nel registro: mostra la previsione come anteprima
                 (utile in locale quando il registro ufficiale e' tenuto online)
  --port N       porta del server locale (predefinita 8765)
"""

import argparse
import functools
import http.server
import socketserver
import sys
import threading
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"


def build(update=True, with_ml=True, register=True, now=None) -> dict:
    import update_data
    from src import ledger, popularity, site, strategies
    from src.data import load_draws
    from src.quote import load_quote, update_quote
    from src.schedule import last_scheduled_draw, next_draw_date

    now = now or datetime.now(timezone.utc)
    notes = []
    if update:
        for label, fn in (("estrazioni", update_data.update), ("quote", update_quote)):
            try:
                n = fn()
                print(f"  {label}: +{n} nuove" if n else f"  {label}: gia' aggiornate", flush=True)
            except Exception as e:  # la rete puo' mancare: si continua con i dati locali
                print(f"  [!] {label} non aggiornate ({e}): uso i dati locali", flush=True)
                notes.append(f"{label} non aggiornate: {e}")

    draws = load_draws()
    try:
        quote = load_quote(draws)
        model = popularity.fit(draws, quote)
    except (OSError, ValueError) as e:
        print(f"  [!] quote non disponibili ({e}): uso il modello di popolarita' salvato", flush=True)
        quote, model = None, _saved_model()
        notes.append(f"quote non disponibili: {e}")

    entries = ledger.load()
    scored = ledger.score(entries, draws, quote)
    if scored:
        print(f"  registro: {scored} previsioni verificate", flush=True)

    slot = site.draw_slot(next_draw_date(now))
    preview = None
    # Se manca l'ultima estrazione (la fonte non e' ancora aggiornata) si aspetta
    # a registrare, finche' mancano almeno 3 ore alla chiusura delle giocate.
    waiting = (draws.data.iloc[-1].date() < last_scheduled_draw(now)
               and now < datetime.fromisoformat(slot["sales_close"]) - timedelta(hours=3))
    if waiting:
        print("  in attesa dei dati dell'ultima estrazione: registro piu' tardi", flush=True)
        notes.append("in attesa dei dati dell'ultima estrazione")
    if not waiting and not ledger.has_prediction(entries, slot["date"]):
        target = datetime.fromisoformat(slot["date"]).date()
        print(f"  calcolo le sestine per {slot['label']}...", flush=True)
        pred = strategies.predict(draws, model, target, with_ml=with_ml)
        entry = {
            "target_date": slot["date"], "sales_close": slot["sales_close"],
            "history_end": str(draws.data.iloc[-1].date()), "history_draws": len(draws),
            "data_sha256": site.data_digest(draws), "code_version": site.code_version(),
            "picks": pred["picks"], "info": pred["info"],
            "superstar_hint": strategies.superstar_hint(model, pred["picks"]["numa"], target),
        }
        if register:
            ledger.append(entries, entry, now)
            print("  previsione registrata nel registro pubblico", flush=True)
        else:
            preview = entry
            notes.append("anteprima locale: previsione non registrata")
    ledger.save(entries)
    return site.write(draws, quote, model, entries, slot, notes, now, preview=preview)


def _saved_model():
    import json

    from src.popularity import PopularityModel
    path = SITE / "data" / "latest.json"
    if not path.exists():
        raise SystemExit("Servono i dati delle quote: esegui senza --no-update almeno una volta.")
    return PopularityModel.from_dict(json.loads(path.read_text())["popularity_model"])


def serve(port=8765, open_browser=True):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}/"
        print(f"\nnuma e' su {url}  (Ctrl+C per chiudere)", flush=True)
        if open_browser:
            threading.Timer(0.6, webbrowser.open, args=(url,)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nA presto!")


def print_picks(latest):
    p = latest["prediction"]
    if not p:
        return
    print(f"\nProssimo concorso: {latest['next_draw']['label']}")
    for name, nums in p["picks"].items():
        print(f"  {latest['labels'].get(name, name):32s} " + "  ".join(f"{n:2d}" for n in nums))
    print(f"  SuperStar suggerito: {p.get('superstar_hint')}")
    print("\nOgni sestina ha la stessa probabilita': 1 su 622.614.630. Quella di Numa,")
    print("se vince, si divide il premio con meno persone. Gioca responsabilmente.")


def cmd_generate(n):
    from src import popularity
    from src.data import NUM_COLS, load_draws
    draws = load_draws()
    model = _saved_model()
    rng = np.random.default_rng()
    tickets = popularity.generate(model, rng, n=n, past=popularity.past_combinations(draws),
                                  last=draws.iloc[-1][NUM_COLS].to_numpy())
    from src.site import baseline_median
    base = baseline_median(model)
    for t in tickets:
        rel = np.exp(model.score(t[None])[0] - base)
        print("  ".join(f"{v:2d}" for v in t), f"   giocata {rel:.2f}x rispetto a una sestina tipica")


def cmd_check(numbers):
    from src.data import NUM_COLS, load_draws
    from src.quote import load_quote
    nums = sorted(set(numbers))
    if len(nums) != 6 or not all(1 <= n <= 90 for n in nums):
        raise SystemExit("Servono 6 numeri diversi tra 1 e 90.")
    draws = load_draws()
    quote = load_quote(draws)
    hits = draws[NUM_COLS].isin(nums).sum(axis=1).to_numpy()
    jolly = draws["jolly"].isin(nums).to_numpy()
    counts = np.bincount(hits, minlength=7)
    won = 0.0
    for k, col in ((2, "payout_2"), (3, "payout_3"), (4, "payout_4"), (6, "payout_6")):
        won += np.nansum(quote[col].to_numpy()[hits == k])
    won += np.nansum(quote["payout_5"].to_numpy()[(hits == 5) & ~jolly])
    won += np.nansum(quote["payout_5_1"].to_numpy()[(hits == 5) & jolly])
    spent = np.where(draws["data"] < "2016-02-02", 0.5, 1.0).sum()
    print(f"Sestina {nums} su {len(draws)} concorsi dal {draws.data.iloc[0].date()}:")
    for k in range(2, 7):
        print(f"  {k} punti: {counts[k]} volte")
    best = int(hits.max())
    when = draws.data[hits == best].dt.date.tolist()[-3:]
    print(f"  miglior risultato: {best} punti (ultime volte: {when})")
    print(f"  giocandola sempre: spesi {spent:,.0f} EUR, vinti {won:,.0f} EUR (lordi)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run",
                    choices=["run", "build", "serve", "genera", "controlla"])
    ap.add_argument("args", nargs="*", type=int)
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()

    if a.command in ("run", "build"):
        print("numa: aggiorno dati e sito...", flush=True)
        latest = build(update=not a.no_update, with_ml=not a.no_ml, register=not a.no_register)
        print_picks(latest)
        if a.command == "run":
            serve(a.port, open_browser=not a.no_browser)
    elif a.command == "serve":
        serve(a.port, open_browser=not a.no_browser)
    elif a.command == "genera":
        cmd_generate(a.args[0] if a.args else 5)
    elif a.command == "controlla":
        cmd_check(a.args)


if __name__ == "__main__":
    sys.exit(main())
