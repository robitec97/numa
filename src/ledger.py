"""Registro pubblico delle previsioni: si scrive prima, si verifica dopo.

Regole:
  - una voce si aggiunge solo prima della chiusura delle giocate del concorso
    a cui si riferisce, e non si modifica mai (dopo l'estrazione si aggiunge
    soltanto il campo "result");
  - ogni voce viene abbinata alla PRIMA estrazione avvenuta dopo il momento
    in cui e' stata registrata, anche se il concorso e' stato spostato;
  - ogni voce contiene l'hash della precedente: cambiare una previsione
    passata spezzerebbe la catena (verify_chain). Su GitHub l'ora del commit
    fatto dall'automazione e' una seconda prova, indipendente, che la
    previsione esisteva prima dell'estrazione.

Il confronto e' quindi prospettico: nessuna scelta puo' essere adattata
agli esiti, a differenza dei backtest sulle stesse finestre gia' esplorate.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import NUM_COLS
from .inference import exact_pvalue
from .schedule import ROME, draw_moment

ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = ROOT / "site" / "data" / "ledger.json"


def _canonical(entry: dict) -> bytes:
    body = {k: v for k, v in entry.items() if k not in ("hash", "result")}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def entry_hash(entry: dict) -> str:
    return hashlib.sha256(_canonical(entry)).hexdigest()


def load(path: Path = LEDGER_PATH) -> list[dict]:
    if not path.exists():
        return []
    entries = json.loads(path.read_text())["entries"]
    verify_chain(entries)
    return entries


def save(entries: list[dict], path: Path = LEDGER_PATH) -> None:
    verify_chain(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"description": "Previsioni registrate prima di ogni concorso e verificate dopo. "
                          "Ogni voce contiene l'hash della precedente.",
           "entries": entries}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    tmp.replace(path)


def verify_chain(entries: list[dict]) -> None:
    prev = None
    for i, e in enumerate(entries):
        if e.get("prev") != prev or e.get("hash") != entry_hash(e):
            raise ValueError(f"registro alterato alla voce {i} ({e.get('target_date')})")
        prev = e["hash"]


def append(entries: list[dict], entry: dict, now: datetime | None = None) -> dict:
    """Aggiunge una previsione, solo se la raccolta del concorso e' ancora aperta."""
    now = now or datetime.now(timezone.utc)
    target = datetime.fromisoformat(entry["sales_close"])
    if now >= target:
        raise ValueError("la raccolta del concorso e' chiusa: troppo tardi per registrare")
    if any(e["target_date"] == entry["target_date"] for e in entries):
        raise ValueError("esiste gia' una previsione per questo concorso")
    entry = {**entry, "created_utc": now.isoformat(timespec="seconds"),
             "prev": entries[-1]["hash"] if entries else None}
    entry["hash"] = entry_hash(entry)
    entries.append(entry)
    return entry


def has_prediction(entries: list[dict], target_date) -> bool:
    return any(e["target_date"] == str(target_date) for e in entries)


def score(entries: list[dict], draws: pd.DataFrame, quote: pd.DataFrame | None = None) -> int:
    """Completa con l'esito le voci il cui concorso e' stato estratto."""
    moments = np.array([draw_moment(d.date()).astimezone(timezone.utc) for d in draws["data"]])
    n_scored = 0
    for e in entries:
        if e.get("result"):
            continue
        created = datetime.fromisoformat(e["created_utc"])
        later = np.flatnonzero(moments > created)
        if not len(later):
            continue
        t = int(later[0])
        row = draws.iloc[t]
        drawn = sorted(int(row[c]) for c in NUM_COLS)
        jolly = int(row["jolly"])
        result = {"date": str(row["data"].date()), "numbers": drawn, "jolly": jolly,
                  "superstar": None if pd.isna(row["superstar"]) else int(row["superstar"]),
                  "hits": {}, "prize": {}}
        q = quote.iloc[t] if quote is not None and t < len(quote) else None
        for name, pick in e["picks"].items():
            hits = len(set(pick) & set(drawn))
            result["hits"][name] = hits
            result["prize"][name] = prize(hits, jolly in pick, q)
        e["result"] = result
        n_scored += 1
    return n_scored


def prize(hits: int, with_jolly: bool, q) -> float | None:
    """Vincita lorda ipotetica di una colonna (None se le quote non sono note)."""
    if q is None or pd.isna(q.get("payout_3")):
        return None
    if hits == 5:
        return float(q["payout_5_1"] if with_jolly else q["payout_5"])
    col = {2: "payout_2", 3: "payout_3", 4: "payout_4"}.get(hits)
    if hits == 6:
        return None  # jackpot: si saprebbe solo dalla quota reale del concorso
    return float(q[col]) if col else 0.0


def leaderboard(entries: list[dict]) -> dict:
    """Totali per strategia sulle sole voci verificate, con p-value esatto."""
    scored = [e for e in entries if e.get("result")]
    names = list(dict.fromkeys(n for e in scored for n in e["picks"]))
    board = {}
    for name in names:
        rows = [e for e in scored if name in e["picks"]]
        hits = [e["result"]["hits"][name] for e in rows]
        prizes = [e["result"]["prize"].get(name) for e in rows]
        n, total = len(hits), int(sum(hits))
        cumulative = np.cumsum(np.array(hits) - 0.4).round(3).tolist()
        board[name] = {
            "draws": n, "hits": total, "expected": round(0.4 * n, 2),
            "p_at_least": exact_pvalue(total, n) if n else None,
            "hit_counts": np.bincount(hits, minlength=7).tolist(),
            "prizes_eur": round(sum(p for p in prizes if p), 2),
            "prizes_known": sum(p is not None for p in prizes),
            "cumulative_excess": cumulative,
            "dates": [e["result"]["date"] for e in rows],
        }
    return board


def now_rome() -> datetime:
    return datetime.now(ROME)
