"""Quote e numero di vincitori per categoria, concorso per concorso.

Fonte: https://superenalottooggi.com/dati/superenalotto.csv (licenza CC BY 4.0,
"Dati: SuperEnalottoOggi.com"), confrontata con le pagine di superenalotto.net
su 80 concorsi estratti a caso dal 2009 al 2026: vincitori e quote identici.

Il file grezzo resta com'e' (data/quote_superenalottooggi.csv); correzioni e
grandezze derivate sono calcolate a runtime, come per le estrazioni.

Grandezze derivate per ogni concorso:
  montepremi  quota della raccolta destinata ai premi, ricavata dal fondo di
              una categoria (vincitori x quota = percentuale fissa del montepremi)
  combos      colonne giocate = montepremi / (percentuale premi x prezzo colonna)
  pop_k       vincitori con k punti / vincitori attesi se tutti giocassero a caso.
              > 1: i numeri estratti erano piu' giocati della media.
  payout_k    quanto avrebbe incassato una colonna in piu' con k punti
              (fondo diviso per vincitori + 1); 5_1 = cinque piu' Jolly;
              6 = jackpot (in palio, se nessuno l'ha vinto).
"""

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from .data import N_BALLS, NUM_COLS

ROOT = Path(__file__).resolve().parent.parent
QUOTE_PATH = ROOT / "data" / "quote_superenalottooggi.csv"
QUOTE_URL = "https://superenalottooggi.com/dati/superenalotto.csv"
ATTRIBUTION = "Dati: SuperEnalottoOggi.com (CC BY 4.0)"

CATEGORIES = ("6", "5_1", "5", "4", "3", "2")

# (inizio, nome, percentuale della raccolta destinata ai premi, prezzo colonna,
#  ripartizione del montepremi tra le categorie)
REGIMES = (
    # 1997-2008: 5, 4 e 3 punti ricevono quote uguali (ricavato dai dati)
    ("1997-12-03", "A_1997", 0.34648, 0.50, {"5": 0.20, "4": 0.20, "3": 0.20}),
    # DD 2009/21729, gia' applicata dal 17/06/2008
    ("2008-06-17", "B_2008", 0.34648, 0.50,
     {"6": 0.20, "5_1": 0.20, "5": 0.15, "4": 0.15, "3": 0.30}),
    # DD 109175 del 16/11/2015, art. 4: nasce la categoria 2 punti, colonna a 1 euro
    ("2016-02-02", "C_2016", 0.60, 1.00,
     {"6": 0.174, "5_1": 0.13, "5": 0.042, "4": 0.042, "3": 0.128, "2": 0.40}),
)

# Errori della fonte trovati con controlli di coerenza interna (le quote
# SuperStar 3 e 4 stelle valgono 100 volte la quota base; i fondi delle
# categorie stanno in rapporti fissi). superenalotto.net riporta gli stessi
# errori: vengono dallo stesso flusso a monte.
FIXES = {
    "2009-02-03": {"vincitori_3": 77222, "quota_3": 14.72, "vincitori_4": 2247, "quota_4": 252.96},
    "2010-11-09": {"quota_4": 420.68},
    "2011-01-04": {"vincitori_5": 21},
    "2011-08-06": {"quota_4": 316.32},
    "2011-08-27": {"quota_3": 19.84},
    "2011-09-15": {"quota_5": 23198.15},
    "2012-06-04": {"vincitori_4": 1523},
}

# Probabilita' che una colonna casuale faccia esattamente k punti.
TOTAL = comb(N_BALLS, 6)
P_HITS = {k: comb(6, k) * comb(N_BALLS - 6, 6 - k) / TOTAL for k in range(7)}
P_HITS_5_1 = 6 / TOTAL          # 5 punti e il sesto numero e' il Jolly
P_HITS_5_ONLY = 6 * 83 / TOTAL  # 5 punti senza Jolly


def regime(date) -> tuple:
    current = REGIMES[0]
    for r in REGIMES:
        if str(date)[:10] >= r[0]:
            current = r
    return current


def read_raw(path: Path = QUOTE_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["data"] = pd.to_datetime(raw["data"])
    for date, values in FIXES.items():
        mask = raw["data"] == pd.Timestamp(date)
        for col, value in values.items():
            raw.loc[mask, col] = value
    return raw.sort_values("data").reset_index(drop=True)


def _montepremi(name, shares, fund, quota2) -> tuple[float, str]:
    """Montepremi del concorso (senza riporti) ricavato dai fondi di categoria."""
    if name == "C_2016" and quota2 > 5.0 and fund["2"] > 0:
        # 2 punti: milioni di vincitori, quota quasi mai arrotondata in modo rilevante;
        # se nessuno fa 5, il fondo del 5 va diviso tra 4, 3 e 2.
        extra = shares["5"] / 3 if fund["5"] == 0 else 0.0
        return fund["2"] / (shares["2"] + extra), "punti2"
    if fund["5"] > 0:
        return fund["5"] / shares["5"], "punti5"
    if fund["4"] > 0:
        others = 3 if name == "C_2016" else 2
        return fund["4"] / (shares["4"] + shares["5"] / others), "punti4"
    return np.nan, "assente"


def derive(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge montepremi, colonne giocate, popolarita' e premi controfattuali."""
    out = []
    for row in raw.itertuples(index=False):
        r = row._asdict()
        name, s, price, shares = regime(r["data"].date())[1:]
        w = {c: _num(r.get(f"vincitori_{c}")) for c in CATEGORIES}
        q = {c: _num(r.get(f"quota_{c}")) for c in CATEGORIES}
        fund = {c: w[c] * q[c] for c in CATEGORIES}
        M, source = _montepremi(name, shares, fund, q["2"])
        N = M / (s * price)
        d = {"data": r["data"], "regime": name, "price": price, "montepremi": M,
             "montepremi_from": source, "combos": N,
             "quota2_floor": name == "C_2016" and q["2"] <= 5.0}
        for k in ("2", "3", "4", "5"):
            available = k in shares and np.isfinite(N) and N > 0
            # Se il montepremi e' ricavato dal 4, pop_4 sarebbe circolare.
            if k == "4" and source == "punti4":
                available = False
            d[f"pop_{k}"] = w[k] / (N * P_HITS[int(k)]) if available else np.nan
        jackpot = _num(r.get("jackpot_in_palio"))
        if w["6"] > 0:
            d["payout_6"] = fund["6"] / (w["6"] + 1)
        else:
            d["payout_6"] = jackpot if jackpot > 0 else np.nan
        for c in CATEGORIES:
            if c == "6":
                continue
            if c not in shares:
                # 2 punti prima del 2016: categoria inesistente. 5+1 nel regime A:
                # quota nota solo se qualcuno l'ha vinta.
                d[f"payout_{c}"] = 0.0 if c == "2" else (fund[c] / (w[c] + 1) if w[c] > 0 else np.nan)
            elif w[c] > 0:
                d[f"payout_{c}"] = fund[c] / (w[c] + 1)
            else:
                d[f"payout_{c}"] = shares[c] * M if np.isfinite(M) else 0.0
        if d["quota2_floor"]:
            d["payout_2"] = q["2"]  # integrata dal fondo di riserva: non dipende dalla popolarita'
        out.append(d)
    return pd.DataFrame(out)


def _num(x) -> float:
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def load_quote(draws: pd.DataFrame, path: Path = QUOTE_PATH) -> pd.DataFrame:
    """Quote allineate riga per riga alle estrazioni di `draws` (NaN se assenti).

    Rifiuta la fonte se i numeri estratti non coincidono con lo storico
    (a parte l'errore noto del 27/12/2004, presente in entrambe le fonti).
    """
    raw = read_raw(path)
    merged = draws[["data", *NUM_COLS]].merge(raw, on="data", how="left", suffixes=("", "_q"))
    have = merged["n1_q"].notna()
    a = np.sort(merged.loc[have, NUM_COLS].to_numpy(), axis=1)
    b = np.sort(merged.loc[have, [f"{c}_q" for c in NUM_COLS]].to_numpy(dtype=int), axis=1)
    bad = merged.loc[have, "data"][(a != b).any(axis=1)]
    bad = bad[bad != pd.Timestamp("2004-12-27")]
    if len(bad):
        raise ValueError(f"le quote riportano numeri diversi dallo storico: {bad.dt.date.tolist()[:5]}")
    derived = derive(raw)
    return draws[["data"]].merge(derived, on="data", how="left")


def update_quote(path: Path = QUOTE_PATH, timeout: int = 30) -> int:
    """Scarica il file aggiornato; le righe gia' note non devono cambiare."""
    import io

    import requests

    resp = requests.get(QUOTE_URL, timeout=timeout,
                        headers={"User-Agent": "numa (progetto didattico; github.com)"})
    resp.raise_for_status()
    remote = pd.read_csv(io.StringIO(resp.text))
    needed = {"data", "n1", "n6", "vincitori_2", "quota_2", "vincitori_3", "quota_3"}
    if not needed <= set(remote.columns):
        raise ValueError("formato delle quote cambiato")
    if not path.exists():
        path.write_text(resp.text)
        return len(remote)
    local = pd.read_csv(path)
    common = local.merge(remote, on="data", suffixes=("_l", "_r"))
    if len(common) < len(local):
        raise ValueError("la fonte delle quote ha perso concorsi gia' noti")
    # I numeri estratti non cambiano mai; le quote possono essere corrette
    # a monte (le correzioni note in FIXES si riapplicano comunque).
    for col in NUM_COLS:
        if not (common[f"{col}_l"] == common[f"{col}_r"]).all():
            raise ValueError("la fonte delle quote contraddice estrazioni gia' note")
    n_new = len(remote) - len(local)
    if n_new > 0:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(resp.text)
        tmp.replace(path)
    return n_new
