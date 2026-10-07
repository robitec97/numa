"""Modello di popolarita': quanto gli altri giocatori scelgono certi numeri.

Ogni sestina ha la stessa probabilita' di uscire (1 su 622.614.630), ma i
premi di ogni categoria sono un fondo fisso diviso tra tutti i vincitori.
Se i numeri estratti sono molto giocati i vincitori sono tanti e la quota
scende. Scegliere numeri poco giocati non aumenta la probabilita' di vincere:
aumenta quanto si incassa quando si vince.

Modello (famiglia esponenziale sulle colonne giocate):
    P(colonna S) ~ exp( sum_{n in S} theta_n + sum_{coppie in S} phi . f(coppia) )
Linearizzando attorno alla scelta uniforme, il logaritmo dei vincitori con
esattamente k punti, rispetto all'atteso, vale
    log pop_k(D) ~ a_k + C1[k] * sum_{n in D} theta_n + C2[k] * sum_{coppie in D} phi . f
con C1[k] = k/6 - (6-k)/84 e C2[k] ~ C(k,2)/C(6,2): le costanti seguono
dal modello, non sono stimate. Per il 6 (jackpot) C1 = C2 = 1.
I parametri si stimano con minimi quadrati penalizzati (ridge) su
log pop_2, pop_3, pop_4, con un'intercetta per categoria e anno che assorbe
variazioni di livello (prezzo, regole, errori del montepremi stimato).
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .data import N_BALLS, N_DRAWN, NUM_COLS

C1 = {k: k / 6 - (6 - k) / 84 for k in range(2, 7)}
C2 = {k: (k * (k - 1) / 30) for k in range(2, 7)}
PAIR_FEATURES = ("consecutivi", "stessa_decina", "stessa_cadenza")
FIT_FROM = "2002-01-01"   # prima del 2002: importi in lire, prezzo incerto
CATEGORIES = (2, 3, 4)

_PAIRS = np.array(list(combinations(range(N_DRAWN), 2)))


def pair_counts(tickets: np.ndarray) -> np.ndarray:
    """(n, 6) numeri 1..90 -> (n, 3) conteggi di coppie: consecutive,
    stessa decina (1-10, 11-20, ...), stessa cadenza (stessa cifra finale)."""
    t = np.sort(np.asarray(tickets), axis=1)
    a, b = t[:, _PAIRS[:, 0]], t[:, _PAIRS[:, 1]]
    return np.stack([
        (b - a == 1).sum(axis=1),
        ((a - 1) // 10 == (b - 1) // 10).sum(axis=1),
        (a % 10 == b % 10).sum(axis=1),
    ], axis=1).astype(float)


@dataclass
class PopularityModel:
    theta: np.ndarray                 # (90,) popolarita' relativa di ogni numero, media zero
    phi: np.ndarray                   # (3,) effetto delle coppie (PAIR_FEATURES)
    fit_end: str = ""                 # ultima estrazione usata
    n_rows: int = 0
    meta: dict = field(default_factory=dict)

    def score(self, tickets: np.ndarray) -> np.ndarray:
        """log-popolarita' della sestina rispetto a una giocata media (0 = media).

        Per il 6: exp(score) = quante volte la sestina e' giocata rispetto
        a una sestina tipica.
        """
        t = np.atleast_2d(np.asarray(tickets))
        return self.theta[t - 1].sum(axis=1) + pair_counts(t) @ self.phi

    def to_dict(self) -> dict:
        return {"theta": [round(float(x), 5) for x in self.theta],
                "phi": dict(zip(PAIR_FEATURES, (round(float(x), 5) for x in self.phi))),
                "fit_end": self.fit_end, "n_rows": self.n_rows, **self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "PopularityModel":
        return cls(np.array(d["theta"], dtype=float),
                   np.array([d["phi"][k] for k in PAIR_FEATURES], dtype=float),
                   d.get("fit_end", ""), d.get("n_rows", 0))


def training_rows(draws: pd.DataFrame, quote: pd.DataFrame, end=None, start=FIT_FROM):
    """Righe (concorso, categoria, log pop_k) utilizzabili per la stima."""
    dates = draws["data"]
    keep = (dates >= pd.Timestamp(start))
    if end is not None:
        keep &= dates <= pd.Timestamp(end)
    rows = []
    for k in CATEGORIES:
        pop = quote[f"pop_{k}"].to_numpy(dtype=float)
        ok = keep.to_numpy() & np.isfinite(pop) & (pop > 0)
        if k == 2:
            ok &= ~quote["quota2_floor"].fillna(False).to_numpy(dtype=bool)
        idx = np.flatnonzero(ok)
        rows.append(pd.DataFrame({"t": idx, "k": k, "y": np.log(pop[idx])}))
    return pd.concat(rows, ignore_index=True)


def design(draws: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    nums = draws[NUM_COLS].to_numpy()[rows["t"].to_numpy()]
    onehot = np.zeros((len(rows), N_BALLS))
    np.put_along_axis(onehot, nums - 1, 1.0, axis=1)
    c1 = rows["k"].map(C1).to_numpy()[:, None]
    c2 = rows["k"].map(C2).to_numpy()[:, None]
    return np.hstack([onehot * c1, pair_counts(nums) * c2])


def _demean(X, y, groups):
    Xc, yc = X.copy(), y.copy()
    for g in np.unique(groups):
        m = groups == g
        Xc[m] -= Xc[m].mean(axis=0)
        yc[m] -= yc[m].mean()
    return Xc, yc


def fit(draws: pd.DataFrame, quote: pd.DataFrame, end=None, ridge=1.0) -> PopularityModel:
    """Stima theta e phi usando solo i concorsi fino a `end` (incluso)."""
    rows = training_rows(draws, quote, end=end)
    if len(rows) < 500:
        raise ValueError("troppi pochi concorsi con quote per stimare la popolarita'")
    X = design(draws, rows)
    years = draws["data"].dt.year.to_numpy()[rows["t"].to_numpy()]
    groups = rows["k"].astype(str).to_numpy() + "_" + years.astype(str)
    Xc, yc = _demean(X, rows["y"].to_numpy(), groups)
    beta = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(X.shape[1]), Xc.T @ yc)
    theta = beta[:N_BALLS] - beta[:N_BALLS].mean()
    last = draws["data"].iloc[rows["t"].max()]
    return PopularityModel(theta, beta[N_BALLS:], str(last.date()), len(rows),
                           {"ridge": ridge, "categories": list(CATEGORIES), "fit_from": FIT_FROM})


def within_r2(model: PopularityModel, draws, quote, start, end=None) -> dict:
    """R^2 dentro ogni anno su un periodo non usato per la stima, per categoria."""
    rows = training_rows(draws, quote, end=end, start=start)
    X = design(draws, rows)
    pred = X @ np.r_[model.theta, model.phi]
    years = draws["data"].dt.year.to_numpy()[rows["t"].to_numpy()]
    out = {}
    for k in CATEGORIES:
        m = rows["k"].to_numpy() == k
        if not m.any():
            continue
        res = tot = 0.0
        for yv in np.unique(years[m]):
            g = m & (years == yv)
            yy = rows["y"].to_numpy()[g] - rows["y"].to_numpy()[g].mean()
            pp = pred[g] - pred[g].mean()
            res += ((yy - pp) ** 2).sum()
            tot += (yy ** 2).sum()
        out[str(k)] = {"r2_within_year": 1 - res / tot, "draws": int(m.sum())}
    return out


# --------------------------------------------------------------- generatore

def past_combinations(draws: pd.DataFrame) -> set:
    """Sestine gia' uscite, per escluderle (molti rigiocano le vincenti)."""
    return {tuple(r) for r in np.sort(draws[NUM_COLS].to_numpy(), axis=1).tolist()}


def pattern_ok(tickets: np.ndarray, past: set | None = None, last=None) -> np.ndarray:
    """Scarta le sestine "a disegno", che molti giocano ma i dati sulle quote
    non possono misurare (non escono quasi mai come combinazione vincente):
      - tre o piu' numeri consecutivi
      - progressioni aritmetiche di 4 o piu' numeri (es. 5 10 15 20)
      - quattro o piu' numeri nella stessa decina
      - sei numeri tutti <= 31 (sestine "di compleanni")
      - tre o piu' numeri dell'ultima estrazione; una sestina gia' uscita
    """
    t = np.sort(np.atleast_2d(tickets), axis=1)
    d = np.diff(t, axis=1)
    run3 = ((d[:, :-1] == 1) & (d[:, 1:] == 1)).any(axis=1)
    decade = (t - 1) // 10
    same_decade = (decade[:, :, None] == decade[:, None, :]).sum(axis=2).max(axis=1) >= 4
    birthdays = (t <= 31).all(axis=1)
    ok = ~(run3 | same_decade | birthdays | _has_progression(t, 4))
    if last is not None:
        ok &= np.isin(t, np.asarray(last)).sum(axis=1) < 3
    if past:
        ok &= np.array([tuple(r) not in past for r in t.tolist()])
    return ok


def _has_progression(t: np.ndarray, length: int) -> np.ndarray:
    """True se la sestina contiene `length` numeri in progressione aritmetica."""
    out = np.zeros(len(t), dtype=bool)
    for i, row in enumerate(t.tolist()):
        s = set(row)
        out[i] = any(all(a + j * (b - a) in s for j in range(2, length))
                     for a, b in combinations(row, 2))
    return out


def random_tickets(rng: np.random.Generator, m: int) -> np.ndarray:
    """m sestine uniformi (ordinate), tutte le combinazioni equiprobabili."""
    keys = rng.random((m, N_BALLS))
    return np.sort(np.argpartition(keys, N_DRAWN, axis=1)[:, :N_DRAWN] + 1, axis=1)


def generate(model: PopularityModel, rng: np.random.Generator, n: int = 1,
             candidates: int = 500, past: set | None = None, last=None,
             max_overlap: int = 2) -> np.ndarray:
    """n sestine: ciascuna e' la meno giocata (secondo il modello) tra
    `candidates` sestine casuali che superano i filtri. Restano casuali, e
    ogni sestina possibile ha la stessa probabilita' di vincere, ma stanno
    nella coda delle combinazioni meno giocate. Due sestine diverse
    condividono al piu' `max_overlap` numeri.
    """
    chosen: list[np.ndarray] = []
    while len(chosen) < n:
        pool = random_tickets(rng, candidates)
        for i in np.argsort(model.score(pool), kind="stable"):
            t = pool[i]
            if chosen and max(np.intersect1d(t, c).size for c in chosen) > max_overlap:
                continue
            if pattern_ok(t[None], past, last)[0]:
                chosen.append(t)
                break
    return np.array(chosen)
