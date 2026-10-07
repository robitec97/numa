"""Inferenza sotto l'ipotesi di estrazioni uniformi e indipendenti.

La coda della somma ipergeometrica e' calcolata per convoluzione (a
precisione floating point), senza errore Monte Carlo. Vale per sestine
scelte usando solo il passato, fissando il protocollo prima del test.
"""

from functools import lru_cache

import numpy as np
from scipy.stats import hypergeom

from .data import N_BALLS, N_DRAWN


@lru_cache(maxsize=12)
def total_hits_pmf(n_draws):
    if not isinstance(n_draws, (int, np.integer)) or n_draws < 0:
        raise ValueError("n_draws deve essere un intero non negativo")
    single = hypergeom.pmf(np.arange(N_DRAWN + 1), N_BALLS, N_DRAWN, N_DRAWN)
    pmf = np.array([1.0])
    for _ in range(n_draws):
        pmf = np.convolve(pmf, single)
    pmf /= pmf.sum()
    pmf.setflags(write=False)
    return pmf


def exact_pvalue(total_hits, n_draws):
    pmf = total_hits_pmf(n_draws)
    if not isinstance(total_hits, (int, np.integer)) or not 0 <= total_hits <= N_DRAWN * n_draws:
        raise ValueError("total_hits fuori range")
    return float(pmf[total_hits:].sum())


def holm_adjust(pvalues):
    """Correzione family-wise valida anche per test dipendenti."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p-value non validi")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted
