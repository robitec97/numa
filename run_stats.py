"""Test statistici di casualita' sullo storico SuperEnalotto.

1. Uniformita' delle frequenze dei 90 numeri (chi-quadro + Monte Carlo)
2. Sovrapposizione tra estrazioni consecutive vs ipergeometrica
3. Distribuzione dei ritardi vs geometrica; P(uscita | ritardo >= g)
4. Autocorrelazione della somma dei 6 numeri estratti

Salva i risultati in results/stats.json per il report e le figure.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.data import N_BALLS, N_DRAWN, load_draws, presence_matrix

OUT = Path("results")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(2026)


def simulate_history(T):
    """Una storia sintetica di T estrazioni eque: presenze (T, 90)."""
    scores = RNG.random((T, N_BALLS))
    picks = np.argpartition(scores, N_DRAWN, axis=1)[:, :N_DRAWN]
    P = np.zeros((T, N_BALLS), dtype=bool)
    P[np.repeat(np.arange(T), N_DRAWN), picks.ravel()] = True
    return P


def test_uniformity(P, n_sims=2000):
    counts = P.sum(axis=0)
    expected = P.shape[0] * N_DRAWN / N_BALLS
    chi2_obs = ((counts - expected) ** 2 / expected).sum()

    chi2_mc = np.empty(n_sims)
    for i in range(n_sims):
        c = simulate_history(P.shape[0]).sum(axis=0)
        chi2_mc[i] = ((c - expected) ** 2 / expected).sum()
    p_mc = (chi2_mc >= chi2_obs).mean()

    # bande al 95% per la figura delle frequenze
    lo, hi = np.quantile(chi2_mc, [0.025, 0.975])
    return {
        "counts": counts.tolist(),
        "expected": expected,
        "chi2_obs": float(chi2_obs),
        "chi2_mc_mean": float(chi2_mc.mean()),
        "p_mc": float(p_mc),
        "p_asintotico": float(stats.chi2.sf(chi2_obs, N_BALLS - 1)),
        "count_band_95": [
            float(stats.binom.ppf(q, P.shape[0], N_DRAWN / N_BALLS)) for q in (0.025, 0.975)
        ],
    }


def test_overlap(P):
    overlap = (P[1:] & P[:-1]).sum(axis=1)
    obs = np.bincount(overlap, minlength=7)[:7].astype(float)
    pmf = stats.hypergeom.pmf(np.arange(7), N_BALLS, N_DRAWN, N_DRAWN)
    exp = pmf * len(overlap)
    # raggruppa le code (>=3) per il chi-quadro
    obs_g = np.array([obs[0], obs[1], obs[2], obs[3:].sum()])
    exp_g = np.array([exp[0], exp[1], exp[2], exp[3:].sum()])
    chi2, p = stats.chisquare(obs_g, exp_g)
    return {
        "observed": obs.tolist(),
        "expected": exp.tolist(),
        "chi2": float(chi2),
        "p": float(p),
    }


def test_gaps(P):
    T = P.shape[0]
    p_geom = N_DRAWN / N_BALLS

    # tutti i ritardi tra uscite consecutive dello stesso numero
    gaps = []
    for n in range(N_BALLS):
        occ = np.flatnonzero(P[:, n])
        gaps.extend(np.diff(occ).tolist())
    gaps = np.asarray(gaps)

    # P(uscita al prossimo concorso | ritardo corrente >= g)
    last_seen = np.full(N_BALLS, -1)
    cond_total = np.zeros(61)
    cond_hit = np.zeros(61)
    for t in range(T):
        gap_now = t - last_seen
        for g in range(61):
            mask = gap_now >= g
            cond_total[g] += mask.sum()
            cond_hit[g] += (mask & P[t]).sum()
        last_seen[P[t]] = t

    # GOF chi-quadro vs geometrica: bin 1..40, coda >=41 raggruppata
    edges = np.arange(1, 41)
    obs = np.array([(gaps == g).sum() for g in edges] + [(gaps >= 41).sum()], dtype=float)
    pmf = stats.geom.pmf(edges, p_geom)
    exp = np.append(pmf, stats.geom.sf(40, p_geom)) * len(gaps)
    chi2_g, p_g = stats.chisquare(obs, exp / exp.sum() * obs.sum())
    return {
        "gaps_mean": float(gaps.mean()),
        "gaps_expected_mean": 1 / p_geom,
        "gaps_max": int(gaps.max()),
        "gof_chi2": float(chi2_g),
        "gof_p": float(p_g),
        "gap_hist": np.bincount(gaps, minlength=100)[:100].tolist(),
        "cond_hit_rate": (cond_hit / np.maximum(cond_total, 1)).tolist(),
        "cond_n": cond_total.tolist(),
        "baseline_rate": p_geom,
    }


def test_autocorr(P, max_lag=10):
    draws_sum = (P * np.arange(1, N_BALLS + 1)).sum(axis=1).astype(float)
    x = draws_sum - draws_sum.mean()
    T = len(x)
    acf = [float(np.dot(x[:-k], x[k:]) / np.dot(x, x)) for k in range(1, max_lag + 1)]
    band = 1.96 / np.sqrt(T)
    q = T * (T + 2) * sum(a**2 / (T - k) for k, a in enumerate(acf, start=1))
    p = float(stats.chi2.sf(q, max_lag))
    return {"acf": acf, "band": float(band), "ljungbox_q": float(q), "ljungbox_p": p,
            "sum_mean": float(draws_sum.mean()), "sum_expected": 6 * (N_BALLS + 1) / 2}


def main():
    df = load_draws()
    P = presence_matrix(df)
    res = {
        "n_draws": len(df),
        "period": [str(df.data.min().date()), str(df.data.max().date())],
        "uniformity": test_uniformity(P),
        "overlap": test_overlap(P),
        "gaps": test_gaps(P),
        "autocorr": test_autocorr(P),
    }
    (OUT / "stats.json").write_text(json.dumps(res, indent=2))

    u, o, g, a = res["uniformity"], res["overlap"], res["gaps"], res["autocorr"]
    print(f"Estrazioni: {res['n_draws']}  ({res['period'][0]} -> {res['period'][1]})")
    print(f"\n1) Uniformita' frequenze: chi2={u['chi2_obs']:.1f} (atteso~{u['chi2_mc_mean']:.0f}), "
          f"p Monte Carlo={u['p_mc']:.3f}, p asintotico={u['p_asintotico']:.3f}")
    print(f"2) Overlap consecutive:   chi2={o['chi2']:.2f}, p={o['p']:.3f}")
    print(f"3) Ritardi: media={g['gaps_mean']:.2f} (attesa {g['gaps_expected_mean']:.0f}), "
          f"max={g['gaps_max']}, GOF geometrica p={g['gof_p']:.3f}")
    print(f"4) Autocorrelazione somma: Ljung-Box p={a['ljungbox_p']:.3f} "
          f"(media somma={a['sum_mean']:.1f}, attesa {a['sum_expected']:.1f})")


if __name__ == "__main__":
    main()
