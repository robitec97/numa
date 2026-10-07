"""Genera le figure del report in figures/ (PNG, tema chiaro).

Stile: superficie #fcfcfb, inchiostro #0b0b0b, griglia recessiva #e1e0d9,
serie in blu #2a78d6 (il colore codifica magnitudine, non identita');
riferimenti teorici tratteggiati in grigio inchiostro.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.data import N_BALLS, N_DRAWN

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"

FIG = Path("figures")
FIG.mkdir(exist_ok=True)
RES = Path("results")

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.axisbelow": True,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 170,
})


def style_ax(ax):
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)


def fig_frequenze(stats_res):
    u = stats_res["uniformity"]
    counts = np.array(u["counts"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    style_ax(ax)
    ax.bar(np.arange(1, 91), counts, width=0.72, color=BLUE, zorder=3)
    lo, hi = u["count_band_95"]
    ax.axhspan(lo, hi, color=GRID, alpha=0.55, zorder=1)
    ax.axhline(u["expected"], color=INK2, ls="--", lw=1.2, zorder=4)
    box = dict(facecolor=SURFACE, edgecolor="none", pad=1.5)
    ax.text(91.3, u["expected"], f"attesa: {u['expected']:.0f}", color=INK2,
            va="center", fontsize=10, bbox=box, zorder=5)
    ax.text(91.3, hi, "banda 95%", color=MUTED, va="top", fontsize=9,
            bbox=box, zorder=5)
    ax.set_xlim(0, 99)
    ax.set_ylim(0, counts.max() * 1.12)
    ax.set_xlabel("numero (1–90)")
    ax.set_ylabel("uscite in 4 239 concorsi")
    ax.set_title("Frequenze dei 90 numeri: tutte dentro la variabilità del caso "
                 f"(χ² p = {u['p_mc']:.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_frequenze.png", bbox_inches="tight")
    plt.close(fig)


def fig_ritardi(stats_res):
    g = stats_res["gaps"]
    rate = np.array(g["cond_hit_rate"])
    n = np.array(g["cond_n"])
    base = g["baseline_rate"]
    gg = np.arange(len(rate))
    keep = n >= 200  # taglia le code con pochi dati
    se = np.sqrt(rate * (1 - rate) / np.maximum(n, 1))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    style_ax(ax)
    ax.errorbar(gg[keep], 100 * rate[keep], yerr=100 * 1.96 * se[keep],
                fmt="o", ms=4.5, color=BLUE, ecolor=BLUE_LIGHT,
                elinewidth=1.4, capsize=0, zorder=3)
    ax.axhline(100 * base, color=INK2, ls="--", lw=1.2, zorder=2)
    ax.text(gg[keep].max() * 0.99, 100 * base + 0.65,
            f"probabilità teorica: {100*base:.2f}%", color=INK2,
            ha="right", fontsize=10,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.set_ylim(0, 14)
    ax.set_xlabel("ritardo attuale del numero (concorsi di assenza ≥ g)")
    ax.set_ylabel("P(esce al prossimo concorso), %")
    ax.set_title("La fallacia del ritardatario: la probabilità non cresce col ritardo")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ritardi.png", bbox_inches="tight")
    plt.close(fig)


def fig_overlap(stats_res):
    o = stats_res["overlap"]
    obs = np.array(o["observed"])
    exp = np.array(o["expected"])
    k = np.arange(len(obs))

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    style_ax(ax)
    bars = ax.bar(k, obs, width=0.6, color=BLUE, zorder=3, label="osservato")
    pts = ax.scatter(k, exp, marker="D", s=42, color=INK, zorder=4,
                     label="atteso (ipergeometrica)")
    for ki, ov in enumerate(obs):
        if ov > 0:
            ax.text(ki, ov * 1.35, f"{ov:.0f}", ha="center", color=INK2, fontsize=9)
    ax.legend(handles=[bars, pts], frameon=False, loc="upper right")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(0, max(obs.max(), exp.max()) * 3)
    ax.set_xlabel("numeri in comune tra due estrazioni consecutive")
    ax.set_ylabel("conteggio (scala log)")
    ax.set_title(f"Nessuna memoria tra estrazioni consecutive (χ² p = {o['p']:.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_overlap.png", bbox_inches="tight")
    plt.close(fig)


def merge_rnn(bt, repl):
    """Integra i risultati LSTM (results/rnn.json) nelle strutture esistenti."""
    rnn_path = RES / "rnn.json"
    if not rnn_path.exists():
        return
    rnn = json.loads(rnn_path.read_text())
    name = "LSTM (rete ricorrente)"
    for wkey, w in rnn["windows"].items():
        repl["windows"][wkey]["hits"][name] = w["hits"]
    c = rnn["windows"]["C (originale)"]
    bt["models"][name] = {
        "total_hits": c["hits"],
        "hits_per_draw": c["hits"] / bt["eval_draws"],
        "p_beats_chance": c["p_beats_chance"],
        "logloss": c["logloss"],
        "brier": c["brier"],
        "balance_eur": c["balance_eur"],
        "hit_counts": c["hit_counts"],
    }


def fig_backtest(bt, repl):
    order = list(bt["models"].keys())
    windows = repl["windows"]
    labels = {}
    for key, w in windows.items():
        a, b = w["period"].split(" -> ")
        labels[key] = f"{key[0]} · {a[:4]}–{b[:4]}"
    # intervallo del caso al 95%
    sd = bt["mc_std"]
    exp = bt["expected_hits_chance"]
    lo, hi = exp - 1.96 * sd, exp + 1.96 * sd

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True, sharex=True)
    for ax, (wkey, wlab) in zip(axes, labels.items()):
        style_ax(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        hits = windows[wkey]["hits"]
        y = np.arange(len(order))[::-1]
        ax.axvspan(lo, hi, color=GRID, alpha=0.55, zorder=1)
        ax.axvline(exp, color=INK2, ls="--", lw=1.2, zorder=2)
        ax.scatter([hits[m] for m in order], y, s=64, color=BLUE, zorder=3)
        for yi, m in zip(y, order):
            ax.text(hits[m], yi + 0.30, f"{hits[m]}", ha="center",
                    color=INK2, fontsize=9)
        ax.set_ylim(-0.6, len(order) - 1 + 0.85)
        if ax is axes[0]:
            ax.set_yticks(y)
            ax.set_yticklabels(order)
        ax.set_title(wlab, fontsize=11, fontweight="bold", color=INK2)
        ax.set_xlabel("hit totali su 600 concorsi")
    axes[0].set_xlim(exp - 4 * sd, exp + 4 * sd)
    fig.suptitle("Backtest su 3 finestre disgiunte da 600 concorsi — banda grigia: 95% del puro caso "
                 f"(attesi {exp:.0f})", fontweight="bold", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_backtest.png", bbox_inches="tight")
    plt.close(fig)


def fig_bilancio(bt):
    models = bt["models"]
    order = sorted(models, key=lambda m: models[m]["balance_eur"])
    vals = [models[m]["balance_eur"] for m in order]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    style_ax(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.barh(y, vals, height=0.62, color=BLUE, zorder=3)
    ax.axvline(0, color=BASELINE, lw=1.2)
    for yi, v in zip(y, vals):
        ax.text(v - 6, yi, f"{v:.0f} €", va="center", ha="right",
                color=INK2, fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlim(min(vals) * 1.25, 60)
    ax.set_xlabel("bilancio dopo 600 concorsi (1 colonna da 1 € a concorso, vincite medie storiche)")
    ax.set_title("Tutte le strategie perdono: costo 600 €, si recupera solo una frazione")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_bilancio.png", bbox_inches="tight")
    plt.close(fig)


def main():
    stats_res = json.loads((RES / "stats.json").read_text())
    bt = json.loads((RES / "backtest.json").read_text())
    repl = json.loads((RES / "replication.json").read_text())
    merge_rnn(bt, repl)
    fig_frequenze(stats_res)
    fig_ritardi(stats_res)
    fig_overlap(stats_res)
    fig_backtest(bt, repl)
    fig_bilancio(bt)
    print("Figure salvate in figures/:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
