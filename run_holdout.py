"""Controlla le estrazioni successive al cutoff con il protocollo congelato.

Scarica una copia separata dello storico, verificando le estrazioni note.
Questo controllo su dati gia' pubblicati non e' una preregistrazione esterna.
Se la fonte aggiunge concorsi mancanti prima del cutoff (e' successo: il
08/06/2026 mancava dallo storico di luglio), lo storico congelato in
data/superenalotto_cutoff_2026-07-30.txt resta quello valutato: le aggiunte
vengono solo registrate, mentre una contraddizione blocca il controllo.
Ripeterlo sugli stessi dati non fornisce una nuova conferma indipendente.
Uso: .venv/bin/python run_holdout.py
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import update_data
from run_improvement import ROOT, summarize
from src.backtest import run_backtest
from src.data import DATA_PATH, NUM_COLS, load_draws, presence_matrix
from src.features import build_features
from src.inference import holm_adjust
from src.models import UniformModel, gbm_model
from src.robust import TemporalEnsembleModel


FROZEN_PATH = ROOT / "data/superenalotto_cutoff_2026-07-30.txt"


def _digest(df):
    return hashlib.sha256(df[["data", *NUM_COLS]].to_csv(index=False).encode()).hexdigest()


def main():
    protocol_path = ROOT / "results/improvement.protocol.json"
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} e' cambiato dopo il congelamento del protocollo")
    out = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cutoff": protocol["history_end"],
        "status": "unavailable",
        "note": "Controllo su estrazioni pubblicate dopo il cutoff storico, scaricate dopo aver fissato il modello. Non e' un test prospettico preregistrato; eventuali repliche sugli stessi dati non sono nuove conferme.",
    }
    output = ROOT / "results/holdout.json"
    try:
        with tempfile.TemporaryDirectory(prefix="numa-holdout-") as directory:
            path = Path(directory) / "history.txt"
            shutil.copyfile(DATA_PATH, path)
            update_data.update(path=path)
            df = load_draws(path)
            snapshot = path.read_bytes()
    except (OSError, ValueError, update_data.requests.RequestException) as error:
        out["error"] = str(error)
        output.write_text(json.dumps(out, indent=2))
        print(f"Controllo non disponibile: {error}", flush=True)
        return
    cutoff = pd.Timestamp(protocol["history_end"])
    known = df[df.data <= cutoff]
    if _digest(known) != protocol["evaluated_data_sha256"]:
        frozen = load_draws(FROZEN_PATH)
        frozen = frozen[frozen.data <= cutoff]
        if _digest(frozen) != protocol["evaluated_data_sha256"]:
            raise ValueError("lo storico congelato non corrisponde al protocollo")
        both = frozen.merge(known, on="data", how="left", suffixes=("", "_new"))
        same = np.sort(both[NUM_COLS].to_numpy(), axis=1) == np.sort(both[[f"{c}_new" for c in NUM_COLS]].to_numpy(), axis=1)
        if not same.all():
            raise ValueError("lo storico prima del cutoff contraddice quello dell'esperimento")
        inserted = known[~known.data.isin(frozen.data)]
        out["inserted_before_cutoff"] = [str(d.date()) for d in inserted.data]
        out["history_used"] = "storico congelato fino al cutoff + estrazioni nuove successive"
        df = pd.concat([frozen, df[df.data > cutoff]], ignore_index=True)
        known = frozen
    if (df.data > pd.Timestamp.now().normalize()).any():
        raise ValueError("la fonte riporta date future")
    start = len(known)
    out["draws"] = len(df) - start
    if start == len(df):
        out["status"] = "no_new_draws"
        output.write_text(json.dumps(out, indent=2))
        print("Nessuna estrazione successiva al cutoff disponibile.", flush=True)
        return
    # Copia integrale per riprodurre il controllo senza cambiare i dati originali.
    snapshot_path = ROOT / "data/superenalotto_holdout.txt"
    snapshot_path.write_bytes(snapshot)
    out["snapshot_sha256"] = hashlib.sha256(snapshot).hexdigest()
    out["period"] = [str(df.data.iloc[start].date()), str(df.data.iloc[-1].date())]
    P = presence_matrix(df)
    X = build_features(P)
    challenger = TemporalEnsembleModel()
    with threadpool_limits(limits=4):
        idx, results = run_backtest(X, P, start, [UniformModel(), gbm_model(), challenger])
    out["status"] = "evaluated"
    out["models"] = {name: summarize(r) for name, r in results.items()}
    competitors = [name for name in out["models"] if name != UniformModel.name]
    adjusted = holm_adjust([out["models"][name]["p_chance_ge_exact"] for name in competitors])
    for name, p in zip(competitors, adjusted):
        out["models"][name]["p_holm_2"] = float(p)
    out["challenger_fits"] = challenger.fit_history
    output.write_text(json.dumps(out, indent=2))
    np.savez_compressed(output.with_suffix(".npz"), eval_idx=idx,
                        **{f"{key}__{name}": r[key] for name, r in results.items()
                           for key in ("hits", "logloss", "brier", "probs", "picks")})
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
