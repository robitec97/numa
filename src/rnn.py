"""Rete ricorrente (LSTM) per la 'predizione' dell'estrazione successiva.

Input: le ultime SEQ_LEN estrazioni come vettori binari a 90 dimensioni.
Output: 90 logit -> sigmoide = P(numero n esce al prossimo concorso).
Loss: binary cross-entropy multi-label (6 positivi su 90 per riga).

Se l'LSTM impara qualcosa oltre p=6/90 uniforme, la log-loss di
validazione scende sotto 0.2449; il backtest verifica se cio' si traduce
in hit reali fuori campione.
"""

import numpy as np
import torch
from torch import nn

from .data import N_BALLS

SEQ_LEN = 50
HIDDEN = 64
MAX_EPOCHS = 40
PATIENCE = 5
LR = 1e-3
BATCH = 128


class _Net(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(N_BALLS, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, N_BALLS)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class LSTMModel:
    """Interfaccia sequenziale per run_backtest (is_sequential=True)."""

    needs_fit = True
    is_sequential = True

    def __init__(self, seed=42, max_epochs=MAX_EPOCHS):
        self.name = "LSTM (rete ricorrente)"
        self.seed = seed
        self.max_epochs = max_epochs
        self.net = None

    @staticmethod
    def _samples(P_hist):
        """(x, y): finestre di SEQ_LEN estrazioni -> estrazione successiva."""
        Pf = torch.from_numpy(P_hist.astype(np.float32))
        n = len(Pf) - SEQ_LEN
        x = Pf.unfold(0, SEQ_LEN, 1)[:n].permute(0, 2, 1)  # (n, SEQ_LEN, 90)
        y = Pf[SEQ_LEN:]
        return x, y

    def fit_seq(self, P_hist):
        torch.manual_seed(self.seed)
        x, y = self._samples(P_hist)
        # split temporale: ultimo 10% come validazione per l'early stopping
        n_val = max(1, len(x) // 10)
        x_tr, y_tr, x_va, y_va = x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]

        self.net = _Net()
        opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        lossf = nn.BCEWithLogitsLoss()

        best_val, best_state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(x_tr))
            for i in range(0, len(x_tr), BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                loss = lossf(self.net(x_tr[idx]), y_tr[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = lossf(self.net(x_va), y_va).item()
            if val < best_val - 1e-5:
                best_val, bad = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.val_logloss = best_val

    def predict_scores_seq(self, P_hist):
        x = torch.from_numpy(P_hist[-SEQ_LEN:].astype(np.float32)).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(x)).squeeze(0).numpy().astype(np.float64)
