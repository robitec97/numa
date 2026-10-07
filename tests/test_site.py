"""Calendario, registro delle previsioni, modello di popolarita', sito."""

import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import ledger, popularity as pop
from src.data import NUM_COLS
from src.quote import P_HITS, derive
from src.schedule import ROME, draw_moment, last_scheduled_draw, next_draw_date
from src.strategies import predict

ROOT = Path(__file__).resolve().parent.parent


def rome(*args):
    return datetime(*args, tzinfo=ROME)


def fake_draws(n=400, start="2015-01-06", seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="3D")
    nums = np.array([np.sort(rng.choice(90, 6, replace=False) + 1) for _ in range(n)])
    df = pd.DataFrame(nums, columns=NUM_COLS)
    df.insert(0, "data", dates)
    df.insert(0, "concorso", np.arange(1, n + 1))
    df["jolly"] = [int(rng.choice(np.setdiff1d(np.arange(1, 91), r))) for r in nums]
    df["superstar"] = rng.integers(1, 91, n)
    return df


class ScheduleTests(unittest.TestCase):
    def test_next_draw(self):
        self.assertEqual(next_draw_date(rome(2026, 10, 7, 22, 0)), date(2026, 10, 8))   # mer -> gio
        self.assertEqual(next_draw_date(rome(2026, 10, 8, 19, 29)), date(2026, 10, 8))  # giocate aperte
        self.assertEqual(next_draw_date(rome(2026, 10, 8, 19, 31)), date(2026, 10, 9))  # chiuse -> ven
        self.assertEqual(next_draw_date(rome(2026, 10, 10, 21, 0)), date(2026, 10, 13))  # sab sera -> mar

    def test_last_scheduled_draw(self):
        self.assertEqual(last_scheduled_draw(rome(2026, 10, 7, 12)), date(2026, 10, 6))   # mer -> mar
        self.assertEqual(last_scheduled_draw(rome(2026, 10, 8, 20, 30)), date(2026, 10, 6))  # troppo presto
        self.assertEqual(last_scheduled_draw(rome(2026, 10, 8, 21, 30)), date(2026, 10, 8))

    def test_draw_moment_follows_daylight_saving(self):
        self.assertEqual(draw_moment(date(2026, 10, 8)).astimezone(timezone.utc).hour, 18)   # CEST
        self.assertEqual(draw_moment(date(2026, 10, 27)).astimezone(timezone.utc).hour, 19)  # CET


class LedgerTests(unittest.TestCase):
    def entry(self, target="2026-10-08", picks=None):
        close = datetime.combine(date.fromisoformat(target), datetime.min.time(), ROME) + timedelta(hours=19, minutes=30)
        return {"target_date": target, "sales_close": close.astimezone(timezone.utc).isoformat(),
                "picks": picks or {"numa": [1, 2, 3, 4, 5, 6], "caso": [10, 20, 30, 40, 50, 60]}}

    def test_chain_detects_tampering(self):
        entries = []
        ledger.append(entries, self.entry("2026-10-08"), now=rome(2026, 10, 7, 9))
        ledger.append(entries, self.entry("2026-10-09"), now=rome(2026, 10, 9, 8))
        ledger.verify_chain(entries)
        entries[0]["picks"]["numa"][0] = 7
        with self.assertRaises(ValueError):
            ledger.verify_chain(entries)

    def test_result_does_not_break_chain(self):
        entries = []
        ledger.append(entries, self.entry(), now=rome(2026, 10, 7, 9))
        entries[0]["result"] = {"hits": {}}
        ledger.verify_chain(entries)

    def test_no_late_or_duplicate_predictions(self):
        entries = []
        with self.assertRaises(ValueError):
            ledger.append(entries, self.entry(), now=rome(2026, 10, 8, 19, 45))
        ledger.append(entries, self.entry(), now=rome(2026, 10, 8, 10))
        with self.assertRaises(ValueError):
            ledger.append(entries, self.entry(), now=rome(2026, 10, 8, 11))

    def draws(self, rows):
        df = pd.DataFrame([[i + 1, pd.Timestamp(d), *nums, j, 1] for i, (d, nums, j) in enumerate(rows)],
                          columns=["concorso", "data", *NUM_COLS, "jolly", "superstar"])
        return df

    def test_scored_against_first_draw_after_registration(self):
        entries = []
        ledger.append(entries, self.entry(), now=rome(2026, 10, 7, 23))
        known = self.draws([("2026-10-06", [1, 2, 3, 40, 50, 60], 7)])
        self.assertEqual(ledger.score(entries, known), 0)  # l'estrazione del 6 era gia' avvenuta
        # concorso spostato al venerdi': conta la prima estrazione successiva
        moved = self.draws([("2026-10-06", [1, 2, 3, 40, 50, 60], 7), ("2026-10-09", [1, 2, 30, 40, 50, 61], 8)])
        self.assertEqual(ledger.score(entries, moved), 1)
        r = entries[0]["result"]
        self.assertEqual(r["date"], "2026-10-09")
        self.assertEqual(r["hits"], {"numa": 2, "caso": 3})
        board = ledger.leaderboard(entries)
        self.assertEqual(board["caso"]["hits"], 3)
        self.assertAlmostEqual(board["caso"]["p_at_least"], P_HITS[3] + P_HITS[4] + P_HITS[5] + P_HITS[6], places=12)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.json"
            entries = []
            ledger.append(entries, self.entry(), now=rome(2026, 10, 7, 9))
            ledger.save(entries, path)
            self.assertEqual(ledger.load(path), entries)


class PopularityTests(unittest.TestCase):
    def test_pair_counts(self):
        np.testing.assert_array_equal(pop.pair_counts(np.array([[1, 2, 11, 21, 22, 90]])), [[2, 2, 4]])

    def test_patterns(self):
        bad = [[1, 2, 3, 40, 50, 60], [5, 10, 15, 20, 61, 77], [1, 5, 9, 12, 25, 31],
               [41, 43, 45, 47, 70, 80], [40, 41, 52, 63, 74, 85]]
        self.assertFalse(pop.pattern_ok(np.array(bad)).any())
        self.assertTrue(pop.pattern_ok(np.array([[7, 23, 38, 52, 69, 84]]))[0])
        self.assertFalse(pop.pattern_ok(np.array([[7, 23, 38, 52, 69, 84]]), last=[7, 23, 38, 1, 2, 4])[0])
        self.assertFalse(pop.pattern_ok(np.array([[7, 23, 38, 52, 69, 84]]), past={(7, 23, 38, 52, 69, 84)})[0])

    def synthetic(self, n=2500, noise=0.02):
        draws = fake_draws(n, start="2005-01-04")
        rng = np.random.default_rng(1)
        theta = rng.normal(0, 0.15, 90)
        theta -= theta.mean()
        nums = draws[NUM_COLS].to_numpy()
        quote = pd.DataFrame({"quota2_floor": False}, index=draws.index)
        for k in pop.CATEGORIES:
            quote[f"pop_{k}"] = np.exp(pop.C1[k] * theta[nums - 1].sum(axis=1) + rng.normal(0, noise, n))
        return draws, quote, theta

    def test_fit_recovers_known_popularity(self):
        draws, quote, theta = self.synthetic()
        model = pop.fit(draws, quote, ridge=0.1)
        self.assertGreater(np.corrcoef(model.theta, theta)[0, 1], 0.98)
        self.assertLess(np.abs(model.phi).max(), 0.02)
        r2 = pop.within_r2(model, draws, quote, start="2015-01-01")
        self.assertGreater(r2["3"]["r2_within_year"], 0.9)

    def test_fit_uses_only_data_until_end(self):
        draws, quote, _ = self.synthetic()
        end = "2010-12-31"
        a = pop.fit(draws, quote, end=end)
        changed = quote.copy()
        late = draws["data"] > pd.Timestamp(end)
        changed.loc[late, ["pop_2", "pop_3", "pop_4"]] *= 3.0
        b = pop.fit(draws, changed, end=end)
        np.testing.assert_allclose(a.theta, b.theta)

    def test_generate_is_unpopular_filtered_and_diverse(self):
        draws, quote, _ = self.synthetic(n=1500)
        model = pop.fit(draws, quote)
        rng = np.random.default_rng(5)
        tickets = pop.generate(model, rng, n=6)
        self.assertTrue(pop.pattern_ok(tickets).all())
        for i in range(6):
            for j in range(i):
                self.assertLessEqual(np.intersect1d(tickets[i], tickets[j]).size, 2)
        typical = np.median(model.score(pop.random_tickets(rng, 20000)))
        self.assertTrue((model.score(tickets) < typical).all())

    def test_random_tickets_are_uniform(self):
        t = pop.random_tickets(np.random.default_rng(0), 90000)
        self.assertTrue((np.diff(t, axis=1) > 0).all())
        counts = np.bincount(t.ravel(), minlength=91)[1:]
        self.assertLess(np.abs(counts / counts.mean() - 1).max(), 0.05)


class QuoteTests(unittest.TestCase):
    def test_popularity_from_prize_pools(self):
        M, w2, w3 = 1_000_000.0, 50_000, None
        N = M / 0.6
        w3 = round(N * P_HITS[3] * 1.2)
        raw = pd.DataFrame([{
            "data": pd.Timestamp("2024-05-07"), "jackpot_in_palio": 9e7,
            "vincitori_6": 0, "quota_6": None, "vincitori_5_1": 0, "quota_5_1": None,
            "vincitori_5": 2, "quota_5": 0.042 * M / 2, "vincitori_4": 300, "quota_4": 0.042 * M / 300,
            "vincitori_3": w3, "quota_3": 0.128 * M / w3, "vincitori_2": w2, "quota_2": 0.40 * M / w2,
        }])
        d = derive(raw).iloc[0]
        self.assertAlmostEqual(d["montepremi"], M, places=3)
        self.assertAlmostEqual(d["pop_3"], 1.2, places=3)
        self.assertAlmostEqual(d["payout_5"], 0.042 * M / 3, places=3)   # una colonna in piu'
        self.assertAlmostEqual(d["payout_5_1"], 0.13 * M, places=3)      # nessun vincitore: tutto il fondo
        self.assertEqual(d["payout_6"], 9e7)


class StrategyTests(unittest.TestCase):
    def test_predictions_are_valid_and_reproducible(self):
        draws = fake_draws(500)
        model = pop.PopularityModel(np.random.default_rng(2).normal(0, 0.1, 90), np.zeros(3))
        a = predict(draws, model, date(2026, 10, 8), with_ml=False)
        b = predict(draws, model, date(2026, 10, 8), with_ml=False)
        self.assertEqual(a, b)
        for picks in a["picks"].values():
            self.assertEqual(len(set(picks)), 6)
            self.assertTrue(all(1 <= n <= 90 for n in picks))
        c = predict(draws, model, date(2026, 10, 9), with_ml=False)
        self.assertNotEqual(a["picks"]["caso"], c["picks"]["caso"])


@unittest.skipUnless(shutil.which("node"), "Node non installato")
class JavaScriptParityTests(unittest.TestCase):
    def test_core_js_matches_python(self):
        rng = np.random.default_rng(11)
        model = pop.PopularityModel(rng.normal(0, 0.2, 90), np.array([-0.13, 0.08, 0.07]))
        tickets = np.vstack([pop.random_tickets(rng, 400),
                             [[1, 2, 3, 40, 50, 60], [5, 10, 15, 20, 61, 77], [1, 5, 9, 12, 25, 31],
                              [41, 43, 45, 47, 70, 80], [7, 23, 38, 52, 69, 84]]])
        last = [7, 23, 38, 1, 2, 4]
        fixture = {"model": model.to_dict(), "tickets": tickets.tolist(), "last": last,
                   "score": model.score(tickets).tolist(),
                   "ok": pop.pattern_ok(tickets).tolist(), "ok_last": pop.pattern_ok(tickets, last=last).tolist(),
                   "pairs": pop.pair_counts(tickets).tolist()}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture.json"
            path.write_text(json.dumps(fixture))
            out = subprocess.run(["node", str(ROOT / "tests" / "core_parity.mjs"), str(path)],
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout), {"score": 0, "ok": 0, "ok_last": 0, "pairs": 0})


if __name__ == "__main__":
    unittest.main()
