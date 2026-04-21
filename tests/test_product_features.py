import math
import os
import tempfile
import unittest
from unittest.mock import patch

from core import correlation, history, ranking, regime


def _make_rows(values, prefix="2025-01-"):
    return [{"date": f"{prefix}{i+1:02d}", "o": v, "h": v * 1.01, "l": v * 0.99, "c": v, "v": 1_000_000} for i, v in enumerate(values)]


class ProductFeatureTests(unittest.TestCase):
    def test_multifactor_ranking_prefers_higher_quality_setup(self):
        rows = [
            {
                "ticker": "AAA", "sig": "strong-buy", "direction": "LONG", "score": 6.0,
                "winRate": 68.0, "avgRet": 2.1, "sharpe": 1.6, "medianFoldWR": 65.0,
                "maxDD": 8.0, "foldSpread": 10.0, "regimeWR": 64.0, "rr": 2.4,
                "signalAge": 1, "volRatio": 1.4, "trades": 20, "dcf": {"mosCons": 22.0},
            },
            {
                "ticker": "BBB", "sig": "buy", "direction": "LONG", "score": 3.8,
                "winRate": 57.0, "avgRet": 0.8, "sharpe": 0.7, "medianFoldWR": 54.0,
                "maxDD": 18.0, "foldSpread": 26.0, "regimeWR": 52.0, "rr": 1.3,
                "signalAge": 4, "volRatio": 0.8, "trades": 8, "dcf": {"mosCons": 5.0},
            },
        ]
        ranked = ranking.rank_scan_results(rows)
        self.assertEqual(ranked[0]["ticker"], "AAA")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertGreater(ranked[0]["rankScore"], ranked[1]["rankScore"])

    def test_scan_history_roundtrip_uses_slim_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "scan_history.json")
            with patch("core.history.SCAN_HISTORY_F", path):
                snap = history.record_scan_snapshot([{
                    "ticker": "AAA", "name": "Alpha", "sig": "strong-buy", "direction": "LONG",
                    "rank": 1, "rankScore": 63.4, "rankPct": 100.0, "winRate": 66.0,
                    "avgRet": 1.9, "sharpe": 1.2, "rr": 2.0, "regime": "ranging",
                    "signalAge": 1, "entry": 100.0, "stop": 95.0, "t1": 108.0,
                    "dcf": {"ok": True, "sig": "buy", "mosCons": 18.0},
                }], {"watchlist": "india", "market": "india", "tickerCount": 1})
                loaded = history.load_scan_history()
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0]["id"], snap["id"])
                self.assertEqual(loaded[0]["summary"]["bestTicker"], "AAA")
                self.assertIn("dcf", loaded[0]["results"][0])

    def test_regime_dashboard_builds_breadth_and_takeaway(self):
        benchmark_prices = [100 + i * 0.6 for i in range(90)]
        breadth_prices = [50 + i * 0.3 for i in range(90)]

        def fake_fetch_ticker(ticker):
            vals = benchmark_prices if ticker in ("SPY", "QQQ", "IWM", "DIA", "XLF", "XLK") else breadth_prices
            return {"ticker": ticker, "name": ticker, "data": _make_rows(vals)}

        with patch("core.regime.fetch_ticker", side_effect=fake_fetch_ticker), patch("core.regime.SP500", ["A", "B", "C", "D"]):
            dashboard = regime.market_regime_dashboard("us")

        self.assertEqual(dashboard["market"], "us")
        self.assertTrue(dashboard["benchmarks"])
        self.assertIn("mrOpportunityScore", dashboard["breadth"])
        self.assertIn("headline", dashboard["takeaway"])

    def test_pair_workspace_returns_cointegration_and_backtest(self):
        rows1 = []
        rows2 = []
        for i in range(160):
            p2 = 100 + i * 0.35 + 2.0 * math.sin(i / 7)
            residual = 0.03 * math.sin(i / 4)
            if i > 150:
                residual -= 0.06
            p1 = math.exp(0.25 + 1.08 * math.log(p2) + residual)
            rows1.append({"date": f"2025-{i:03d}", "o": p1, "h": p1 * 1.01, "l": p1 * 0.99, "c": p1, "v": 1_000_000})
            rows2.append({"date": f"2025-{i:03d}", "o": p2, "h": p2 * 1.01, "l": p2 * 0.99, "c": p2, "v": 1_000_000})

        cache = {
            "AAA": {"data": {"ticker": "AAA", "data": rows1}},
            "BBB": {"data": {"ticker": "BBB", "data": rows2}},
        }
        with patch.dict(correlation.CACHE, cache, clear=True):
            pair = correlation.pair_workspace("AAA", "BBB")

        self.assertIsNotNone(pair)
        self.assertIn("backtest", pair)
        self.assertIn("plan", pair)
        self.assertGreater(len(pair["series"]["dates"]), 50)


if __name__ == "__main__":
    unittest.main()
