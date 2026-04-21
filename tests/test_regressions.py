import unittest
from datetime import date, timedelta
from unittest.mock import patch

from core import dcf_engine, ml_optimizer, mr_engine


class MathRegressionTests(unittest.TestCase):
    def test_wacc_zero_debt_uses_full_equity_cost(self):
        fund = {"beta": 1.0, "marketCap": 1_000.0, "totalDebt": 0.0}
        expected = dcf_engine.cost_of_equity(1.0)
        self.assertAlmostEqual(dcf_engine.wacc(fund), expected, places=8)

    def test_trade_levels_keep_stop_on_correct_side_of_entry(self):
        band = {"lower": 642.3333, "middle": 679.26, "upper": 716.0}

        long_stop, long_t1, _, _ = mr_engine._trade_levels(614.34, 6.15, band, "strong_down", "LONG")
        short_stop, short_t1, _, _ = mr_engine._trade_levels(614.34, 6.15, band, "strong_up", "SHORT")

        self.assertLess(long_stop, 614.34)
        self.assertGreater(long_t1, 614.34)
        self.assertGreater(short_stop, 614.34)
        self.assertLess(short_t1, 614.34)

    def test_analyze_uses_shrunk_win_rate_for_ev_not_raw_rate(self):
        raw = {
            "ticker": "TEST",
            "name": "Test",
            "source": "synthetic",
            "data": [
                {"date": f"2024-01-{(i % 28) + 1:02d}", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0, "v": 1_000_000}
                for i in range(60)
            ],
            "earningsNext": None,
        }
        params = {"rsiP": 14, "bbStd": 2.0, "zWin": 20, "volMin": 1.0}
        bt = {
            "winRate": 60.0,
            "winRateRaw": 100.0,
            "winRateBayes": 60.0,
            "avgReturn": 1.5,
            "trades": 6,
            "sharpe": 1.2,
            "maxDD": 5.0,
            "pf": 1.4,
            "method": "purged-walk-forward",
            "medianFoldWR": 60.0,
            "foldSpread": 0.0,
            "regimeStats": {},
            "execCostBps": 8.0,
            "avgHold": 4.0,
        }

        with (
            patch("core.mr_engine.rsi", return_value=20.0),
            patch("core.mr_engine.bb", return_value={"lower": 90.0, "middle": 110.0, "upper": 130.0, "std": 10.0}),
            patch("core.mr_engine.z_score", return_value=-2.0),
            patch("core.mr_engine.atr", return_value=5.0),
            patch("core.mr_engine.kalman_mean", return_value=(99.0, -2.0)),
            patch("core.mr_engine.sector_rel_z", return_value=None),
            patch("core.mr_engine.macd", return_value=None),
            patch("core.mr_engine.stoch_rsi", return_value=(None, None)),
            patch("core.mr_engine.volume_ratio", return_value=1.0),
            patch("core.mr_engine.market_regime", return_value="ranging"),
            patch("core.mr_engine.ou_halflife", return_value=10.0),
            patch("core.mr_engine.adf_stat", return_value=-3.0),
            patch("core.mr_engine.gap_detect", return_value=0.0),
            patch("core.mr_engine.day_of_week_score", return_value=(1.0, "Tue")),
            patch("core.mr_engine.near_earnings", return_value=(False, None)),
            patch("core.mr_engine.signal_age", return_value=1),
            patch("core.mr_engine.backtest", return_value=bt),
        ):
            result = mr_engine.analyze(raw, params, meta={})

        reward = abs(result["t1"] - result["entry"])
        risk = abs(result["entry"] - result["stop"])
        expected_ev = 0.60 * reward - 0.40 * risk
        wrong_ev = 1.00 * reward - 0.00 * risk

        self.assertEqual(result["winRate"], 60.0)
        self.assertEqual(result["winRateRaw"], 100.0)
        self.assertAlmostEqual(result["ev"], round(expected_ev, 4), places=4)
        self.assertNotAlmostEqual(result["ev"], round(wrong_ev, 4), places=4)

    def test_run_ga_mr_uses_real_price_series_and_preserves_fold_metrics(self):
        series = [
            {"date": f"2024-01-{(i % 28) + 1:02d}", "o": 100.0 + i, "h": 101.0 + i, "l": 99.0 + i, "c": 100.0 + i, "v": 1_000_000 + i}
            for i in range(80)
        ]
        chrom_a = {"rsiP": 10, "bbStd": 2.0, "zWin": 20, "rsiOS": 30, "rsiOB": 70, "zThresh": 1.5, "volMin": 1.0}
        chrom_b = {"rsiP": 11, "bbStd": 2.1, "zWin": 21, "rsiOS": 31, "rsiOB": 69, "zThresh": 1.4, "volMin": 1.1}
        seen_inputs = []

        def fake_backtest(input_series, _chrom):
            seen_inputs.append(input_series)
            return {
                "winRate": 62.0,
                "winRateRaw": 70.0,
                "winRateBayes": 62.0,
                "avgReturn": 1.1,
                "sharpe": 0.9,
                "pf": 1.3,
                "maxDD": 4.0,
                "trades": 3,
                "medianFoldWR": 58.0,
                "foldSpread": 7.0,
            }

        with (
            patch("core.ml_optimizer.rnd_mr", side_effect=[chrom_a, chrom_b]),
            patch("core.ml_optimizer.mr_backtest", side_effect=fake_backtest),
            patch("core.ml_optimizer.load_profiles", return_value={}),
            patch("core.ml_optimizer.save_profiles"),
            patch("core.ml_optimizer.time.sleep"),
        ):
            ml_optimizer.run_ga_mr([series], {"popSize": 2, "nGen": 1, "fw": 50, "fr": 50}, ticker="TEST")

        best = ml_optimizer.ga_state["best"]
        self.assertTrue(seen_inputs)
        self.assertIs(seen_inputs[0], series)
        self.assertEqual(best["bt"]["medianFoldWR"], 58.0)
        self.assertEqual(best["bt"]["foldSpread"], 7.0)
        self.assertEqual(best["bt"]["trades"], 3)

    def test_dcf_backtest_uses_dated_snapshots_without_future_history(self):
        fund = {
            "ticker": "X",
            "name": "Example",
            "sharesOut": 10.0,
            "fcfSeries": [
                {"date": "2025-12-31", "value": 100.0},
                {"date": "2024-12-31", "value": 90.0},
                {"date": "2023-12-31", "value": 80.0},
            ],
        }

        start = date(2023, 1, 1)
        price_data = []
        for i in range(1_100):
            dt = start + timedelta(days=i)
            price_data.append({"date": dt.isoformat(), "c": 50.0 + i * 0.01})

        seen_histories = []
        seen_prices = []

        def fake_analyze_dcf(snap, _params):
            seen_histories.append(list(snap["fcfHistory"]))
            seen_prices.append(snap["currentPrice"])
            self.assertEqual(snap["beta"], 1.0)
            self.assertEqual(snap["netDebt"], 0)
            self.assertGreater(snap["marketCap"], 0)
            return {"ok": True, "sig": "buy", "mosCons": 15.0}

        with patch("core.dcf_engine.analyze_dcf", side_effect=fake_analyze_dcf):
            result = dcf_engine.dcf_backtest(fund, price_data, params={})

        self.assertEqual(seen_histories[0], [80.0])
        self.assertEqual(seen_histories[1], [90.0, 80.0])
        self.assertEqual(result["method"], "dated-fcf-walkforward")
        self.assertGreaterEqual(result["signals"], 2)
        self.assertEqual(len(seen_histories), len(seen_prices))

    def test_reverse_dcf_respects_active_params(self):
        fund = {
            "fcfTTM": 100.0,
            "sharesOut": 10.0,
            "netDebt": 0.0,
            "marketCap": 1_000.0,
            "totalDebt": 0.0,
            "beta": 1.0,
        }
        price = 50.0
        low_far = dcf_engine.reverse_dcf(fund, price, params={"gFar": 0.0, "wacc": 0.10, "nNear": 5, "nFar": 5})
        high_far = dcf_engine.reverse_dcf(fund, price, params={"gFar": 0.10, "wacc": 0.10, "nNear": 5, "nFar": 5})

        self.assertIsNotNone(low_far)
        self.assertIsNotNone(high_far)
        self.assertNotAlmostEqual(low_far, high_far, places=6)


if __name__ == "__main__":
    unittest.main()
