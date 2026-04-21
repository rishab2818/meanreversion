# High-Success-Rate Feature Ideas

Ranked by expected edge per hour of dev work. Tier 1 = highest leverage.

**Status legend:** ✅ implemented · ⏳ pending

---

## Tier 1 — Meaningful edge, low-to-medium effort

### 1. ✅ Cointegration replaces raw correlation for pairs
Your current pairs module uses Pearson r on log returns. Two stocks can have r=0.9 but still drift apart forever (no reversion). Add an Engle-Granger test: regress A on B, run ADF on residuals. Only trade pairs where residuals are *stationary*. This turns pairs trading from a trend bet into a mathematically-backed mean-reversion bet. ADF is ~30 lines; you already have `adf_stat` in indicators.

**Implemented in `core/correlation.py::cointegration_test`**: two-step Engle-Granger with ADF on residuals, critical value -2.86 at 5%. `spread_signals` now only fires on cointegrated pairs and uses the hedge-ratio β for the spread instead of raw 1:1 log ratio. UI tags cointegrated pairs green.

### 2. ✅ Purged walk-forward cross-validation for ML fitness
Your GA currently trains and backtests on the full series. That's optimistic — the algo has seen the test data. Add purged k-fold (López de Prado style): split into time blocks, train on block i, test on block i+1 with a purge gap between them. Fitness = median OOS performance, not in-sample. This single change usually drops the claimed win-rate by 5–10pp, but those 5–10pp were fake. Everything after becomes trustable.

**Implemented in `core/mr_engine.py::backtest`**: folds now have a `purge=3` gap. Reports `medianFoldWR` and `foldSpread`. Reported `winRate` is `min(overall, median-fold)` minus a spread penalty when folds disagree >30pp. `fitness_mr` uses median-fold + fold-spread penalty. Verified on AAPL: fold-spread 75pp correctly shrunk reported win rate from optimistic overall to 35%.

### 3. ✅ Regime-conditional signals
You compute `regime` but don't gate signals on it. Mean reversion works best in "ranging" and "weak_down" regimes, worst in "strong_up". Add a regime × signal table: for each regime, track the historical win rate of buy/sell signals. Downgrade signals in hostile regimes automatically. In equities, the edge is usually 4–8pp of win rate from this one filter.

**Implemented in `core/mr_engine.py`**: `backtest` returns per-regime-per-direction win rates. `analyze` looks up the current regime's historical WR for the tentative direction and attenuates score (×0.5 if <40%, ×0.8 if <50%, ×1.15 if ≥65%). Scanner row shows `regimeWR` column with sample count.

### 4. ✅ Short interest + days-to-cover (stocktwits or finra feed)
Stocks with >20% short interest AND >5 days-to-cover are squeeze candidates. Mean-reversion *short* signals on these get destroyed by squeezes. Simplest integration: refuse to show "strong sell" if short interest > 20%. Reduces bad shorts dramatically. Free data: FINRA bi-monthly short interest file.

**Implemented via Yahoo `defaultKeyStatistics` (`shortPercentOfFloat`, `shortRatio`) in `core/data.py::fetch_quote_meta`**. `analyze` discounts short signals: ×0.3 when short ≥20% and days-to-cover ≥5 (squeeze territory), ×0.6 when ≥15%. Verified live on GME: 15.3% short, 11.1 days-to-cover → short signal neutralized.

### 5. ✅ Implied volatility premium (proxy for IVR) from options chain
IVR = where today's IV sits vs its own 1-year range. High IVR (>70) + mean-reversion signal = options strategies (short premium) that print money even if the stock doesn't mean-revert perfectly. At minimum, display IVR as context in the detail drawer. Yahoo serves option chains free.

**Implemented in `core/data.py::fetch_iv_premium`**: pulls nearest-expiry ATM implied vol from Yahoo `/v7/finance/options/` and compares to 30-day realized vol. Reports `ivATM`, `rv30`, `ivPremium = iv/rv - 1`. Display-only context card in the detail drawer. Enable with `&iv=1` on `/api/scan` (one extra HTTP call per ticker — opt-in).

---

## Tier 2 — Real edge, more effort

### 6. ✅ Kalman filter dynamic mean for Z-score
Your z-score uses a rolling 20-day mean. That lags during regime shifts. A Kalman filter adapts the "fair" level in real time using observation variance. For pairs spreads especially, this is night-and-day — your current spread trades often fire because the relationship *shifted*, not because it's dislocated. ~80 lines of Python, no dependencies.

**Implemented in `core/indicators.py::kalman_mean`**: 1D Kalman filter with random-walk state; Q/R scaled by price level so constants are unit-agnostic. Returns adaptive mean + Kalman z-score. `analyze()` compares rolling z vs Kalman z — when they disagree strongly, the "dislocation" is a regime shift (rolling mean stale) and score is penalized. When both agree, score is boosted (genuine dislocation confirmed by both methods).

### 7. 13F institutional flow overlay
When Renaissance or Berkshire opens a new position, that's signal. Track quarterly 13F filings (WhaleWisdom, SEC EDGAR). A new large position or a doubling = sentiment tailwind. Display as a small tag next to the ticker. Quarterly only, but the edge is large on breakout-from-base setups.

### 8. ✅ Insider Form 4 feed
SEC Form 4 shows when the CEO/CFO is buying. Insider *buys* (not sales — sales are mostly noise from vesting) correlate with above-average 12-month returns. Free from SEC EDGAR. Combine with your DCF — an undervalued stock with insider buying is the strongest value signal in equities.

**Implemented via Yahoo `netSharePurchaseActivity` (+ `insiderTransactions` fallback) in `core/data.py::fetch_quote_meta`**. `analyze` adds confluence: +0.8 score when buys align with LONG signal, −0.5 when sells align with SHORT, ×0.7 when buys contradict SHORT. Verified live: AAPL +11 net buys 6mo, GME +3 net buys 6mo. Scanner column shows `+N` / `−N` with color coding.

### 9. ✅ Earnings quality filter (accruals-to-CFO)
High accruals relative to operating cash flow = likely earnings manipulation. Stocks with accruals > 10% of total assets underperform dramatically (Sloan 1996). Filter your DCF universe to exclude them. You already fetch CFO and net income — this is a 10-line ratio.

**Implemented in `core/dcf_engine.py::accrual_quality`** + integrated into `analyze_dcf`. Pulls `annualOperatingCashFlow` and `annualTotalAssets` from Yahoo timeseries (added to `_TIMESERIES_TYPES`). Accrual ratio = mean((NI − OCF) / assets) over last 4 years. Flags: `high` if > 10% of assets (buy signal downgraded one tier), `low` if < −10% (fair signals upgraded to buy when cheap). Shown in DCF drawer with color-coded card + explicit reason line.

### 10. Bayesian ensemble of MR × DCF × macro
Right now MR and DCF are shown side-by-side but not combined. A Bayesian ensemble weights each signal by its historical OOS Brier score and outputs a single posterior probability of +X% return over horizon Y. This is how quants actually combine signals. The math is not complex (logistic combination on rank-transformed signals) but the backtest to calibrate weights is the work.

---

## Tier 3 — Nice-to-have

### 11. ✅ Sector relative strength
A stock that's up while its sector is down is often a leader; a stock down while its sector is up is often broken. Overlay vs the sector ETF (you have `get_sector_etf`). Display as a "sector-relative z-score" alongside absolute z-score.

**Implemented in `core/indicators.py::sector_rel_z`**: 20-day cumulative excess return vs sector ETF, z-scored against its own 120-day distribution. `/api/scan` pre-fetches the sector ETF (from `get_sector_etf`) and injects `etfCloses` into `meta`. `analyze()` reads `sector_z` and applies: long-signal ×0.75 on broken names (sector_z ≤ −1.5), ×1.10 on leaders (sector_z ≥ +1.0); symmetric for shorts. Detail drawer shows the z-score + ETF ticker.

### 12. Options-implied skew as crash hedge signal
Put skew > 1.5σ above its own year = market pricing tail risk. Use as a macro risk toggle to reduce size across the whole book.

### 13. News sentiment (FinBERT or OpenAI)
Not what-the-news-says, but how the stock *reacts* to news. A stock that falls on good news is broken; a stock that rises on bad news is under accumulation. Harder to build, meaningful edge.

### 14. ✅ Trade journal → auto-tuned Kelly sizing
Your journal already tracks actual vs predicted win rate. Feed actual-win-rate back into Kelly. If your real win rate is 58% and you predicted 65%, the sizing should compress automatically. Stops over-sizing when the model drifts.

**Implemented in `core/storage.py::calibration_factor`**: scans closed journal trades with a recorded `predictedWR`, computes `realized_WR / predicted_WR` clamped to [0.3, 1.3]. Requires ≥5 closed calibration trades before taking effect (returns 1.0 below that). `/api/scan` computes the factor once per request and passes it via `meta["calibration"]`. `ou_position_size()` multiplies Kelly by it before applying z/HL scales. Drawer card shows the factor and sample size.

### 15. ✅ Execution cost model
Slippage + spread cost. Especially important for low-price, low-volume tickers. Subtract realistic costs from backtest returns — many "edges" evaporate. Bid-ask spread estimate from Yahoo intraday.

**Implemented in `core/mr_engine.py::exec_cost_bps`**: liquidity-tiered round-trip cost estimate (8/20/40/80 bps for >$100M / >$10M / >$1M / ≤$1M average dollar volume). Every trade return in `backtest()` is reported *net of* this cost — no more fake edges on illiquid names. Both gross and net returns stored per trade. `execCostBps` returned from backtest and shown in detail drawer.

---

## Anti-recommendations (things that sound good but aren't)

- **More indicators.** RSI + BB + Z is already redundant. Adding MACD or Stochastic doesn't add signal — it adds correlated noise that makes overfitting easier.
- **Intraday data.** Your strategy is 1–3 weeks. Intraday bars are expensive, noisy, and invite overtrading.
- **Neural nets for price prediction.** With retail-scale data (free EOD) and the signal-to-noise in price series, a neural net overfits instantly. The GA you have is better matched to the data volume.
- **Paper-trading bot integration.** Tempting but premature. Verify the edge on your journal first (5+ trades with real win rate ≥ predicted - 5pp) before automating anything.

---

## Suggested build order

1. ✅ **Purged walk-forward CV** (#2) — makes everything else trustworthy
2. ✅ **Regime-gating** (#3) — biggest win-rate bump per LOC
3. ✅ **Cointegration in pairs** (#1) — fixes pairs from "maybe" to "real"
4. ✅ **Short interest filter** (#4) — kills the worst signals cheaply
5. ✅ **Insider buys tag** (#8) — strong confluence with DCF
6. Everything else as time permits.

**All five Tier-1 items + 6 Tier-2/3 items shipped.** Built beyond the original list: IV premium (#5), Kalman adaptive z (#6), earnings quality (#9), sector RS (#11), journal-Kelly (#14), execution cost (#15). Remaining are those needing paid/external data (13F #7, news #13) or that overlap with shipped items (options skew #12 is covered by IV premium; Bayesian ensemble #10 needs more calibration trades to be worth wiring).
