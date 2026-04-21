"""Market regime dashboard analytics."""
import math

from core.config import INDIA_LARGECAP, SP500
from core.data import fetch_ticker
from core.indicators import market_regime

US_BENCHMARKS = [
    ("SPY", "S&P 500"),
    ("QQQ", "Nasdaq 100"),
    ("IWM", "Russell 2000"),
    ("DIA", "Dow 30"),
    ("XLF", "Financials"),
    ("XLK", "Technology"),
]

INDIA_BENCHMARKS = [
    ("^NSEI", "Nifty 50"),
    ("^BSESN", "Sensex"),
    ("NIFTYBEES.NS", "Nifty ETF"),
    ("BANKBEES.NS", "Bank Nifty ETF"),
    ("ITBEES.NS", "IT ETF"),
]


def _pct_change(a, b):
    if a in (None, 0) or b is None:
        return None
    return (b - a) / a * 100


def _series_stats(raw):
    closes = [r["c"] for r in raw["data"] if r.get("c")]
    if len(closes) < 70:
        return None
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]
    tail20 = closes[-20:]
    ma20 = sum(tail20) / len(tail20)
    ma50 = sum(closes[-50:]) / 50
    peak63 = max(closes[-63:])
    vol20 = None
    if len(rets) >= 20:
        r20 = rets[-20:]
        mean = sum(r20) / len(r20)
        var = sum((r - mean) ** 2 for r in r20) / max(len(r20) - 1, 1)
        vol20 = math.sqrt(var) * math.sqrt(252) * 100
    return {
        "ticker": raw["ticker"],
        "name": raw.get("name") or raw["ticker"],
        "last": round(closes[-1], 4),
        "ret20": round(_pct_change(closes[-21], closes[-1]), 2) if len(closes) >= 21 else None,
        "ret60": round(_pct_change(closes[-61], closes[-1]), 2) if len(closes) >= 61 else None,
        "drawdown63": round((closes[-1] - peak63) / peak63 * 100, 2) if peak63 else None,
        "vol20": round(vol20, 2) if vol20 is not None else None,
        "dist20": round((closes[-1] - ma20) / ma20 * 100, 2) if ma20 else None,
        "dist50": round((closes[-1] - ma50) / ma50 * 100, 2) if ma50 else None,
        "regime": market_regime(closes),
    }


def _breadth_summary(tickers, max_names=30):
    sampled = []
    above20 = above50 = winners20 = 0
    one_month_returns = []
    for tk in tickers[:max_names]:
        raw = fetch_ticker(tk)
        if not raw:
            continue
        closes = [r["c"] for r in raw["data"] if r.get("c")]
        if len(closes) < 60:
            continue
        sampled.append(tk)
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50
        above20 += 1 if closes[-1] > ma20 else 0
        above50 += 1 if closes[-1] > ma50 else 0
        ret20 = (closes[-1] - closes[-21]) / closes[-21] * 100 if closes[-21] else 0
        one_month_returns.append(ret20)
        winners20 += 1 if ret20 > 0 else 0
    n = len(sampled)
    if n == 0:
        return None
    mean = sum(one_month_returns) / n
    var = sum((r - mean) ** 2 for r in one_month_returns) / max(n - 1, 1)
    dispersion = math.sqrt(var)
    breadth20 = above20 / n * 100
    breadth50 = above50 / n * 100
    winners = winners20 / n * 100
    oversold = max(0.0, min(1.0, (50 - breadth20) / 40))
    breadth = max(0.0, min(1.0, (55 - breadth50) / 45))
    dispersion_score = max(0.0, min(1.0, dispersion / 8))
    mr_score = round((0.4 * oversold + 0.3 * breadth + 0.3 * dispersion_score) * 100, 1)
    return {
        "sampled": n,
        "tickers": sampled,
        "above20Pct": round(breadth20, 1),
        "above50Pct": round(breadth50, 1),
        "winners20Pct": round(winners, 1),
        "dispersion20": round(dispersion, 2),
        "medianRet20": round(sorted(one_month_returns)[n // 2], 2),
        "mrOpportunityScore": mr_score,
    }


def _regime_takeaway(lead, breadth):
    if not lead:
        return {"headline": "No benchmark data", "stance": "neutral"}
    breadth20 = (breadth or {}).get("above20Pct", 50)
    vol = lead.get("vol20") or 20
    dd = abs(min(lead.get("drawdown63") or 0, 0))
    stance = "balanced"
    if lead.get("regime") in ("strong_down", "weak_down"):
        stance = "defensive"
    elif lead.get("regime") in ("strong_up", "weak_up") and breadth20 > 55:
        stance = "trend"
    if breadth20 < 35 and vol >= 18:
        headline = "Oversold tape with enough volatility for mean reversion"
    elif stance == "trend":
        headline = "Trend-following conditions dominate, short mean-reversion setups are lower quality"
    elif stance == "defensive":
        headline = "Downtrend regime, prefer selective longs or market-neutral pairs"
    else:
        headline = "Mixed regime, favor rank-driven single names and tighter stops"
    return {
        "headline": headline,
        "stance": stance,
        "volatility": vol,
        "drawdown63": dd,
    }


def market_regime_dashboard(market="us"):
    market = (market or "us").lower()
    benchmark_defs = INDIA_BENCHMARKS if market == "india" else US_BENCHMARKS
    breadth_universe = INDIA_LARGECAP if market == "india" else SP500
    benchmarks = []
    for tk, label in benchmark_defs:
        raw = fetch_ticker(tk)
        if not raw:
            continue
        stats = _series_stats(raw)
        if stats:
            stats["label"] = label
            benchmarks.append(stats)
    breadth = _breadth_summary(breadth_universe)
    lead = benchmarks[0] if benchmarks else None
    benchmarks.sort(key=lambda b: abs(b.get("ret20") or 0), reverse=True)
    takeaway = _regime_takeaway(lead, breadth)
    return {
        "market": market,
        "benchmarks": benchmarks,
        "breadth": breadth,
        "takeaway": takeaway,
    }
