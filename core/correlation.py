"""Pair correlation + spread (pairs-trading) signals from cached price data.

Adds Engle-Granger cointegration: high correlation alone doesn't mean the spread
is stationary; two assets can have r=0.9 and still drift apart forever.
Cointegration says the regression residual y - β·x is mean-reverting — which
is what pairs trading actually needs. We now gate spread signals on it.
"""
import math
from core.data import CACHE

def log_ret(prices):
    return [math.log(prices[i]/prices[i-1]) for i in range(1, len(prices))
            if prices[i-1] > 0 and prices[i] > 0]

def pearson(a, b):
    n = min(len(a), len(b))
    if n < 10:
        return 0
    a, b = a[-n:], b[-n:]
    ma, mb = sum(a)/n, sum(b)/n
    num = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x-ma)**2 for x in a))
    db = math.sqrt(sum((x-mb)**2 for x in b))
    return round(num/(da*db), 3) if da*db > 0 else 0

def _ols_slope_intercept(x, y):
    """OLS regression of y on x. Returns (beta, alpha)."""
    n = len(x)
    if n < 5:
        return 0.0, 0.0
    mx = sum(x)/n; my = sum(y)/n
    num = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    den = sum((x[i]-mx)**2 for i in range(n))
    if den == 0:
        return 0.0, my
    beta = num/den
    alpha = my - beta*mx
    return beta, alpha

def _adf_on_series(c):
    """ADF t-statistic on a levels series. Engle-Granger critical value at 5%
    for n≈100 is ≈ -2.86 (more negative than vanilla ADF because residuals
    come from a regression)."""
    if len(c) < 15:
        return None
    y_lag = c[:-1]
    dy = [c[i]-c[i-1] for i in range(1, len(c))]
    n = len(dy)
    mean_lag = sum(y_lag)/n; mean_dy = sum(dy)/n
    num = sum((y_lag[i]-mean_lag)*(dy[i]-mean_dy) for i in range(n))
    den = sum((y_lag[i]-mean_lag)**2 for i in range(n))
    if den == 0:
        return None
    lam = num/den
    res = [dy[i]-lam*(y_lag[i]-mean_lag)-mean_dy for i in range(n)]
    se = math.sqrt(sum(r**2 for r in res)/(n-2))/math.sqrt(den) if n > 2 else 1
    return lam/se if se > 0 else None

def cointegration_test(px1, px2):
    """Engle-Granger two-step: regress log(px1) on log(px2), then ADF on residuals.
    Returns (adf_stat, beta, is_cointegrated). adf_stat < -2.86 means stationary
    residuals at ~5% significance — the spread is mean-reverting."""
    n = min(len(px1), len(px2))
    if n < 40:
        return None, None, False
    lp1 = [math.log(p) for p in px1[-n:] if p > 0]
    lp2 = [math.log(p) for p in px2[-n:] if p > 0]
    n2 = min(len(lp1), len(lp2))
    if n2 < 40:
        return None, None, False
    lp1 = lp1[-n2:]; lp2 = lp2[-n2:]
    beta, alpha = _ols_slope_intercept(lp2, lp1)
    residuals = [lp1[i] - (beta*lp2[i] + alpha) for i in range(n2)]
    adf = _adf_on_series(residuals)
    if adf is None:
        return None, round(beta, 4), False
    is_coint = adf < -2.86
    return round(adf, 2), round(beta, 4), is_coint

def build_correlation(tickers):
    closes_map = {}
    price_map  = {}
    for tk in tickers:
        if tk in CACHE:
            d = CACHE[tk]["data"]
            closes_map[tk] = log_ret([r["c"] for r in d["data"]])
            price_map[tk]  = [r["c"] for r in d["data"]]
    if len(closes_map) < 2:
        return {}, []
    tks = list(closes_map.keys())
    matrix = {t1: {t2: pearson(closes_map[t1], closes_map[t2]) for t2 in tks} for t1 in tks}
    pairs = []
    for i in range(len(tks)):
        for j in range(i+1, len(tks)):
            t1, t2 = tks[i], tks[j]
            c = matrix[t1][t2]
            if abs(c) >= 0.65:
                adf, beta, is_coint = cointegration_test(price_map[t1], price_map[t2])
                pairs.append({"t1": t1, "t2": t2, "corr": c,
                              "cointAdf": adf, "cointBeta": beta,
                              "cointegrated": is_coint})
    # Sort: cointegrated pairs first (most tradable), then by |corr|
    pairs.sort(key=lambda x: (-1 if x["cointegrated"] else 0, -abs(x["corr"])))
    return matrix, pairs

def spread_signals(pairs):
    """Generate pairs-trading signals ONLY on cointegrated pairs (stationary spread).
    Uses the cointegration beta to weight the spread instead of raw 1:1 log ratio."""
    sigs = []
    # Filter to only cointegrated pairs, then take top 8 by |corr|
    coint_pairs = [p for p in pairs if p.get("cointegrated")]
    candidates = coint_pairs[:8] if coint_pairs else []
    for p in candidates:
        t1, t2 = p["t1"], p["t2"]
        if t1 not in CACHE or t2 not in CACHE:
            continue
        c1 = [r["c"] for r in CACHE[t1]["data"]["data"]]
        c2 = [r["c"] for r in CACHE[t2]["data"]["data"]]
        n = min(len(c1), len(c2)); c1 = c1[-n:]; c2 = c2[-n:]
        beta = p.get("cointBeta") or 1.0
        # Cointegrated spread: log(c1) - beta*log(c2)
        spread = [math.log(c1[i]) - beta*math.log(c2[i])
                  for i in range(n) if c1[i] > 0 and c2[i] > 0]
        if len(spread) < 25:
            continue
        w = 20; s = spread[-w:]; mean = sum(s)/w
        sd = math.sqrt(sum((x-mean)**2 for x in s)/w)
        if sd < 1e-8:
            continue
        z = (spread[-1]-mean)/sd
        if abs(z) < 1.5:
            continue
        direction = f"BUY {t1} / SELL {t2}" if z < 0 else f"BUY {t2} / SELL {t1}"
        # Historical mean-reversion test on the cointegrated spread
        revs = tests = 0
        for i in range(w, len(spread)-3):
            sw = spread[i-w:i]; sm = sum(sw)/w
            ssd = math.sqrt(sum((x-sm)**2 for x in sw)/w)
            if ssd < 1e-8:
                continue
            sz = (spread[i]-sm)/ssd
            if abs(sz) < 1.2:
                continue
            tests += 1
            future = spread[i+1:i+4]
            if   sz < 0 and any(s2 > sm for s2 in future): revs += 1
            elif sz > 0 and any(s2 < sm for s2 in future): revs += 1
        wr = round(revs/tests*100) if tests >= 5 else 50
        plain = (f"{t1} is unusually cheap vs {t2} right now" if z < 0
                 else f"{t2} is unusually cheap vs {t1} right now")
        plain += f" — cointegrated pair (ADF={p.get('cointAdf')}), so spread reliably snaps back"
        sigs.append({"t1": t1, "t2": t2, "z": round(z, 2),
                     "direction": direction, "winRate": wr,
                     "tests": tests, "plain": plain,
                     "cointAdf": p.get("cointAdf"),
                     "cointBeta": beta})
    return sigs
