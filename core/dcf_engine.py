"""Discounted Cash Flow engine.

- Two-stage DCF with terminal value (Gordon growth).
- WACC from CAPM (β) + weighted cost of debt with tax shield.
- Monte-Carlo DCF: 5000 sims across g, gT, WACC drawn from truncated normals
  whose priors come from historical FCF growth + volatility.
- Reverse DCF: bisection for the market-implied growth rate.
- Walk-forward historical backtest over available annual FCF snapshots.

No charts. All numerical.
"""
import math, random, statistics
from core.config import (RISK_FREE, MKT_PREMIUM, CORP_TAX, DEFAULT_COST_DEBT)

# ─── Sloan accrual quality ─────────────────────────────────────────────────────
def accrual_quality(fund):
    """Sloan (1996) accrual anomaly: accruals = NI - OCF. Normalized by total
    assets (preferred) or revenue as fallback. Ratio > 0.10 is the "high
    accruals" red zone; those firms have historically underperformed by ~10pp.

    Returns {ratio, denom, trend, flag, years} or None when data is too thin.
    """
    ni_hist = fund.get("niHistory") or []
    ocf_hist = fund.get("ocfHistory") or []
    assets_hist = fund.get("totalAssetsHistory") or []
    rev_hist = fund.get("revHistory") or []
    if not ni_hist or not ocf_hist:
        return None
    n = min(len(ni_hist), len(ocf_hist))
    if n < 2:
        return None
    # Accrual ratio per year, most recent first
    ratios = []
    for i in range(min(n, 4)):
        ni = ni_hist[i]; ocf = ocf_hist[i]
        if ni is None or ocf is None:
            continue
        accr = ni - ocf
        denom = None
        if i < len(assets_hist) and assets_hist[i] and assets_hist[i] > 0:
            denom = assets_hist[i]
            scale = "assets"
        elif i < len(rev_hist) and rev_hist[i] and rev_hist[i] > 0:
            denom = rev_hist[i]
            scale = "revenue"
        if not denom:
            continue
        ratios.append({"ratio": accr / denom, "scale": scale, "accruals": accr, "denom": denom})
    if not ratios:
        return None
    avg_ratio = sum(r["ratio"] for r in ratios) / len(ratios)
    # Sloan flag: high-accrual red zone. Stricter on assets (0.10) than revenue (0.20).
    scale = ratios[0]["scale"]
    hi = 0.10 if scale == "assets" else 0.20
    lo = -0.10 if scale == "assets" else -0.20
    if avg_ratio > hi:
        flag = "high"    # earnings flattering cash flow — warning
    elif avg_ratio < lo:
        flag = "low"     # cash > earnings, conservative bookkeeping — bullish
    else:
        flag = "ok"
    # Trend: positive = accruals rising (worse)
    trend = None
    if len(ratios) >= 2:
        trend = ratios[0]["ratio"] - ratios[-1]["ratio"]
    return {
        "ratio": round(avg_ratio, 3),
        "scale": scale,
        "years": len(ratios),
        "flag": flag,
        "trend": round(trend, 3) if trend is not None else None,
        "threshold": hi,
    }

# ─── CAPM / WACC ───────────────────────────────────────────────────────────────
def cost_of_equity(beta, rf=RISK_FREE, mkt_prem=MKT_PREMIUM):
    return rf + (beta or 1.0) * mkt_prem

def wacc(fund, cost_debt=DEFAULT_COST_DEBT, tax=CORP_TAX,
         rf=RISK_FREE, mkt_prem=MKT_PREMIUM):
    """WACC from capital structure. Falls back to 80/20 if debt/equity unknown."""
    beta = fund.get("beta") if fund else 1.0
    re = cost_of_equity(beta or 1.0, rf, mkt_prem)
    # equity weight
    mc = fund.get("marketCap") if fund else None
    td = fund.get("totalDebt") if fund else None
    if mc and td and (mc + td) > 0:
        we = mc / (mc + td)
        wd = td / (mc + td)
    else:
        we, wd = 0.80, 0.20
    return we * re + wd * cost_debt * (1 - tax)

# ─── growth prior from history ─────────────────────────────────────────────────
def fcf_growth_prior(fcf_hist):
    """Returns (mean_growth, sigma) from log-returns of historical FCF.
    Handles negative/missing values by fall-through to conservative defaults."""
    clean = [f for f in (fcf_hist or []) if f is not None]
    positives = [f for f in clean if f > 0]
    if len(positives) < 2:
        return 0.05, 0.20  # conservative prior
    chron = list(reversed(positives))  # oldest first
    grs = []
    for i in range(1, len(chron)):
        if chron[i-1] > 0 and chron[i] > 0:
            grs.append(math.log(chron[i] / chron[i-1]))
    if not grs:
        return 0.05, 0.20
    m = sum(grs) / len(grs)
    v = sum((g - m) ** 2 for g in grs) / max(len(grs) - 1, 1)
    g = max(-0.20, min(0.50, math.exp(m) - 1))
    sigma = max(0.05, min(0.50, math.sqrt(v)))
    return g, sigma

# ─── two-stage DCF ─────────────────────────────────────────────────────────────
def two_stage_ev(fcf0, g_near, g_far, g_term, wacc_v, n_near=5, n_far=5):
    """Enterprise value via 2-stage DCF. Guards WACC > g_term."""
    if wacc_v <= g_term + 0.005:
        wacc_v = g_term + 0.015
    pv = 0.0
    f = fcf0
    # Stage 1: near-term growth
    for t in range(1, n_near + 1):
        f *= (1 + g_near)
        pv += f / (1 + wacc_v) ** t
    # Stage 2: fade toward terminal
    for t in range(n_near + 1, n_near + n_far + 1):
        f *= (1 + g_far)
        pv += f / (1 + wacc_v) ** t
    # Terminal value (Gordon growth at end of stage 2)
    tv = f * (1 + g_term) / (wacc_v - g_term)
    pv_tv = tv / (1 + wacc_v) ** (n_near + n_far)
    return pv + pv_tv

def dcf_per_share(fund, g_near, g_far, g_term, wacc_v, n_near=5, n_far=5):
    fcf0 = fund.get("fcfTTM")
    shares = fund.get("sharesOut")
    if not fcf0 or not shares or shares <= 0:
        return None
    ev = two_stage_ev(fcf0, g_near, g_far, g_term, wacc_v, n_near, n_far)
    equity = ev - (fund.get("netDebt") or 0)
    return equity / shares

# ─── Monte Carlo DCF ───────────────────────────────────────────────────────────
def _truncated_normal(mean, sigma, lo, hi):
    for _ in range(40):
        x = random.gauss(mean, sigma)
        if lo <= x <= hi:
            return x
    return max(lo, min(hi, mean))

def monte_carlo_dcf(fund, params=None, n_sims=4000):
    """Returns distribution of per-share intrinsic values."""
    if not fund or not fund.get("fcfTTM") or not fund.get("sharesOut"):
        return None
    fcf_hist = fund.get("fcfHistory") or []
    g_prior, g_sigma = fcf_growth_prior(fcf_hist)
    base_wacc = wacc(fund)

    p = params or {}
    g_mu   = p.get("gNear", g_prior)
    g_s    = p.get("gNearSigma", g_sigma)
    gF_mu  = p.get("gFar", g_prior * 0.5)
    gF_s   = p.get("gFarSigma", g_sigma * 0.7)
    gT_mu  = p.get("gTerm", 0.025)
    gT_s   = p.get("gTermSigma", 0.005)
    wacc_mu= p.get("wacc", base_wacc)
    wacc_s = p.get("waccSigma", 0.010)
    n_near = int(p.get("nNear", 5))
    n_far  = int(p.get("nFar", 5))

    ivs = []
    for _ in range(n_sims):
        g_n = _truncated_normal(g_mu,  g_s,  -0.20, 0.50)
        g_f = _truncated_normal(gF_mu, gF_s, -0.10, 0.20)
        g_t = _truncated_normal(gT_mu, gT_s,  0.000, 0.045)
        w   = _truncated_normal(wacc_mu, wacc_s, 0.04, 0.25)
        iv = dcf_per_share(fund, g_n, g_f, g_t, w, n_near, n_far)
        if iv is not None and iv > 0 and iv < 1e7:  # sanity cap
            ivs.append(iv)

    if not ivs:
        return None
    ivs.sort()
    def pct(a, q): return a[min(int(q/100*len(a)), len(a)-1)]
    return {
        "p5":  pct(ivs, 5),  "p25": pct(ivs, 25),
        "p50": pct(ivs, 50), "p75": pct(ivs, 75),
        "p95": pct(ivs, 95),
        "mean": sum(ivs)/len(ivs),
        "std":  math.sqrt(sum((x - sum(ivs)/len(ivs))**2 for x in ivs)/len(ivs)),
        "n":    len(ivs),
        "gPrior":   g_prior,
        "gSigma":   g_sigma,
        "waccBase": base_wacc,
    }

# ─── reverse DCF (implied growth) ──────────────────────────────────────────────
def reverse_dcf(fund, price, g_term=0.025, wacc_override=None, n_near=5, n_far=5):
    """Bisection solve: what g_near makes IV per share = price?"""
    shares = fund.get("sharesOut")
    if not shares or shares <= 0 or not fund.get("fcfTTM"):
        return None
    w = wacc_override if wacc_override is not None else wacc(fund)
    target_equity = price * shares
    target_ev = target_equity + (fund.get("netDebt") or 0)

    def ev_at(g):
        return two_stage_ev(fund["fcfTTM"], g, g * 0.5, g_term, w, n_near, n_far)

    lo, hi = -0.40, 0.80
    # sanity: EV monotonic in g near this range; otherwise clamp
    if ev_at(lo) > target_ev:
        return lo
    if ev_at(hi) < target_ev:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if ev_at(mid) < target_ev:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ─── main analyze ──────────────────────────────────────────────────────────────
def analyze_dcf(fund, params=None):
    """Produce DCF signal + full breakdown for a ticker."""
    if not fund:
        return None
    price = fund.get("currentPrice")
    if not price or not fund.get("fcfTTM") or not fund.get("sharesOut") or fund["fcfTTM"] <= 0:
        return {"ticker": fund.get("ticker"), "name": fund.get("name"),
                "ok": False, "price": price, "error": "Insufficient fundamentals (need positive FCF + shares)",
                "sig": "unknown", "plain": "Can't value this company with DCF — fundamentals missing or FCF is negative."}

    mc = monte_carlo_dcf(fund, params)
    if not mc:
        return {"ticker": fund.get("ticker"), "name": fund.get("name"),
                "ok": False, "price": price, "error": "Monte Carlo DCF failed",
                "sig": "unknown", "plain": "DCF computation failed."}

    iv_p25 = mc["p25"]; iv_med = mc["p50"]; iv_p75 = mc["p75"]
    mos_cons = (iv_p25 - price) / price * 100   # conservative (uses p25)
    mos_med  = (iv_med - price) / price * 100
    mos_opt  = (iv_p75 - price) / price * 100

    margin_th = (params or {}).get("marginThreshold", 20)
    if   mos_cons >=  margin_th * 2: sig = "strong-buy"
    elif mos_cons >=  margin_th:     sig = "buy"
    elif mos_cons <= -margin_th * 2: sig = "strong-sell"
    elif mos_cons <= -margin_th:     sig = "sell"
    else:                            sig = "fair"

    # ── Sloan accrual anomaly: high accruals → downgrade buy signals
    aq = accrual_quality(fund)
    if aq:
        if aq["flag"] == "high" and sig in ("strong-buy", "buy"):
            sig = "buy" if sig == "strong-buy" else "fair"
        elif aq["flag"] == "high" and sig == "fair":
            # push toward sell — earnings quality is suspect
            sig = "sell" if mos_cons < 0 else "fair"
        elif aq["flag"] == "low" and sig == "fair":
            # cash-backed earnings with slight undervaluation — upgrade
            if mos_cons > 0:
                sig = "buy"

    impl_g = reverse_dcf(fund, price)
    g_prior = mc["gPrior"]
    g_sigma = mc["gSigma"]

    # How unrealistic is implied growth vs historical?
    if impl_g is not None and g_sigma > 0:
        z_impl = (impl_g - g_prior) / g_sigma
    else:
        z_impl = 0

    # Probability IV > price (rough, from MC)
    # using normal approx on IV distribution
    iv_mean = mc["mean"]; iv_std = mc["std"] or 1.0
    # prob undervalued: P(IV > price) = 1 - Phi((price - mean)/std)
    def _phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    prob_under = 1 - _phi((price - iv_mean) / iv_std) if iv_std > 0 else 0.5

    direction = "LONG" if sig in ("strong-buy","buy") else "SHORT" if sig in ("strong-sell","sell") else "WAIT"

    # plain language
    if sig == "strong-buy":
        plain = (f"Significantly undervalued — conservative DCF ({iv_p25:.2f}) shows ~{mos_cons:.0f}% upside "
                 f"even under pessimistic assumptions. Market pricing in just {impl_g*100:.0f}% growth vs "
                 f"historical {g_prior*100:.0f}%.")
    elif sig == "buy":
        plain = (f"Undervalued — DCF median fair value ${iv_med:.2f} implies {mos_med:.0f}% upside. "
                 f"Market pricing in {impl_g*100:.0f}% growth.")
    elif sig == "fair":
        plain = (f"Roughly fair — DCF range ${iv_p25:.2f}–${iv_p75:.2f} brackets current price. "
                 f"Market pricing in {impl_g*100:.0f}% growth vs historical {g_prior*100:.0f}%.")
    elif sig == "sell":
        plain = (f"Overvalued by DCF — market demands {impl_g*100:.0f}% growth ({z_impl:+.1f}σ vs historical). "
                 f"Even the optimistic p75 (${iv_p75:.2f}) is {abs(mos_opt):.0f}% below price.")
    else:
        plain = (f"Significantly overvalued — market demands {impl_g*100:.0f}% growth "
                 f"({z_impl:+.1f}σ vs historical) which is historically implausible.")

    # Reasons list — mirrors MR reasons style
    reasons = []
    reasons.append(f"FCF TTM: ${fund['fcfTTM']/1e9:.2f}B; shares out: {fund['sharesOut']/1e6:.1f}M")
    reasons.append(f"Historical FCF growth: {g_prior*100:+.1f}% ± {g_sigma*100:.1f}%")
    reasons.append(f"WACC (CAPM, β={fund.get('beta') or 1:.2f}): {mc['waccBase']*100:.2f}%")
    reasons.append(f"DCF IV range (Monte Carlo, {mc['n']} sims): ${iv_p25:.2f} – ${iv_med:.2f} – ${iv_p75:.2f}")
    reasons.append(f"Implied growth from current price: {impl_g*100:+.1f}% ({z_impl:+.1f}σ vs history)")
    if z_impl > 1.5:
        reasons.append("⚠ Market pricing in growth > 1.5σ above historical — heroic assumptions required")
    if z_impl < -0.5 and sig in ("buy", "strong-buy"):
        reasons.append(f"Market pricing in growth {abs(z_impl):.1f}σ BELOW historical — conservative")
    if fund.get("profitMargin"):
        reasons.append(f"Profit margin: {fund['profitMargin']*100:.1f}%")
    if fund.get("returnOnEquity"):
        reasons.append(f"Return on equity: {fund['returnOnEquity']*100:.1f}%")
    if fund.get("debtToEquity") and fund["debtToEquity"] > 150:
        reasons.append(f"⚠ High debt/equity: {fund['debtToEquity']:.0f} — leverage risk amplifies DCF sensitivity")
    if aq:
        if aq["flag"] == "high":
            reasons.append(f"⚠ Sloan accrual ratio {aq['ratio']*100:+.1f}% of {aq['scale']} over {aq['years']}y "
                           f"— earnings > cash flow, quality suspect (signal downgraded)")
        elif aq["flag"] == "low":
            reasons.append(f"✓ Accruals {aq['ratio']*100:+.1f}% of {aq['scale']} — conservative bookkeeping, "
                           f"cash flow backing earnings")

    return {
        "ticker": fund.get("ticker"),
        "name": fund.get("name"),
        "ok": True,
        "sig": sig,
        "direction": direction,
        "price": round(price, 2),
        "ivP5":   round(mc["p5"], 2),
        "ivP25":  round(iv_p25, 2),
        "ivMed":  round(iv_med, 2),
        "ivP75":  round(iv_p75, 2),
        "ivP95":  round(mc["p95"], 2),
        "ivMean": round(iv_mean, 2),
        "ivStd":  round(iv_std, 2),
        "mosCons": round(mos_cons, 1),
        "mosMed":  round(mos_med,  1),
        "mosOpt":  round(mos_opt,  1),
        "impliedGrowth": round(impl_g*100, 2) if impl_g is not None else None,
        "impliedZ":      round(z_impl, 2),
        "probUndervalued": round(prob_under, 3),
        "wacc":    round(mc["waccBase"]*100, 2),
        "gPrior":  round(g_prior*100, 2),
        "gSigma":  round(g_sigma*100, 2),
        "fcfTTM":  fund.get("fcfTTM"),
        "sharesOut": fund.get("sharesOut"),
        "netDebt": fund.get("netDebt"),
        "beta":    fund.get("beta"),
        "marketCap": fund.get("marketCap"),
        "params":  {"gNear": (params or {}).get("gNear", g_prior),
                    "gFar":  (params or {}).get("gFar",  g_prior*0.5),
                    "gTerm": (params or {}).get("gTerm", 0.025),
                    "wacc":  (params or {}).get("wacc",  mc["waccBase"]),
                    "nNear": int((params or {}).get("nNear", 5)),
                    "nFar":  int((params or {}).get("nFar",  5)),
                    "marginThreshold": margin_th},
        "plain": plain,
        "reasons": reasons,
        "accrualQuality": aq,
    }

# ─── historical backtest ───────────────────────────────────────────────────────
def dcf_backtest(fund, price_data, params=None):
    """Walk-forward on annual FCF snapshots.
    For each historical FCF entry (most recent → oldest), reconstruct a DCF
    using THAT year's FCF as fcf0 and price at that time, then check 12-mo
    forward return. Small sample but mathematically honest."""
    fcf_hist = fund.get("fcfHistory") or []
    if len(fcf_hist) < 2 or not price_data or len(price_data) < 252:
        return {"trades": 0, "winRate": 50, "avgReturn": 0, "spearman": 0,
                "method": "insufficient"}

    rows = []
    n = len(price_data)
    for i, fcf in enumerate(fcf_hist):
        if i + 1 >= len(fcf_hist):
            break
        if not fcf or fcf <= 0:
            continue
        years_ago = i + 1
        idx_then = n - 1 - years_ago * 252
        idx_fwd  = idx_then + 252
        if idx_then < 0 or idx_fwd >= n:
            continue
        price_then = price_data[idx_then]["c"]
        price_fwd  = price_data[idx_fwd]["c"]
        snap = dict(fund)
        snap["fcfTTM"] = fcf
        snap["fcfHistory"] = fcf_hist[i:]
        snap["currentPrice"] = price_then
        r = analyze_dcf(snap, params)
        if not r or not r.get("ok"):
            continue
        fwd_ret = (price_fwd - price_then) / price_then * 100
        d = 1 if r["sig"] in ("strong-buy", "buy") else (-1 if r["sig"] in ("strong-sell","sell") else 0)
        if d == 0:
            # record for rank-correlation even if no trade
            rows.append({"mos": r["mosCons"], "fwd": fwd_ret, "trade": 0, "ret": 0})
            continue
        trade_ret = fwd_ret * d
        rows.append({"mos": r["mosCons"], "fwd": fwd_ret, "trade": d, "ret": trade_ret})

    if not rows:
        return {"trades": 0, "winRate": 50, "avgReturn": 0, "spearman": 0,
                "method": "no_signals"}

    # Spearman rank correlation of MoS vs forward return (signal quality)
    def _rank(arr):
        s = sorted(range(len(arr)), key=lambda i: arr[i])
        r = [0]*len(arr)
        for pos, idx in enumerate(s):
            r[idx] = pos
        return r
    mos = [x["mos"] for x in rows]; fwd = [x["fwd"] for x in rows]
    rm = _rank(mos); rf = _rank(fwd)
    nn = len(rm)
    if nn > 1:
        mm = sum(rm)/nn; mf = sum(rf)/nn
        num = sum((rm[i]-mm)*(rf[i]-mf) for i in range(nn))
        da = math.sqrt(sum((r-mm)**2 for r in rm))
        db = math.sqrt(sum((r-mf)**2 for r in rf))
        spearman = num/(da*db) if da*db > 0 else 0
    else:
        spearman = 0

    trades = [x for x in rows if x["trade"] != 0]
    if trades:
        wins = sum(1 for t in trades if t["ret"] > 0)
        wr = round(wins / len(trades) * 100)
        avg_ret = sum(t["ret"] for t in trades) / len(trades)
    else:
        wr = 50
        avg_ret = 0

    return {
        "trades": len(trades),
        "signals": len(rows),
        "winRate": wr,
        "avgReturn": round(avg_ret, 2),
        "spearman": round(spearman, 3),
        "method": "annual-fcf-walkforward",
    }
