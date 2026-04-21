"""Correlation and pair-trading analytics."""
import math

from core.data import CACHE


def log_ret(prices):
    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))
            if prices[i - 1] > 0 and prices[i] > 0]


def pearson(a, b):
    n = min(len(a), len(b))
    if n < 10:
        return 0
    a, b = a[-n:], b[-n:]
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return round(num / (da * db), 3) if da * db > 0 else 0


def _ols_slope_intercept(x, y):
    n = len(x)
    if n < 5:
        return 0.0, 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    if den == 0:
        return 0.0, my
    beta = num / den
    alpha = my - beta * mx
    return beta, alpha


def _adf_on_series(levels):
    if len(levels) < 15:
        return None
    y_lag = levels[:-1]
    dy = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    n = len(dy)
    mean_lag = sum(y_lag) / n
    mean_dy = sum(dy) / n
    num = sum((y_lag[i] - mean_lag) * (dy[i] - mean_dy) for i in range(n))
    den = sum((y_lag[i] - mean_lag) ** 2 for i in range(n))
    if den == 0:
        return None
    lam = num / den
    res = [dy[i] - lam * (y_lag[i] - mean_lag) - mean_dy for i in range(n)]
    se = math.sqrt(sum(r ** 2 for r in res) / (n - 2)) / math.sqrt(den) if n > 2 else 1
    return lam / se if se > 0 else None


def _ou_halflife_series(levels):
    if len(levels) < 20:
        return None
    lag = levels[:-1]
    diff = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    n = len(diff)
    ml = sum(lag) / n
    md = sum(diff) / n
    num = sum((lag[i] - ml) * (diff[i] - md) for i in range(n))
    den = sum((lag[i] - ml) ** 2 for i in range(n))
    if den == 0:
        return None
    lam = num / den
    if lam >= 0:
        return None
    return round(-math.log(2) / lam, 1)


def _align_rows(rows1, rows2):
    map1 = {r["date"]: r for r in rows1 if r.get("c")}
    map2 = {r["date"]: r for r in rows2 if r.get("c")}
    dates = sorted(set(map1).intersection(map2))
    aligned = []
    for dt in dates:
        p1 = map1[dt]["c"]
        p2 = map2[dt]["c"]
        if p1 and p2 and p1 > 0 and p2 > 0:
            aligned.append((dt, map1[dt], map2[dt]))
    return aligned


def _rolling_zscores(series, window=60):
    out = []
    for idx, value in enumerate(series):
        if idx < window:
            out.append(None)
            continue
        hist = series[idx - window:idx]
        mean = sum(hist) / len(hist)
        sd = math.sqrt(sum((x - mean) ** 2 for x in hist) / len(hist))
        out.append((value - mean) / sd if sd > 1e-12 else 0.0)
    return out


def _pair_core_metrics(rows1, rows2, lookback=220):
    aligned = _align_rows(rows1, rows2)
    if len(aligned) < 80:
        return None
    aligned = aligned[-lookback:]
    dates = [d for d, _, _ in aligned]
    p1 = [r1["c"] for _, r1, _ in aligned]
    p2 = [r2["c"] for _, _, r2 in aligned]
    lr1 = log_ret(p1)
    lr2 = log_ret(p2)
    corr = pearson(lr1, lr2)
    lp1 = [math.log(v) for v in p1]
    lp2 = [math.log(v) for v in p2]
    beta, alpha = _ols_slope_intercept(lp2, lp1)
    spread = [lp1[i] - (beta * lp2[i] + alpha) for i in range(len(lp1))]
    adf = _adf_on_series(spread)
    half_life = _ou_halflife_series(spread)
    z_hist = _rolling_zscores(spread, window=min(60, max(20, len(spread) // 3)))
    latest_z = next((z for z in reversed(z_hist) if z is not None), None)
    return {
        "dates": dates,
        "p1": p1,
        "p2": p2,
        "corr": corr,
        "beta": round(beta, 4),
        "alpha": alpha,
        "spread": spread,
        "zHist": z_hist,
        "latestZ": round(latest_z, 2) if latest_z is not None else None,
        "cointAdf": round(adf, 2) if adf is not None else None,
        "cointegrated": bool(adf is not None and adf < -2.86),
        "halfLife": half_life,
    }


def cointegration_test(px1, px2):
    metrics = _pair_core_metrics(
        [{"date": str(i), "c": v} for i, v in enumerate(px1)],
        [{"date": str(i), "c": v} for i, v in enumerate(px2)],
        lookback=min(len(px1), len(px2)),
    )
    if not metrics:
        return None, None, False
    return metrics["cointAdf"], metrics["beta"], metrics["cointegrated"]


def _pair_trade_return(p1_entry, p1_exit, p2_entry, p2_exit, beta, direction):
    leg = ((p1_exit / p1_entry) - 1.0) - beta * ((p2_exit / p2_entry) - 1.0)
    leg = leg if direction > 0 else -leg
    gross = 1.0 + abs(beta)
    return leg / gross * 100


def _pair_backtest(metrics, entry_z=1.8, exit_z=0.35, stop_z=3.2, max_hold=15):
    spread = metrics["spread"]
    z_hist = metrics["zHist"]
    p1 = metrics["p1"]
    p2 = metrics["p2"]
    beta = metrics["beta"]
    trades = []
    eq = 100.0
    peak = 100.0
    max_dd = 0.0
    i = 0
    while i < len(spread):
        z0 = z_hist[i]
        if z0 is None or abs(z0) < entry_z:
            i += 1
            continue
        direction = 1 if z0 < 0 else -1
        entry_idx = i
        exit_idx = min(len(spread) - 1, entry_idx + max_hold)
        reason = "max-hold"
        for j in range(entry_idx + 1, min(len(spread), entry_idx + max_hold + 1)):
            zj = z_hist[j]
            if zj is None:
                continue
            if abs(zj) <= exit_z:
                exit_idx = j
                reason = "mean"
                break
            if abs(zj) >= stop_z:
                exit_idx = j
                reason = "stop"
                break
        ret = _pair_trade_return(p1[entry_idx], p1[exit_idx], p2[entry_idx], p2[exit_idx], beta, direction)
        eq *= (1 + ret / 100)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak else 0
        if dd > max_dd:
            max_dd = dd
        trades.append({
            "entryIdx": entry_idx,
            "exitIdx": exit_idx,
            "entryZ": round(z0, 2),
            "exitZ": round(z_hist[exit_idx] or 0, 2),
            "direction": direction,
            "ret": round(ret, 3),
            "barsHeld": exit_idx - entry_idx,
            "reason": reason,
        })
        i = exit_idx + 1
    if not trades:
        return {"trades": 0, "winRate": 50, "avgReturn": 0.0, "avgHold": 0.0, "maxDD": 0.0, "pf": 1.0, "sharpe": 0.0, "history": []}
    wins = [t for t in trades if t["ret"] > 0]
    rets = [t["ret"] for t in trades]
    avg_ret = sum(rets) / len(rets)
    avg_hold = sum(t["barsHeld"] for t in trades) / len(trades)
    mean = avg_ret
    std = math.sqrt(sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1))
    sharpe = round(mean / std * math.sqrt(252 / max(avg_hold, 1)), 2) if std > 0 else 0.0
    gross_w = sum(t["ret"] for t in trades if t["ret"] > 0)
    gross_l = abs(sum(t["ret"] for t in trades if t["ret"] <= 0))
    return {
        "trades": len(trades),
        "winRate": round(len(wins) / len(trades) * 100, 1),
        "avgReturn": round(avg_ret, 3),
        "avgHold": round(avg_hold, 2),
        "maxDD": round(max_dd, 2),
        "pf": round(gross_w / max(gross_l, 0.01), 2),
        "sharpe": sharpe,
        "history": trades[-12:],
    }


def build_correlation(tickers):
    rows_map = {}
    for tk in tickers:
        if tk in CACHE:
            rows_map[tk] = CACHE[tk]["data"]["data"]
    if len(rows_map) < 2:
        return {}, []
    tks = list(rows_map.keys())
    matrix = {t1: {} for t1 in tks}
    pairs = []
    for i, t1 in enumerate(tks):
        matrix[t1][t1] = 1.0
        for j in range(i + 1, len(tks)):
            t2 = tks[j]
            metrics = _pair_core_metrics(rows_map[t1], rows_map[t2])
            corr = metrics["corr"] if metrics else 0
            matrix[t1][t2] = corr
            matrix[t2][t1] = corr
            if metrics and abs(corr) >= 0.65:
                pairs.append({
                    "t1": t1,
                    "t2": t2,
                    "corr": corr,
                    "cointAdf": metrics["cointAdf"],
                    "cointBeta": metrics["beta"],
                    "cointegrated": metrics["cointegrated"],
                    "latestZ": metrics["latestZ"],
                    "halfLife": metrics["halfLife"],
                })
    pairs.sort(key=lambda x: (-1 if x["cointegrated"] else 0, -abs(x["corr"]), -abs(x.get("latestZ") or 0)))
    return matrix, pairs


def spread_signals(pairs):
    sigs = []
    for p in [x for x in pairs if x.get("cointegrated")][:8]:
        raw1 = CACHE.get(p["t1"], {}).get("data", {}).get("data")
        raw2 = CACHE.get(p["t2"], {}).get("data", {}).get("data")
        if not raw1 or not raw2:
            continue
        metrics = _pair_core_metrics(raw1, raw2)
        if not metrics or metrics["latestZ"] is None or abs(metrics["latestZ"]) < 1.5:
            continue
        bt = _pair_backtest(metrics)
        z = metrics["latestZ"]
        direction = f"BUY {p['t1']} / SELL {p['t2']}" if z < 0 else f"BUY {p['t2']} / SELL {p['t1']}"
        plain = (f"{p['t1']} is unusually cheap vs {p['t2']} right now" if z < 0
                 else f"{p['t2']} is unusually cheap vs {p['t1']} right now")
        plain += f" — spread z-score is {z:+.2f}, ADF={metrics['cointAdf']}, half-life {metrics['halfLife'] or '?'}d."
        sigs.append({
            "t1": p["t1"],
            "t2": p["t2"],
            "z": z,
            "direction": direction,
            "winRate": bt["winRate"],
            "tests": bt["trades"],
            "plain": plain,
            "cointAdf": metrics["cointAdf"],
            "cointBeta": metrics["beta"],
            "halfLife": metrics["halfLife"],
            "avgReturn": bt["avgReturn"],
        })
    return sigs


def pair_workspace(t1, t2):
    raw1 = CACHE.get(t1, {}).get("data")
    raw2 = CACHE.get(t2, {}).get("data")
    if not raw1 or not raw2:
        return None
    metrics = _pair_core_metrics(raw1["data"], raw2["data"])
    if not metrics:
        return None
    bt = _pair_backtest(metrics)
    latest_z = metrics["latestZ"] or 0.0
    action = "Wait"
    if metrics["cointegrated"] and abs(latest_z) >= 1.8:
        action = f"Buy {t1} / Sell {t2}" if latest_z < 0 else f"Buy {t2} / Sell {t1}"
    plan = {
        "action": action,
        "entryZ": 1.8,
        "exitZ": 0.35,
        "stopZ": 3.2,
        "hedgeRatio": metrics["beta"],
        "latestZ": latest_z,
        "halfLife": metrics["halfLife"],
        "cointegrated": metrics["cointegrated"],
    }
    series_tail = 120
    return {
        "t1": t1,
        "t2": t2,
        "corr": metrics["corr"],
        "cointAdf": metrics["cointAdf"],
        "cointBeta": metrics["beta"],
        "cointegrated": metrics["cointegrated"],
        "halfLife": metrics["halfLife"],
        "latestZ": latest_z,
        "plan": plan,
        "backtest": bt,
        "series": {
            "dates": metrics["dates"][-series_tail:],
            "p1": metrics["p1"][-series_tail:],
            "p2": metrics["p2"][-series_tail:],
            "spread": [round(x, 5) for x in metrics["spread"][-series_tail:]],
            "z": [round(x, 3) if x is not None else None for x in metrics["zHist"][-series_tail:]],
        },
    }
