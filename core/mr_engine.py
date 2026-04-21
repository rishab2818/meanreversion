"""Mean-reversion engine — analyze + walk-forward backtest."""
import math
from core.indicators import (rsi, bb, z_score, atr, macd, stoch_rsi,
                             volume_ratio, market_regime, ou_halflife,
                             adf_stat, signal_age, gap_detect,
                             day_of_week_score, near_earnings,
                             kalman_mean, sector_rel_z)

def exec_cost_bps(data):
    """Estimate round-trip execution cost (bps) from 20-day dollar volume.
    Bid-ask + slippage rises sharply for illiquid names — a "50% win rate" on a
    micro-cap can be a losing strategy after costs.

    Rough tiers calibrated to retail-size fills:
      > $100M ADV  →  8 bps round-trip  (liquid mega-caps)
      > $10M  ADV  → 20 bps
      > $1M   ADV  → 40 bps
      <= $1M  ADV  → 80 bps  (illiquid; wide spreads + impact)
    """
    if not data or len(data) < 20:
        return 25.0
    recent = data[-20:]
    adv = sum((d["c"] * d["v"]) for d in recent) / 20
    if   adv > 100_000_000: return 8.0
    elif adv >  10_000_000: return 20.0
    elif adv >   1_000_000: return 40.0
    else:                   return 80.0

def ou_position_size(capital, z, half_life, win_rate, rr, calibration=1.0):
    """Scale position by Z-score distance and half-life reliability.

    `calibration` = realized_WR / predicted_WR from closed journal trades. Below
    1.0 = model is overconfident → shrink Kelly; above 1.0 = model conservative.
    Clamp is applied at the source (storage.calibration_factor)."""
    W = win_rate/100
    kelly = max(0, W - (1-W)/max(rr, 0.01)) * 0.25  # quarter-Kelly
    kelly *= max(0.3, min(1.3, calibration))
    z_scale = min(abs(z)/2.0, 1.5)
    hl_scale = 1.0
    if half_life and 5 <= half_life <= 15: hl_scale = 1.2
    elif half_life and 15 < half_life <= 30: hl_scale = 1.0
    elif half_life and half_life > 30: hl_scale = 0.8
    size = capital * kelly * z_scale * hl_scale
    return min(round(size, 2), capital * 0.15)

def _agg_trades(trades):
    if not trades:
        return {"winRate": 50, "avgReturn": 0.0, "trades": 0}
    wins = sum(1 for t in trades if t["win"])
    n = len(trades)
    wr = wins/n*100
    ar = sum(t["ret"] for t in trades)/n
    return {"winRate": round(wr, 1), "avgReturn": round(ar, 3), "trades": n}

def backtest(data, params, n_folds=4, purge=3):
    """Walk-forward backtest with purged k-fold stats + per-regime win rates.

    - Chronological walk-forward (never peek ahead).
    - Each fold block is separated by a `purge` gap to avoid overlap leak from
      the 5-bar forward holding window.
    - Fitness metrics reported as OOS MEDIAN across folds, not in-sample mean.
    - Also reports per-regime win rate for use by regime-gating in analyze().
    """
    closes = [d["c"] for d in data]
    rsi_p = params.get("rsiP", 14); bb_std = params.get("bbStd", 2.0)
    z_win = params.get("zWin", 20); vol_min = params.get("volMin", 1.0)
    warmup = max(rsi_p+5, z_win, 25, 30)
    if len(closes) < warmup + 20:
        return {"winRate":50,"avgReturn":0.0,"trades":0,"sharpe":0,
                "maxDD":0,"pf":1,"method":"insufficient",
                "medianFoldWR":50,"foldSpread":0,"regimeStats":{},"execCostBps":0}

    fold_sz = max(15, (len(closes)-warmup)//n_folds)
    trades = []
    fold_trades = {i: [] for i in range(n_folds)}
    # Round-trip execution cost applied to every trade return. This turns many
    # marginal "edges" on illiquid names into the losses they actually are.
    cost_pct = exec_cost_bps(data) / 100.0  # bps → %

    for fold in range(n_folds):
        ts = warmup + fold*(fold_sz + purge)
        te = min(ts + fold_sz, len(closes)-5)
        if ts >= te:
            break
        for i in range(ts, te):
            sl = closes[:i+1]; dl = data[:i+1]
            r_v = rsi(sl, rsi_p); z_v = z_score(sl, z_win); b_v = bb(sl, 20, bb_std)
            vr = volume_ratio(dl, 20)
            if r_v is None or z_v is None or b_v is None:
                continue
            if vr is not None and vr < vol_min:
                continue
            cur = closes[i]; gp = gap_detect(dl)
            if gp < -2.5 and r_v < 45:
                continue
            direction = 0
            if r_v < params.get("rsiOS", 35) and z_v < -params.get("zThresh", 1.5) and cur < b_v["lower"]:
                direction = 1
            elif r_v > params.get("rsiOB", 65) and z_v > params.get("zThresh", 1.5) and cur > b_v["upper"]:
                direction = -1
            if direction == 0:
                continue
            reg = market_regime(sl)
            if direction == 1 and reg in ("strong_down","weak_down"):
                continue
            if direction == -1 and reg in ("strong_up","weak_up"):
                continue
            atr_v = atr(dl, 14) or cur*0.02
            atr_m = 1.2 if "rang" in reg else 2.0
            stop = cur - atr_v*atr_m if direction == 1 else cur + atr_v*atr_m
            target = b_v["middle"]; future = closes[i+1:i+6]
            if not future:
                continue
            ret = 0.0
            for f in future:
                if direction == 1:
                    if f >= target: ret = (f-cur)/cur*100; break
                    if f <= stop:   ret = (f-cur)/cur*100; break
                else:
                    if f <= target: ret = (cur-f)/cur*100; break
                    if f >= stop:   ret = (cur-f)/cur*100; break
            else:
                f = future[-1]
                ret = ((f-cur)/cur*100) if direction == 1 else ((cur-f)/cur*100)
            ret_net = ret - cost_pct   # subtract round-trip cost
            tr = {"ret": ret_net, "retGross": ret, "win": ret_net > 0, "fold": fold,
                  "regime": reg, "direction": "LONG" if direction == 1 else "SHORT"}
            trades.append(tr)
            fold_trades[fold].append(tr)

    if not trades:
        return {"winRate":50,"avgReturn":0.0,"trades":0,"sharpe":0,
                "maxDD":0,"pf":1,"method":"no_signals",
                "medianFoldWR":50,"foldSpread":0,"regimeStats":{},"execCostBps":0}

    # Overall (in-sample-ish) stats
    wins = sum(1 for t in trades if t["win"]); n = len(trades)
    wr_raw = wins/n*100
    ar = sum(t["ret"] for t in trades)/n
    rets = [t["ret"] for t in trades]
    std_r = math.sqrt(sum((r-ar)**2 for r in rets)/max(n-1, 1))
    sharpe = round(ar/std_r*math.sqrt(52), 2) if std_r > 0 else 0
    gross_w = sum(t["ret"] for t in trades if t["win"])
    gross_l = abs(sum(t["ret"] for t in trades if not t["win"]))
    pf = round(gross_w/max(gross_l, 0.01), 2)
    eq = 100.0; pk = 100.0; mdd = 0.0
    for t in trades:
        eq *= (1 + t["ret"]/100)
        if eq > pk: pk = eq
        dd = (pk-eq)/pk*100
        if dd > mdd: mdd = dd

    # Per-fold OOS stats for purged CV: take MEDIAN fold WR, penalize spread
    fold_wrs = []
    for fold, ft in fold_trades.items():
        if len(ft) >= 2:
            fw = sum(1 for t in ft if t["win"])/len(ft)*100
            fold_wrs.append(fw)
    if fold_wrs:
        fold_wrs_sorted = sorted(fold_wrs)
        m = len(fold_wrs_sorted)//2
        median_fold_wr = fold_wrs_sorted[m] if len(fold_wrs_sorted) % 2 else \
                         (fold_wrs_sorted[m-1]+fold_wrs_sorted[m])/2
        fold_spread = max(fold_wrs) - min(fold_wrs)
    else:
        median_fold_wr = wr_raw
        fold_spread = 0

    # Conservative reported win rate: shrink toward 50 using the pessimistic
    # of (overall, median-fold). This is what gets shown to the user.
    wr_reported = min(wr_raw, median_fold_wr)
    # Small honesty-penalty when folds disagree wildly (overfitting symptom)
    if fold_spread > 30:
        wr_reported -= (fold_spread - 30) * 0.3
    wr_bounded = min(85, max(35, round(wr_reported)))

    # Per-regime breakdown
    regime_stats = {}
    for reg in ("ranging", "strong_up", "weak_up", "strong_down", "weak_down", "unknown"):
        for drc in ("LONG", "SHORT"):
            key = f"{reg}_{drc}"
            bucket = [t for t in trades if t["regime"] == reg and t["direction"] == drc]
            if len(bucket) >= 2:
                bw = sum(1 for t in bucket if t["win"]) / len(bucket) * 100
                br = sum(t["ret"] for t in bucket) / len(bucket)
                regime_stats[key] = {"wr": round(bw, 1), "avgRet": round(br, 2), "n": len(bucket)}

    return {"winRate": wr_bounded,
            "avgReturn": round(ar, 3), "trades": n,
            "sharpe": sharpe, "maxDD": round(mdd, 1), "pf": pf,
            "method": "purged-walk-forward",
            "medianFoldWR": round(median_fold_wr, 1),
            "foldSpread": round(fold_spread, 1),
            "regimeStats": regime_stats,
            "execCostBps": round(cost_pct * 100, 1)}

def analyze(raw, params, meta=None):
    """Produce a scanner row. `meta` may contain short interest / insider data
    pulled via data.fetch_quote_meta()."""
    closes = [d["c"] for d in raw["data"]]
    data = raw["data"]
    if len(closes) < 30:
        return None
    meta = meta or {}

    cur = closes[-1]
    rp = params.get("rsiP", 14); bs = params.get("bbStd", 2.0); zw = params.get("zWin", 20)

    r_v = rsi(closes, rp); b_v = bb(closes, 20, bs); z_v = z_score(closes, zw)
    atr_v = atr(data, 14)
    if any(v is None for v in [r_v, b_v, z_v, atr_v]):
        return None

    # Adaptive mean via Kalman filter — responsive to regime shifts
    k_mean, z_kalman = kalman_mean(closes)
    # Sector relative strength (if ETF closes passed through meta)
    etf_closes = meta.get("etfCloses") if meta else None
    sector_z = sector_rel_z(closes, etf_closes) if etf_closes else None

    ma20 = sum(closes[-20:])/20 if len(closes) >= 20 else cur
    ma50 = sum(closes[-50:])/50 if len(closes) >= 50 else cur
    m_v = macd(closes); sk, sd_ = stoch_rsi(closes, rp)
    vr = volume_ratio(data, 20); reg = market_regime(closes)
    hl = ou_halflife(closes); adf = adf_stat(closes)
    gap = gap_detect(data); dow_score, dow_name = day_of_week_score(data)
    earn_near, earn_days = near_earnings(raw)

    bt = backtest(data, params)

    score = 0.0; reasons = []

    if   r_v < 28: score += 3.0; reasons.append(f"RSI {r_v:.0f} — very oversold (buyers often take over here)")
    elif r_v < 38: score += 2.0; reasons.append(f"RSI {r_v:.0f} — oversold")
    elif r_v < 45: score += 1.0; reasons.append(f"RSI {r_v:.0f} — mildly oversold")
    elif r_v > 72: score -= 3.0; reasons.append(f"RSI {r_v:.0f} — very overbought (sellers often dominate)")
    elif r_v > 62: score -= 2.0; reasons.append(f"RSI {r_v:.0f} — overbought")

    if   z_v < -2.5: score += 3.0; reasons.append(f"Z-Score {z_v:.2f} — price extremely far below average")
    elif z_v < -2.0: score += 2.5; reasons.append(f"Z-Score {z_v:.2f} — price well below average")
    elif z_v < -1.5: score += 1.5; reasons.append(f"Z-Score {z_v:.2f} — price below average")
    elif z_v < -1.0: score += 0.8
    elif z_v >  2.5: score -= 3.0; reasons.append(f"Z-Score {z_v:.2f} — price extremely far above average")
    elif z_v >  2.0: score -= 2.5
    elif z_v >  1.5: score -= 1.5

    # Kalman z-score should AGREE with rolling z for a high-quality MR setup.
    # If they disagree strongly, the "dislocation" is actually a regime shift —
    # the rolling mean is stale but the adaptive mean has already moved with it.
    if z_kalman is not None:
        if z_v < -1.5 and z_kalman > -0.5:
            score -= 1.0
            reasons.append(f"⚠ Kalman z {z_kalman:+.2f} ≫ rolling z {z_v:+.2f} — adaptive mean has shifted, dislocation may be regime change not overshoot")
        elif z_v > 1.5 and z_kalman < 0.5:
            score += 1.0
            reasons.append(f"⚠ Kalman z {z_kalman:+.2f} ≪ rolling z {z_v:+.2f} — adaptive mean has shifted upward, signal weaker than it appears")
        elif abs(z_v) > 1.5 and ((z_v < 0) == (z_kalman < 0)) and abs(z_kalman) > 1.0:
            # both methods agree the stock is genuinely dislocated
            score += 0.5 if z_v < 0 else -0.5
            reasons.append(f"✓ Rolling z {z_v:+.2f} and Kalman z {z_kalman:+.2f} both agree — genuine dislocation")

    bb_pct = ((cur-b_v["lower"])/(b_v["upper"]-b_v["lower"])*100) if b_v["std"] > 0 else 50
    bb_pos = "below" if cur < b_v["lower"] else "above" if cur > b_v["upper"] else "inside"
    if   cur < b_v["lower"]*0.98: score += 2.0; reasons.append("Price below lower band — statistically stretched too far down")
    elif cur < b_v["lower"]:      score += 1.5; reasons.append("Price touching lower band")
    elif cur > b_v["upper"]*1.02: score -= 2.0; reasons.append("Price above upper band — statistically stretched too far up")
    elif cur > b_v["upper"]:      score -= 1.5

    if sk is not None:
        if   sk < 25 and sd_ and sk > sd_: score += 1.5; reasons.append("StochRSI turning up — reversal may have started")
        elif sk < 20:                       score += 0.8
        elif sk > 75 and sd_ and sk < sd_: score -= 1.5; reasons.append("StochRSI turning down — momentum fading")
        elif sk > 80:                       score -= 0.8

    if m_v:
        if   m_v.get("crossed_up"): score += 1.0; reasons.append("MACD just crossed up — momentum confirming")
        elif m_v["hist"] > 0:       score += 0.4
        elif m_v["hist"] < 0:       score -= 0.4

    if vr is not None:
        if   vr >= 1.5: score += 1.0; reasons.append(f"Volume {vr:.1f}x average — big players moving")
        elif vr >= 1.2: score += 0.5
        elif vr <  0.5: score -= 1.0; reasons.append("Very low volume — weak signal")

    if   hl and 5 <= hl <= 20: score += 0.8; reasons.append(f"OU half-life {hl}d — this stock reliably snaps back fast")
    elif hl and 20 < hl <= 35: score += 0.3
    elif not hl:                score -= 0.5; reasons.append("No mean-reversion detected mathematically — risky trade")

    if adf and adf < -2.5: score += 0.5; reasons.append("ADF test confirms price is stationary — genuine MR asset")

    if score > 0:
        if   reg == "strong_down": score -= 2.5; reasons.append("⚠ Market in strong downtrend — mean reversion risky")
        elif reg == "weak_down":   score -= 1.2; reasons.append("⚠ Weak downtrend — use tighter stop")
    elif score < 0:
        if reg == "strong_up": score += 2.5; reasons.append("⚠ Market in strong uptrend — short is risky")

    # ── Regime-gating: attenuate score by how well this direction worked
    # historically in THIS regime (per-regime conditional win rate).
    tentative_dir = "LONG" if score > 0 else ("SHORT" if score < 0 else None)
    regime_wr = None; regime_n = 0
    if tentative_dir:
        rs = bt.get("regimeStats", {}) or {}
        bucket = rs.get(f"{reg}_{tentative_dir}")
        if bucket and bucket.get("n", 0) >= 3:
            regime_wr = bucket["wr"]; regime_n = bucket["n"]
            # Shrink score toward 0 when this regime historically kills this direction
            if regime_wr < 40:
                score *= 0.5
                reasons.append(f"⚠ Regime-conditional win rate only {regime_wr:.0f}% in {reg.replace('_',' ')} ({regime_n} hist trades) — edge penalized")
            elif regime_wr < 50:
                score *= 0.8
                reasons.append(f"⚠ Regime-conditional win rate {regime_wr:.0f}% — weaker than blended")
            elif regime_wr >= 65:
                score *= 1.15
                reasons.append(f"✓ Regime-conditional win rate {regime_wr:.0f}% in {reg.replace('_',' ')} — this direction works here historically")

    # ── Sector relative strength: is this a leader or a broken name?
    # Suggestion #11: "A stock up while its sector is down = leader; a stock
    # down while its sector is up = broken."
    if sector_z is not None:
        if score > 0 and sector_z < -1.5:
            # long signal on name that has been badly underperforming sector
            score *= 0.75
            reasons.append(f"⚠ Sector-relative z {sector_z:+.2f} — this name is broken vs its sector (long edge reduced)")
        elif score > 0 and sector_z > 1.0:
            score *= 1.10
            reasons.append(f"✓ Sector-relative z {sector_z:+.2f} — leader in its sector, oversold leaders revert fastest")
        elif score < 0 and sector_z > 1.5:
            # short signal on name outperforming sector = fighting strength
            score *= 0.75
            reasons.append(f"⚠ Sector-relative z {sector_z:+.2f} — name is a sector leader, short edge reduced")
        elif score < 0 and sector_z < -1.0:
            score *= 1.10
            reasons.append(f"✓ Sector-relative z {sector_z:+.2f} — weak vs sector, short confluence")

    # ── Short interest filter: don't short heavily-shorted names (squeeze risk)
    short_pct = meta.get("shortPctFloat")   # 0..1
    short_ratio = meta.get("shortRatio")    # days-to-cover
    if short_pct is not None and score < 0:
        if short_pct >= 0.20 and (short_ratio or 0) >= 5:
            score *= 0.3
            reasons.append(f"⚠ Short interest {short_pct*100:.0f}% of float, {short_ratio or 0:.1f} days-to-cover — squeeze risk, short signal heavily discounted")
        elif short_pct >= 0.15:
            score *= 0.6
            reasons.append(f"⚠ Elevated short interest {short_pct*100:.0f}% — short signal reduced")

    # ── Insider transactions: net buys add confluence to LONG, net sells to SHORT
    insider_net = meta.get("insiderNetCount")       # can be negative
    insider_buy_pct = meta.get("insiderBuyPct")     # 0..1 of buys vs total
    if insider_net is not None and abs(insider_net) >= 2:
        if insider_net > 0 and score > 0:
            score += 0.8
            reasons.append(f"✓ Insiders net buying ({insider_net:+d} txns, 6mo) — confluence with buy signal")
        elif insider_net < 0 and score < 0:
            score -= 0.5
            reasons.append(f"✓ Insiders net selling ({insider_net:+d} txns, 6mo) — confluence with sell signal")
        elif insider_net > 0 and score < 0:
            score *= 0.7
            reasons.append(f"⚠ Insiders net buying ({insider_net:+d}) contradicts sell signal — reduced")

    if earn_near:
        score *= 0.3; reasons.append(f"⚠ Earnings in {earn_days:.0f} days — signal heavily discounted (gaps bypass stops)")

    if   gap < -3.0 and score > 0: score *= 0.6; reasons.append(f"⚠ Today gapped down {gap:.1f}% — gaps often continue, not revert")
    elif gap >  3.0 and score < 0: score *= 0.6

    if dow_score < 0.7 and abs(score) > 2:
        score *= 0.85; reasons.append(f"⚠ {dow_name} signal — historically weaker on this day")

    direction_tentative = "LONG" if score > 0 else "SHORT" if score < 0 else "WAIT"
    age = signal_age(closes, b_v, r_v, direction_tentative)
    if   age >= 4: score *= 0.7; reasons.append(f"⚠ Signal day {age} — been oversold {age} days, edge weakens")
    elif age == 1: reasons.append("Fresh signal (day 1) — best odds")

    if   score >=  5.5: sig = "strong-buy"
    elif score >=  3.5: sig = "buy"
    elif score <= -5.5: sig = "strong-sell"
    elif score <= -3.5: sig = "sell"
    else:               sig = "neutral"

    direction = "LONG"  if sig in ("strong-buy","buy")   else \
                "SHORT" if sig in ("strong-sell","sell") else "WAIT"

    atr_mult = 1.2 if "rang" in reg else (2.0 if "strong" in reg else 1.5)
    if direction == "LONG":
        stop = max(cur - atr_v*atr_mult, b_v["lower"]*0.96)
        t1 = b_v["middle"]; t2 = b_v["upper"]*0.96
        t3 = b_v["upper"] + (b_v["upper"]-b_v["lower"])*0.4
    elif direction == "SHORT":
        stop = min(cur + atr_v*atr_mult, b_v["upper"]*1.04)
        t1 = b_v["middle"]; t2 = b_v["lower"]*1.04
        t3 = b_v["lower"] - (b_v["upper"]-b_v["lower"])*0.4
    else:
        stop = cur - atr_v*1.5; t1 = b_v["middle"]; t2 = b_v["upper"]; t3 = b_v["upper"]

    risk = abs(cur-stop); reward = abs(t1-cur)
    rr = reward/risk if risk > 0 else 0
    ev_raw = (bt["winRate"]/100)*reward - (1-bt["winRate"]/100)*risk

    # Journal-tuned Kelly: realized-WR / predicted-WR calibration
    calib = meta.get("calibration") if meta else None
    calib_factor = (calib or {}).get("factor", 1.0)
    pos_size = ou_position_size(1000, z_v, hl, bt["winRate"], rr, calibration=calib_factor)
    shares_at_1k = round(pos_size/cur, 4) if cur > 0 else 0

    if   sig == "strong-buy":  plain = "Strong buy — multiple indicators say this is unusually cheap right now. Good risk/reward."
    elif sig == "buy":         plain = "Buy — stock looks oversold. Below its normal range. Consider entering with proper stop."
    elif sig == "strong-sell": plain = "Strong sell/short — stock looks unusually expensive. Multiple indicators align."
    elif sig == "sell":        plain = "Sell/short signal — stock stretched above normal range."
    else:                      plain = "No trade yet — not extreme enough. Wait for better setup."

    return {
        "ticker": raw["ticker"], "name": raw["name"], "source": raw["source"],
        "cur": round(cur, 4), "rsi": round(r_v, 1),
        "z": round(z_v, 3), "bbPos": bb_pos, "bbPct": round(bb_pct, 1),
        "bbUpper": round(b_v["upper"], 4), "bbLower": round(b_v["lower"], 4),
        "bbMid": round(b_v["middle"], 4),
        "sig": sig, "score": round(score, 2), "direction": direction,
        "plain": plain,
        "winRate": bt["winRate"], "avgRet": bt["avgReturn"],
        "trades": bt["trades"], "sharpe": bt["sharpe"],
        "maxDD": bt["maxDD"], "pf": bt["pf"], "btMethod": bt["method"],
        "medianFoldWR": bt.get("medianFoldWR"),
        "foldSpread":   bt.get("foldSpread"),
        "regimeWR": regime_wr, "regimeN": regime_n,
        "shortPctFloat": short_pct,
        "shortRatio":    short_ratio,
        "insiderNetCount": insider_net,
        "insiderBuyPct":   insider_buy_pct,
        "kalmanMean":  k_mean,
        "zKalman":     z_kalman,
        "sectorZ":     sector_z,
        "sectorETF":   meta.get("sectorETF") if meta else None,
        "execCostBps": bt.get("execCostBps"),
        "calibFactor": calib_factor,
        "calibN":      (calib or {}).get("n") if calib else 0,
        "ivATM":       meta.get("ivATM") if meta else None,
        "ivPremium":   meta.get("ivPremium") if meta else None,
        "rv30":        meta.get("rv30") if meta else None,
        "entry": round(cur, 4), "stop": round(stop, 4),
        "t1": round(t1, 4), "t2": round(t2, 4), "t3": round(t3, 4),
        "rr": round(rr, 3), "atr": round(atr_v, 4),
        "ma20": round(ma20, 4), "ma50": round(ma50, 4),
        "macdHist": round(m_v["hist"], 4) if m_v else None,
        "macdCross": m_v.get("crossed_up") if m_v else False,
        "stochK": round(sk, 1) if sk else None,
        "volRatio": round(vr, 2) if vr else None,
        "halfLife": hl, "adfStat": adf,
        "regime": reg, "gap": round(gap, 2), "dow": dow_name,
        "signalAge": age, "nearEarnings": earn_near, "earningsDays": earn_days,
        "ev": round(ev_raw, 4),
        "posSize": pos_size, "sharesAt1k": shares_at_1k,
        "riskPct": round(risk/cur*100, 2) if cur > 0 else 0,
        "rewPct":  round(reward/cur*100, 2) if cur > 0 else 0,
        "reasons": reasons,
        "closes60": closes[-60:],
    }
