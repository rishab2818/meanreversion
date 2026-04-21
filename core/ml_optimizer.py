"""GA optimizer — tunes MR params AND DCF params.

Two entry points:
  run_ga_mr(closes_list, params, ticker) — original MR optimization.
  run_ga_dcf(funds, price_data_map, params, ticker) — NEW DCF tuning.

Both write into the same module-level `ga_state` dict so the UI
polls one /api/ga_status endpoint.
"""
import random, time
from datetime import datetime
from core.mr_engine import backtest as mr_backtest
from core.dcf_engine import dcf_backtest, analyze_dcf
from core.storage import load_profiles, save_profiles

ga_state = {"running": False, "log": [], "pct": 0, "best": None,
            "top10": [], "history": [], "ticker": "", "mode": "mr"}

def _addlog(m):
    ga_state["log"].append(m)
    ga_state["log"] = ga_state["log"][-200:]

# ═══════════════════════════════════════════════════════════════════════════
#                              MR GA
# ═══════════════════════════════════════════════════════════════════════════
def rnd_mr():
    return {"rsiP":    random.randint(7, 21),
            "bbStd":   round(random.uniform(1.5, 3.0), 2),
            "zWin":    random.randint(10, 30),
            "rsiOS":   random.randint(25, 40),
            "rsiOB":   random.randint(60, 75),
            "zThresh": round(random.uniform(1.0, 2.5), 2),
            "volMin":  round(random.uniform(0.7, 1.5), 2)}

def xover(a, b):
    return {k: (a[k] if random.random() < .5 else b[k]) for k in a}

def mutate_mr(c, rate):
    c = dict(c)
    if random.random() < rate: c["rsiP"]    = max(5,  min(25, c["rsiP"]    + random.randint(-2, 2)))
    if random.random() < rate: c["bbStd"]   = max(1.0, min(3.5, round(c["bbStd"]   + random.uniform(-.3, .3), 2)))
    if random.random() < rate: c["zWin"]    = max(5,  min(35, c["zWin"]    + random.randint(-3, 3)))
    if random.random() < rate: c["rsiOS"]   = max(20, min(45, c["rsiOS"]   + random.randint(-3, 3)))
    if random.random() < rate: c["rsiOB"]   = max(55, min(80, c["rsiOB"]   + random.randint(-3, 3)))
    if random.random() < rate: c["zThresh"] = max(.5, min(3.0, round(c["zThresh"] + random.uniform(-.2, .2), 2)))
    if random.random() < rate: c["volMin"]  = max(0.5, min(2.0, round(c["volMin"]  + random.uniform(-.15, .15), 2)))
    return c

def fitness_mr(bt, fw, fr):
    """Purged-CV fitness: uses MEDIAN out-of-sample fold win rate (not in-sample),
    plus a stability penalty on fold-spread to punish overfitted parameter sets."""
    # Prefer median OOS fold win rate when available; else fall back to overall
    oos_wr = bt.get("medianFoldWR", bt.get("winRate", 50))
    ws = (oos_wr - 50) / 50
    rs = max(-1, min(2, bt["avgReturn"]/5))
    sh = min(bt.get("sharpe", 0)/3, 1)
    pf = min((bt.get("pf", 1) - 1)/2, 1)
    dd = bt.get("maxDD", 0)/100
    # Fold-spread penalty: if OOS folds disagree >30pp it's likely lucky params
    spread_pen = max(0, (bt.get("foldSpread", 0) - 30)) / 100
    return (fw/100)*ws + (fr/100)*rs + 0.12*sh + 0.08*pf - 0.08*dd - 0.15*spread_pen

def run_ga_mr(closes_list, params, ticker="portfolio"):
    global ga_state
    ga_state.update({"running": True, "log": [], "pct": 0, "best": None,
                     "top10": [], "history": [], "ticker": ticker, "mode": "mr"})
    pop_size = params.get("popSize", 40)
    n_gen    = params.get("nGen", 80)
    mut_rate = params.get("mutRate", 0.15)
    elite_k  = max(2, int(pop_size * params.get("eliteK", 0.2)))
    fw = params.get("fw", 50); fr = params.get("fr", 50)

    def eval_c(c):
        tot_w = tot_r = tot_s = tot_p = tot_d = 0.0; n = 0
        for closes in closes_list:
            d = [{"c": v, "h": v*1.01, "l": v*0.99, "v": 1e6, "o": v, "date": "2020-01-01"} for v in closes]
            bt = mr_backtest(d, c)
            tot_w += bt["winRate"]; tot_r += bt["avgReturn"]; tot_s += bt.get("sharpe", 0)
            tot_p += bt.get("pf", 1); tot_d += bt.get("maxDD", 0); n += 1
        if n == 0:
            return {"winRate":50,"avgReturn":0,"sharpe":0,"pf":1,"maxDD":0,"trades":0}, -999
        bt2 = {"winRate":   round(tot_w/n, 1),
               "avgReturn": round(tot_r/n, 3),
               "sharpe":    round(tot_s/n, 2),
               "pf":        round(tot_p/n, 2),
               "maxDD":     round(tot_d/n, 1),
               "trades":    n*10}
        return bt2, fitness_mr(bt2, fw, fr)

    pop = [rnd_mr() for _ in range(pop_size)]
    best_ever = None; best_fit = -999
    _addlog(f"MR OPTIMIZER | Stocks: {len(closes_list)} | Pop: {pop_size} | Gens: {n_gen}")
    _addlog("Using walk-forward backtesting — honest numbers only")
    _addlog("─"*48)

    for gen in range(n_gen):
        if not ga_state["running"]:
            break
        # evaluate each member once (cache eval)
        cache = {}
        scored = []
        for c in pop:
            key = tuple(sorted(c.items()))
            if key not in cache:
                cache[key] = eval_c(c)
            bt, fit = cache[key]
            scored.append((fit, bt, c))
        scored.sort(key=lambda x: -x[0])
        gb = scored[0]
        if gb[0] > best_fit:
            best_fit = gb[0]
            best_ever = {"params": gb[2], "bt": gb[1], "fit": round(gb[0], 5)}
        ga_state["history"].append(round(best_fit, 5))
        ga_state["pct"] = round((gen+1)/n_gen*100)
        ga_state["best"] = best_ever
        if gen % 5 == 0 or gen == n_gen-1:
            c = gb[2]; bt = gb[1]
            _addlog(f"Gen {gen+1:3d} | win={bt['winRate']:.0f}% ret={bt['avgReturn']:+.2f}% "
                    f"sh={bt.get('sharpe', 0):.2f} | RSI={c['rsiP']} BB={c['bbStd']} Z={c['zWin']}")
        elites = [s[2] for s in scored[:elite_k]]
        children = []
        while len(children) < pop_size - elite_k:
            a = random.choice(scored[:max(elite_k*2, 3)])[2]
            b = random.choice(scored[:max(elite_k*2, 3)])[2]
            children.append(mutate_mr(xover(a, b), mut_rate))
        pop = elites + children
        ga_state["top10"] = [{"params": s[2], "bt": s[1], "fit": round(s[0], 5)} for s in scored[:10]]
        time.sleep(0)

    _addlog("─"*48); _addlog("DONE")
    if best_ever:
        p = best_ever["params"]; bt = best_ever["bt"]
        _addlog(f"Best: RSI={p['rsiP']} BB={p['bbStd']} Z={p['zWin']} vol≥{p['volMin']}")
        _addlog(f"Win={bt['winRate']}% Ret={bt['avgReturn']:+.2f}% Sharpe={bt.get('sharpe', 0):.2f}")
        try:
            profiles = load_profiles()
            profiles.setdefault(ticker, {})
            profiles[ticker]["mr"] = {"params": best_ever["params"], "bt": best_ever["bt"],
                                      "updated": datetime.now().strftime("%Y-%m-%d")}
            # Also keep legacy flat shape so existing /api/scan keeps working
            profiles[ticker]["params"] = best_ever["params"]
            profiles[ticker]["bt"]     = best_ever["bt"]
            profiles[ticker]["updated"] = datetime.now().strftime("%Y-%m-%d")
            save_profiles(profiles)
            _addlog(f"Saved MR profile for {ticker}")
        except Exception as e:
            _addlog(f"(could not save profile: {e})")
    ga_state["running"] = False

# ═══════════════════════════════════════════════════════════════════════════
#                              DCF GA
# ═══════════════════════════════════════════════════════════════════════════
def rnd_dcf(base_wacc=0.09, g_prior=0.07, g_sigma=0.20):
    """Seed DCF chromosomes around the observed priors for faster convergence."""
    return {
        "gNear":       round(max(-0.05, min(0.35, g_prior + random.uniform(-0.05, 0.08))), 3),
        "gNearSigma":  round(max(0.05, min(0.35, g_sigma * random.uniform(0.6, 1.4))), 3),
        "gFar":        round(max(0.00, min(0.12, g_prior*0.5 + random.uniform(-0.03, 0.03))), 3),
        "gFarSigma":   round(max(0.04, min(0.20, g_sigma*0.7 * random.uniform(0.6, 1.4))), 3),
        "gTerm":       round(random.uniform(0.015, 0.035), 3),
        "gTermSigma":  round(random.uniform(0.003, 0.010), 4),
        "wacc":        round(max(0.05, min(0.16, base_wacc + random.uniform(-0.02, 0.02))), 3),
        "waccSigma":   round(random.uniform(0.008, 0.018), 4),
        "nNear":       random.choice([3, 5, 7, 10]),
        "nFar":        random.choice([3, 5, 7, 10]),
        "marginThreshold": random.choice([15, 20, 25, 30]),
    }

def mutate_dcf(c, rate):
    c = dict(c)
    if random.random() < rate: c["gNear"]      = max(-0.10, min(0.40, round(c["gNear"]      + random.uniform(-0.03, 0.03), 3)))
    if random.random() < rate: c["gNearSigma"] = max(0.03,  min(0.40, round(c["gNearSigma"] + random.uniform(-0.04, 0.04), 3)))
    if random.random() < rate: c["gFar"]       = max(0.00,  min(0.15, round(c["gFar"]       + random.uniform(-0.02, 0.02), 3)))
    if random.random() < rate: c["gFarSigma"]  = max(0.02,  min(0.20, round(c["gFarSigma"]  + random.uniform(-0.02, 0.02), 3)))
    if random.random() < rate: c["gTerm"]      = max(0.005, min(0.04, round(c["gTerm"]      + random.uniform(-0.005, 0.005), 3)))
    if random.random() < rate: c["gTermSigma"] = max(0.001, min(0.015, round(c["gTermSigma"]+ random.uniform(-0.002, 0.002), 4)))
    if random.random() < rate: c["wacc"]       = max(0.04,  min(0.20, round(c["wacc"]       + random.uniform(-0.015, 0.015), 3)))
    if random.random() < rate: c["waccSigma"]  = max(0.004, min(0.03, round(c["waccSigma"]  + random.uniform(-0.004, 0.004), 4)))
    if random.random() < rate: c["nNear"]      = max(3, min(10, c["nNear"] + random.choice([-1, 1])))
    if random.random() < rate: c["nFar"]       = max(3, min(10, c["nFar"]  + random.choice([-1, 1])))
    if random.random() < rate: c["marginThreshold"] = max(10, min(40, c["marginThreshold"] + random.choice([-5, 5])))
    return c

def fitness_dcf(bt_list):
    """Cross-sectional fitness across stocks. Rewards: high Spearman (monotonic
    MoS→forward-return), high win-rate, decent avg return; penalizes low trade counts."""
    if not bt_list:
        return -999
    sp   = sum(b.get("spearman", 0)   for b in bt_list) / len(bt_list)
    wr   = sum(b.get("winRate",   50) for b in bt_list) / len(bt_list)
    ret  = sum(b.get("avgReturn",  0) for b in bt_list) / len(bt_list)
    nts  = sum(b.get("trades",     0) for b in bt_list)
    wr_norm  = (wr - 50) / 50
    ret_norm = max(-1, min(2, ret / 10))
    n_pen = -0.5 if nts < max(3, len(bt_list)) else 0.0
    return 0.45*sp + 0.35*wr_norm + 0.15*ret_norm + 0.05 + n_pen

def run_ga_dcf(funds_map, prices_map, params, ticker="portfolio"):
    """funds_map: {ticker: fund_dict}; prices_map: {ticker: price_data_list}."""
    global ga_state
    ga_state.update({"running": True, "log": [], "pct": 0, "best": None,
                     "top10": [], "history": [], "ticker": ticker, "mode": "dcf"})
    pop_size = params.get("popSize", 30)
    n_gen    = params.get("nGen", 40)
    mut_rate = params.get("mutRate", 0.20)
    elite_k  = max(2, int(pop_size * params.get("eliteK", 0.25)))

    # seed hints from first available fund
    from core.dcf_engine import wacc, fcf_growth_prior
    seeds = []
    for tk, f in funds_map.items():
        if f and f.get("fcfHistory"):
            g, s = fcf_growth_prior(f.get("fcfHistory"))
            seeds.append((wacc(f), g, s))
    if seeds:
        w_s = sum(s[0] for s in seeds)/len(seeds)
        g_s = sum(s[1] for s in seeds)/len(seeds)
        sg_s= sum(s[2] for s in seeds)/len(seeds)
    else:
        w_s, g_s, sg_s = 0.09, 0.07, 0.20

    def eval_c(c):
        bts = []
        for tk, f in funds_map.items():
            if not f: continue
            prices = prices_map.get(tk)
            if not prices: continue
            bt = dcf_backtest(f, prices, c)
            if bt.get("trades", 0) > 0 or bt.get("signals", 0) > 0:
                bts.append(bt)
        if not bts:
            return {"winRate":50,"avgReturn":0,"spearman":0,"trades":0,
                    "method":"empty"}, -999
        avg = {"winRate":   round(sum(b["winRate"]   for b in bts)/len(bts), 1),
               "avgReturn": round(sum(b["avgReturn"] for b in bts)/len(bts), 2),
               "spearman":  round(sum(b["spearman"]  for b in bts)/len(bts), 3),
               "trades":    sum(b["trades"] for b in bts),
               "method":    "cross-section"}
        return avg, fitness_dcf(bts)

    pop = [rnd_dcf(w_s, g_s, sg_s) for _ in range(pop_size)]
    best_ever = None; best_fit = -999
    _addlog(f"DCF OPTIMIZER | Stocks: {len(funds_map)} | Pop: {pop_size} | Gens: {n_gen}")
    _addlog(f"Priors: g≈{g_s*100:.1f}% σ≈{sg_s*100:.1f}% WACC≈{w_s*100:.1f}%")
    _addlog("─"*48)

    for gen in range(n_gen):
        if not ga_state["running"]:
            break
        cache = {}
        scored = []
        for c in pop:
            key = tuple(sorted((k, round(v, 5) if isinstance(v, float) else v) for k, v in c.items()))
            if key not in cache:
                cache[key] = eval_c(c)
            bt, fit = cache[key]
            scored.append((fit, bt, c))
        scored.sort(key=lambda x: -x[0])
        gb = scored[0]
        if gb[0] > best_fit:
            best_fit = gb[0]
            best_ever = {"params": gb[2], "bt": gb[1], "fit": round(gb[0], 5)}
        ga_state["history"].append(round(best_fit, 5))
        ga_state["pct"] = round((gen+1)/n_gen*100)
        ga_state["best"] = best_ever
        if gen % 3 == 0 or gen == n_gen-1:
            c = gb[2]; bt = gb[1]
            _addlog(f"Gen {gen+1:3d} | wr={bt['winRate']:.0f}% spearman={bt['spearman']:+.2f} "
                    f"ret={bt['avgReturn']:+.1f}% | g={c['gNear']*100:.1f}% WACC={c['wacc']*100:.1f}% mos≥{c['marginThreshold']}%")
        elites = [s[2] for s in scored[:elite_k]]
        children = []
        while len(children) < pop_size - elite_k:
            a = random.choice(scored[:max(elite_k*2, 3)])[2]
            b = random.choice(scored[:max(elite_k*2, 3)])[2]
            children.append(mutate_dcf(xover(a, b), mut_rate))
        pop = elites + children
        ga_state["top10"] = [{"params": s[2], "bt": s[1], "fit": round(s[0], 5)} for s in scored[:10]]
        time.sleep(0)

    _addlog("─"*48); _addlog("DONE")
    if best_ever:
        p = best_ever["params"]; bt = best_ever["bt"]
        _addlog(f"Best: g_near={p['gNear']*100:.1f}% g_far={p['gFar']*100:.1f}% gT={p['gTerm']*100:.1f}% "
                f"WACC={p['wacc']*100:.1f}% mos≥{p['marginThreshold']}%")
        _addlog(f"Win={bt['winRate']}% Spearman={bt['spearman']:+.2f} Ret={bt['avgReturn']:+.1f}%")
        try:
            profiles = load_profiles()
            profiles.setdefault(ticker, {})
            profiles[ticker]["dcf"] = {"params": best_ever["params"], "bt": best_ever["bt"],
                                       "updated": datetime.now().strftime("%Y-%m-%d")}
            save_profiles(profiles)
            _addlog(f"Saved DCF profile for {ticker}")
        except Exception as e:
            _addlog(f"(could not save DCF profile: {e})")
    ga_state["running"] = False
