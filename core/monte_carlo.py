"""Account-level Monte Carlo — 'what could happen to my $1000?' sim."""
import random

def monte_carlo(win_rate, rr, n_trades, capital, risk_pct, n_sims=1000):
    finals = []; dds = []; curves = []
    for sim in range(n_sims):
        cap = capital; pk = capital; mdd = 0.0
        curve = [capital]
        for _ in range(n_trades):
            risk = cap * risk_pct / 100
            cap += risk * rr if random.random() < win_rate/100 else -risk
            cap = max(cap, 0)
            curve.append(round(cap, 2))
            if cap > pk: pk = cap
            dd = (pk-cap)/pk*100 if pk > 0 else 0
            if dd > mdd: mdd = dd
        finals.append(cap); dds.append(mdd)
        if sim < 25:
            curves.append(curve[::max(1, n_trades//20)])
    finals.sort(); dds.sort()
    def pct(arr, p): return arr[min(int(p/100*n_sims), n_sims-1)]
    return {
        "p5":  round(pct(finals, 5), 0),
        "p25": round(pct(finals, 25), 0),
        "p50": round(pct(finals, 50), 0),
        "p75": round(pct(finals, 75), 0),
        "p95": round(pct(finals, 95), 0),
        "medMaxDD":   round(pct(dds, 50), 1),
        "worstMaxDD": round(pct(dds, 95), 1),
        "ruinPct":    round(sum(1 for f in finals if f < capital*0.5)/n_sims*100, 1),
        "profitPct":  round(sum(1 for f in finals if f > capital)/n_sims*100, 1),
        "curves":     curves,
    }
