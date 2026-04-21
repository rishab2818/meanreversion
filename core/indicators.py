"""Technical indicators — pure functions, no external deps."""
import math, time
from datetime import datetime

def rsi(c, p=14):
    if len(c) < p+1:
        return None
    ag = al = 0.0
    for i in range(1, p+1):
        d = c[i] - c[i-1]
        if d > 0: ag += d
        else:     al -= d
    ag /= p; al /= p
    for i in range(p+1, len(c)):
        d = c[i] - c[i-1]
        ag = (ag*(p-1) + max(d, 0)) / p
        al = (al*(p-1) + max(-d, 0)) / p
    return 100.0 if al == 0 else 100 - 100/(1 + ag/al)

def bb(c, p=20, mult=2.0):
    if len(c) < p:
        return None
    s = c[-p:]
    mean = sum(s)/p
    sd = math.sqrt(sum((x-mean)**2 for x in s)/p)
    return {"upper": mean+mult*sd, "middle": mean, "lower": mean-mult*sd, "std": sd}

def z_score(c, w=20):
    if len(c) < w:
        return None
    s = c[-w:]
    mean = sum(s)/w
    sd = math.sqrt(sum((x-mean)**2 for x in s)/w)
    return 0.0 if sd == 0 else (c[-1]-mean)/sd

def atr(data, p=14):
    if len(data) < p+1:
        return None
    trs = [max(data[i]["h"]-data[i]["l"],
               abs(data[i]["h"]-data[i-1]["c"]),
               abs(data[i]["l"]-data[i-1]["c"])) for i in range(1, len(data))]
    return sum(trs[-p:])/p

def ema_series(c, p):
    if len(c) < p:
        return []
    k = 2/(p+1); e = c[0]; out = []
    for v in c:
        e = v*k + e*(1-k)
        out.append(e)
    return out

def macd(c):
    if len(c) < 35:
        return None
    fast = ema_series(c, 12); slow = ema_series(c, 26)
    ml = [f-s for f, s in zip(fast, slow)]
    sig = ema_series(ml, 9)
    if not sig:
        return None
    hist = ml[-1] - sig[-1]
    prev_hist = ml[-2] - sig[-2] if len(ml) > 1 and len(sig) > 1 else hist
    return {"hist": hist, "crossed_up": hist > 0 and prev_hist <= 0}

def stoch_rsi(c, rsi_p=14, stoch_p=14, smooth=3):
    if len(c) < rsi_p + stoch_p + smooth + 5:
        return None, None
    rv = []
    for i in range(rsi_p, len(c)):
        v = rsi(c[:i+1], rsi_p)
        if v is not None:
            rv.append(v)
    if len(rv) < stoch_p:
        return None, None
    kr = []
    for i in range(stoch_p-1, len(rv)):
        w = rv[i-stoch_p+1:i+1]
        mn, mx = min(w), max(w)
        kr.append(0.0 if mx == mn else (rv[i]-mn)/(mx-mn)*100)
    def sm(a, n):
        r = []
        for i in range(n-1, len(a)):
            r.append(sum(a[i-n+1:i+1])/n)
        return r
    k = sm(kr, smooth); d = sm(k, smooth)
    return (k[-1] if k else None, d[-1] if d else None)

def volume_ratio(data, p=20):
    if len(data) < p+1:
        return None
    avg = sum(d["v"] for d in data[-p-1:-1])/p
    return data[-1]["v"]/avg if avg > 0 else None

def market_regime(c, fast=10, slow=50):
    if len(c) < slow+5:
        return "unknown"
    sl = (c[-1]-c[-fast])/c[-fast]*100 if c[-fast] > 0 else 0
    if abs(sl) > 8: return "strong_up" if sl > 0 else "strong_down"
    if abs(sl) > 4: return "weak_up"   if sl > 0 else "weak_down"
    return "ranging"

def ou_halflife(c):
    """OU half-life via OLS regression of Δy on y(t-1). Negative coef = mean-reverting.
    Returns half-life in days; 5-30 days ideal for swing MR."""
    if len(c) < 30:
        return None
    prices = [math.log(x) for x in c if x > 0]
    if len(prices) < 20:
        return None
    y_lag = prices[:-1]
    dy = [prices[i]-prices[i-1] for i in range(1, len(prices))]
    n = len(dy)
    mean_lag = sum(y_lag)/n; mean_dy = sum(dy)/n
    num = sum((y_lag[i]-mean_lag)*(dy[i]-mean_dy) for i in range(n))
    den = sum((y_lag[i]-mean_lag)**2 for i in range(n))
    if den == 0:
        return None
    lam = num/den
    if lam >= 0:
        return None
    return round(-math.log(2)/lam, 1)

def kalman_mean(c, Q=0.01, R=1.0):
    """1D Kalman filter with random-walk state. Returns (mean_now, z_kalman).

    Unlike a rolling 20d mean, the Kalman mean ADAPTS to regime shifts in real
    time — which matters for MR: your current z-score fires on a stale reference
    level during a regime change, while the Kalman z-score only fires on a
    genuine dislocation from the adaptive "fair" level.

    Q = process noise (how fast the mean can drift — higher = faster adaptation)
    R = observation noise (how noisy individual prices are)
    """
    if not c or len(c) < 15:
        return None, None
    # Scale Q and R by recent price level so the constants are units-agnostic
    level = sum(c[-20:]) / min(20, len(c))
    Qv = Q * level * level * 1e-4
    Rv = R * level * level * 1e-4
    x = c[0]; P = Rv
    means = []
    for z_obs in c:
        # predict
        P_pred = P + Qv
        # update
        K = P_pred / (P_pred + Rv)
        x = x + K * (z_obs - x)
        P = (1 - K) * P_pred
        means.append(x)
    residuals = [c[i] - means[i] for i in range(len(c))]
    window = min(30, len(residuals))
    rs = residuals[-window:]
    mean_r = sum(rs) / window
    sd = math.sqrt(sum((r - mean_r) ** 2 for r in rs) / max(window - 1, 1))
    z_k = (c[-1] - means[-1]) / sd if sd > 0 else 0.0
    return round(means[-1], 4), round(z_k, 3)

def sector_rel_z(stock_closes, etf_closes, window=20, lookback=120):
    """Z-score of the stock's cumulative excess return over its sector ETF,
    measured vs its own 120-day distribution.

    Positive = stock has outperformed sector more than usual (leadership).
    Negative = stock has underperformed sector more than usual (broken/weak).

    For MR trading: leaders with oversold MR signals are higher-probability
    longs than broken names; broken names with short signals are higher-
    probability shorts. Returns a float z-score or None.
    """
    n = min(len(stock_closes), len(etf_closes))
    if n < lookback + window + 2:
        return None
    s = stock_closes[-n:]
    e = etf_closes[-n:]
    # daily log returns
    sr = [math.log(s[i]/s[i-1]) for i in range(1, n) if s[i-1] > 0 and s[i] > 0]
    er = [math.log(e[i]/e[i-1]) for i in range(1, n) if e[i-1] > 0 and e[i] > 0]
    m = min(len(sr), len(er))
    if m < lookback + window:
        return None
    sr = sr[-m:]; er = er[-m:]
    excess = [sr[i] - er[i] for i in range(m)]
    # rolling cumulative over `window` bars
    def roll(arr, w):
        return [sum(arr[i-w+1:i+1]) for i in range(w-1, len(arr))]
    rolled = roll(excess, window)
    if len(rolled) < 30:
        return None
    hist = rolled[-lookback:] if len(rolled) >= lookback else rolled
    mh = sum(hist)/len(hist)
    sdh = math.sqrt(sum((x-mh)**2 for x in hist)/max(len(hist)-1, 1))
    if sdh == 0:
        return None
    return round((rolled[-1] - mh) / sdh, 2)

def adf_stat(c):
    """Simple ADF t-statistic. < -2.5 = likely stationary (good for MR)."""
    if len(c) < 20:
        return None
    prices = [math.log(x) for x in c if x > 0]
    if len(prices) < 15:
        return None
    y_lag = prices[:-1]
    dy = [prices[i]-prices[i-1] for i in range(1, len(prices))]
    n = len(dy)
    mean_lag = sum(y_lag)/n; mean_dy = sum(dy)/n
    num = sum((y_lag[i]-mean_lag)*(dy[i]-mean_dy) for i in range(n))
    den = sum((y_lag[i]-mean_lag)**2 for i in range(n))
    if den == 0:
        return None
    lam = num/den
    res = [dy[i]-lam*(y_lag[i]-mean_lag)-mean_dy for i in range(n)]
    se = math.sqrt(sum(r**2 for r in res)/(n-2))/math.sqrt(den) if n > 2 else 1
    return round(lam/se, 3) if se > 0 else None

def signal_age(closes, bb_val, rsi_val, direction):
    if len(closes) < 6:
        return 1
    age = 0
    for i in range(len(closes)-1, max(len(closes)-8, 0), -1):
        slice_c = closes[:i+1]
        b = bb(slice_c)
        r = rsi(slice_c)
        if b is None or r is None:
            break
        if direction == "LONG" and closes[i] < b["lower"] and r < 50:
            age += 1
        elif direction == "SHORT" and closes[i] > b["upper"] and r > 50:
            age += 1
        else:
            break
    return max(age, 1)

def gap_detect(data):
    if len(data) < 2:
        return 0
    prev_c = data[-2]["c"]; today_o = data[-1]["o"]
    if prev_c == 0:
        return 0
    return (today_o-prev_c)/prev_c*100

def day_of_week_score(data):
    if not data:
        return 0, "unknown"
    try:
        dt = datetime.strptime(data[-1]["date"], "%Y-%m-%d").weekday()
        labels = ["Mon","Tue","Wed","Thu","Fri"]
        scores = [0.5, 1.0, 1.0, 1.0, 0.6]
        return scores[dt], labels[dt]
    except Exception:
        return 0.8, "?"

def near_earnings(raw):
    e = raw.get("earningsNext")
    if not e:
        return False, None
    days = (e - time.time()) / 86400
    return abs(days) <= 5, round(days, 0)
