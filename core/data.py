"""Data fetching — prices (Yahoo/Stooq) + fundamentals (Yahoo quoteSummary)."""
import json, math, urllib.request, urllib.parse, http.cookiejar, csv, io, time
from datetime import datetime
from core.config import CACHE_TTL, FUND_TTL, log

CACHE = {}       # ticker -> {data, ts}
FUND_CACHE = {}  # ticker -> {data, ts}
META_CACHE = {}  # ticker -> {data, ts}  — short interest + insider activity

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"

# ─── Yahoo session (cookie jar + crumb) ──────────────────────────────────────
_yahoo_cookie_jar = http.cookiejar.CookieJar()
_yahoo_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_yahoo_cookie_jar))
_yahoo_crumb = {"crumb": None, "ts": 0}

def _yahoo_request(url, timeout=15):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with _yahoo_opener.open(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")

def _get_yahoo_crumb():
    """Fetch a crumb (cached 1h). Required for v10 endpoints."""
    now = time.time()
    if _yahoo_crumb["crumb"] and now - _yahoo_crumb["ts"] < 3600:
        return _yahoo_crumb["crumb"]
    try:
        # Step 1 — hit fc.yahoo.com to set A3/B cookies
        try:
            _yahoo_request("https://fc.yahoo.com/", timeout=10)
        except Exception:
            pass  # often returns 404 but still sets cookies
        # Step 2 — fetch crumb
        crumb = _yahoo_request("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=10)
        crumb = crumb.strip()
        if crumb and len(crumb) < 40 and "<" not in crumb:
            _yahoo_crumb["crumb"] = crumb
            _yahoo_crumb["ts"] = now
            return crumb
    except Exception as e:
        log(f"  ! crumb fetch failed: {e}")
    return None

def fetch_url(url, timeout=12):
    """Plain unauthenticated fetch (used by chart endpoint, stooq)."""
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")

# ─── price data ───────────────────────────────────────────────────────────────
def fetch_yahoo(ticker, period="2y"):
    for host in ["query1", "query2"]:
        try:
            url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range={period}&includePrePost=false"
            d = json.loads(fetch_url(url))
            res = d.get("chart", {}).get("result", [None])[0]
            if not res:
                continue
            q, ts = res["indicators"]["quote"][0], res["timestamp"]
            meta = res.get("meta", {})
            rows = []
            for i, c in enumerate(q.get("close", [])):
                if c and c > 0:
                    rows.append({
                        "date": datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d"),
                        "o": q.get("open", [c]*len(ts))[i] or c,
                        "h": q["high"][i] or c,
                        "l": q["low"][i] or c,
                        "c": c,
                        "v": q.get("volume", [1e6]*len(ts))[i] or 1e6,
                    })
            if len(rows) >= 60:
                return {
                    "ticker": ticker, "name": meta.get("shortName", ticker),
                    "data": rows, "source": "yahoo",
                    "earningsNext": meta.get("earningsTimestampEnd"),
                }
        except Exception:
            pass
    return None

def fetch_stooq(ticker):
    try:
        sym = ticker.lower().replace("-", ".") + ".us"
        raw = fetch_url(f"https://stooq.com/q/d/l/?s={sym}&i=d")
        if not raw or "No data" in raw or len(raw.strip()) < 50:
            return None
        rows = []
        for row in csv.DictReader(io.StringIO(raw.strip())):
            try:
                c = float(row.get("Close", 0) or 0)
                if c > 0:
                    rows.append({
                        "date": row["Date"],
                        "o": float(row.get("Open", c) or c),
                        "h": float(row.get("High", c) or c),
                        "l": float(row.get("Low", c) or c),
                        "c": c,
                        "v": float(row.get("Volume", 1e6) or 1e6),
                    })
            except Exception:
                pass
        if len(rows) >= 60:
            return {"ticker": ticker, "name": ticker, "data": rows[-500:], "source": "stooq"}
    except Exception:
        pass
    return None

def fetch_ticker(ticker):
    now = time.time()
    if ticker in CACHE and now - CACHE[ticker]["ts"] < CACHE_TTL:
        return CACHE[ticker]["data"]
    result = fetch_yahoo(ticker) or fetch_stooq(ticker)
    if result:
        CACHE[ticker] = {"data": result, "ts": now}
        log(f"  ✓ {ticker:6s} {len(result['data'])} days [{result['source']}]")
    else:
        log(f"  ✗ {ticker}")
    return result

# ─── fundamentals (Yahoo quoteSummary) ────────────────────────────────────────
def _raw(d, key, default=None):
    if not d:
        return default
    v = d.get(key)
    if isinstance(v, dict):
        return v.get("raw", default)
    return v if v is not None else default

_FUND_MODULES = ",".join([
    "financialData", "defaultKeyStatistics", "summaryDetail", "price",
    "cashflowStatementHistory", "balanceSheetHistory", "incomeStatementHistory",
])

def _parse_fundamentals(res, ticker):
    fd = res.get("financialData", {}) or {}
    ks = res.get("defaultKeyStatistics", {}) or {}
    sd = res.get("summaryDetail", {}) or {}
    pr = res.get("price", {}) or {}
    cfs = (res.get("cashflowStatementHistory", {}) or {}).get("cashflowStatements", []) or []
    ish = (res.get("incomeStatementHistory", {}) or {}).get("incomeStatementHistory", []) or []
    bsh = (res.get("balanceSheetHistory", {}) or {}).get("balanceSheetStatements", []) or []

    fcf_ttm = _raw(fd, "freeCashflow")
    ocf_ttm = _raw(fd, "operatingCashflow")

    # Historical FCF (most recent first)
    fcf_hist = []
    for s in cfs:
        ocf = _raw(s, "totalCashFromOperatingActivities")
        capex = _raw(s, "capitalExpenditures")  # Yahoo returns negative
        if ocf is not None and capex is not None:
            fcf_hist.append(ocf + capex)
        elif ocf is not None:
            fcf_hist.append(ocf)  # fall back to OCF if capex missing

    rev_hist = [_raw(s, "totalRevenue") for s in ish]
    rev_hist = [r for r in rev_hist if r]
    ni_hist  = [_raw(s, "netIncome") for s in ish]
    ni_hist  = [n for n in ni_hist if n is not None]

    total_cash = _raw(fd, "totalCash") or 0
    total_debt = _raw(fd, "totalDebt") or 0
    net_debt = total_debt - total_cash

    shares = _raw(ks, "sharesOutstanding") or _raw(pr, "sharesOutstanding")
    market_cap = _raw(pr, "marketCap") or _raw(sd, "marketCap")
    beta = _raw(sd, "beta") or _raw(ks, "beta") or 1.0
    cur_price = _raw(fd, "currentPrice") or _raw(pr, "regularMarketPrice")

    return {
        "ticker": ticker,
        "name": _raw(pr, "longName") or _raw(pr, "shortName") or ticker,
        "fcfTTM": fcf_ttm,
        "ocfTTM": ocf_ttm,
        "fcfHistory": fcf_hist,     # most recent first
        "revHistory": rev_hist,
        "niHistory":  ni_hist,
        "sharesOut": shares,
        "marketCap": market_cap,
        "beta": beta,
        "totalCash": total_cash,
        "totalDebt": total_debt,
        "netDebt": net_debt,
        "currentPrice": cur_price,
        "revGrowth": _raw(fd, "revenueGrowth"),
        "earningsGrowth": _raw(fd, "earningsGrowth"),
        "profitMargin": _raw(fd, "profitMargins"),
        "returnOnEquity": _raw(fd, "returnOnEquity"),
        "returnOnAssets": _raw(fd, "returnOnAssets"),
        "debtToEquity": _raw(fd, "debtToEquity"),
        "currentRatio": _raw(fd, "currentRatio"),
        "source": "yahoo",
    }

_TIMESERIES_TYPES = ",".join([
    "annualFreeCashFlow", "annualNetIncome", "annualTotalRevenue",
    "annualOperatingCashFlow", "annualCapitalExpenditure",
    "annualTotalAssets",
])

def _fetch_timeseries(ticker, crumb):
    """Pull 4y historical annual series from Yahoo fundamentals-timeseries."""
    end = int(time.time())
    url = (f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/"
           f"{ticker}?symbol={ticker}&type={_TIMESERIES_TYPES}"
           f"&period1=946684800&period2={end}")
    if crumb:
        url += f"&crumb={urllib.parse.quote(crumb)}"
    try:
        raw = _yahoo_request(url, timeout=15)
        d = json.loads(raw)
        result = d.get("timeseries", {}).get("result", []) or []
    except Exception:
        return {}
    out = {}
    for series in result:
        meta_types = series.get("meta", {}).get("type") or []
        if not meta_types:
            continue
        type_name = meta_types[0]
        values = series.get(type_name) or []
        parsed = []
        for entry in values:
            if not entry:
                continue
            rv = (entry.get("reportedValue") or {}).get("raw")
            if rv is not None:
                parsed.append({"date": entry.get("asOfDate"), "value": rv})
        out[type_name] = parsed
    return out

def fetch_fundamentals(ticker):
    """Fetch fundamentals: quoteSummary (TTM snapshot) + timeseries (history)."""
    now = time.time()
    if ticker in FUND_CACHE and now - FUND_CACHE[ticker]["ts"] < FUND_TTL:
        return FUND_CACHE[ticker]["data"]
    crumb = _get_yahoo_crumb()
    f = None
    last_err = None
    for host in ["query1", "query2"]:
        try:
            base = f"https://{host}.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={_FUND_MODULES}"
            url = base + (f"&crumb={urllib.parse.quote(crumb)}" if crumb else "")
            raw = _yahoo_request(url, timeout=20)
            d = json.loads(raw)
            err = (d.get("quoteSummary", {}) or {}).get("error")
            if err:
                last_err = err; continue
            res = (d.get("quoteSummary", {}) or {}).get("result", [None])
            if not res or not res[0]:
                continue
            f = _parse_fundamentals(res[0], ticker)
            break
        except Exception as e:
            last_err = str(e)

    if not f:
        log(f"  $ {ticker} fundamentals unavailable ({last_err or 'unknown'})")
        FUND_CACHE[ticker] = {"data": None, "ts": now}
        return None

    # Enrich with timeseries history (FCF, revenue, NI history)
    try:
        ts = _fetch_timeseries(ticker, crumb)
        fcf_series = ts.get("annualFreeCashFlow", [])
        # most-recent-first
        fcf_hist = [x["value"] for x in reversed(fcf_series) if x.get("value") is not None]
        if fcf_hist:
            f["fcfHistory"] = fcf_hist
            # If TTM FCF is missing, use most recent annual as fallback
            if not f.get("fcfTTM"):
                f["fcfTTM"] = fcf_hist[0]
        rev_series = ts.get("annualTotalRevenue", [])
        if rev_series:
            f["revHistory"] = [x["value"] for x in reversed(rev_series) if x.get("value") is not None]
        ni_series = ts.get("annualNetIncome", [])
        if ni_series:
            f["niHistory"] = [x["value"] for x in reversed(ni_series) if x.get("value") is not None]
        ocf_series = ts.get("annualOperatingCashFlow", [])
        if ocf_series:
            f["ocfHistory"] = [x["value"] for x in reversed(ocf_series) if x.get("value") is not None]
        ta_series = ts.get("annualTotalAssets", [])
        if ta_series:
            f["totalAssetsHistory"] = [x["value"] for x in reversed(ta_series) if x.get("value") is not None]
    except Exception as e:
        log(f"  $ {ticker} timeseries enrichment failed: {e}")

    if not (f.get("fcfTTM") and f.get("sharesOut")):
        log(f"  $ {ticker} missing FCF or shares — insufficient for DCF")
        FUND_CACHE[ticker] = {"data": None, "ts": now}
        return None
    FUND_CACHE[ticker] = {"data": f, "ts": now}
    log(f"  $ {ticker:6s} fundamentals [yahoo] ({len(f.get('fcfHistory') or [])} yrs hist)")
    return f

# ─── lightweight per-ticker meta: short interest + insider activity ────────────
_META_MODULES = "defaultKeyStatistics,netSharePurchaseActivity,insiderTransactions"

def fetch_quote_meta(ticker):
    """Light quoteSummary call for non-DCF enrichment of the MR scanner.
    Returns: {shortPctFloat, shortRatio, insiderNetCount, insiderNetShares,
    insiderBuyPct, insiderPeriod} or None. Cached FUND_TTL (6h).
    Never raises — silently returns None on any failure so the main scan keeps
    working even if Yahoo rejects these modules for a given ticker.
    """
    now = time.time()
    if ticker in META_CACHE and now - META_CACHE[ticker]["ts"] < FUND_TTL:
        return META_CACHE[ticker]["data"]
    crumb = _get_yahoo_crumb()
    out = None
    try:
        for host in ["query1", "query2"]:
            url = (f"https://{host}.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
                   f"?modules={_META_MODULES}")
            if crumb:
                url += f"&crumb={urllib.parse.quote(crumb)}"
            try:
                raw = _yahoo_request(url, timeout=10)
                d = json.loads(raw)
            except Exception:
                continue
            res = (d.get("quoteSummary", {}) or {}).get("result", [None])
            if not res or not res[0]:
                continue
            r = res[0]
            ks = r.get("defaultKeyStatistics", {}) or {}
            nsp = r.get("netSharePurchaseActivity", {}) or {}
            ins = (r.get("insiderTransactions", {}) or {}).get("transactions", []) or []

            short_pct = _raw(ks, "shortPercentOfFloat")
            short_ratio = _raw(ks, "shortRatio")

            insider_buys = _raw(nsp, "buyInfoCount") or 0
            insider_sells = _raw(nsp, "sellInfoCount") or 0
            insider_net = insider_buys - insider_sells
            insider_net_shares = _raw(nsp, "netInfoShares") or 0
            period = nsp.get("period") or "6mo"
            total = insider_buys + insider_sells
            buy_pct = (insider_buys / total) if total > 0 else None

            # Also count raw insider Form 4 buys (fallback if netSharePurchaseActivity empty)
            if total == 0 and ins:
                buys = 0; sells = 0
                for t in ins[:40]:
                    txt = (t.get("transactionText") or "").lower()
                    if "purchase" in txt or "buy" in txt: buys += 1
                    elif "sale" in txt or "sell" in txt: sells += 1
                insider_buys = buys; insider_sells = sells
                insider_net = buys - sells
                if (buys+sells) > 0:
                    buy_pct = buys/(buys+sells)

            out = {
                "shortPctFloat":   short_pct,
                "shortRatio":      short_ratio,
                "insiderBuys":     insider_buys,
                "insiderSells":    insider_sells,
                "insiderNetCount": insider_net,
                "insiderNetShares": insider_net_shares,
                "insiderBuyPct":   buy_pct,
                "insiderPeriod":   period,
            }
            break
    except Exception as e:
        log(f"  ? {ticker} meta fetch failed: {e}")
    META_CACHE[ticker] = {"data": out, "ts": now}
    return out

# ─── IV premium (ATM implied vol / realized vol - 1) ──────────────────────────
IV_CACHE = {}

def fetch_iv_premium(ticker):
    """Pull nearest-expiry ATM implied vol from Yahoo options chain and compare
    to 30d realized vol. Returns {ivATM, rv30, ivPremium, daysToExpiry} or None.
    Cached 6h. Free proxy for paid IVR. Display-only — not a trade gate."""
    now = time.time()
    if ticker in IV_CACHE and now - IV_CACHE[ticker]["ts"] < FUND_TTL:
        return IV_CACHE[ticker]["data"]
    crumb = _get_yahoo_crumb()
    out = None
    try:
        for host in ["query1", "query2"]:
            url = f"https://{host}.finance.yahoo.com/v7/finance/options/{ticker}"
            if crumb:
                url += f"?crumb={urllib.parse.quote(crumb)}"
            try:
                raw = _yahoo_request(url, timeout=10)
                d = json.loads(raw)
            except Exception:
                continue
            res = (d.get("optionChain", {}) or {}).get("result", [None])
            if not res or not res[0]:
                continue
            r0 = res[0]
            quote = r0.get("quote", {}) or {}
            spot = _raw(quote, "regularMarketPrice")
            opts = r0.get("options", []) or []
            if not opts or not spot:
                continue
            chain = opts[0]
            expiry = chain.get("expirationDate") or 0
            dte = max(1, int((expiry - now) / 86400))
            # Find ATM call (smallest |strike-spot|)
            calls = chain.get("calls", []) or []
            if not calls:
                continue
            atm = min(calls, key=lambda c: abs((_raw(c, "strike") or 0) - spot))
            iv_atm = _raw(atm, "impliedVolatility")
            if not iv_atm or iv_atm <= 0:
                continue
            # Realized vol (30d) from price cache if available
            rv30 = None
            if ticker in CACHE:
                closes = [r["c"] for r in CACHE[ticker]["data"]["data"][-31:]]
                if len(closes) >= 20:
                    rets = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))
                            if closes[i-1] > 0 and closes[i] > 0]
                    if rets:
                        m = sum(rets)/len(rets)
                        v = sum((r-m)**2 for r in rets)/max(len(rets)-1, 1)
                        rv30 = math.sqrt(v) * math.sqrt(252)
            iv_premium = (iv_atm / rv30 - 1) if rv30 and rv30 > 0 else None
            out = {
                "ivATM": round(iv_atm, 4),
                "rv30": round(rv30, 4) if rv30 else None,
                "ivPremium": round(iv_premium, 3) if iv_premium is not None else None,
                "daysToExpiry": dte,
                "atmStrike": _raw(atm, "strike"),
            }
            break
    except Exception as e:
        log(f"  ? {ticker} IV fetch failed: {e}")
    IV_CACHE[ticker] = {"data": out, "ts": now}
    return out
