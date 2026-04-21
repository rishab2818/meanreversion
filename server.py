#!/usr/bin/env python3
"""Mean Reversion + DCF Scanner v5 — Professional Edition
Run:  python server.py
Open: http://localhost:7432

Thin HTTP shell — all engines live in the `core/` package.
"""
import http.server, socketserver, json, urllib.parse, os, sys, time, threading
from datetime import datetime

from core.config    import PORT, BASE_DIR, log, SP500, TOP50_VOL, YOUR_STOCKS, get_sector_etf
from core.data      import (fetch_ticker, fetch_fundamentals, fetch_quote_meta,
                            fetch_iv_premium, CACHE)
from core.mr_engine import analyze
from core.dcf_engine import analyze_dcf, dcf_backtest
from core.monte_carlo import monte_carlo
from core.correlation import build_correlation, spread_signals
from core.ml_optimizer import ga_state, run_ga_mr, run_ga_dcf
from core.storage import (load_journal, save_journal,
                          load_profiles, save_profiles,
                          get_mr_params, get_dcf_params, calibration_factor)

# ═══ HTTP ═══════════════════════════════════════════════════════════════════
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers(); self.wfile.write(body)

    def send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ─── GET ──────────────────────────────────────────────────────────────
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            ui = os.path.join(BASE_DIR, "ui.html")
            if os.path.exists(ui):
                with open(ui, encoding="utf-8") as f:
                    self.send_html(f.read())
            else:
                self.send_html("<h1>ui.html missing</h1>")
            return

        if path == "/api/scan":
            tickers = [t.strip().upper() for t in qs.get("tickers", [""])[0].split(",") if t.strip()]
            rp = int(qs.get("rsiP",  ["14"])[0]);  bs = float(qs.get("bbStd", ["2.0"])[0])
            zw = int(qs.get("zWin",  ["20"])[0]);  vm = float(qs.get("volMin", ["1.0"])[0])
            include_dcf = qs.get("dcf", ["0"])[0] in ("1", "true", "yes")
            include_meta = qs.get("meta", ["1"])[0] in ("1", "true", "yes")
            include_iv  = qs.get("iv",   ["0"])[0] in ("1", "true", "yes")
            include_sector = qs.get("sector", ["1"])[0] in ("1", "true", "yes")
            calib = calibration_factor()
            results = []
            for tk in tickers:
                raw = fetch_ticker(tk)
                if not raw: continue
                mr_params = get_mr_params(tk) or {"rsiP": rp, "bbStd": bs, "zWin": zw, "volMin": vm}
                meta = dict(fetch_quote_meta(tk) or {}) if include_meta else {}
                # Sector relative strength: pull/cache ETF closes, inject into meta
                if include_sector:
                    etf = get_sector_etf(tk)
                    if etf:
                        etf_raw = fetch_ticker(etf)
                        if etf_raw:
                            meta["sectorETF"] = etf
                            meta["etfCloses"] = [d["c"] for d in etf_raw["data"]]
                # IV premium (optional — one extra HTTP call per ticker)
                if include_iv:
                    ivm = fetch_iv_premium(tk)
                    if ivm:
                        meta.update({k: ivm.get(k) for k in ("ivATM","rv30","ivPremium","daysToExpiry")})
                meta["calibration"] = calib
                a = analyze(raw, mr_params, meta=meta or None)
                if not a: continue
                # Don't leak heavy ETF series back to client
                a.pop("etfCloses", None)
                a["hasProfile"] = bool(get_mr_params(tk))
                a.pop("closes60", None)
                if include_dcf:
                    f = fetch_fundamentals(tk)
                    if f:
                        dcf_params = get_dcf_params(tk)
                        da = analyze_dcf(f, dcf_params)
                        if da:
                            a["dcf"] = {
                                "ok": da.get("ok"),
                                "sig": da.get("sig"),
                                "ivMed": da.get("ivMed"),
                                "ivP25": da.get("ivP25"),
                                "ivP75": da.get("ivP75"),
                                "mosCons": da.get("mosCons"),
                                "mosMed":  da.get("mosMed"),
                                "impliedGrowth": da.get("impliedGrowth"),
                                "wacc": da.get("wacc"),
                                "accrualQuality": da.get("accrualQuality"),
                                "hasProfile": bool(dcf_params),
                            }
                    else:
                        a["dcf"] = {"ok": False, "sig": "unknown"}
                results.append(a)
            self.send_json({"ok": True, "results": results, "n": len(tickers), "got": len(results)})
            return

        if path == "/api/dcf":
            tickers = [t.strip().upper() for t in qs.get("tickers", [""])[0].split(",") if t.strip()]
            results = []
            for tk in tickers:
                f = fetch_fundamentals(tk)
                if not f:
                    results.append({"ticker": tk, "ok": False,
                                    "error": "fundamentals unavailable",
                                    "plain": "Fundamentals fetch failed — try again or check ticker."})
                    continue
                dcf_params = get_dcf_params(tk)
                a = analyze_dcf(f, dcf_params)
                if a:
                    a["hasProfile"] = bool(dcf_params)
                    results.append(a)
            self.send_json({"ok": True, "results": results, "n": len(tickers), "got": len(results)})
            return

        if path == "/api/dcf_detail":
            tk = qs.get("ticker", [""])[0].strip().upper()
            if not tk:
                self.send_json({"ok": False, "error": "ticker required"}, 400); return
            f = fetch_fundamentals(tk)
            if not f:
                self.send_json({"ok": False, "error": "fundamentals unavailable"}); return
            dcf_params = get_dcf_params(tk)
            a = analyze_dcf(f, dcf_params)
            price_data = fetch_ticker(tk)
            bt = None
            if price_data:
                bt = dcf_backtest(f, price_data["data"], dcf_params)
            a["backtest"] = bt
            a["hasProfile"] = bool(dcf_params)
            self.send_json({"ok": True, "result": a})
            return

        if path == "/api/correlation":
            tickers = [t.strip().upper() for t in qs.get("tickers", [""])[0].split(",") if t.strip()]
            if not tickers:
                tickers = list(CACHE.keys())
            matrix, pairs = build_correlation(tickers)
            spreads = spread_signals(pairs)
            self.send_json({"ok": True, "matrix": matrix, "pairs": pairs, "spreads": spreads})
            return

        if path == "/api/monte_carlo":
            wr   = float(qs.get("wr",   ["60"])[0])
            rr   = float(qs.get("rr",   ["2.0"])[0])
            nt   = int(  qs.get("nt",   ["52"])[0])
            cap  = float(qs.get("cap",  ["1000"])[0])
            risk = float(qs.get("risk", ["2.5"])[0])
            self.send_json({"ok": True, "result": monte_carlo(wr, rr, nt, cap, risk)})
            return

        if path == "/api/watchlists":
            self.send_json({"sp500": SP500, "top50vol": TOP50_VOL, "yours": YOUR_STOCKS})
            return

        if path == "/api/ga_status":
            self.send_json({**ga_state, "log": ga_state["log"][-40:]})
            return

        if path == "/api/journal":
            self.send_json({"ok": True, "trades": load_journal()})
            return

        if path == "/api/profiles":
            self.send_json({"ok": True, "profiles": load_profiles()})
            return

        self.send_json({"error": "not found"}, 404)

    # ─── POST ─────────────────────────────────────────────────────────────
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/ga_start":
            if ga_state["running"]:
                self.send_json({"ok": False, "error": "Already running"}); return
            tickers = body.get("tickers", [])
            closes_list = []
            for tk in [t.strip().upper() for t in tickers if t.strip()]:
                raw = fetch_ticker(tk)
                if raw and len(raw["data"]) >= 60:
                    closes_list.append([d["c"] for d in raw["data"]])
            if not closes_list:
                self.send_json({"ok": False, "error": "No data — run scanner first"}); return
            t = threading.Thread(
                target=run_ga_mr,
                args=(closes_list, body.get("gaParams", {}), body.get("ticker", "portfolio")),
                daemon=True)
            t.start()
            self.send_json({"ok": True, "stocks": len(closes_list)})
            return

        if parsed.path == "/api/ga_dcf_start":
            if ga_state["running"]:
                self.send_json({"ok": False, "error": "Already running"}); return
            tickers = [t.strip().upper() for t in body.get("tickers", []) if t.strip()]
            funds_map = {}; prices_map = {}
            for tk in tickers:
                f = fetch_fundamentals(tk)
                raw = fetch_ticker(tk)
                if f and raw:
                    funds_map[tk] = f
                    prices_map[tk] = raw["data"]
            if not funds_map:
                self.send_json({"ok": False, "error": "No fundamentals could be fetched for these tickers"}); return
            t = threading.Thread(
                target=run_ga_dcf,
                args=(funds_map, prices_map, body.get("gaParams", {}),
                      body.get("ticker", "portfolio")),
                daemon=True)
            t.start()
            self.send_json({"ok": True, "stocks": len(funds_map)})
            return

        if parsed.path == "/api/ga_stop":
            ga_state["running"] = False
            self.send_json({"ok": True}); return

        if parsed.path == "/api/journal/add":
            trades = load_journal()
            tr = body.get("trade", {})
            tr["id"] = int(time.time()*1000)
            tr["openDate"] = datetime.now().strftime("%Y-%m-%d")
            tr["status"] = "open"
            trades.append(tr); save_journal(trades)
            self.send_json({"ok": True, "id": tr["id"]}); return

        if parsed.path == "/api/journal/close":
            trades = load_journal()
            tid = body.get("id")
            ep = float(body.get("exitPrice", 0))
            for t in trades:
                if t["id"] == tid:
                    t["status"] = "closed"
                    t["closeDate"] = datetime.now().strftime("%Y-%m-%d")
                    t["exitPrice"] = ep
                    en = float(t.get("entry", ep))
                    t["pnlPct"] = round((ep-en)/en*100, 2) if t.get("direction") == "LONG" else round((en-ep)/en*100, 2)
                    t["pnl"] = round(float(t.get("size", 0)) * t["pnlPct"]/100, 2)
                    break
            save_journal(trades); self.send_json({"ok": True}); return

        if parsed.path == "/api/journal/delete":
            trades = [t for t in load_journal() if t["id"] != body.get("id")]
            save_journal(trades); self.send_json({"ok": True}); return

        self.send_json({"error": "not found"}, 404)

# ═══ main ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    ui_path = os.path.join(base_dir, "ui.html")
    print("="*60)
    print("  MR + DCF SCANNER v5 — Professional Edition")
    print("="*60)
    print(f"  URL   : http://localhost:{PORT}")
    print(f"  Folder: {base_dir}")
    if os.path.exists(ui_path): print("  UI    : ui.html OK")
    else:
        print("  ERROR : ui.html not found. Both files must be in same folder.")
        sys.exit(1)
    print("  Stop  : Ctrl+C")
    print("="*60)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.allow_reuse_address = True
        log(f"Server ready — http://localhost:{PORT}")
        try: httpd.serve_forever()
        except KeyboardInterrupt: print("\nStopped.")
