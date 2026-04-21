"""Scan history persistence and snapshot summaries."""
import hashlib
from datetime import datetime

from core.config import SCAN_HISTORY_F
from core.storage import load_json_file, save_json_file

MAX_SCAN_SNAPSHOTS = 180
MAX_RESULTS_PER_SNAPSHOT = 80


def load_scan_history():
    data = load_json_file(SCAN_HISTORY_F, [])
    return data if isinstance(data, list) else []


def save_scan_history(history):
    save_json_file(SCAN_HISTORY_F, history[-MAX_SCAN_SNAPSHOTS:])


def get_scan_snapshot(snapshot_id):
    for snap in load_scan_history():
        if snap.get("id") == snapshot_id:
            return snap
    return None


def _slim_result(row):
    dcf = row.get("dcf") or {}
    return {
        "ticker": row.get("ticker"),
        "name": row.get("name"),
        "sig": row.get("sig"),
        "direction": row.get("direction"),
        "cur": row.get("cur"),
        "score": row.get("score"),
        "rank": row.get("rank"),
        "rankScore": row.get("rankScore"),
        "rankPct": row.get("rankPct"),
        "winRate": row.get("winRate"),
        "avgRet": row.get("avgRet"),
        "sharpe": row.get("sharpe"),
        "rr": row.get("rr"),
        "regime": row.get("regime"),
        "signalAge": row.get("signalAge"),
        "dcf": {
            "ok": dcf.get("ok"),
            "sig": dcf.get("sig"),
            "mosCons": dcf.get("mosCons"),
        } if dcf else None,
        "entry": row.get("entry"),
        "stop": row.get("stop"),
        "t1": row.get("t1"),
    }


def _summary(results):
    action = [r for r in results if r.get("sig") in ("strong-buy", "buy", "strong-sell", "sell")]
    buys = [r for r in results if r.get("direction") == "LONG"]
    sells = [r for r in results if r.get("direction") == "SHORT"]
    best = max(results, key=lambda r: r.get("rankScore", 0), default=None)
    avg_rank = round(sum(r.get("rankScore", 0) for r in results) / len(results), 2) if results else 0
    avg_wr = round(sum(r.get("winRate", 0) for r in results) / len(results), 1) if results else 0
    return {
        "scanned": len(results),
        "actionable": len(action),
        "buys": len(buys),
        "sells": len(sells),
        "avgRankScore": avg_rank,
        "avgWinRate": avg_wr,
        "bestTicker": best.get("ticker") if best else None,
        "bestScore": best.get("rankScore") if best else None,
    }


def record_scan_snapshot(results, request_meta=None):
    results = list(results or [])
    if not results:
        return None
    meta = request_meta or {}
    ts = datetime.now().isoformat(timespec="seconds")
    digest_src = f"{ts}|{meta.get('watchlist')}|{','.join(sorted(r.get('ticker') or '' for r in results[:10]))}"
    snapshot_id = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:12]
    snap = {
        "id": snapshot_id,
        "createdAt": ts,
        "meta": {
            "watchlist": meta.get("watchlist"),
            "market": meta.get("market"),
            "tickerCount": meta.get("tickerCount"),
            "scanMode": meta.get("scanMode", "mr"),
        },
        "summary": _summary(results),
        "results": [_slim_result(r) for r in results[:MAX_RESULTS_PER_SNAPSHOT]],
    }
    history = load_scan_history()
    history.append(snap)
    save_scan_history(history)
    return snap
