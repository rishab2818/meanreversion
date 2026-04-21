"""Persistence â€” trade journal + per-stock ML profiles."""
import json
import os

from core.config import JOURNAL_F, PROFILES_F


def load_json_file(path, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def save_json_file(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_journal():
    return load_json_file(JOURNAL_F, [])


def save_journal(t):
    save_json_file(JOURNAL_F, t)


def load_profiles():
    return load_json_file(PROFILES_F, {})


def save_profiles(p):
    save_json_file(PROFILES_F, p)


def get_mr_params(ticker):
    """Returns tuned MR params for ticker, or None."""
    profiles = load_profiles()
    prof = profiles.get(ticker, {})
    if "mr" in prof:
        return prof["mr"].get("params")
    return prof.get("params")


def get_dcf_params(ticker):
    profiles = load_profiles()
    prof = profiles.get(ticker, {})
    if "dcf" in prof:
        return prof["dcf"].get("params")
    return None


def calibration_factor():
    """Calibration factor for Kelly sizing, based on CLOSED journal trades."""
    trades = [t for t in load_journal() if t.get("status") == "closed"]
    calib_trades = [t for t in trades if t.get("predictedWR") is not None]
    if len(calib_trades) < 5:
        return {"factor": 1.0, "n": len(calib_trades), "predWR": None, "realWR": None}
    pred = sum(t["predictedWR"] for t in calib_trades) / len(calib_trades)
    wins = sum(1 for t in calib_trades if (t.get("pnlPct") or 0) > 0)
    real = wins / len(calib_trades) * 100
    if pred <= 0:
        f = 1.0
    else:
        f = max(0.3, min(1.3, real / pred))
    return {
        "factor": round(f, 3),
        "n": len(calib_trades),
        "predWR": round(pred, 1),
        "realWR": round(real, 1),
    }
