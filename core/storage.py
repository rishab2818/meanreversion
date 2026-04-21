"""Persistence — trade journal + per-stock ML profiles."""
import os, json
from core.config import JOURNAL_F, PROFILES_F

def load_journal():
    if os.path.exists(JOURNAL_F):
        try:
            with open(JOURNAL_F, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_journal(t):
    with open(JOURNAL_F, "w", encoding="utf-8") as f:
        json.dump(t, f, indent=2)

def load_profiles():
    if os.path.exists(PROFILES_F):
        try:
            with open(PROFILES_F, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_profiles(p):
    with open(PROFILES_F, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2)

def get_mr_params(ticker):
    """Returns tuned MR params for ticker, or None."""
    profiles = load_profiles()
    prof = profiles.get(ticker, {})
    if "mr" in prof:
        return prof["mr"].get("params")
    # legacy shape
    return prof.get("params")

def get_dcf_params(ticker):
    profiles = load_profiles()
    prof = profiles.get(ticker, {})
    if "dcf" in prof:
        return prof["dcf"].get("params")
    return None

def calibration_factor():
    """Calibration factor for Kelly sizing, based on CLOSED journal trades.

    factor = realized_WR / predicted_WR, clamped to [0.3, 1.3].

    <1.0 → model is overconfident; shrink size.
    >1.0 → model is conservative; allow slightly larger size.
    With <5 closed trades we return 1.0 (insufficient data — don't bias yet).
    """
    trades = [t for t in load_journal() if t.get("status") == "closed"]
    # Need enough trades with a recorded predicted win rate to calibrate
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
    return {"factor": round(f, 3), "n": len(calib_trades),
            "predWR": round(pred, 1), "realWR": round(real, 1)}
