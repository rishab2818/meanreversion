"""Cross-sectional multi-factor ranking for scan results."""
import math


def _median(values):
    seq = sorted(values)
    n = len(seq)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return seq[mid]
    return (seq[mid - 1] + seq[mid]) / 2


def _mad(values, center):
    return _median([abs(v - center) for v in values])


def _robust_z_map(rows, field_fn):
    prepared = []
    for row in rows:
        value = field_fn(row)
        if value is not None and math.isfinite(value):
            prepared.append((row, value))
    if not prepared:
        return {id(row): 0.0 for row in rows}
    vals = [v for _, v in prepared]
    med = _median(vals)
    mad = _mad(vals, med)
    if mad > 1e-9:
        scale = 1.4826 * mad
        return {id(row): max(-3.0, min(3.0, (field_fn(row) - med) / scale)) if field_fn(row) is not None else 0.0 for row in rows}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
    sd = math.sqrt(var)
    if sd < 1e-9:
        return {id(row): 0.0 for row in rows}
    return {id(row): max(-3.0, min(3.0, ((field_fn(row) or mean) - mean) / sd)) for row in rows}


def _dcf_alignment(row):
    dcf = row.get("dcf") or {}
    mos = dcf.get("mosCons")
    if mos is None:
        return None
    direction = row.get("direction")
    if direction == "LONG":
        return mos
    if direction == "SHORT":
        return -mos
    return mos * 0.3


def _signal_quality(row):
    sig = row.get("sig")
    if sig in ("strong-buy", "strong-sell"):
        return 1.0
    if sig in ("buy", "sell"):
        return 0.55
    return 0.2


def rank_scan_results(results):
    rows = [dict(r) for r in (results or [])]
    if not rows:
        return []

    factor_defs = [
        ("winRate", 0.22, lambda r: r.get("winRate")),
        ("avgRet", 0.12, lambda r: r.get("avgRet")),
        ("sharpe", 0.14, lambda r: r.get("sharpe")),
        ("signal", 0.11, lambda r: abs(r.get("score") or 0)),
        ("dcf", 0.12, _dcf_alignment),
        ("stability", 0.08, lambda r: r.get("medianFoldWR")),
        ("drawdown", 0.07, lambda r: -(r.get("maxDD") or 0)),
        ("foldSpread", 0.05, lambda r: -(r.get("foldSpread") or 0)),
        ("regime", 0.05, lambda r: r.get("regimeWR") if r.get("regimeWR") is not None else r.get("winRate")),
        ("riskReward", 0.03, lambda r: r.get("rr")),
        ("freshness", 0.03, lambda r: -(r.get("signalAge") or 0)),
        ("liquidity", 0.02, lambda r: r.get("volRatio")),
        ("sample", 0.04, lambda r: math.sqrt(max(r.get("trades") or 0, 0))),
    ]

    z_maps = {name: _robust_z_map(rows, fn) for name, _, fn in factor_defs}
    ranked = []
    for row in rows:
        factors = {}
        total = 0.0
        quality = _signal_quality(row)
        for name, weight, _ in factor_defs:
            raw = z_maps[name][id(row)]
            factors[name] = round(raw, 3)
            total += weight * raw
        total *= (0.7 + 0.3 * quality)
        if row.get("direction") == "WAIT":
            total *= 0.8
        row["rankFactors"] = factors
        row["rankScore"] = round(50 + total * 12, 2)
        ranked.append(row)

    ranked.sort(key=lambda r: (-r.get("rankScore", 0), -abs(r.get("score") or 0), -(r.get("winRate") or 0), r.get("ticker") or ""))
    n = len(ranked)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
        row["rankPct"] = round((n - idx + 1) / n * 100, 1)
    return ranked
