"""Machine health score (0-100), maintenance-risk indicator and simple trend extrapolation."""
import numpy as np
import pandas as pd
from config import PARAMS, NOMINAL, DEFAULT_THRESHOLDS

# Piece-wise linear sub-score for each parameter:
#   at nominal value   -> 100
#   at WARN threshold  ->  79   (top of the WARNING band)
#   at FAULT threshold ->  59   (just inside the FAULT band)
#   one more (fault-warn) beyond fault -> 20
SUB_POINTS = [100, 79, 59, 20]


def _subscores(values, param, thr):
    nom, w, f = NOMINAL[param], thr[param]["warn"], thr[param]["fault"]
    w = max(w, nom + 1e-6)
    f = max(f, w + 1e-6)
    return np.interp(values, [nom, w, f, f + (f - w)], SUB_POINTS)


def health_scores(df: pd.DataFrame, thr=DEFAULT_THRESHOLDS):
    """Weakest-link score:  Health = min(sub-scores) - 5 x (extra abnormal parameters).

    Engineering meaning: a machine is only as healthy as its worst parameter
    (one failing bearing matters even if current is perfect). If more than one
    parameter is already abnormal, an extra 5-point penalty per additional
    parameter reflects the higher risk of combined faults.
    """
    subs = np.vstack([_subscores(df[p].to_numpy(float), p, thr) for p in PARAMS])
    n_abn = (subs <= 79.5).sum(axis=0)
    score = np.clip(subs.min(axis=0) - 5 * np.maximum(n_abn - 1, 0), 0, 100)
    return score, subs


def health_score_single(reading: dict, thr=DEFAULT_THRESHOLDS):
    s, subs = health_scores(pd.DataFrame([reading]), thr)
    return float(s[0]), {p: float(subs[i, 0]) for i, p in enumerate(PARAMS)}


def health_status(score: float) -> str:
    return "NORMAL" if score >= 80 else ("WARNING" if score >= 60 else "FAULT")


def trend_slope(series, window=30) -> float:
    """Least-squares slope (units per sample/minute) over the last `window` points."""
    y = np.asarray(series, float)[-window:]
    return float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) >= 5 else 0.0


def maintenance_risk(health, vib_slope, temp_slope, thr=DEFAULT_THRESHOLDS, horizon=30):
    """Heuristic 0-100 maintenance-risk indicator (NOT Remaining Useful Life).

    risk = 0.7 * (100 - health) / 0.6  [scaled so health=60 -> 47]  +  0.3 * trend_score
    trend_score = how much of the (fault - nominal) gap vibration and temperature
    would close in `horizon` minutes if they keep rising at the current slope.
    """
    gap_v = max(thr["vibration_rms"]["fault"] - NOMINAL["vibration_rms"], 1e-6)
    gap_t = max(thr["temperature"]["fault"] - NOMINAL["temperature"], 1e-6)
    trend = np.clip(np.mean([vib_slope * horizon / gap_v, temp_slope * horizon / gap_t]), 0, 1) * 100
    risk = float(np.clip(0.7 * (100 - health) / 0.6 + 0.3 * trend, 0, 100))
    level = "LOW" if risk < 20 else ("MEDIUM" if risk < 40 else "HIGH")
    return risk, level


def time_to_threshold(value, slope, threshold):
    """Indicative linear extrapolation (minutes) until `threshold`; None if not rising."""
    if value >= threshold:
        return 0.0
    return (threshold - value) / slope if slope > 1e-6 else None
