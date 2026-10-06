"""Transparent rule-based fault detection using PROJECT-DEFINED DEMONSTRATION THRESHOLDS."""
import numpy as np
import pandas as pd
from config import PARAMS, DEFAULT_THRESHOLDS

PARAM_NAME = {"vibration_rms": "High vibration", "temperature": "Overheating",
              "current": "Overload / high current", "rpm": "Overspeed"}

POSSIBLE_CAUSE = {
    "vibration_rms": "Possible imbalance, misalignment, looseness or bearing-related abnormality.",
    "temperature":   "Possible lubrication problem, poor cooling, bearing friction or excessive load.",
    "current":       "Possible overload, abnormal mechanical resistance or electrical supply issue.",
    "rpm":           "Possible control/drive setting problem or loss of load.",
}
ACTION = {
    "vibration_rms": "Inspect alignment, balance and bearing condition.",
    "temperature":   "Check lubrication, cooling and operating load.",
    "current":       "Inspect for overload or abnormal mechanical resistance.",
    "rpm":           "Check drive/speed settings and coupling/load condition.",
}
COMBINED_ACTION = ("Stop/inspect according to the applicable safety procedure and escalate "
                   "to qualified maintenance personnel.")
SAFETY_NOTE = ("Demonstration output only - not a substitute for actual industrial "
               "safety procedures or qualified judgement.")


def validate_thresholds(thr: dict) -> list:
    """Return list of problems (empty list = OK)."""
    return [f"{p}: warn must be lower than fault" for p in PARAMS if thr[p]["warn"] >= thr[p]["fault"]]


def _level(values, warn, fault):
    return np.select([values >= fault, values >= warn], [2, 1], default=0)


def evaluate_dataframe(df: pd.DataFrame, thr=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """Add level_<param> (0 ok / 1 warn / 2 fault) and rule_status columns."""
    out = df.copy()
    for p in PARAMS:
        out[f"level_{p}"] = _level(out[p].to_numpy(float), thr[p]["warn"], thr[p]["fault"])
    worst = out[[f"level_{p}" for p in PARAMS]].max(axis=1)
    out["rule_status"] = worst.map({0: "NORMAL", 1: "WARNING", 2: "FAULT"})
    return out


def diagnose(reading: dict, thr=DEFAULT_THRESHOLDS) -> dict:
    """Diagnosis for a single reading (dict with the 4 parameters)."""
    lv = evaluate_dataframe(pd.DataFrame([reading]), thr).iloc[0]
    levels = {p: int(lv[f"level_{p}"]) for p in PARAMS}
    abnormal = [p for p, l in levels.items() if l > 0]
    status = str(lv["rule_status"])
    if not abnormal:
        return dict(status=status, levels=levels, detected="No abnormality detected",
                    cause="-", action="Continue operation and monitor machine parameters.",
                    note=SAFETY_NOTE)
    if len(abnormal) >= 2:
        names = ", ".join(PARAM_NAME[p].lower() for p in abnormal)
        return dict(status=status, levels=levels, detected=f"Combined abnormal condition ({names})",
                    cause="Several parameters are abnormal together - possible developing mechanical "
                          "fault (e.g. bearing damage, severe misalignment, overload). The exact cause "
                          "cannot be determined from these four signals alone.",
                    action=COMBINED_ACTION, note=SAFETY_NOTE)
    p = abnormal[0]
    return dict(status=status, levels=levels, detected=PARAM_NAME[p],
                cause=POSSIBLE_CAUSE[p], action=ACTION[p], note=SAFETY_NOTE)
