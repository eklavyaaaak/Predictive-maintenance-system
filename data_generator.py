"""Synthetic sensor data for a rotating machine (motor/pump).

ALL DATA IS SIMULATED. It is built from simple engineering-inspired rules,
not measured on a real machine.

Design idea
-----------
The dataset is made of "episodes". Each episode is one operating run with one
scenario (normal, bearing degradation, imbalance, ...). Inside a fault episode
a hidden SEVERITY goes 0 -> 1 (gradual degradation). Sensor values respond to
severity with an accelerating curve (severity**1.5), because real faults
typically speed up as they develop. Ground-truth condition labels come from
severity (NOT from the detection thresholds), so the rule-based detector and
the ML model are both tested against an independent reference.
"""
import numpy as np
import pandas as pd
from config import NOMINAL, SEED, DATA_PATH

FAULT_TYPES = ["normal", "bearing_degradation", "imbalance", "misalignment",
               "overload", "overspeed", "combined"]

# Change in each signal at full severity (severity = 1)
EFFECTS = {
    "normal":              dict(vibration_rms=0.0, temperature=0.0,  current=0.0, rpm=0.0),
    "bearing_degradation": dict(vibration_rms=4.0, temperature=45.0, current=1.0, rpm=0.0),
    "imbalance":           dict(vibration_rms=5.0, temperature=4.0,  current=0.3, rpm=0.0),
    "misalignment":        dict(vibration_rms=4.0, temperature=12.0, current=2.0, rpm=0.0),
    "overload":            dict(vibration_rms=1.0, temperature=28.0, current=6.5, rpm=-60.0),
    "overspeed":           dict(vibration_rms=1.8, temperature=5.0,  current=0.5, rpm=190.0),
    "combined":            dict(vibration_rms=3.8, temperature=35.0, current=4.5, rpm=-20.0),
}


def severity_to_condition(s: float) -> str:
    """Ground-truth label from hidden severity (simulation design choice)."""
    return "NORMAL" if s < 0.35 else ("WARNING" if s < 0.70 else "FAULT")


def make_reading(fault_type, severity, prev_temp, t_index, rng):
    """One sensor reading. Also used by the dashboard's 'Generate New Reading'."""
    eff = EFFECTS[fault_type]
    g = float(severity) ** 1.5                       # accelerating degradation
    load = 1 + 0.04 * np.sin(2 * np.pi * t_index / 120)   # slow load variation

    rpm = NOMINAL["rpm"] + eff["rpm"] * g + rng.normal(0, 5)
    vib_add = eff["vibration_rms"] * g
    vib = max(0.2, NOMINAL["vibration_rms"] + vib_add + rng.normal(0, 0.08 + 0.03 * vib_add))
    cur = NOMINAL["current"] * load + eff["current"] * g + rng.normal(0, 0.25)

    # Temperature has thermal inertia: it moves slowly toward its target
    # (first-order lag) instead of jumping, like real machine housings.
    target = NOMINAL["temperature"] + 25 * (load - 1) + eff["temperature"] * g
    prev = target if prev_temp is None else prev_temp
    temp = prev + 0.15 * (target - prev) + rng.normal(0, 0.5)

    return dict(rpm=round(rpm, 1), temperature=round(temp, 2),
                vibration_rms=round(vib, 3), current=round(cur, 2))


def generate_dataset(episodes_per_type=4, episode_len=220, seed=SEED) -> pd.DataFrame:
    """7 scenarios x 4 episodes x 220 samples = 6160 rows (1 sample/minute)."""
    rng = np.random.default_rng(seed)
    order = [ft for ft in FAULT_TYPES for _ in range(episodes_per_type)]
    rng.shuffle(order)
    t0, k, rows = pd.Timestamp("2025-01-01 00:00"), 0, []
    for ep, ft in enumerate(order):
        prev = None
        for i in range(episode_len):
            s = 0.0 if ft == "normal" else i / (episode_len - 1)
            r = make_reading(ft, s, prev, i, rng)
            prev = r["temperature"]
            rows.append(dict(timestamp=t0 + pd.Timedelta(minutes=k), episode_id=ep, **r,
                             machine_condition=severity_to_condition(s),
                             fault_type=ft, severity=round(s, 4)))
            k += 1
    return pd.DataFrame(rows)


def save_dataset(df=None, path=DATA_PATH):
    df = generate_dataset() if df is None else df
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    try:
        return pd.read_csv(path, parse_dates=["timestamp"])
    except FileNotFoundError:
        return save_dataset()


if __name__ == "__main__":
    d = save_dataset()
    print(f"Saved {len(d)} simulated rows to {DATA_PATH}")
    print(d["machine_condition"].value_counts().to_string())
