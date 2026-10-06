"""Feature extraction (causal: uses only past samples, so no future leakage).

Features are computed per episode so one run never leaks into another.
"""
import numpy as np
import pandas as pd

BASE = ["rpm", "temperature", "vibration_rms", "current"]
FEATURE_COLUMNS = BASE + [
    "rpm_ma", "temp_ma", "vib_ma", "curr_ma",                 # moving averages
    "temp_rate", "vib_rate", "curr_rate",                      # trend: change per sample (minute)
    "vib_roll_rms",                                            # RMS of vibration over the window
]


def add_features(df: pd.DataFrame, window: int = 10, group_col: str = "episode_id") -> pd.DataFrame:
    df = df.copy()
    if group_col not in df.columns:
        df[group_col] = 0
    g = df.groupby(group_col, sort=False)
    for col, short in [("rpm", "rpm"), ("temperature", "temp"),
                       ("vibration_rms", "vib"), ("current", "curr")]:
        df[f"{short}_ma"] = g[col].transform(lambda s: s.rolling(window, min_periods=1).mean())
    for col, short in [("temperature", "temp"), ("vibration_rms", "vib"), ("current", "curr")]:
        # Rate of change = (x_now - x_(window-1 samples ago)) / (window-1)  -> units per minute
        df[f"{short}_rate"] = g[col].transform(lambda s: (s - s.shift(window - 1)) / (window - 1))
    # Rolling RMS: sqrt(mean(v^2)) over the window
    df["vib_roll_rms"] = g["vibration_rms"].transform(
        lambda s: np.sqrt((s ** 2).rolling(window, min_periods=1).mean()))
    rate_cols = ["temp_rate", "vib_rate", "curr_rate"]
    df[rate_cols] = df[rate_cols].fillna(0.0)
    return df
