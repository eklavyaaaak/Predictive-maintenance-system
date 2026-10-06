"""Central configuration: paths, machine parameters and demonstration thresholds.

NOTE: All values here are PROJECT-DEFINED DEMONSTRATION VALUES for a
hypothetical 4-pole, 50 Hz induction motor (~1500 rpm synchronous speed).
They are NOT taken from a standard or a manufacturer datasheet.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # no hardcoded absolute paths
DATA_PATH = ROOT / "data" / "machine_sensor_data.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"
SEED = 42                                       # reproducibility

PARAMS = ["vibration_rms", "temperature", "current", "rpm"]
UNITS = {"vibration_rms": "mm/s RMS", "temperature": "°C", "current": "A", "rpm": "rpm"}

# "Healthy" operating point of the simulated machine
NOMINAL = {"rpm": 1480.0, "temperature": 55.0, "vibration_rms": 1.8, "current": 12.0}

# Project-defined demonstration thresholds (editable from the dashboard)
DEFAULT_THRESHOLDS = {
    "vibration_rms": {"warn": 2.8, "fault": 4.5},
    "temperature":   {"warn": 75.0, "fault": 90.0},
    "current":       {"warn": 14.5, "fault": 17.0},
    "rpm":           {"warn": 1540.0, "fault": 1600.0},   # overspeed side only
}

LABELS = ["NORMAL", "WARNING", "FAULT"]
