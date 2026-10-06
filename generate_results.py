"""One command to rebuild everything: data -> model -> result figures -> flowchart.
Run:  python generate_results.py
"""
import json
import matplotlib
matplotlib.use("Agg")                      # no display needed (works on any laptop)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import RESULTS_DIR, ASSETS_DIR, DEFAULT_THRESHOLDS, LABELS, PARAMS
from data_generator import generate_dataset, save_dataset
from ml_model import train_and_evaluate
from fault_detection import diagnose
from health_score import health_scores

COL = {"NORMAL": "#2e9d5b", "WARNING": "#f0a500", "FAULT": "#d64545"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / name, dpi=130)
    plt.close(fig)


def make_flowchart(path):
    steps = ["Rotating Machine (motor / pump)", "Sensor Parameters\n(temperature, vibration, RPM, current)",
             "Data Acquisition (simulated here)", "Data Preprocessing\n(cleaning, timestamps)",
             "Feature Extraction\n(moving avg, rolling RMS, trends)",
             "Fault Detection (rules)  +  ML Model (Random Forest)", "Machine Health Score (0-100)",
             "Condition Classification\nNORMAL / WARNING / FAULT", "Maintenance Recommendation"]
    fig, ax = plt.subplots(figsize=(7, 11)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, len(steps))
    for i, t in enumerate(steps):
        y = len(steps) - i - 0.5
        ax.text(0.5, y, t, ha="center", va="center", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.6", fc="#e8f0fe" if i not in (5, 8) else "#fff4d6", ec="#1f4e99"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.62), xytext=(0.5, y - 0.38),
                        arrowprops=dict(arrowstyle="-|>", color="#1f4e99", lw=2))
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True); ASSETS_DIR.mkdir(exist_ok=True)
    df = save_dataset(generate_dataset())
    m = train_and_evaluate(df)
    make_flowchart(ASSETS_DIR / "flowchart.png")

    # 1. Fault / condition distribution
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    df["fault_type"].value_counts().plot.bar(ax=ax[0], color="#1f4e99"); ax[0].set_title("Samples per scenario (simulated)")
    df["machine_condition"].value_counts()[LABELS].plot.bar(ax=ax[1], color=[COL[l] for l in LABELS]); ax[1].set_title("Condition distribution")
    save(fig, "fault_distribution.png")

    # 2. Sensor trends for one degradation episode (bearing) + health score
    ep = df[df.fault_type == "bearing_degradation"].episode_id.iloc[0]
    e = df[df.episode_id == ep].reset_index(drop=True)
    hs, _ = health_scores(e)
    fig, axs = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for a, (c, t) in zip(axs, [("temperature", "Temperature (°C)"), ("vibration_rms", "Vibration (mm/s RMS)"),
                               ("rpm", "RPM"), ("current", "Current (A)")]):
        a.plot(e[c], lw=1); a.set_ylabel(t)
        if c in DEFAULT_THRESHOLDS:
            a.axhline(DEFAULT_THRESHOLDS[c]["warn"], color="orange", ls="--"); a.axhline(DEFAULT_THRESHOLDS[c]["fault"], color="red", ls="--")
    axs[4].plot(hs, color="k"); axs[4].axhline(80, color="orange", ls="--"); axs[4].axhline(60, color="red", ls="--")
    axs[4].set_ylabel("Health score"); axs[4].set_xlabel("Sample (minute)")
    axs[0].set_title("Simulated bearing-degradation run (dashed = demo warn/fault thresholds)")
    save(fig, "sensor_trends_bearing_episode.png")

    # 3. Confusion matrices: ML vs rules
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, (cm, t) in zip(ax, [(m["confusion_matrix"], f"Random Forest (acc {m['accuracy']:.3f})"),
                               (m["rule_confusion_matrix"], f"Rule-based (acc {m['rule_based_accuracy']:.3f})")]):
        a.imshow(cm, cmap="Blues"); a.set_xticks(range(3), LABELS); a.set_yticks(range(3), LABELS)
        a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title(t)
        for i in range(3):
            for j in range(3):
                a.text(j, i, cm[i][j], ha="center", va="center", color="black")
    save(fig, "confusion_matrices.png")

    # 4. Feature importance
    fi = pd.Series(m["feature_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5)); fi.plot.barh(ax=ax, color="#1f4e99"); ax.set_title("Random Forest feature importance")
    save(fig, "feature_importance.png")

    # 5. Example fault cases (one reading per scenario near severity 0.9)
    rows = []
    for ft, g in df[df.fault_type != "normal"].groupby("fault_type"):
        r = g.iloc[(g.severity - 0.9).abs().argsort().iloc[0]]
        d = diagnose({p: r[p] for p in PARAMS})
        sc, _ = health_scores(pd.DataFrame([r]))
        rows.append(dict(scenario=ft, rpm=r.rpm, temperature=r.temperature, vibration_rms=r.vibration_rms,
                         current=r.current, health_score=round(float(sc[0]), 1), rule_status=d["status"],
                         detected=d["detected"], recommended_action=d["action"]))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "example_fault_cases.csv", index=False)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                      if k in ("n_train", "n_test", "accuracy", "precision_macro", "recall_macro",
                               "f1_macro", "cv_accuracy_mean", "cv_accuracy_std", "rule_based_accuracy")}, indent=2))
    print("Results written to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
