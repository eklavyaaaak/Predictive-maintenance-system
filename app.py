"""Rotating Machinery Predictive Maintenance System - Streamlit dashboard.
Run:  streamlit run app.py      (ALL DATA IS SIMULATED)
"""
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (DATA_PATH, MODEL_PATH, RESULTS_DIR, SEED, PARAMS, UNITS, NOMINAL,
                    DEFAULT_THRESHOLDS, LABELS)
from data_generator import FAULT_TYPES, make_reading, load_dataset
from features import add_features
from fault_detection import diagnose, validate_thresholds, evaluate_dataframe
from health_score import (health_scores, health_score_single, health_status, trend_slope,
                          maintenance_risk, time_to_threshold)
from ml_model import train_and_evaluate, predict_latest

st.set_page_config(page_title="Rotating Machinery Predictive Maintenance", page_icon="⚙️", layout="wide")
COLORS = {"NORMAL": "#2e9d5b", "WARNING": "#f0a500", "FAULT": "#d64545",
          "LOW": "#2e9d5b", "MEDIUM": "#f0a500", "HIGH": "#d64545"}


# ---------- cached loaders ----------
@st.cache_data
def get_dataset():
    return load_dataset()


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():                       # first run: train automatically
        train_and_evaluate(get_dataset())
    return joblib.load(MODEL_PATH)


def get_metrics():
    p = RESULTS_DIR / "metrics.json"
    if not p.exists():
        train_and_evaluate(get_dataset())
    return json.loads(p.read_text(encoding="utf-8"))


def badge(text):
    return (f"<span style='background:{COLORS.get(text, '#888')};color:white;padding:4px 14px;"
            f"border-radius:14px;font-weight:700'>{text}</span>")


# ---------- simulation state ----------
def reset_simulation():
    rng = np.random.default_rng(SEED + 1)
    start, temp, rows = pd.Timestamp.now().floor("min") - pd.Timedelta(minutes=120), None, []
    for i in range(120):
        r = make_reading("normal", 0.0, temp, i, rng)
        temp = r["temperature"]
        rows.append({"timestamp": start + pd.Timedelta(minutes=i), **r})
    st.session_state.history = pd.DataFrame(rows)
    st.session_state.rng = rng


def add_reading(scenario, severity):
    h = st.session_state.history
    r = make_reading(scenario, severity, float(h.temperature.iloc[-1]), len(h), st.session_state.rng)
    row = {"timestamp": h.timestamp.iloc[-1] + pd.Timedelta(minutes=1), **r}
    st.session_state.history = pd.concat([h, pd.DataFrame([row])], ignore_index=True).tail(600).reset_index(drop=True)


if "history" not in st.session_state:
    reset_simulation()

# ---------- sidebar ----------
with st.sidebar:
    st.header("Simulator controls")
    scenario = st.selectbox("Scenario to inject", FAULT_TYPES, format_func=lambda s: s.replace("_", " ").title())
    severity = st.slider("Fault severity (0 = healthy, 1 = severe)", 0.0, 1.0, 0.0, 0.05,
                         disabled=(scenario == "normal"))
    if st.button("Generate New Reading", type="primary"):
        add_reading(scenario, 0.0 if scenario == "normal" else severity)
    if st.button("Simulate degradation (30 readings)", disabled=(scenario == "normal")):
        for s in np.linspace(severity, 1.0, 30):
            add_reading(scenario, float(s))
    if st.button("Reset simulation"):
        reset_simulation()
        st.rerun()

    st.header("Thresholds")
    st.caption("Project-defined demonstration thresholds - editable. Not from any standard.")
    thr = {}
    for p in PARAMS:
        with st.expander(f"{p.replace('_', ' ').title()} ({UNITS[p]})"):
            w = st.number_input("Warning at", value=float(DEFAULT_THRESHOLDS[p]["warn"]), key=f"w_{p}")
            f = st.number_input("Fault at", value=float(DEFAULT_THRESHOLDS[p]["fault"]), key=f"f_{p}")
            thr[p] = {"warn": w, "fault": f}
    problems = validate_thresholds(thr) or [f"{p}: warn must be above nominal ({NOMINAL[p]})"
                                            for p in PARAMS if thr[p]["warn"] <= NOMINAL[p]]
    if problems:
        st.error("Invalid thresholds - using defaults.\n\n" + "\n".join(problems))
        thr = DEFAULT_THRESHOLDS

# ---------- computations ----------
hist = st.session_state.history
feats = add_features(hist)
latest, prev = hist.iloc[-1], hist.iloc[-2]
reading = {p: float(latest[p]) for p in PARAMS}
diag = diagnose(reading, thr)
score, subs = health_score_single(reading, thr)
h_status = health_status(score)
v_slope, t_slope = trend_slope(hist.vibration_rms), trend_slope(hist.temperature)
risk, risk_level = maintenance_risk(score, v_slope, t_slope, thr)
bundle = get_model()
ml_pred, ml_proba = predict_latest(bundle, feats.tail(1))

# ---------- header + overview ----------
st.title("Rotating Machinery Predictive Maintenance System")
st.warning("**SIMULATED DATA** - no real machine or sensor is connected. Thresholds are project-defined "
           "demonstration values. This is a learning/portfolio project, not an industrial-certified system.")

st.subheader("A. Machine overview")
c = st.columns(6)
c[0].metric("RPM", f"{latest.rpm:.0f}", f"{latest.rpm - prev.rpm:+.1f}")
c[1].metric("Temperature", f"{latest.temperature:.1f} °C", f"{latest.temperature - prev.temperature:+.2f}", delta_color="inverse")
c[2].metric("Vibration", f"{latest.vibration_rms:.2f} mm/s", f"{latest.vibration_rms - prev.vibration_rms:+.3f}", delta_color="inverse")
c[3].metric("Current", f"{latest.current:.2f} A", f"{latest.current - prev.current:+.2f}", delta_color="inverse")
c[4].metric("Health score", f"{score:.0f} / 100")
c[5].markdown("**Status (rules)**<br>" + badge(diag["status"]), unsafe_allow_html=True)
st.caption("Use **Generate New Reading** in the sidebar (B. live/simulated monitoring).")

tabs = st.tabs(["C. Trends", "D. Machine health", "E. Fault diagnosis", "F. Predictive model", "Dataset & results"])

# ---------- C. trends ----------
with tabs[0]:
    view = hist.tail(200)
    cols = st.columns(2)
    for i, (p, title) in enumerate([("temperature", "Temperature vs Time"), ("vibration_rms", "Vibration vs Time"),
                                    ("rpm", "RPM vs Time"), ("current", "Current vs Time")]):
        fig = px.line(view, x="timestamp", y=p, title=title, labels={p: UNITS[p]})
        fig.add_hline(y=thr[p]["warn"], line_dash="dash", line_color="orange", annotation_text="warn")
        fig.add_hline(y=thr[p]["fault"], line_dash="dash", line_color="red", annotation_text="fault")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        cols[i % 2].plotly_chart(fig)
    st.caption("Dashed lines = project-defined demonstration thresholds.")

# ---------- D. health ----------
with tabs[1]:
    a, b = st.columns([1, 1])
    g = go.Figure(go.Indicator(mode="gauge+number", value=score, title={"text": "Machine Health Score"},
                               gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=COLORS[h_status]),
                                          steps=[dict(range=[0, 60], color="#f8d7d7"),
                                                 dict(range=[60, 80], color="#fdecc8"),
                                                 dict(range=[80, 100], color="#d8f0e0")])))
    g.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
    a.plotly_chart(g)
    b.markdown("**Status (from score):** " + badge(h_status), unsafe_allow_html=True)
    b.markdown("**Maintenance risk level:** " + badge(risk_level) + f" &nbsp; (risk index {risk:.0f}/100)",
               unsafe_allow_html=True)
    b.caption("Risk index = heuristic mix of current health and rising vibration/temperature trends. "
              "It is NOT a Remaining Useful Life estimate.")
    b.markdown("**Sub-scores (100 = at nominal, 79 = at warning threshold, 59 = at fault threshold)**")
    b.dataframe(pd.DataFrame({"parameter": PARAMS, "sub-score": [round(subs[p], 1) for p in PARAMS]}),
                hide_index=True)
    hs, _ = health_scores(hist, thr)
    fig = px.line(x=hist.timestamp.tail(200), y=hs[-200:], labels={"x": "time", "y": "health score"},
                  title="Health score over time")
    fig.add_hline(y=80, line_dash="dash", line_color="orange"); fig.add_hline(y=60, line_dash="dash", line_color="red")
    fig.update_layout(height=280, yaxis_range=[0, 105], margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig)
    st.markdown("**Trend-based indication** (linear extrapolation of last 30 readings, indicative only):")
    t_v = time_to_threshold(reading["vibration_rms"], v_slope, thr["vibration_rms"]["fault"])
    t_t = time_to_threshold(reading["temperature"], t_slope, thr["temperature"]["fault"])
    st.write(f"- Vibration slope: {v_slope:+.4f} mm/s per min -> "
             + (f"~{t_v:.0f} min to fault threshold" if t_v is not None else "not rising"))
    st.write(f"- Temperature slope: {t_slope:+.4f} °C per min -> "
             + (f"~{t_t:.0f} min to fault threshold" if t_t is not None else "not rising"))
    with st.expander("How the health score is calculated"):
        st.markdown("""
1. Each parameter gets a **sub-score** by linear interpolation: 100 at its nominal value, 79 at the warning
   threshold, 59 at the fault threshold, ~20 one more (fault-warning) span beyond that.
2. **Health = min(sub-scores) - 5 x (number of additional abnormal parameters)**, limited to 0-100.
3. Status: >= 80 NORMAL, 60-79 WARNING, < 60 FAULT.

*Weakest-link logic:* a machine is only as healthy as its worst parameter.""")

# ---------- E. diagnosis ----------
with tabs[2]:
    st.markdown("**Detected condition:** " + diag["detected"])
    st.markdown("**Possible cause:** " + diag["cause"])
    st.markdown("**Recommended action:** " + diag["action"])
    st.info(diag["note"])
    lv = {0: "OK", 1: "WARNING", 2: "FAULT"}
    st.dataframe(pd.DataFrame({"parameter": PARAMS, "value": [reading[p] for p in PARAMS],
                               "warn at": [thr[p]["warn"] for p in PARAMS],
                               "fault at": [thr[p]["fault"] for p in PARAMS],
                               "level": [lv[diag["levels"][p]] for p in PARAMS]}),
                 hide_index=True)

# ---------- F. model ----------
with tabs[3]:
    m = get_metrics()
    x, y = st.columns(2)
    x.markdown("**Predicted condition (ML):** " + badge(ml_pred), unsafe_allow_html=True)
    y.markdown("**Rule-based status:** " + badge(diag["status"]), unsafe_allow_html=True)
    st.plotly_chart(px.bar(x=list(ml_proba), y=list(ml_proba.values()), labels={"x": "class", "y": "probability"},
                           title="Class probabilities (latest reading)", range_y=[0, 1]).update_layout(height=250),
                    )
    k = st.columns(5)
    k[0].metric("Accuracy", f"{m['accuracy']:.3f}"); k[1].metric("Precision (macro)", f"{m['precision_macro']:.3f}")
    k[2].metric("Recall (macro)", f"{m['recall_macro']:.3f}"); k[3].metric("Group-CV accuracy", f"{m['cv_accuracy_mean']:.3f}")
    k[4].metric("Rule-based accuracy", f"{m['rule_based_accuracy']:.3f}")
    st.caption(f"Held-out test set: {m['n_test']} samples from whole runs never seen in training "
               f"(train: {m['n_train']}). {m['data_note']}")
    p, q = st.columns(2)
    cm = px.imshow(m["confusion_matrix"], x=LABELS, y=LABELS, text_auto=True, color_continuous_scale="Blues",
                   labels=dict(x="Predicted", y="True"), title="Confusion matrix (Random Forest, test set)")
    p.plotly_chart(cm)
    fi = pd.Series(m["feature_importance"]).sort_values()
    q.plotly_chart(px.bar(fi, orientation="h", title="Feature importance", labels={"value": "importance", "index": ""})
                   .update_layout(showlegend=False))

# ---------- dataset ----------
with tabs[4]:
    df = get_dataset()
    st.markdown(f"**Simulated dataset:** {len(df):,} rows, 1 sample/minute, {df.episode_id.nunique()} operating runs.")
    r1, r2 = st.columns(2)
    r1.plotly_chart(px.histogram(df, x="fault_type", color="machine_condition", color_discrete_map=COLORS,
                                 title="Scenarios and ground-truth condition"))
    ep = st.selectbox("Browse a simulated run", sorted(df.episode_id.unique()),
                      format_func=lambda e: f"Run {e} - {df[df.episode_id == e].fault_type.iloc[0]}")
    d = df[df.episode_id == ep]
    r2.plotly_chart(px.line(d, x="timestamp", y=["temperature", "vibration_rms", "current"],
                            title="Selected run (gradual degradation)"))
    ex = RESULTS_DIR / "example_fault_cases.csv"
    if ex.exists():
        st.markdown("**Example fault cases (severity ~0.9)**")
        st.dataframe(pd.read_csv(ex), hide_index=True)
