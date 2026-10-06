"""Random Forest condition classifier with leakage-safe evaluation.

Leakage control
---------------
* Split is BY EPISODE (whole runs held out), never by random rows - neighbouring
  samples in a time series are almost identical, so a random split would leak.
* One episode per scenario is held out for testing; all rolling features are causal.
* The hidden `severity` and `fault_type` columns are NOT used as features.
"""
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import GroupKFold, cross_val_score
from config import LABELS, MODEL_PATH, RESULTS_DIR, SEED, DEFAULT_THRESHOLDS
from features import add_features, FEATURE_COLUMNS
from fault_detection import evaluate_dataframe
from data_generator import load_dataset


def split_by_episode(df):
    test_ids = df.groupby("fault_type")["episode_id"].max().to_numpy()   # 1 held-out run per scenario
    mask = df["episode_id"].isin(test_ids)
    return df[~mask], df[mask]


def train_and_evaluate(df=None, save=True) -> dict:
    df = load_dataset() if df is None else df
    train, test = split_by_episode(add_features(df))
    X_tr, y_tr, X_te, y_te = train[FEATURE_COLUMNS], train["machine_condition"], \
        test[FEATURE_COLUMNS], test["machine_condition"]

    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                   class_weight="balanced", random_state=SEED, n_jobs=-1)
    cv = cross_val_score(model, X_tr, y_tr, groups=train["episode_id"],
                         cv=GroupKFold(n_splits=5), scoring="accuracy")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    rule_pred = evaluate_dataframe(test, DEFAULT_THRESHOLDS)["rule_status"]
    metrics = dict(
        n_train=int(len(train)), n_test=int(len(test)),
        accuracy=float(accuracy_score(y_te, pred)),
        precision_macro=float(precision_score(y_te, pred, average="macro", labels=LABELS, zero_division=0)),
        recall_macro=float(recall_score(y_te, pred, average="macro", labels=LABELS, zero_division=0)),
        f1_macro=float(f1_score(y_te, pred, average="macro", labels=LABELS, zero_division=0)),
        cv_accuracy_mean=float(cv.mean()), cv_accuracy_std=float(cv.std()),
        rule_based_accuracy=float(accuracy_score(y_te, rule_pred)),
        labels=LABELS,
        confusion_matrix=confusion_matrix(y_te, pred, labels=LABELS).tolist(),
        rule_confusion_matrix=confusion_matrix(y_te, rule_pred, labels=LABELS).tolist(),
        per_class=classification_report(y_te, pred, labels=LABELS, output_dict=True, zero_division=0),
        feature_importance=dict(sorted(zip(FEATURE_COLUMNS, map(float, model.feature_importances_)),
                                       key=lambda kv: -kv[1])),
        data_note="Metrics are on SIMULATED data only; they do not predict real-machine performance.",
    )
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(model=model, features=FEATURE_COLUMNS, labels=LABELS), MODEL_PATH)
        (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def predict_latest(bundle, feature_row):
    """Predict condition + class probabilities for one feature row (DataFrame with 1 row)."""
    X = feature_row[bundle["features"]]
    proba = bundle["model"].predict_proba(X)[0]
    classes = list(bundle["model"].classes_)
    return classes[int(np.argmax(proba))], dict(zip(classes, map(float, proba)))


if __name__ == "__main__":
    m = train_and_evaluate()
    print(f"Test accuracy {m['accuracy']:.3f} | macro precision {m['precision_macro']:.3f} | "
          f"macro recall {m['recall_macro']:.3f} | rule-based accuracy {m['rule_based_accuracy']:.3f}")
