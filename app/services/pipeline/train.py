from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.services.pipeline.feature_engineering import build_preprocessor


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, classes: int
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    if classes == 2 and y_prob is not None:
        try:
            proba = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            out["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:  # noqa: BLE001
            out["roc_auc"] = None
    elif classes > 2 and y_prob is not None:
        try:
            out["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            )
        except Exception:  # noqa: BLE001
            out["roc_auc_ovr"] = None
    return out


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any], dict[str, float], list[str], LabelEncoder | None]:
    """
    Fit preprocessor + RandomForest. Returns pipeline, metrics dict,
    feature_importance (name -> score), feature names, optional LabelEncoder for y.
    """
    le: LabelEncoder | None = None
    if task == "classification":
        if y.dtype == object or str(y.dtype).startswith("category"):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y.astype(str))
        else:
            y_encoded = np.asarray(y)
        _, counts = np.unique(y_encoded, return_counts=True)
        stratify = y_encoded if counts.min() >= 2 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=0.2,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=0.2,
                random_state=random_state,
            )
    else:
        y_num = pd.to_numeric(y, errors="coerce")
        mask = y_num.notna()
        X = X.loc[mask].reset_index(drop=True)
        y_num = y_num.loc[mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_num,
            test_size=0.2,
            random_state=random_state,
        )

    preprocessor = build_preprocessor(X_train)
    if task == "classification":
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=20,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    else:
        clf = RandomForestRegressor(
            n_estimators=120,
            max_depth=20,
            random_state=random_state,
            n_jobs=-1,
        )

    pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    if task == "classification":
        y_prob = None
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            try:
                y_prob = pipe.predict_proba(X_test)
            except Exception:  # noqa: BLE001
                y_prob = None
        metrics = _classification_metrics(
            y_test, y_pred, y_prob, len(np.unique(y_train))
        )
    else:
        metrics = _regression_metrics(np.asarray(y_test), y_pred)

    model = pipe.named_steps["model"]
    importances = model.feature_importances_
    try:
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
    except Exception:  # noqa: BLE001
        names = np.array([f"f{i}" for i in range(len(importances))])
    fi = {str(names[i]): float(importances[i]) for i in range(len(importances))}
    fi_sorted = dict(sorted(fi.items(), key=lambda x: -x[1])[:50])

    return pipe, metrics, fi_sorted, list(names), le
