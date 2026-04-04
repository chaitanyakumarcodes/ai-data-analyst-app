from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.config import Config
from app.services.data_layer import load_csv_from_path, validate_target
from app.services.insights.generator import generate_insights_openai
from app.services.insights.risk import compute_deterministic_risks
from app.services.memory.analysis_store import AnalysisStore, set_store
from app.services.pipeline import cleaning, eda
from app.services.pipeline.plots import (
    ensure_plot_dir,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_target_distribution,
)
from app.services.pipeline.train import train_model

logger = logging.getLogger(__name__)


def run_full_analysis(
    session_id: str,
    csv_path: Path,
    target: str,
    upload_filename: str,
    app_root: Path,
) -> AnalysisStore:
    store = AnalysisStore(session_id=session_id, upload_filename=upload_filename)
    df, read_err = load_csv_from_path(csv_path)
    if read_err or df.empty:
        store.error = read_err or "Empty dataset."
        set_store(store)
        return store

    err = validate_target(df, target)
    if err:
        store.error = err
        set_store(store)
        return store

    store.columns = list(df.columns)
    store.target = target

    df = cleaning.drop_missing_target(df, target)
    if len(df) < 5:
        store.error = "Not enough rows after removing missing target values."
        set_store(store)
        return store

    store.df_raw = df.copy()
    X, y = cleaning.separate_xy(df, target)
    task = cleaning.infer_task(y)
    store.task = task

    store.summary = eda.build_summary(df)
    store.correlations = eda.correlation_matrix_numeric(df)
    store.metadata = {
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "nunique": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
        "upload_filename": upload_filename,
    }
    store.risks = compute_deterministic_risks(df, target, task)

    plot_dir = ensure_plot_dir(session_id, app_root / Config.GENERATED_FOLDER)
    rel = f"generated/{session_id}"
    tdist = plot_target_distribution(y, task, plot_dir / "target_dist.png")
    store.plot_paths["target_dist"] = f"/static/{rel}/target_dist.png"

    ch = plot_correlation_heatmap(df, plot_dir / "corr_heatmap.png")
    if ch:
        store.plot_paths["corr_heatmap"] = f"/static/{rel}/corr_heatmap.png"

    try:
        pipe, metrics, fi, feat_names, _le = train_model(X, y, task)
    except Exception as e:  # noqa: BLE001
        logger.exception("Training failed")
        store.error = f"Model training failed: {e}"
        set_store(store)
        return store

    store.model = pipe
    store.model_metrics = metrics
    store.feature_importance = fi
    store.feature_names = feat_names
    store.df_processed = df

    fi_path = plot_dir / "feature_importance.png"
    plot_feature_importance(fi, fi_path)
    store.plot_paths["feature_importance"] = f"/static/{rel}/feature_importance.png"

    try:
        hi, det = generate_insights_openai(store)
        store.insight_highlights = hi
        store.insight_details = det
        store.insights = [*hi, *det]
    except Exception as e:  # noqa: BLE001
        logger.exception("Insight generation failed")
        err = [f"Insight generation failed: {e}"]
        store.insight_highlights = []
        store.insight_details = err
        store.insights = err

    set_store(store)
    return store
