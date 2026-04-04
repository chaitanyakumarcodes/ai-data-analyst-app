from __future__ import annotations

import csv
from pathlib import Path

from app.services.analysis_service import run_full_analysis
from app.services.memory.analysis_store import get_store


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_classification_pipeline(tmp_path: Path) -> None:
    rows = []
    for i in range(80):
        rows.append(
            {
                "num": i % 10,
                "cat": "A" if i % 2 == 0 else "B",
                "target": "yes" if i % 3 == 0 else "no",
            }
        )
    csv_path = tmp_path / "d.csv"
    _write_csv(csv_path, rows)
    sid = "test-cls"
    app_root = tmp_path
    store = run_full_analysis(sid, csv_path, "target", "d.csv", app_root)
    assert store.error is None
    assert store.task == "classification"
    assert "accuracy" in store.model_metrics
    assert store.feature_importance
    assert get_store(sid) is not None


def test_regression_pipeline(tmp_path: Path) -> None:
    rows = [{"x": i, "y": 0.5 * i + (i % 4)} for i in range(60)]
    csv_path = tmp_path / "r.csv"
    _write_csv(csv_path, rows)
    sid = "test-reg"
    store = run_full_analysis(sid, csv_path, "y", "r.csv", tmp_path)
    assert store.error is None
    assert store.task == "regression"
    assert "r2" in store.model_metrics
    assert "rmse" in store.model_metrics
