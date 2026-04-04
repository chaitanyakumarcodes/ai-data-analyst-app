from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class AnalysisStore:
    """Session-scoped memory for analysis artifacts (Phase 2 schema)."""

    session_id: str
    columns: list[str] = field(default_factory=list)
    target: str = ""
    task: str = ""  # "classification" | "regression"
    summary: dict[str, Any] = field(default_factory=dict)
    correlations: dict[str, Any] = field(default_factory=dict)
    model_metrics: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)
    insight_highlights: list[str] = field(default_factory=list)
    insight_details: list[str] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    plot_paths: dict[str, str] = field(default_factory=dict)
    df_raw: pd.DataFrame | None = None
    df_processed: pd.DataFrame | None = None
    model: Pipeline | None = None
    feature_names: list[str] = field(default_factory=list)
    upload_filename: str = ""
    error: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "target": self.target,
            "task": self.task,
            "summary": self.summary,
            "correlations": self.correlations,
            "model_metrics": self.model_metrics,
            "feature_importance": self.feature_importance,
            "metadata": self.metadata,
            "insights": self.insights,
            "insight_highlights": self.insight_highlights,
            "insight_details": self.insight_details,
            "risks": self.risks,
            "plot_paths": self.plot_paths,
        }


_registry: dict[str, AnalysisStore] = {}
_lock = threading.Lock()


def get_store(session_id: str) -> AnalysisStore | None:
    with _lock:
        return _registry.get(session_id)


def set_store(store: AnalysisStore) -> None:
    with _lock:
        _registry[store.session_id] = store


def delete_store(session_id: str) -> None:
    with _lock:
        _registry.pop(session_id, None)
