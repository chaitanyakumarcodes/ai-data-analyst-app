"""Presentation helpers for the analysis dashboard (metrics layout, plot order)."""

from __future__ import annotations

from typing import Any

# Preferred visual order; unknown keys appended at end.
_PLOT_ORDER = (
    "target_dist",
    "feature_importance",
    "corr_heatmap",
)

_PLOT_TITLES = {
    "target_dist": "Target distribution",
    "feature_importance": "Feature importance",
    "corr_heatmap": "Correlation heatmap",
}

_METRIC_META: dict[str, dict[str, Any]] = {
    "accuracy": {"label": "Accuracy", "format": "percent"},
    "f1_macro": {"label": "F1 score (macro)", "format": "float"},
    "roc_auc": {"label": "ROC AUC", "format": "float"},
    "roc_auc_ovr": {"label": "ROC AUC (OvR, weighted)", "format": "float"},
    "rmse": {"label": "RMSE", "format": "float"},
    "mae": {"label": "MAE", "format": "float"},
    "r2": {"label": "R²", "format": "float"},
}


def _fmt_value(fmt: str, v: Any) -> str:
    if v is None:
        return "—"
    try:
        if fmt == "percent":
            return f"{float(v) * 100:.2f}%"
        if fmt == "float":
            return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)
    return str(v)


def metric_cards(model_metrics: dict[str, Any]) -> list[dict[str, str]]:
    """One card per metric with label, display value, raw key."""
    cards: list[dict[str, str]] = []
    seen: set[str] = set()
    for key, meta in _METRIC_META.items():
        if key not in model_metrics:
            continue
        seen.add(key)
        val = model_metrics[key]
        cards.append(
            {
                "key": key,
                "label": str(meta["label"]),
                "value": _fmt_value(str(meta["format"]), val),
            }
        )
    for key, val in model_metrics.items():
        if key in seen:
            continue
        cards.append(
            {
                "key": key,
                "label": key.replace("_", " ").title(),
                "value": _fmt_value("float", val) if isinstance(val, (int, float)) else str(val),
            }
        )
    return cards


def ordered_plots(plot_paths: dict[str, str]) -> list[dict[str, str]]:
    """Stable order and human titles for charts."""
    out: list[dict[str, str]] = []
    used: set[str] = set()
    for key in _PLOT_ORDER:
        if key in plot_paths:
            used.add(key)
            out.append(
                {
                    "key": key,
                    "title": _PLOT_TITLES.get(key, key.replace("_", " ").title()),
                    "src": plot_paths[key],
                }
            )
    for key, src in plot_paths.items():
        if key in used:
            continue
        out.append(
            {
                "key": key,
                "title": key.replace("_", " ").title(),
                "src": src,
            }
        )
    return out
