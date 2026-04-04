from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from app.config import Config
from app.services.memory.analysis_store import AnalysisStore

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_stats",
            "description": "Get dataset shape, column names, and missingness. Optionally pass column for dtype and summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Optional column name for focused stats.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_rows",
            "description": "Filter processed dataset by one condition. Ops: eq, ne, gt, lt, gte, lte, contains.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "op": {"type": "string"},
                    "value": {"description": "String or number"},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["column", "op", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "column_value_counts",
            "description": "Top value frequencies for a column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "top_n": {"type": "integer", "default": 15},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_metrics",
            "description": "Return trained model performance metrics on holdout test split.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_feature_importance",
            "description": "Return top N feature importances from the random forest.",
            "parameters": {
                "type": "object",
                "properties": {"top_n": {"type": "integer", "default": 15}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_correlation",
            "description": "Pearson correlation between two numeric columns in the processed dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_a": {"type": "string"},
                    "column_b": {"type": "string"},
                },
                "required": ["column_a", "column_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_insights",
            "description": "List precomputed business insights.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_flags",
            "description": "Return deterministic data-quality and leakage risk flags.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_row",
            "description": "Run one prediction given feature values as JSON object (column names to values). Exclude target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values_json": {
                        "type": "string",
                        "description": 'JSON object e.g. {"age": 30, "city": "NYC"}',
                    }
                },
                "required": ["values_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_plots",
            "description": "List available plot URLs for this analysis session.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _df_or_error(store: AnalysisStore) -> tuple[pd.DataFrame | None, str | None]:
    if store.df_processed is None:
        return None, "No dataset loaded for this session."
    return store.df_processed, None


def run_tool(store: AnalysisStore, name: str, arguments: str | dict | None) -> str:
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            args = {}
    elif isinstance(arguments, dict):
        args = arguments
    else:
        args = {}

    try:
        if name == "get_summary_stats":
            return _tool_summary(store, args)
        if name == "filter_rows":
            return _tool_filter(store, args)
        if name == "column_value_counts":
            return _tool_value_counts(store, args)
        if name == "get_model_metrics":
            return json.dumps({"metrics": store.model_metrics}, default=str)
        if name == "get_feature_importance":
            return _tool_fi(store, args)
        if name == "get_correlation":
            return _tool_corr(store, args)
        if name == "list_insights":
            return json.dumps(
                {
                    "highlights": store.insight_highlights,
                    "details": store.insight_details,
                    "insights": store.insights,
                },
                default=str,
            )
        if name == "get_risk_flags":
            return json.dumps({"risks": store.risks}, default=str)
        if name == "predict_row":
            return _tool_predict(store, args)
        if name == "list_plots":
            return json.dumps({"plots": store.plot_paths}, default=str)
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


def _tool_summary(store: AnalysisStore, args: dict[str, Any]) -> str:
    df, err = _df_or_error(store)
    if err:
        return json.dumps({"error": err})
    col = args.get("column")
    if col:
        if col not in df.columns:
            return json.dumps({"error": f"Unknown column: {col}"})
        s = df[col]
        out = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_pct": float(s.isna().mean() * 100),
            "nunique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            out["describe"] = s.describe().to_dict()
        else:
            out["top_values"] = s.value_counts(dropna=True).head(10).to_dict()
        return json.dumps(out, default=str)
    return json.dumps(
        {
            "columns": list(df.columns),
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "missing_pct": {k: float(v) for k, v in (df.isna().mean() * 100).items()},
        },
        default=str,
    )


def _tool_filter(store: AnalysisStore, args: dict[str, Any]) -> str:
    df, err = _df_or_error(store)
    if err:
        return json.dumps({"error": err})
    col = args.get("column")
    op = (args.get("op") or "eq").lower()
    val = args.get("value")
    limit = min(int(args.get("limit") or 50), Config.MAX_CHAT_ROWS_RETURN)
    if col not in df.columns:
        return json.dumps({"error": f"Unknown column: {col}"})
    s = df[col]
    try:
        if op == "eq":
            mask = s == val
        elif op == "ne":
            mask = s != val
        elif op == "gt":
            mask = pd.to_numeric(s, errors="coerce") > float(val)
        elif op == "lt":
            mask = pd.to_numeric(s, errors="coerce") < float(val)
        elif op == "gte":
            mask = pd.to_numeric(s, errors="coerce") >= float(val)
        elif op == "lte":
            mask = pd.to_numeric(s, errors="coerce") <= float(val)
        elif op == "contains":
            mask = s.astype(str).str.contains(str(val), na=False)
        else:
            return json.dumps({"error": f"Unsupported op: {op}"})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})
    sub = df.loc[mask].head(limit)
    return json.dumps(
        {"rows": sub.to_dict(orient="records"), "returned": len(sub)}, default=str
    )


def _tool_value_counts(store: AnalysisStore, args: dict[str, Any]) -> str:
    df, err = _df_or_error(store)
    if err:
        return json.dumps({"error": err})
    col = args.get("column")
    top_n = int(args.get("top_n") or 15)
    if col not in df.columns:
        return json.dumps({"error": f"Unknown column: {col}"})
    vc = df[col].value_counts(dropna=True).head(top_n)
    return json.dumps({"counts": vc.to_dict()}, default=str)


def _tool_fi(store: AnalysisStore, args: dict[str, Any]) -> str:
    top_n = int(args.get("top_n") or 15)
    items = sorted(store.feature_importance.items(), key=lambda x: -x[1])[:top_n]
    return json.dumps({"feature_importance": dict(items)}, default=str)


def _tool_corr(store: AnalysisStore, args: dict[str, Any]) -> str:
    df, err = _df_or_error(store)
    if err:
        return json.dumps({"error": err})
    a = args.get("column_a")
    b = args.get("column_b")
    if a not in df.columns or b not in df.columns:
        return json.dumps({"error": "Unknown column(s)."})
    pair = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 2:
        return json.dumps({"error": "Not enough numeric data for correlation."})
    c = pair[a].corr(pair[b])
    return json.dumps({"correlation": float(c) if c == c else None})


def _tool_predict(store: AnalysisStore, args: dict[str, Any]) -> str:
    if store.model is None:
        return json.dumps({"error": "No trained model."})
    raw = args.get("values_json") or "{}"
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    if not isinstance(values, dict):
        return json.dumps({"error": "values_json must be a JSON object."})
    df, err = _df_or_error(store)
    if err:
        return json.dumps({"error": err})
    cols = [c for c in df.columns if c != store.target]
    row = {}
    for c in cols:
        if c in values:
            row[c] = values[c]
        else:
            row[c] = np.nan
    X = pd.DataFrame([row], columns=cols)
    try:
        pred = store.model.predict(X)[0]
        out: dict[str, Any] = {"prediction": pred}
        if store.task == "classification" and hasattr(
            store.model.named_steps["model"], "predict_proba"
        ):
            proba = store.model.predict_proba(X)
            out["probabilities_shape"] = list(proba.shape)
            out["class_index_argmax"] = int(np.argmax(proba[0]))
        return json.dumps(out, default=str)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})
