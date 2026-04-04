from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from app.config import Config
from app.services.memory.analysis_store import AnalysisStore


def _excerpt_for_llm(store: AnalysisStore, max_chars: int = 12000) -> str:
    pub = store.to_public_dict()
    top_fi = sorted(
        store.feature_importance.items(), key=lambda x: -x[1]
    )[:15]
    payload = {
        "task": store.task,
        "target": store.target,
        "metrics": store.model_metrics,
        "top_feature_importance": dict(top_fi),
        "summary_snippet": {
            "n_rows": store.summary.get("n_rows"),
            "n_columns": store.summary.get("n_columns"),
            "missing_pct": store.summary.get("missing_pct", {}),
        },
        "risks": store.risks,
        "correlations_sample": _corr_sample(store.correlations),
    }
    s = json.dumps(payload, default=str)
    if len(s) > max_chars:
        return s[:max_chars] + "\n...[truncated]"
    return s


def _corr_sample(correlations: dict[str, Any], limit: int = 30) -> dict[str, Any]:
    if not correlations:
        return {}
    pairs: list[tuple[str, str, float]] = []
    for a, row in list(correlations.items())[:15]:
        if not isinstance(row, dict):
            continue
        for b, v in row.items():
            if a >= b:
                continue
            try:
                pairs.append((a, b, float(v)))
            except (TypeError, ValueError):
                continue
    pairs.sort(key=lambda x: -abs(x[2]))
    return {"top_pairs": pairs[:limit]}


def _normalize_lists(highlights: list[Any], details: list[Any]) -> tuple[list[str], list[str]]:
    h = [str(x).strip() for x in highlights if str(x).strip()]
    d = [str(x).strip() for x in details if str(x).strip()]
    return h, d


def generate_insights_openai(store: AnalysisStore) -> tuple[list[str], list[str]]:
    """
    Returns (highlights, details). Highlights are short headlines; details are fuller points.
    Also used to populate store.insights as highlights + details for backward compatibility.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        msg = (
            "Set OPENAI_API_KEY to enable LLM-generated insights. "
            "Deterministic metrics and plots are still available."
        )
        return ([], [msg])

    client = OpenAI(api_key=key)
    excerpt = _excerpt_for_llm(store)
    prompt = (
        "You are a senior data scientist. Given the analysis excerpt, return JSON only with:\n"
        '- "highlights": array of 3–5 very short headline bullets (max ~120 characters each) '
        "capturing the main takeaways for an executive.\n"
        '- "details": array of 4–7 fuller sentences or short paragraphs that expand on drivers, '
        "risks, metrics, and recommended next steps. Be actionable, not generic.\n"
        "Do not repeat the exact same wording in highlights and details; details should add depth."
    )
    resp = client.chat.completions.create(
        model=Config.INSIGHTS_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": excerpt},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    text = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(text)
        highlights, details = _normalize_lists(
            data.get("highlights", []),
            data.get("details", []),
        )
        if highlights or details:
            return highlights, details
        # Legacy shape
        legacy = data.get("insights", [])
        if isinstance(legacy, list) and legacy:
            lines = [str(x).strip() for x in legacy if str(x).strip()]
            if not lines:
                return [], ["No insights returned."]
            return lines[: min(3, len(lines))], lines
    except json.JSONDecodeError:
        pass
    return [], ["Could not parse insight response; check API configuration."]
