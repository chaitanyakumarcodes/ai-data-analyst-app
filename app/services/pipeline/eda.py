from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_summary(df: pd.DataFrame) -> dict[str, Any]:
    num = df.select_dtypes(include=[np.number])
    summary: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_pct": {k: float(v) for k, v in (df.isna().mean() * 100).items()},
    }
    if len(num.columns) > 0:
        summary["numeric_describe"] = num.describe().to_dict()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        summary.setdefault("categorical_top", {})[col] = (
            df[col].value_counts(dropna=True).head(10).to_dict()
        )
    return summary


def correlation_matrix_numeric(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return {}
    corr = num.corr(numeric_only=True)
    return {
        str(i): {str(c): float(corr.loc[i, c]) for c in corr.columns}
        for i in corr.index
    }
