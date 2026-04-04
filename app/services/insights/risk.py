from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_deterministic_risks(
    df: pd.DataFrame,
    target: str,
    task: str,
) -> list[dict[str, Any]]:
    """Flags for imbalance, missingness, potential target leakage via correlation."""
    risks: list[dict[str, Any]] = []
    miss = df.isna().mean() * 100
    for col, pct in miss.items():
        if pct > 30:
            risks.append(
                {
                    "type": "high_missingness",
                    "column": col,
                    "detail": f"Column '{col}' has {pct:.1f}% missing values.",
                }
            )

    if task == "classification":
        vc = df[target].value_counts()
        if len(vc) > 1:
            ratio = float(vc.max() / vc.min())
            if ratio > 3:
                risks.append(
                    {
                        "type": "class_imbalance",
                        "detail": f"Class imbalance: max/min count ratio is {ratio:.2f}.",
                    }
                )

    num = df.select_dtypes(include=[np.number])
    if target in num.columns and num.shape[1] >= 2:
        for col in num.columns:
            if col == target:
                continue
            c = num[[col, target]].dropna()
            if len(c) < 10:
                continue
            corr = c[col].corr(c[target])
            if corr is not None and abs(corr) >= 0.999:
                risks.append(
                    {
                        "type": "possible_leakage",
                        "column": col,
                        "detail": f"Very high correlation ({corr:.4f}) with target — possible leakage.",
                    }
                )

    return risks
