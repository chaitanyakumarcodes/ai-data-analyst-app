from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def ensure_plot_dir(session_id: str, root: Path) -> Path:
    d = root / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_target_distribution(
    y: pd.Series, task: str, out_path: Path, title: str = "Target distribution"
) -> str:
    plt.figure(figsize=(8, 4))
    if task == "regression":
        sns.histplot(pd.to_numeric(y, errors="coerce").dropna(), kde=True)
    else:
        y.value_counts().head(30).plot(kind="bar")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return str(out_path.as_posix())


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path, max_cols: int = 20) -> str | None:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None
    cols = num.columns.tolist()[:max_cols]
    sub = num[cols]
    plt.figure(figsize=(10, 8))
    sns.heatmap(sub.corr(), annot=len(cols) <= 12, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (numeric features)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return str(out_path.as_posix())


def plot_feature_importance(
    feature_importance: dict[str, float], out_path: Path, top_n: int = 15
) -> str:
    items = sorted(feature_importance.items(), key=lambda x: -x[1])[:top_n]
    names = [k[:40] for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(9, 5))
    sns.barplot(x=vals, y=names, orient="h")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} feature importances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return str(out_path.as_posix())
