from __future__ import annotations

import pandas as pd


def drop_missing_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df.dropna(subset=[target]).reset_index(drop=True)


def separate_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def infer_task(y: pd.Series) -> str:
    """Infer classification vs regression."""
    if pd.api.types.is_float_dtype(y):
        return "regression"
    if pd.api.types.is_integer_dtype(y):
        nu = y.nunique()
        if nu > 50:
            return "regression"
        return "classification"
    # object, bool, category
    return "classification"
