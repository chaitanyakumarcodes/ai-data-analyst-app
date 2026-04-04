from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from app.config import Config


def load_csv_from_path(path: Path) -> tuple[pd.DataFrame, str | None]:
    """Load CSV with encoding fallbacks. Returns (df, error_message)."""
    encodings = ("utf-8-sig", "utf-8", "latin-1", "cp1252")
    last_err: str | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            if len(df) > Config.MAX_ROWS:
                return (
                    pd.DataFrame(),
                    f"Dataset exceeds maximum of {Config.MAX_ROWS} rows.",
                )
            return df, None
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
    return pd.DataFrame(), last_err or "Failed to read CSV."


def load_csv_from_stream(stream: BinaryIO) -> tuple[pd.DataFrame, str | None]:
    raw = stream.read()
    encodings = ("utf-8-sig", "utf-8", "latin-1", "cp1252")
    last_err: str | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
            if len(df) > Config.MAX_ROWS:
                return (
                    pd.DataFrame(),
                    f"Dataset exceeds maximum of {Config.MAX_ROWS} rows.",
                )
            return df, None
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
    return pd.DataFrame(), last_err or "Failed to read CSV."


def validate_target(df: pd.DataFrame, target: str) -> str | None:
    if not target or target.strip() == "":
        return "Target column is required."
    if target not in df.columns:
        return f"Target column '{target}' not found in dataset."
    return None
