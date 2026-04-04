import os
from pathlib import Path


class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    UPLOAD_FOLDER = Path(os.environ.get("UPLOAD_FOLDER", "uploads"))
    GENERATED_FOLDER = Path("app/static/generated")
    ALLOWED_EXTENSIONS = {".csv"}
    MAX_ROWS = 100_000
    MAX_CHAT_ROWS_RETURN = 100
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    INSIGHTS_MODEL = os.environ.get("INSIGHTS_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))


def ensure_directories(app_root: Path) -> None:
    cfg = Config
    (app_root / cfg.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    (app_root / cfg.GENERATED_FOLDER).mkdir(parents=True, exist_ok=True)
