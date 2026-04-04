from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from app.config import Config, ensure_directories
from app.jinja_filters import highlight_numerics


def create_app() -> Flask:
    load_dotenv()
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.config.from_object(Config)
    app.jinja_env.filters["highlight_numerics"] = highlight_numerics

    app_root = Path(__file__).resolve().parent.parent
    ensure_directories(app_root)

    from app.routes.chat import bp as chat_bp
    from app.routes.main import bp as main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp, url_prefix="/api")

    logging.basicConfig(level=logging.INFO)
    return app
