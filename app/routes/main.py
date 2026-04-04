from __future__ import annotations

import uuid
from pathlib import Path

import logging

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from app.config import Config
from app.services.analysis_service import run_full_analysis
from app.services.dashboard_presenter import metric_cards, ordered_plots
from app.services.memory.analysis_store import get_store
from app.services.pdf_report import render_dashboard_pdf

bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in Config.ALLOWED_EXTENSIONS


def _upload_template_context() -> dict:
    return {
        "max_upload_bytes": Config.MAX_CONTENT_LENGTH,
        "max_upload_mb": Config.MAX_CONTENT_LENGTH // (1024 * 1024),
    }


@bp.route("/")
def index():
    return render_template("landing.html")


@bp.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            flash("No file selected.", "danger")
            return render_template("upload.html", **_upload_template_context()), 400
        if not _allowed(f.filename):
            flash("Only CSV files are allowed.", "danger")
            return render_template("upload.html", **_upload_template_context()), 400
        session_id = str(uuid.uuid4())
        root = Path(current_app.root_path).parent
        up = root / Config.UPLOAD_FOLDER / session_id
        up.mkdir(parents=True, exist_ok=True)
        name = secure_filename(f.filename)
        path = up / "data.csv"
        f.save(path)
        return redirect(url_for("main.preview", session_id=session_id, filename=name))
    return render_template("upload.html", **_upload_template_context())


@bp.route("/preview/<session_id>")
def preview(session_id: str):
    filename = request.args.get("filename", "")
    root = Path(current_app.root_path).parent
    path = root / Config.UPLOAD_FOLDER / session_id / "data.csv"
    if not path.is_file():
        abort(404)
    import pandas as pd

    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:  # noqa: BLE001
        flash("Could not preview CSV.", "danger")
        return redirect(url_for("main.upload"))
    def _column_kind(series: "pd.Series") -> str:
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "text"

    column_rows = [{"name": c, "kind": _column_kind(df[c])} for c in df.columns]

    return render_template(
        "preview.html",
        session_id=session_id,
        column_rows=column_rows,
        filename=filename,
        preview_html=df.to_html(
            classes="table table-hover table-striped table-sm mb-0",
            index=False,
        ),
    )


@bp.route("/analyze", methods=["POST"])
def analyze():
    session_id = request.form.get("session_id", "").strip()
    target = request.form.get("target", "").strip()
    filename = request.form.get("filename", "")
    if not session_id or not target:
        flash("Session and target are required.", "danger")
        return redirect(url_for("main.upload"))
    root = Path(current_app.root_path).parent
    csv_path = root / Config.UPLOAD_FOLDER / session_id / "data.csv"
    if not csv_path.is_file():
        abort(404)
    run_full_analysis(session_id, csv_path, target, filename or csv_path.name, root)
    return redirect(url_for("main.dashboard", session_id=session_id))


@bp.route("/dashboard/<session_id>")
def dashboard(session_id: str):
    store = get_store(session_id)
    if store is None:
        flash("No analysis for this session. Upload again.", "warning")
        return redirect(url_for("main.upload"))
    top_fi = sorted(store.feature_importance.items(), key=lambda x: -x[1])[:20]
    return render_template(
        "dashboard.html",
        store=store,
        session_id=session_id,
        top_fi=top_fi,
        metric_cards=metric_cards(store.model_metrics),
        plot_items=ordered_plots(store.plot_paths),
    )


@bp.route("/dashboard/<session_id>/report.pdf")
def dashboard_report_pdf(session_id: str):
    store = get_store(session_id)
    if store is None:
        flash("No analysis for this session.", "warning")
        return redirect(url_for("main.upload"))
    if store.error:
        flash("Fix analysis errors before exporting a report.", "warning")
        return redirect(url_for("main.dashboard", session_id=session_id))
    root = Path(current_app.root_path).parent
    try:
        data = render_dashboard_pdf(store, root)
    except Exception:  # noqa: BLE001
        logger.exception("PDF report generation failed")
        flash("Could not generate the PDF report.", "danger")
        return redirect(url_for("main.dashboard", session_id=session_id))
    fname = f"analysis-report-{session_id[:8]}.pdf"
    return Response(
        data,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{fname}"',
            "Cache-Control": "no-store",
        },
    )


@bp.route("/chat/<session_id>")
def chat_page(session_id: str):
    store = get_store(session_id)
    if store is None:
        flash("No analysis for this session.", "warning")
        return redirect(url_for("main.upload"))
    return render_template("chat.html", session_id=session_id, store=store)
