"""Single-page landscape PDF report (metrics + charts) for dashboard export."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.config import Config
from app.services.dashboard_presenter import metric_cards, ordered_plots
from app.services.memory.analysis_store import AnalysisStore

logger = logging.getLogger(__name__)

METRICS_PER_ROW = 4
METRIC_ROW_H = 24
MAX_METRICS = 8


def _safe_pdf_text(s: str) -> str:
    return (
        str(s)
        .replace("\u00b2", "2")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def _plot_file(app_root: Path, session_id: str, key: str) -> Path | None:
    p = app_root / Config.GENERATED_FOLDER / session_id / f"{key}.png"
    return p if p.is_file() else None


def _wrap_line(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        test = (" ".join(cur + [w])) if cur else w
        if len(test) <= max_chars:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w] if len(w) <= max_chars else [w[: max_chars - 3] + "..."]
    if cur:
        lines.append(" ".join(cur))
    return lines[:3]


def render_dashboard_pdf(store: AnalysisStore, app_root: Path) -> bytes:
    """One landscape A4 page: header, KPIs, metrics, charts, top features, highlights."""
    buf = BytesIO()
    W, H = landscape(A4)
    c = canvas.Canvas(buf, pagesize=(W, H))
    margin = 36
    inner_w = W - 2 * margin

    # --- Header bar ---
    bar_h = 52
    c.setFillColor(colors.HexColor("#0c4a6e"))
    c.rect(0, H - bar_h, W, bar_h, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, H - bar_h + 18, "Analysis report")
    c.setFont("Helvetica", 9)
    gen_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sub = (
        f"{_safe_pdf_text(store.upload_filename or 'dataset.csv')}  |  "
        f"Target: {_safe_pdf_text(store.target)}  |  "
        f"{store.task.title()}  |  {gen_at}"
    )
    if len(sub) > 118:
        sub = sub[:115] + "..."
    c.drawString(margin, H - bar_h + 4, sub)

    # yb = vertical baseline from bottom (ReportLab convention)
    yb = H - bar_h - 20

    # --- KPI line ---
    c.setFillColor(colors.HexColor("#1e293b"))
    c.setFont("Helvetica-Bold", 8)
    rows_n = store.summary.get("n_rows", "—")
    cols_n = store.summary.get("n_columns", "—")
    kpi_parts = [
        f"Task: {store.task}",
        f"Target: {_safe_pdf_text(str(store.target))[:32]}",
        f"Rows: {rows_n}",
        f"Columns: {cols_n}",
    ]
    col_w = inner_w / max(len(kpi_parts), 1)
    for i, part in enumerate(kpi_parts):
        c.drawString(margin + i * col_w + 2, yb, _safe_pdf_text(part)[:44])
    yb -= 26

    # --- Metrics ---
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(margin, yb, "Model metrics")
    yb -= 16

    cards = metric_cards(store.model_metrics)[:MAX_METRICS]
    n_m = len(cards)
    n_metric_rows = max(1, (n_m + METRICS_PER_ROW - 1) // METRICS_PER_ROW)
    cell_w = inner_w / METRICS_PER_ROW
    metrics_row0_y = yb

    for idx, card in enumerate(cards):
        r, col = divmod(idx, METRICS_PER_ROW)
        line_y = metrics_row0_y - r * METRIC_ROW_H
        x = margin + col * cell_w + 4
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(colors.HexColor("#334155"))
        c.drawString(x, line_y, _safe_pdf_text(card["label"])[:30])
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#0369a1"))
        c.drawString(x, line_y - 11, _safe_pdf_text(str(card["value"]))[:22])
        c.setFillColor(colors.HexColor("#334155"))

    yb = metrics_row0_y - n_metric_rows * METRIC_ROW_H - 12

    # --- Charts ---
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(margin, yb, "Charts")
    yb -= 10

    plot_items = ordered_plots(store.plot_paths)
    n_plots = len(plot_items)
    footer_reserve = 88
    chart_title_h = 14
    chart_bot = margin + footer_reserve
    chart_h_max = max(72, min(158, yb - chart_bot - chart_title_h))

    if n_plots == 0:
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColor(colors.HexColor("#94a3b8"))
        c.drawString(margin, yb - 20, "No charts available for this session.")
        yb = chart_bot + footer_reserve - 40
    else:
        gap = 8
        slot_w = (inner_w - gap * (n_plots - 1)) / n_plots
        title_y = yb
        img_top = title_y - chart_title_h
        img_bottom = img_top - chart_h_max

        for i, item in enumerate(plot_items):
            key = item["key"]
            title = _safe_pdf_text(item["title"])[:36]
            x0 = margin + i * (slot_w + gap)
            cx = x0 + slot_w / 2
            c.setFont("Helvetica-Bold", 7)
            c.setFillColor(colors.HexColor("#475569"))
            c.drawCentredString(cx, title_y - 2, title)

            path = _plot_file(app_root, store.session_id, key)
            if path is None:
                c.setFont("Helvetica-Oblique", 7)
                c.drawString(x0 + 4, (img_top + img_bottom) / 2, "(missing)")
                continue
            try:
                ir = ImageReader(str(path))
                iw, ih = ir.getSize()
                scale = min((slot_w - 6) / iw, chart_h_max / ih)
                dw, dh = iw * scale, ih * scale
                ix = x0 + (slot_w - dw) / 2
                iy = img_bottom + (chart_h_max - dh) / 2
                c.drawImage(ir, ix, iy, width=dw, height=dh, mask="auto")
            except Exception as e:  # noqa: BLE001
                logger.warning("PDF chart embed failed %s: %s", path, e)
                c.setFont("Helvetica-Oblique", 7)
                c.drawString(x0 + 4, (img_top + img_bottom) / 2, "(load error)")

        yb = img_bottom - 10

    # --- Top features ---
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(margin, yb, "Top features (importance)")
    yb -= 12
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#475569"))
    top_fi = sorted(store.feature_importance.items(), key=lambda x: -x[1])[:5]
    for j, (name, score) in enumerate(top_fi, 1):
        line = f"{j}. {_safe_pdf_text(name)[:40]} — {score:.4f}"
        c.drawString(margin + 4, yb, line)
        yb -= 10
        if yb < margin + 6:
            break

    yb -= 4
    if yb > margin + 8 and store.insight_highlights:
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.drawString(margin, yb, "Highlights")
        yb -= 10
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#64748b"))
        for line in store.insight_highlights[:2]:
            for chunk in _wrap_line(_safe_pdf_text(line), 105):
                if yb < margin:
                    break
                c.drawString(margin + 4, yb, chunk)
                yb -= 9

    c.showPage()
    c.save()
    return buf.getvalue()
