from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.services.chat.orchestrator import run_chat_turn

bp = Blueprint("chat_api", __name__)


@bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    message = (data.get("message") or "").strip()
    if not session_id or not message:
        return jsonify({"error": "session_id and message required"}), 400
    reply = run_chat_turn(session_id, message)
    return jsonify({"reply": reply})
