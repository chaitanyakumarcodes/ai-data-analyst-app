from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from app.config import Config
from app.services.chat.tools import TOOL_DEFINITIONS, run_tool
from app.services.memory.analysis_store import AnalysisStore, get_store


def run_chat_turn(session_id: str, user_message: str) -> str:
    store = get_store(session_id)
    if store is None:
        return "No analysis found for this session. Upload a dataset and run analysis first."
    if store.error:
        return f"Analysis error: {store.error}"

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return (
            "OPENAI_API_KEY is not set. Configure it to use the chat agent. "
            "You can still view the dashboard for deterministic results."
        )

    client = OpenAI(api_key=key)
    context = json.dumps(store.to_public_dict(), default=str)[:8000]
    system = (
        "You are a data analyst assistant. Answer using tools only for factual "
        "claims about the data and model. After tools return, explain clearly to "
        "the user. If a tool returns an error, say so. Do not invent metrics. "
        "Format answers with Markdown: use ## or ### headings, numbered or bullet "
        "lists, and **bold** for key terms and figures so they display cleanly in chat."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {
            "role": "system",
            "content": f"Session analysis snapshot (may be partial):\n{context}",
        },
        {"role": "user", "content": user_message},
    ]

    max_rounds = 6
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.3,
        )
        choice = resp.choices[0]
        msg = choice.message

        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                name = tc.function.name
                args = tc.function.arguments or "{}"
                result = run_tool(store, name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result[:15000],
                    }
                )
            continue

        return (msg.content or "").strip() or "No response."

    return "Stopped after maximum tool rounds; try a simpler question."
